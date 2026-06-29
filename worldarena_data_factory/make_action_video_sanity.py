#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PRIORITY_TASK_FAMILIES = [
    "pick_place",
    "object_to_container",
    "button_press_click",
    "stacking",
    "articulated_open_close",
    "tool_use",
    "lifting",
]

CSV_FIELDS = [
    "episode_id",
    "task_family",
    "robotwin_task_name",
    "video_path",
    "action_ee16_path",
    "action_joint14_path",
    "T_action",
    "N_video_frames",
    "fps",
    "action_video_length_ratio",
    "max_ee_motion_energy",
    "mean_ee_motion_energy",
    "gripper_transition_count",
    "visual_motion_mean",
    "visual_motion_peak",
    "action_visual_peak_time_gap",
    "projection_status",
    "output_overview_sheet",
    "output_action_peak_sheet",
    "output_timeline_plot",
    "output_overlay_sheet",
    "output_preview_video",
]


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return default


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported manifest format: {path}")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_errors(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        fields = ["episode_id", "stage", "error"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def video_meta(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    meta = {
        "readable": False,
        "fps": 0.0,
        "frame_count": 0,
        "width": 0,
        "height": 0,
    }
    if cap.isOpened():
        meta.update(
            {
                "readable": True,
                "fps": safe_float(cap.get(cv2.CAP_PROP_FPS), 24.0) or 24.0,
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            }
        )
    cap.release()
    return meta


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def action_t_to_frame(action_t: int, T: int, frame_count: int) -> int:
    if T <= 1 or frame_count <= 1:
        return 0
    return int(round(float(action_t) / float(T - 1) * float(frame_count - 1)))


def load_ee16(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError(f"invalid ee16 shape: {arr.shape}")
    return arr.astype(np.float32)


def load_joint14(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 14:
        return None
    return arr.astype(np.float32)


def compute_ee16_metrics(ee16: np.ndarray) -> dict[str, Any]:
    T = int(ee16.shape[0])
    left_delta = np.zeros(T, dtype=np.float32)
    right_delta = np.zeros(T, dtype=np.float32)
    grip_transition = np.zeros(T, dtype=np.float32)
    if T > 1:
        left_delta[1:] = np.linalg.norm(np.diff(ee16[:, 0:3], axis=0), axis=1)
        right_delta[1:] = np.linalg.norm(np.diff(ee16[:, 8:11], axis=0), axis=1)
        grip_transition[1:] = np.abs(np.diff(ee16[:, 7])) + np.abs(np.diff(ee16[:, 15]))
    ee_motion_energy = left_delta + right_delta
    action_peak_score = ee_motion_energy + 0.2 * grip_transition
    return {
        "left_delta": left_delta,
        "right_delta": right_delta,
        "ee_motion_energy": ee_motion_energy,
        "left_gripper": ee16[:, 7],
        "right_gripper": ee16[:, 15],
        "gripper_transition": grip_transition,
        "action_peak_score": action_peak_score,
        "gripper_transition_count": int(np.sum(grip_transition > 0.15)),
    }


def compute_joint14_metrics(
    joint14: np.ndarray | None, T: int
) -> dict[str, np.ndarray]:
    zeros = np.zeros(T, dtype=np.float32)
    if joint14 is None or joint14.shape[0] < 2:
        return {
            "joint_delta_l2": zeros,
            "left_joint_delta": zeros,
            "right_joint_delta": zeros,
            "joint_gripper_transition": zeros,
        }
    n = min(T, joint14.shape[0])
    arr = joint14[:n].astype(np.float32)
    out = {
        k: zeros.copy()
        for k in [
            "joint_delta_l2",
            "left_joint_delta",
            "right_joint_delta",
            "joint_gripper_transition",
        ]
    }
    delta = np.diff(arr, axis=0)
    out["joint_delta_l2"][1:n] = np.linalg.norm(delta, axis=1)
    out["left_joint_delta"][1:n] = np.linalg.norm(delta[:, :6], axis=1)
    out["right_joint_delta"][1:n] = np.linalg.norm(delta[:, 7:13], axis=1)
    out["joint_gripper_transition"][1:n] = np.abs(np.diff(arr[:, 6])) + np.abs(
        np.diff(arr[:, 13])
    )
    return out


def select_action_peak_steps(metrics: dict[str, Any], top_k: int) -> list[int]:
    score = np.asarray(metrics["action_peak_score"], dtype=np.float32)
    gripper = np.asarray(metrics["gripper_transition"], dtype=np.float32)
    chosen: list[int] = []
    for idx in np.argsort(-gripper).tolist():
        if gripper[idx] <= 0.15:
            break
        chosen.append(int(idx))
        if len(chosen) >= max(2, top_k // 3):
            break
    for idx in np.argsort(-score).tolist():
        chosen.append(int(idx))
        if len(set(chosen)) >= top_k:
            break
    return sorted(set(chosen))[:top_k]


def draw_labeled_sheet(
    frames: list[dict[str, Any]],
    out_path: Path,
    title: str,
    cols: int = 4,
    thumb_size: tuple[int, int] = (260, 195),
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h = thumb_size
    label_h = 48
    header_h = 28
    if not frames:
        img = Image.new("RGB", (640, 160), "white")
        ImageDraw.Draw(img).text((20, 70), f"{title}: no frames", fill=(0, 0, 0))
        img.save(out_path, quality=92)
        return
    rows = int(math.ceil(len(frames) / cols))
    sheet = Image.new(
        "RGB", (cols * thumb_w, header_h + rows * (thumb_h + label_h)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    for i, item in enumerate(frames):
        x = (i % cols) * thumb_w
        y = header_h + (i // cols) * (thumb_h + label_h)
        im = Image.fromarray(item["frame"]).convert("RGB")
        im.thumbnail((thumb_w, thumb_h))
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        canvas.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
        sheet.paste(canvas, (x, y))
        for j, text in enumerate(item.get("labels", [])[:3]):
            draw.text((x + 4, y + 4 + j * 13), text, fill=(0, 0, 0), font=font)
    sheet.save(out_path, quality=92)


def make_overview_sheet(video_path: Path, out_path: Path, num_frames: int = 12):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    if cap.isOpened():
        fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 24.0) or 24.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for idx in (
            np.linspace(0, max(0, count - 1), min(num_frames, max(1, count)))
            .astype(int)
            .tolist()
        ):
            frame = read_frame(cap, idx)
            if frame is not None:
                frames.append(
                    {
                        "frame": frame,
                        "labels": [f"frame {idx} / {idx / fps:.2f}s"],
                    }
                )
    cap.release()
    draw_labeled_sheet(frames, out_path, "overview")


def make_action_peak_sheet(
    row: pd.Series,
    video_path: Path,
    out_path: Path,
    ee16: np.ndarray,
    ee_metrics: dict[str, Any],
    peak_steps: list[int],
):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    if cap.isOpened():
        fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 24.0) or 24.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        T = int(ee16.shape[0])
        for t in peak_steps:
            frame_idx = action_t_to_frame(t, T, count)
            frame = read_frame(cap, frame_idx)
            if frame is None:
                continue
            frames.append(
                {
                    "frame": frame,
                    "labels": [
                        f"{safe_str(row.get('episode_id'))} {safe_str(row.get('task_family'))}",
                        f"t={t} frame={frame_idx} {frame_idx / fps:.2f}s",
                        (
                            f"Ld={ee_metrics['left_delta'][t]:.4f} "
                            f"Rd={ee_metrics['right_delta'][t]:.4f} "
                            f"Lg={ee_metrics['left_gripper'][t]:.2f} "
                            f"Rg={ee_metrics['right_gripper'][t]:.2f}"
                        ),
                    ],
                }
            )
    cap.release()
    draw_labeled_sheet(frames, out_path, "action peaks")


def compute_visual_motion(video_path: Path, max_frames: int = 240) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"x": np.asarray([]), "energy": np.asarray([]), "mean": 0.0, "peak": 0.0}
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count < 2:
        cap.release()
        return {"x": np.asarray([]), "energy": np.asarray([]), "mean": 0.0, "peak": 0.0}
    indices = np.linspace(0, count - 1, min(max_frames, count)).astype(int).tolist()
    prev = None
    xs = []
    energy = []
    for idx in indices:
        frame = read_frame(cap, idx)
        if frame is None:
            continue
        gray = cv2.cvtColor(
            cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA),
            cv2.COLOR_RGB2GRAY,
        )
        if prev is not None:
            diff = cv2.absdiff(gray, prev)
            xs.append(float(idx) / float(max(count - 1, 1)))
            energy.append(float(np.mean(diff)))
        prev = gray
    cap.release()
    arr = np.asarray(energy, dtype=np.float32)
    return {
        "x": np.asarray(xs, dtype=np.float32),
        "energy": arr,
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "peak": float(np.max(arr)) if len(arr) else 0.0,
    }


def normalize_curve(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == 0:
        return arr
    denom = float(np.max(arr) - np.min(arr))
    if denom < 1e-8:
        return np.zeros_like(arr)
    return (arr - np.min(arr)) / denom


def make_timeline_plot(
    out_path: Path,
    ee_metrics: dict[str, Any],
    joint_metrics: dict[str, np.ndarray],
    visual_motion: dict[str, Any],
    peak_steps: list[int],
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = len(ee_metrics["ee_motion_energy"])
    x_action = np.linspace(0.0, 1.0, max(T, 1))
    plt.figure(figsize=(11, 4.8))
    plt.plot(
        x_action,
        normalize_curve(ee_metrics["ee_motion_energy"]),
        label="ee_motion_energy",
    )
    if np.max(joint_metrics["joint_delta_l2"]) > 0:
        plt.plot(
            x_action,
            normalize_curve(joint_metrics["joint_delta_l2"][:T]),
            label="joint_delta_l2",
            alpha=0.85,
        )
    plt.plot(
        x_action,
        normalize_curve(ee_metrics["left_gripper"]),
        label="left_gripper",
        alpha=0.7,
    )
    plt.plot(
        x_action,
        normalize_curve(ee_metrics["right_gripper"]),
        label="right_gripper",
        alpha=0.7,
    )
    if len(visual_motion["energy"]):
        plt.plot(
            visual_motion["x"],
            normalize_curve(visual_motion["energy"]),
            label="visual_motion_energy",
            alpha=0.9,
        )
    for t in peak_steps:
        plt.axvline(
            float(t) / float(max(T - 1, 1)), color="red", alpha=0.16, linewidth=1
        )
    plt.xlabel("normalized time")
    plt.ylabel("normalized signal")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.2)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def as_matrix(obj: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for key in ["matrix", "intrinsic", "extrinsic", "K", "camera_matrix", "value"]:
            if key in obj:
                mat = as_matrix(obj[key], shape)
                if mat is not None:
                    return mat
    try:
        arr = np.asarray(obj, dtype=np.float32)
        if arr.shape == shape:
            return arr
    except Exception:
        return None
    return None


def project_points(
    xyz: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray | None
) -> np.ndarray | None:
    pts = xyz.astype(np.float32)
    if extrinsic is not None:
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        homo = np.concatenate([pts, ones], axis=1)
        pts = (extrinsic @ homo.T).T[:, :3]
    z = pts[:, 2]
    valid = z > 1e-5
    if not np.any(valid):
        return None
    uvw = (intrinsic @ pts.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
    uv[~valid] = np.nan
    return uv


def make_projection_overlay(
    row: pd.Series,
    ee16: np.ndarray,
    out_path: Path,
    skip_projection: bool,
) -> str:
    if skip_projection:
        return "skipped_by_arg"
    intrinsic_path = Path(safe_str(row.get("intrinsic_path")))
    extrinsic_path = Path(safe_str(row.get("extrinsic_path")))
    first_frame_path = Path(
        safe_str(row.get("first_frame_320x240_path") or row.get("first_frame_path"))
    )
    if not intrinsic_path.exists() or not first_frame_path.exists():
        return "skipped_missing_camera"
    intrinsic_json = load_json(intrinsic_path)
    extrinsic_json = load_json(extrinsic_path)
    if (
        "fallback" in str(intrinsic_json).lower()
        or "fallback" in str(extrinsic_json).lower()
    ):
        return "skipped_fallback_camera"
    K = as_matrix(intrinsic_json, (3, 3))
    E = as_matrix(extrinsic_json, (4, 4))
    if K is None:
        return "failed"
    try:
        base = Image.open(first_frame_path).convert("RGB")
        scale_x = base.width / 640.0 if base.width <= 320 else 1.0
        scale_y = base.height / 480.0 if base.height <= 240 else 1.0
        K = K.copy()
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        left = project_points(ee16[:, 0:3], K, E)
        right = project_points(ee16[:, 8:11], K, E)
        if left is None or right is None:
            return "unreliable"
        all_uv = np.concatenate([left, right], axis=0)
        finite = np.isfinite(all_uv).all(axis=1)
        if np.mean(finite) < 0.5:
            return "unreliable"
        inside = (
            (all_uv[:, 0] >= -0.25 * base.width)
            & (all_uv[:, 0] <= 1.25 * base.width)
            & (all_uv[:, 1] >= -0.25 * base.height)
            & (all_uv[:, 1] <= 1.25 * base.height)
            & finite
        )
        if np.mean(inside) < 0.5:
            return "unreliable"
        draw = ImageDraw.Draw(base)
        for uv, color, tag in [(left, (220, 30, 30), "L"), (right, (30, 80, 220), "R")]:
            pts = [
                (float(x), float(y))
                for x, y in uv
                if math.isfinite(float(x)) and math.isfinite(float(y))
            ]
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=2)
                for j, p in enumerate(pts[:: max(1, len(pts) // 12)]):
                    draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=color)
                    if j == 0:
                        draw.text((p[0] + 4, p[1] + 4), tag, fill=color)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        base.save(out_path, quality=92)
        return "reliable"
    except Exception:
        return "failed"


def draw_metric_bar(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    value: float,
    color: tuple[int, int, int],
    label: str,
):
    value = float(max(0.0, min(1.0, value)))
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (70, 70, 70), 1)
    cv2.rectangle(canvas, (x, y), (x + int(w * value), y + h), color, -1)
    cv2.putText(
        canvas,
        label,
        (x, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


def draw_curve(
    canvas: np.ndarray,
    curve: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
):
    x, y, w, h = rect
    values = normalize_curve(curve)
    if len(values) < 2:
        return
    pts = []
    for i, v in enumerate(values):
        px = x + int(round(i / max(len(values) - 1, 1) * (w - 1)))
        py = y + h - 1 - int(round(float(v) * (h - 1)))
        pts.append([px, py])
    cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, color, thickness)


def make_preview_video(
    video_path: Path,
    out_path: Path,
    ee16: np.ndarray,
    ee_metrics: dict[str, Any],
    joint_metrics: dict[str, np.ndarray],
    peak_steps: list[int],
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 24.0) or 24.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    panel_h = 160
    out_h = height + panel_h
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, out_h)
    )
    T = int(ee16.shape[0])
    ee_curve = np.asarray(ee_metrics["ee_motion_energy"], dtype=np.float32)
    left_curve = np.asarray(ee_metrics["left_delta"], dtype=np.float32)
    right_curve = np.asarray(ee_metrics["right_delta"], dtype=np.float32)
    joint_curve = np.asarray(
        joint_metrics.get("joint_delta_l2", np.zeros(T)), dtype=np.float32
    )[:T]
    left_grip = np.asarray(ee_metrics["left_gripper"], dtype=np.float32)
    right_grip = np.asarray(ee_metrics["right_gripper"], dtype=np.float32)
    max_ee = max(float(np.max(ee_curve)) if len(ee_curve) else 0.0, 1e-8)
    max_left = max(float(np.max(left_curve)) if len(left_curve) else 0.0, 1e-8)
    max_right = max(float(np.max(right_curve)) if len(right_curve) else 0.0, 1e-8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = int(round(float(idx) / float(max(count - 1, 1)) * float(max(T - 1, 1))))
        t = max(0, min(T - 1, t))
        canvas = np.zeros((out_h, width, 3), dtype=np.uint8)
        canvas[:height] = frame
        canvas[height:] = (28, 28, 28)

        header = (
            f"frame {idx}/{max(count - 1, 0)}  action_t {t}/{max(T - 1, 0)}  "
            f"ee={ee_curve[t]:.5f}  Ld={left_curve[t]:.5f}  Rd={right_curve[t]:.5f}  "
            f"Lg={left_grip[t]:.2f}  Rg={right_grip[t]:.2f}"
        )
        cv2.putText(canvas, header, (12, 24), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            canvas, header, (12, 24), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )

        py = height + 28
        draw_metric_bar(
            canvas, 12, py, 120, 12, ee_curve[t] / max_ee, (0, 210, 255), "EE motion"
        )
        draw_metric_bar(
            canvas,
            150,
            py,
            100,
            12,
            left_curve[t] / max_left,
            (80, 180, 255),
            "L delta",
        )
        draw_metric_bar(
            canvas,
            268,
            py,
            100,
            12,
            right_curve[t] / max_right,
            (255, 160, 80),
            "R delta",
        )
        draw_metric_bar(canvas, 386, py, 90, 12, left_grip[t], (80, 230, 80), "L grip")
        draw_metric_bar(
            canvas, 494, py, 90, 12, right_grip[t], (80, 120, 255), "R grip"
        )

        plot_rect = (12, height + 66, width - 24, 78)
        x0, y0, w, h = plot_rect
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (85, 85, 85), 1)
        for frac in [0.25, 0.5, 0.75]:
            gx = x0 + int(frac * w)
            cv2.line(canvas, (gx, y0), (gx, y0 + h), (52, 52, 52), 1)
        draw_curve(canvas, ee_curve, plot_rect, (0, 220, 255), 2)
        if np.max(joint_curve) > 0:
            draw_curve(canvas, joint_curve, plot_rect, (255, 210, 40), 1)
        draw_curve(canvas, left_grip, plot_rect, (80, 230, 80), 1)
        draw_curve(canvas, right_grip, plot_rect, (80, 120, 255), 1)
        for peak_t in peak_steps:
            px = x0 + int(round(float(peak_t) / float(max(T - 1, 1)) * w))
            cv2.line(canvas, (px, y0), (px, y0 + h), (0, 70, 255), 1)
        cursor_x = x0 + int(round(float(t) / float(max(T - 1, 1)) * w))
        cv2.line(canvas, (cursor_x, y0 - 6), (cursor_x, y0 + h + 6), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            "yellow=EE motion  cyan=joint delta  green=L gripper  blue=R gripper  red=current/action peaks",
            (12, height + panel_h - 12),
            font,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        writer.write(canvas)
        idx += 1
    cap.release()
    writer.release()


def sample_manifest(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.episode_ids:
        ids = {x.strip() for x in args.episode_ids.split(",") if x.strip()}
        return df[df["episode_id"].astype(str).isin(ids)].copy()
    if args.max_samples:
        args.num_samples = min(args.num_samples, args.max_samples)
    rng = random.Random(args.seed)
    selected_indices: list[int] = []
    if args.sample_by == "task_family" and "task_family" in df.columns:
        grouped = {safe_str(k): g for k, g in df.groupby("task_family", dropna=False)}
        for family in PRIORITY_TASK_FAMILIES:
            group = grouped.get(family)
            if group is None or group.empty:
                continue
            take = min(
                args.max_per_task,
                max(1, math.ceil(args.num_samples / max(1, len(grouped)))),
                len(group),
            )
            selected_indices.extend(
                group.sample(
                    take, random_state=args.seed + len(selected_indices)
                ).index.tolist()
            )
            if len(selected_indices) >= args.num_samples:
                break
        remaining = df.drop(index=selected_indices, errors="ignore")
        families = list(remaining.groupby("task_family", dropna=False))
        rng.shuffle(families)
        while len(selected_indices) < args.num_samples and families:
            next_families = []
            for _, group in families:
                group = group.drop(index=selected_indices, errors="ignore")
                if group.empty:
                    continue
                selected_indices.append(
                    group.sample(
                        1, random_state=args.seed + len(selected_indices)
                    ).index[0]
                )
                if len(selected_indices) >= args.num_samples:
                    break
                next_families.append((None, group.iloc[1:]))
            families = [(k, g) for k, g in next_families if not g.empty]
        if len(selected_indices) < args.num_samples and not remaining.empty:
            fill = remaining.drop(index=selected_indices, errors="ignore")
            if not fill.empty:
                selected_indices.extend(
                    fill.sample(
                        min(args.num_samples - len(selected_indices), len(fill)),
                        random_state=args.seed,
                    ).index.tolist()
                )
        return df.loc[selected_indices[: args.num_samples]].copy()
    return df.sample(min(args.num_samples, len(df)), random_state=args.seed).copy()


def process_episode(
    row: pd.Series, out: Path, args: argparse.Namespace
) -> dict[str, Any]:
    episode_id = safe_str(row.get("episode_id"))
    video_path = Path(safe_str(row.get("video_640x480_path") or row.get("video_path")))
    ee16_path = Path(
        safe_str(row.get("action_ee16_raw_path") or row.get("action_ee16_path"))
    )
    joint14_path = Path(
        safe_str(row.get("action_joint14_raw_path") or row.get("action_joint14_path"))
    )
    if not ee16_path.exists():
        raise FileNotFoundError(f"missing_ee16:{ee16_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"missing_video:{video_path}")

    overview_path = out / "overview_sheets" / f"{episode_id}.jpg"
    action_sheet_path = out / "action_peak_sheets" / f"{episode_id}.jpg"
    timeline_path = out / "timeline_plots" / f"{episode_id}.png"
    overlay_path = out / "ee16_overlay_sheets" / f"{episode_id}.jpg"
    preview_path = out / "action_overlay_videos" / f"{episode_id}.mp4"

    meta = video_meta(video_path)
    ee16 = load_ee16(ee16_path)
    joint14 = load_joint14(joint14_path)
    ee_metrics = compute_ee16_metrics(ee16)
    joint_metrics = compute_joint14_metrics(joint14, ee16.shape[0])
    peaks = select_action_peak_steps(ee_metrics, args.action_top_k)
    visual_motion = compute_visual_motion(video_path)

    if not args.resume or not overview_path.exists():
        make_overview_sheet(video_path, overview_path, args.overview_frames)
    if not args.resume or not action_sheet_path.exists():
        make_action_peak_sheet(
            row, video_path, action_sheet_path, ee16, ee_metrics, peaks
        )
    if not args.resume or not timeline_path.exists():
        make_timeline_plot(
            timeline_path, ee_metrics, joint_metrics, visual_motion, peaks
        )
    projection_status = make_projection_overlay(
        row, ee16, overlay_path, args.skip_projection
    )
    if args.make_preview_videos and (not args.resume or not preview_path.exists()):
        make_preview_video(
            video_path, preview_path, ee16, ee_metrics, joint_metrics, peaks
        )

    action_peak_x = float(np.argmax(ee_metrics["action_peak_score"])) / float(
        max(ee16.shape[0] - 1, 1)
    )
    if len(visual_motion["energy"]):
        visual_peak_x = float(
            visual_motion["x"][int(np.argmax(visual_motion["energy"]))]
        )
        peak_gap = abs(action_peak_x - visual_peak_x)
    else:
        peak_gap = math.nan

    return {
        "episode_id": episode_id,
        "task_family": safe_str(row.get("task_family")),
        "robotwin_task_name": safe_str(row.get("robotwin_task_name")),
        "video_path": str(video_path),
        "action_ee16_path": str(ee16_path),
        "action_joint14_path": str(joint14_path) if joint14_path.exists() else "",
        "T_action": int(ee16.shape[0]),
        "N_video_frames": int(meta["frame_count"]),
        "fps": float(meta["fps"]),
        "action_video_length_ratio": float(meta["frame_count"])
        / float(max(ee16.shape[0], 1)),
        "max_ee_motion_energy": (
            float(np.max(ee_metrics["ee_motion_energy"]))
            if len(ee_metrics["ee_motion_energy"])
            else 0.0
        ),
        "mean_ee_motion_energy": (
            float(np.mean(ee_metrics["ee_motion_energy"]))
            if len(ee_metrics["ee_motion_energy"])
            else 0.0
        ),
        "gripper_transition_count": int(ee_metrics["gripper_transition_count"]),
        "visual_motion_mean": float(visual_motion["mean"]),
        "visual_motion_peak": float(visual_motion["peak"]),
        "action_visual_peak_time_gap": peak_gap,
        "projection_status": projection_status,
        "output_overview_sheet": str(overview_path),
        "output_action_peak_sheet": str(action_sheet_path),
        "output_timeline_plot": str(timeline_path),
        "output_overlay_sheet": (
            str(overlay_path) if projection_status == "reliable" else ""
        ),
        "output_preview_video": str(preview_path) if args.make_preview_videos else "",
    }


def write_report(out: Path, rows: list[dict[str, Any]], errors: list[dict[str, Any]]):
    status_counts = {}
    for row in rows:
        status = row.get("projection_status", "")
        status_counts[status] = status_counts.get(status, 0) + 1
    reliable = status_counts.get("reliable", 0)
    fallback = status_counts.get("skipped_fallback_camera", 0)
    report = [
        "# Action-Video Consistency Sanity Check",
        "",
        f"Processed episodes: {len(rows)}",
        f"Errors: {len(errors)}",
        f"Projection reliable: {reliable}",
        f"Projection skipped because of fallback camera: {fallback}",
        "",
        "## Output Paths",
        "",
        f"- Overview sheets: `{out / 'overview_sheets'}`",
        f"- Action peak sheets: `{out / 'action_peak_sheets'}`",
        f"- Timeline plots: `{out / 'timeline_plots'}`",
        f"- EE16 overlays: `{out / 'ee16_overlay_sheets'}`",
        f"- Action overlay videos: `{out / 'action_overlay_videos'}`",
        "",
        "## Projection Status Counts",
        "",
    ]
    for key, value in sorted(status_counts.items()):
        report.append(f"- `{key}`: {value}")
    report += [
        "",
        "## Manual Inspection Checklist",
        "",
        "- Action peak sheet: action peak 时视频里是否发生实际操作。",
        "- Gripper transition: 夹爪开合时是否靠近物体或接触区域。",
        "- Timeline plot: video motion peak 是否和 action peak 大致同步。",
        "- Left/right: 是否疑似左右臂反了。",
        "- Failure pattern: 是否存在 action 很大但视频不动。",
        "- Failure pattern: 是否存在视频里物体动了但 action 没对应。",
    ]
    (out / "action_video_sanity_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=30)
    ap.add_argument(
        "--sample-by", choices=["task_family", "random"], default="task_family"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episode-ids", default="")
    ap.add_argument("--max-per-task", type=int, default=3)
    ap.add_argument("--make-preview-videos", action="store_true")
    ap.add_argument("--skip-projection", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--overview-frames", type=int, default=12)
    ap.add_argument("--action-top-k", type=int, default=12)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    df = read_table(args.manifest).reset_index(drop=True)
    if "episode_id" not in df.columns:
        df["episode_id"] = [f"episode_{i:06d}" for i in range(len(df))]
    sample = sample_manifest(df, args).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for i, row in sample.iterrows():
        episode_id = safe_str(row.get("episode_id"))
        try:
            print(f"[{i + 1}/{len(sample)}] {episode_id}", flush=True)
            rows.append(process_episode(row, out, args))
        except Exception as exc:
            errors.append(
                {
                    "episode_id": episode_id,
                    "stage": "process_episode",
                    "error": repr(exc),
                }
            )
            print(f"[ERROR] {episode_id}: {exc}", flush=True)
    write_csv(out / "action_video_sanity_scores.csv", rows, CSV_FIELDS)
    append_errors(out / "errors.csv", errors)
    write_report(out, rows, errors)
    print(out / "action_video_sanity_report.md", flush=True)
    print(out / "action_video_sanity_scores.csv", flush=True)


if __name__ == "__main__":
    main()
