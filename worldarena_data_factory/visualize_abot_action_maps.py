#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
    "action_path",
    "representation",
    "T_action",
    "N_video_frames",
    "action_map_frames",
    "fps",
    "camera_source",
    "action_map_source",
    "action_map_convention",
    "action_map_convention_path",
    "ee_local_z_offset",
    "use_abot_default_offset",
    "action_map_quat_order",
    "action_map_nonzero_ratio",
    "action_map_bbox_xmin",
    "action_map_bbox_ymin",
    "action_map_bbox_xmax",
    "action_map_bbox_ymax",
    "action_map_out_of_frame_ratio",
    "output_action_map_video",
    "output_overlay_video",
    "output_side_by_side_video",
    "output_contact_sheet",
    "status",
    "error",
]


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported manifest format: {path}")


def read_jsonl_as_manifest(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            video = (
                obj.get("video")
                or obj.get("first_frame_image")
                or obj.get("image")
                or ""
            )
            action = obj.get("action_path") or ""
            rows.append(
                {
                    "episode_id": obj.get("episode_id") or f"readme_a2v_{i:06d}",
                    "task_family": obj.get("task_family") or "readme_a2v_example",
                    "robotwin_task_name": obj.get("robotwin_task_name")
                    or obj.get("prompt", "")[:80],
                    "video_640x480_path": video,
                    "action_ee16_raw_path": action,
                    "action_joint14_raw_path": action,
                    "intrinsic_path": obj.get("intrinsic_path", ""),
                    "extrinsic_path": obj.get("extrinsic_path", ""),
                    "prompt_worldarena_style": obj.get("prompt", ""),
                    "original_size": obj.get("original_size", ""),
                }
            )
    return pd.DataFrame(rows)


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
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode_id", "stage", "error"], extrasaction="ignore"
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def default_action_map_convention_path() -> Path:
    return Path(__file__).resolve().parent / "action_map_conventions" / "robotwin_hdf5_z0.json"


def load_action_map_convention(args: argparse.Namespace) -> dict[str, Any]:
    path = args.action_map_convention or default_action_map_convention_path()
    convention = {}
    if path and Path(path).exists():
        convention = load_json(Path(path)) or {}
        convention["_path"] = str(Path(path))
    elif args.action_map_convention:
        raise FileNotFoundError(f"action map convention not found: {path}")
    offset = convention.get("ee_local_z_offset", args.ee_local_z_offset)
    if args.use_abot_default_offset:
        offset = 0.23
    convention["ee_local_z_offset"] = float(offset)
    convention.setdefault("camera_name", "manifest_camera")
    convention.setdefault("intrinsic_mode", "manifest")
    convention.setdefault("extrinsic_mode", "manifest")
    convention.setdefault("camera_source", "manifest")
    convention.setdefault("quat_order", args.quat_order or "xyzw")
    if args.quat_order:
        convention["quat_order"] = args.quat_order
    convention.setdefault("name", Path(convention.get("_path", "inline_convention")).stem)
    convention.setdefault("_path", "")
    return convention


def video_meta(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    meta = {"readable": False, "fps": 0.0, "frame_count": 0, "width": 0, "height": 0}
    if cap.isOpened():
        meta.update(
            {
                "readable": True,
                "fps": float(cap.get(cv2.CAP_PROP_FPS) or 24.0),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            }
        )
    cap.release()
    return meta


def sample_manifest(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.episode_ids:
        ids = {x.strip() for x in args.episode_ids.split(",") if x.strip()}
        return df[df["episode_id"].astype(str).isin(ids)].copy()
    n = min(args.num_samples, len(df))
    if args.sample_by != "task_family" or "task_family" not in df.columns:
        return df.sample(n, random_state=args.seed).copy()
    selected: list[int] = []
    grouped = {safe_str(k): g for k, g in df.groupby("task_family", dropna=False)}
    for family in PRIORITY_TASK_FAMILIES:
        group = grouped.get(family)
        if group is None or group.empty:
            continue
        take = min(
            args.max_per_task, max(1, math.ceil(n / max(1, len(grouped)))), len(group)
        )
        selected.extend(
            group.sample(take, random_state=args.seed + len(selected)).index.tolist()
        )
        if len(selected) >= n:
            return df.loc[selected[:n]].copy()
    remaining = df.drop(index=selected, errors="ignore")
    for _, group in remaining.groupby("task_family", dropna=False):
        if len(selected) >= n:
            break
        group = group.drop(index=selected, errors="ignore")
        if not group.empty:
            selected.append(
                group.sample(1, random_state=args.seed + len(selected)).index[0]
            )
    if len(selected) < n:
        fill = df.drop(index=selected, errors="ignore")
        if not fill.empty:
            selected.extend(
                fill.sample(
                    min(n - len(selected), len(fill)), random_state=args.seed
                ).index.tolist()
            )
    return df.loc[selected[:n]].copy()


def import_abot_action_utils(abot_root: Path):
    candidates = [abot_root / "inference", abot_root]
    for path in candidates:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from diffsynth.utils.action_utils import (  # type: ignore
        get_vace_traj_maps_with_scaled_intrinsic,
        simple_radius_gen_func,
    )

    return get_vace_traj_maps_with_scaled_intrinsic, simple_radius_gen_func


def default_camera(num_frames: int, height: int, width: int):
    import torch

    fx = fy = width / (2.0 * np.tan(np.radians(30.0)))
    intrinsic = torch.eye(3, dtype=torch.float32)
    intrinsic[0, 0] = float(fx)
    intrinsic[1, 1] = float(fy)
    intrinsic[0, 2] = width / 2.0
    intrinsic[1, 2] = height / 2.0
    extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num_frames, 1, 1)
    return intrinsic, extrinsics, "fallback"


def matrix_from_json_value(obj: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for key in ["matrix", "intrinsic", "extrinsic", "K", "camera_matrix", "value"]:
            if key in obj:
                mat = matrix_from_json_value(obj[key], shape)
                if mat is not None:
                    return mat
        if {"fx", "fy"}.issubset(obj.keys()):
            fx = float(obj["fx"])
            fy = float(obj["fy"])
            cx = float(obj.get("ppx", obj.get("cx", 0.0)))
            cy = float(obj.get("ppy", obj.get("cy", 0.0)))
            return np.asarray(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        if {"rotation_matrix", "translation_vector"}.issubset(obj.keys()) and shape == (
            4,
            4,
        ):
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = np.asarray(obj["rotation_matrix"], dtype=np.float32)
            mat[:3, 3] = np.asarray(
                obj["translation_vector"], dtype=np.float32
            ).reshape(3)
            return mat
    try:
        arr = np.asarray(obj, dtype=np.float32)
        if arr.shape == shape:
            return arr
    except Exception:
        return None
    return None


def load_camera(row: pd.Series, num_frames: int, height: int, width: int):
    import torch

    intrinsic_path = Path(safe_str(row.get("intrinsic_path")))
    extrinsic_path = Path(safe_str(row.get("extrinsic_path")))
    intrinsic_obj = load_json(intrinsic_path)
    extrinsic_obj = load_json(extrinsic_path)
    if (
        "fallback" in str(intrinsic_obj).lower()
        or "fallback" in str(extrinsic_obj).lower()
    ):
        return default_camera(num_frames, height, width)
    if not intrinsic_path.exists() or not extrinsic_path.exists():
        return default_camera(num_frames, height, width)
    K = None
    E = None
    if intrinsic_path.suffix == ".npy":
        K = np.load(intrinsic_path).astype(np.float32)
    else:
        K = matrix_from_json_value(intrinsic_obj, (3, 3))
    if extrinsic_path.suffix == ".npy":
        E = np.load(extrinsic_path).astype(np.float32)
    else:
        if isinstance(extrinsic_obj, list):
            mats = [matrix_from_json_value(x, (4, 4)) for x in extrinsic_obj]
            mats = [x for x in mats if x is not None]
            E = np.stack(mats, axis=0) if mats else None
        else:
            E = matrix_from_json_value(extrinsic_obj, (4, 4))
    if K is None or E is None:
        return default_camera(num_frames, height, width)
    K_t = torch.as_tensor(K, dtype=torch.float32)
    E_arr = np.asarray(E, dtype=np.float32)
    if E_arr.ndim == 2:
        E_arr = np.repeat(E_arr[None], num_frames, axis=0)
    if E_arr.shape[0] < num_frames:
        E_arr = np.concatenate(
            [E_arr, np.repeat(E_arr[-1:], num_frames - E_arr.shape[0], axis=0)], axis=0
        )
    elif E_arr.shape[0] > num_frames:
        E_arr = E_arr[:num_frames]
    return K_t, torch.as_tensor(E_arr, dtype=torch.float32), "camera_runtime_verified"


def load_action(row: pd.Series, representation: str) -> tuple[Path, np.ndarray]:
    if representation == "ee16":
        path = Path(
            safe_str(row.get("action_ee16_raw_path") or row.get("action_ee16_path"))
        )
        expected_dim = 16
    else:
        path = Path(
            safe_str(
                row.get("action_joint14_raw_path") or row.get("action_joint14_path")
            )
        )
        expected_dim = 14
    if not path.exists():
        raise FileNotFoundError(f"missing action: {path}")
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"invalid {representation} shape: {arr.shape}")
    return path, arr


def resample_action_to_video(action: np.ndarray, n_frames: int) -> np.ndarray:
    if n_frames <= 0:
        return action[:0]
    if len(action) == n_frames:
        return action
    idx = np.linspace(0, max(len(action) - 1, 0), n_frames).round().astype(int)
    return action[idx]


def convert_action_quat_order(action: np.ndarray, quat_order: str) -> np.ndarray:
    quat_order = (quat_order or "xyzw").lower()
    if quat_order == "xyzw" or action.shape[1] < 16:
        return action
    if quat_order != "wxyz":
        raise ValueError(f"unsupported quat_order: {quat_order}")
    out = action.copy()
    for start in [3, 11]:
        q = out[:, start : start + 4].copy()
        out[:, start : start + 4] = q[:, [1, 2, 3, 0]]
    return out


def joint14_to_fake_ee16(joint14: np.ndarray) -> np.ndarray:
    # Fallback-only visualization helper. This is not FK; it creates a simple 2D-ish
    # signal map from joint deltas when user explicitly asks for joint14 visualization.
    out = np.zeros((joint14.shape[0], 16), dtype=np.float32)
    left = np.cumsum(joint14[:, :3], axis=0)
    right = np.cumsum(joint14[:, 7:10], axis=0)
    left = left / max(float(np.std(left)), 1e-6) * 0.05 + np.asarray([-0.1, 0.0, 0.8])
    right = right / max(float(np.std(right)), 1e-6) * 0.05 + np.asarray([0.1, 0.0, 0.8])
    out[:, 0:3] = left
    out[:, 3:7] = np.asarray([0, 0, 0, 1], dtype=np.float32)
    out[:, 7] = joint14[:, 6]
    out[:, 8:11] = right
    out[:, 11:15] = np.asarray([0, 0, 0, 1], dtype=np.float32)
    out[:, 15] = joint14[:, 13]
    return out


def generate_abot_action_maps(
    action: np.ndarray,
    representation: str,
    row: pd.Series,
    abot_root: Path,
    n_frames: int,
    height: int,
    width: int,
    ee_local_z_offset: float,
    quat_order: str,
) -> tuple[np.ndarray, str, str]:
    import torch

    if representation == "joint14":
        pose = joint14_to_fake_ee16(resample_action_to_video(action, n_frames))
        source_suffix = "joint14_fake_ee16_fallback"
    else:
        pose = convert_action_quat_order(resample_action_to_video(action, n_frames), quat_order)
        source_suffix = f"ee16:quat={quat_order}"
    K, extrinsics, camera_source = load_camera(row, n_frames, height, width)
    try:
        gen, radius_func = import_abot_action_utils(abot_root)
        pose_t = torch.as_tensor(pose, dtype=torch.float32)
        c2w = extrinsics.unsqueeze(0)
        w2c = torch.linalg.inv(c2w)
        K_v = K.unsqueeze(0)
        cond = gen(
            pose_t,
            w2c,
            c2w,
            K_v,
            (height, width),
            (height, width),
            radius_gen_func=radius_func,
            ee_local_z_offset=ee_local_z_offset,
        )
        rgb = cond[:3, 0].detach().cpu().numpy()  # C,T,H,W in [0,1]
        maps = np.transpose(rgb, (1, 2, 3, 0))
        maps = np.clip(maps * 255.0, 0, 255).astype(np.uint8)
        return (
            maps,
            camera_source,
            f"abot_get_vace_traj_maps_with_scaled_intrinsic:{source_suffix}:z={ee_local_z_offset:g}",
        )
    except Exception:
        maps = fallback_draw_maps(pose, height, width)
        return maps, camera_source, f"fallback_draw:{source_suffix}"


def fallback_draw_maps(pose: np.ndarray, height: int, width: int) -> np.ndarray:
    maps = []
    xs_l = pose[:, 0]
    ys_l = pose[:, 1]
    xs_r = pose[:, 8]
    ys_r = pose[:, 9]
    all_x = np.concatenate([xs_l, xs_r])
    all_y = np.concatenate([ys_l, ys_r])
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    sx = max(x_max - x_min, 1e-6)
    sy = max(y_max - y_min, 1e-6)
    prev_l = prev_r = None
    for i in range(len(pose)):
        img = np.zeros((height, width, 3), dtype=np.uint8) + 50
        pl = (
            int((xs_l[i] - x_min) / sx * (width * 0.75) + width * 0.125),
            int((ys_l[i] - y_min) / sy * (height * 0.75) + height * 0.125),
        )
        pr = (
            int((xs_r[i] - x_min) / sx * (width * 0.75) + width * 0.125),
            int((ys_r[i] - y_min) / sy * (height * 0.75) + height * 0.125),
        )
        if prev_l is not None:
            cv2.line(img, prev_l, pl, (40, 220, 80), 6)
            cv2.line(img, prev_r, pr, (220, 60, 60), 6)
        cv2.circle(img, pl, 7 if pose[i, 7] > 0.5 else 4, (40, 255, 120), -1)
        cv2.circle(img, pr, 7 if pose[i, 15] > 0.5 else 4, (255, 80, 80), -1)
        prev_l, prev_r = pl, pr
        maps.append(img)
    return np.stack(maps, axis=0)


def action_map_stats(maps: np.ndarray) -> dict[str, Any]:
    if maps.size == 0:
        return {
            "action_map_nonzero_ratio": 0.0,
            "action_map_bbox_xmin": -1,
            "action_map_bbox_ymin": -1,
            "action_map_bbox_xmax": -1,
            "action_map_bbox_ymax": -1,
            "action_map_out_of_frame_ratio": 1.0,
        }
    diff = np.max(np.abs(maps.astype(np.int16) - 50), axis=-1)
    mask = diff > 10
    nz = np.argwhere(mask)
    empty_frames = np.mean(mask.reshape(mask.shape[0], -1).sum(axis=1) < 5)
    if len(nz) == 0:
        bbox = (-1, -1, -1, -1)
    else:
        _, ys, xs = nz[:, 0], nz[:, 1], nz[:, 2]
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return {
        "action_map_nonzero_ratio": float(mask.mean()),
        "action_map_bbox_xmin": bbox[0],
        "action_map_bbox_ymin": bbox[1],
        "action_map_bbox_xmax": bbox[2],
        "action_map_bbox_ymax": bbox[3],
        "action_map_out_of_frame_ratio": float(empty_frames),
    }


def put_label(frame: np.ndarray, lines: list[str]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 22
    for line in lines:
        cv2.putText(frame, line, (8, y), font, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (8, y), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18


def write_videos_and_sheet(
    row: pd.Series,
    video_path: Path,
    maps_rgb: np.ndarray,
    out: Path,
    args: argparse.Namespace,
    camera_source: str,
    action_map_source: str,
) -> tuple[str, str, str, str]:
    episode_id = safe_str(row.get("episode_id"))
    map_path = out / "action_maps" / f"{episode_id}.mp4"
    overlay_path = out / "overlays" / f"{episode_id}.mp4"
    sbs_path = out / "side_by_side" / f"{episode_id}.mp4"
    sheet_path = out / "contact_sheets" / f"{episode_id}.jpg"
    for p in [map_path, overlay_path, sbs_path, sheet_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"unreadable video: {video_path}")
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = args.fps or float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width, height = args.width, args.height
    writer_map = cv2.VideoWriter(
        str(map_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    writer_overlay = cv2.VideoWriter(
        str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    writer_sbs = None
    if not args.no_side_by_side:
        writer_sbs = cv2.VideoWriter(
            str(sbs_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 3, height)
        )
    sheet_items = []
    sheet_indices = set(
        np.linspace(0, max(n_video - 1, 0), min(12, max(n_video, 1)))
        .round()
        .astype(int)
        .tolist()
    )
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
        map_idx = min(
            int(round(idx / max(n_video - 1, 1) * max(len(maps_rgb) - 1, 0))),
            len(maps_rgb) - 1,
        )
        amap_rgb = maps_rgb[map_idx]
        amap_rgb = cv2.resize(amap_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        amap_bgr = cv2.cvtColor(amap_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(
            frame_bgr, 1.0 - args.alpha, amap_bgr, args.alpha, 0.0
        )
        label_lines = [
            f"{episode_id} frame={idx} action_t={map_idx}",
            f"{safe_str(row.get('task_family'))} {safe_str(row.get('robotwin_task_name'))}",
            f"camera={camera_source}",
            f"map={action_map_source[:58]}",
        ]
        put_label(amap_bgr, label_lines)
        put_label(overlay, label_lines)
        writer_map.write(amap_bgr)
        writer_overlay.write(overlay)
        if writer_sbs is not None:
            sbs = np.concatenate([frame_bgr, amap_bgr, overlay], axis=1)
            writer_sbs.write(sbs)
        if idx in sheet_indices:
            triplet = np.concatenate([frame_bgr, amap_bgr, overlay], axis=1)
            sheet_items.append(cv2.cvtColor(triplet, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    writer_map.release()
    writer_overlay.release()
    if writer_sbs is not None:
        writer_sbs.release()
    make_contact_sheet(sheet_items, sheet_path, episode_id)
    return (
        str(map_path),
        str(overlay_path),
        str(sbs_path) if not args.no_side_by_side else "",
        str(sheet_path),
    )


def make_contact_sheet(frames_rgb: list[np.ndarray], out_path: Path, title: str):
    cols = 2
    thumb_w, thumb_h = 960, 240
    header = 26
    rows = max(1, int(math.ceil(len(frames_rgb) / cols)))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + header) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text(
        (8, 8), f"{title}: [RGB | action map | overlay]", fill=(0, 0, 0), font=font
    )
    for i, arr in enumerate(frames_rgb):
        im = Image.fromarray(arr).convert("RGB")
        im.thumbnail((thumb_w, thumb_h))
        x = (i % cols) * thumb_w
        y = 30 + (i // cols) * (thumb_h + header)
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        canvas.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
        sheet.paste(canvas, (x, y))
        draw.text((x + 4, y + 4), f"sample {i}", fill=(0, 0, 0), font=font)
    sheet.save(out_path, quality=92)


def process_episode(
    row: pd.Series, out: Path, args: argparse.Namespace
) -> dict[str, Any]:
    episode_id = safe_str(row.get("episode_id"))
    video_path = Path(safe_str(row.get("video_640x480_path") or row.get("video_path")))
    action_path, action = load_action(row, args.representation)
    meta = video_meta(video_path)
    if not meta["readable"]:
        raise ValueError(f"unreadable video: {video_path}")
    n_frames = int(meta["frame_count"])
    maps_rgb, camera_source, map_source = generate_abot_action_maps(
        action,
        args.representation,
        row,
        args.abot_root,
        n_frames,
        args.height,
        args.width,
        args.convention["ee_local_z_offset"],
        args.convention.get("quat_order", "xyzw"),
    )
    stats = action_map_stats(maps_rgb)
    map_video, overlay_video, sbs_video, sheet = write_videos_and_sheet(
        row, video_path, maps_rgb, out, args, camera_source, map_source
    )
    return {
        "episode_id": episode_id,
        "task_family": safe_str(row.get("task_family")),
        "robotwin_task_name": safe_str(row.get("robotwin_task_name")),
        "video_path": str(video_path),
        "action_path": str(action_path),
        "representation": args.representation,
        "T_action": int(action.shape[0]),
        "N_video_frames": n_frames,
        "action_map_frames": int(maps_rgb.shape[0]),
        "fps": float(args.fps or meta["fps"]),
        "camera_source": camera_source,
        "action_map_source": map_source,
        "action_map_convention": args.convention.get("name", ""),
        "action_map_convention_path": args.convention.get("_path", ""),
        "ee_local_z_offset": args.convention["ee_local_z_offset"],
        "use_abot_default_offset": bool(args.use_abot_default_offset),
        "action_map_quat_order": args.convention.get("quat_order", "xyzw"),
        **stats,
        "output_action_map_video": map_video,
        "output_overlay_video": overlay_video,
        "output_side_by_side_video": sbs_video,
        "output_contact_sheet": sheet,
        "status": "ok",
        "error": "",
    }


def write_report(out: Path, rows: list[dict[str, Any]], errors: list[dict[str, Any]], args: argparse.Namespace):
    total = len(rows) + len(errors)
    official = sum(
        str(r.get("action_map_source", "")).startswith("abot_get_vace") for r in rows
    )
    fallback = sum(
        str(r.get("action_map_source", "")).startswith("fallback_draw") for r in rows
    )
    camera_counts: dict[str, int] = {}
    suspicious = []
    for r in rows:
        camera_counts[str(r.get("camera_source", "unknown"))] = (
            camera_counts.get(str(r.get("camera_source", "unknown")), 0) + 1
        )
        flags = []
        if float(r.get("action_map_nonzero_ratio", 0.0)) < 0.0005:
            flags.append("almost_empty_action_map")
        if float(r.get("action_map_out_of_frame_ratio", 0.0)) > 0.8:
            flags.append("mostly_empty_or_out_of_frame")
        xmin, ymin = int(r.get("action_map_bbox_xmin", -1)), int(
            r.get("action_map_bbox_ymin", -1)
        )
        xmax, ymax = int(r.get("action_map_bbox_xmax", -1)), int(
            r.get("action_map_bbox_ymax", -1)
        )
        if xmin >= 0 and (xmax < 80 or ymax < 60 or xmin > 560 or ymin > 420):
            flags.append("bbox_near_corner")
        if flags:
            suspicious.append((r.get("episode_id"), ",".join(flags)))
    lines = [
        "# ABot/VACE Action Map Visualization Report",
        "",
        f"Total requested/processed: {total}",
        f"Successful action map videos: {len(rows)}",
        f"ABot official action map logic: {official}",
        f"Fallback draw: {fallback}",
        "",
        "## Action Map Convention",
        "",
        f"- convention: `{args.convention.get('name', '')}`",
        f"- convention path: `{args.convention.get('_path', '')}`",
        f"- ee_local_z_offset: `{args.convention.get('ee_local_z_offset')}`",
        f"- use_abot_default_offset: `{bool(args.use_abot_default_offset)}`",
        f"- camera_name: `{args.convention.get('camera_name', '')}`",
        f"- intrinsic_mode: `{args.convention.get('intrinsic_mode', '')}`",
        f"- extrinsic_mode: `{args.convention.get('extrinsic_mode', '')}`",
        f"- quat_order: `{args.convention.get('quat_order', '')}`",
        "",
        "## Camera Source Counts",
        "",
    ]
    for key, val in sorted(camera_counts.items()):
        lines.append(f"- `{key}`: {val}")
    lines += [
        "",
        "## Suspicious Action Map Cases",
        "",
    ]
    if suspicious:
        for eid, flag in suspicious[:80]:
            lines.append(f"- `{eid}`: {flag}")
    else:
        lines.append("- none flagged by simple map geometry heuristics")
    lines += [
        "",
        "## Output Paths",
        "",
        f"- action maps: `{out / 'action_maps'}`",
        f"- overlays: `{out / 'overlays'}`",
        f"- side-by-side: `{out / 'side_by_side'}`",
        f"- contact sheets: `{out / 'contact_sheets'}`",
        "",
        "## Manual Inspection Checklist",
        "",
        "- action map 是否落在机械臂/物体附近。",
        "- action map 峰值是否对应视频里的实际操作。",
        "- gripper open/close 变化是否出现在接触附近。",
        "- action map 是否全部偏到桌面外或画面角落。",
        "- 左右臂颜色/轨迹是否疑似反了。",
        "",
        "Note: `camera_source=fallback` means the map is generated with fallback camera parameters; geometry may be unreliable. `--use-abot-default-offset` is required to use the official z_offset=0.23 in this visualization script.",
    ]
    (out / "action_map_vis_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, help="Parquet/CSV episode manifest")
    ap.add_argument(
        "--jsonl-path",
        type=Path,
        help="ABot A2V JSONL metadata, e.g. inference/assets/demo_a2v.jsonl",
    )
    ap.add_argument("--abot-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument(
        "--sample-by", choices=["task_family", "random"], default="task_family"
    )
    ap.add_argument("--representation", choices=["ee16", "joint14"], default="ee16")
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episode-ids", default="")
    ap.add_argument("--max-per-task", type=int, default=3)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no-side-by-side", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--action-map-convention", type=Path, help="JSON convention config. Defaults to action_map_conventions/robotwin_hdf5_z0.json if present.")
    ap.add_argument("--ee-local-z-offset", type=float, default=0.0, help="EE local z offset for ABot/VACE maps. Data-factory default is 0.0.")
    ap.add_argument("--use-abot-default-offset", action="store_true", help="Use official ABot z_offset=0.23 for ablation.")
    ap.add_argument("--quat-order", choices=["wxyz", "xyzw"], help="Source quaternion order in ee16 action. Convention config defaults to wxyz for RoboTwin.")
    args = ap.parse_args()
    args.convention = load_action_map_convention(args)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.jsonl_path:
        df = read_jsonl_as_manifest(args.jsonl_path).reset_index(drop=True)
    elif args.manifest:
        df = read_table(args.manifest).reset_index(drop=True)
    else:
        raise SystemExit("Either --manifest or --jsonl-path is required")
    if "episode_id" not in df.columns:
        df["episode_id"] = [f"episode_{i:06d}" for i in range(len(df))]
    sample = sample_manifest(df, args).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for i, row in sample.iterrows():
        eid = safe_str(row.get("episode_id"))
        try:
            existing = args.out / "side_by_side" / f"{eid}.mp4"
            if args.resume and existing.exists():
                print(
                    f"[{i + 1}/{len(sample)}] {eid} exists, recomputing metadata",
                    flush=True,
                )
            else:
                print(f"[{i + 1}/{len(sample)}] {eid}", flush=True)
            rows.append(process_episode(row, args.out, args))
        except Exception as exc:
            err = {"episode_id": eid, "stage": "process_episode", "error": repr(exc)}
            errors.append(err)
            print(f"[ERROR] {eid}: {exc}", flush=True)
    write_csv(args.out / "action_map_vis_scores.csv", rows, CSV_FIELDS)
    append_errors(args.out / "errors.csv", errors)
    write_report(args.out, rows, errors, args)
    print(args.out / "action_map_vis_report.md", flush=True)
    print(args.out / "action_map_vis_scores.csv", flush=True)


if __name__ == "__main__":
    main()
