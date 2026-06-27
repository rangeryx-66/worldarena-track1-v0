#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


QC_FIELDS = [
    "video_readable", "fps", "width", "height", "frame_count",
    "brightness_mean", "brightness_temporal_std", "contrast_mean",
    "color_shift_score", "temporal_flicker_score", "compression_artifact_score",
    "sharpness_laplacian", "arm_visible_ratio", "left_gripper_visible_ratio",
    "right_gripper_visible_ratio", "end_effector_visible_ratio",
    "contact_region_visible_ratio", "object_motion_without_visible_contact_score",
    "bad_frame_ratio", "motion_score",
    "deterministic_hard_fail", "hard_fail_reason", "action_joint14_valid",
    "action_has_nan", "action_has_inf", "heuristic_candidate_labels", "rule_qc_context",
    "qc_status", "qc_reason",
]


def resolve_manifest(path: Path) -> Path:
    if path.exists():
        return path
    alt = path.parent / "manifests" / path.name
    if alt.exists():
        return alt
    if path.name == "episode_manifest.parquet":
        alt = path / "manifests" / "episode_manifest.parquet"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"manifest not found: {path}")


def read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported manifest format: {path}")


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def validate_action_joint14(row: dict[str, Any]) -> dict[str, Any]:
    path = str(row.get("action_joint14_norm_path") or row.get("action_joint14_raw_path") or "")
    out = {"action_joint14_valid": False, "action_has_nan": False, "action_has_inf": False, "action_reason": "action_missing"}
    if not path or not Path(path).exists():
        return out
    try:
        arr = np.load(path, mmap_mode="r")
        shape = tuple(arr.shape)
        out["action_has_nan"] = bool(np.isnan(arr).any())
        out["action_has_inf"] = bool(np.isinf(arr).any())
        valid = len(shape) == 2 and shape[1] == 14 and shape[0] >= 60 and not out["action_has_nan"] and not out["action_has_inf"]
        out["action_joint14_valid"] = bool(valid)
        out["action_reason"] = "ok" if valid else f"invalid_action_shape_or_values:{shape}"
    except Exception as exc:
        out["action_reason"] = f"action_load_error:{type(exc).__name__}"
    return out


def frame_foreground_metrics(frame_bgr: np.ndarray) -> dict[str, float]:
    h, w = frame_bgr.shape[:2]
    rgb = frame_bgr[:, :, ::-1]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    # White workspaces are normal in WorldArena/RoboTwin. This mask is a proxy for visible non-background structure.
    non_white = np.any(rgb < 210, axis=2)
    saturated = sat > 35
    edges = cv2.Canny(gray, 60, 140) > 0
    fg = non_white | saturated | edges

    def region_ratio(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float]:
        xs, xe = int(w * x0), int(w * x1)
        ys, ye = int(h * y0), int(h * y1)
        if xe <= xs or ye <= ys:
            return 0.0, 0.0
        reg_fg = fg[ys:ye, xs:xe]
        reg_edges = edges[ys:ye, xs:xe]
        return float(reg_fg.mean()), float(reg_edges.mean())

    # Broad dual-arm gripper visibility proxies. Regions are deliberately generous.
    left_fg, left_edge = region_ratio(0.00, 0.25, 0.58, 1.00)
    right_fg, right_edge = region_ratio(0.42, 0.25, 1.00, 1.00)
    center_fg, center_edge = region_ratio(0.20, 0.25, 0.80, 0.95)
    contact_fg, contact_edge = region_ratio(0.25, 0.35, 0.75, 0.90)
    lower_fg, lower_edge = region_ratio(0.00, 0.45, 1.00, 1.00)

    return {
        "foreground_frac": float(fg.mean()),
        "edge_density": float(edges.mean()),
        "left_visible": float((left_fg > 0.018) or (left_edge > 0.010)),
        "right_visible": float((right_fg > 0.018) or (right_edge > 0.010)),
        "end_effector_visible": float((max(left_fg, right_fg, center_fg) > 0.022) or (max(left_edge, right_edge, center_edge) > 0.012)),
        "contact_visible": float((contact_fg > 0.018) or (contact_edge > 0.010) or (lower_fg > 0.025 and lower_edge > 0.010)),
    }


def blockiness_score(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    if g.shape[0] < 16 or g.shape[1] < 16:
        return 0.0
    v_boundary = np.abs(g[:, 8::8] - g[:, 7:-1:8]).mean() if g.shape[1] > 16 else 0.0
    h_boundary = np.abs(g[8::8, :] - g[7:-1:8, :]).mean() if g.shape[0] > 16 else 0.0
    v_inner = np.abs(g[:, 4::8] - g[:, 3:-1:8]).mean() if g.shape[1] > 16 else 0.0
    h_inner = np.abs(g[4::8, :] - g[3:-1:8, :]).mean() if g.shape[0] > 16 else 0.0
    return float(max(0.0, ((v_boundary + h_boundary) * 0.5) - ((v_inner + h_inner) * 0.5)))


def sample_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    n = min(max_frames, frame_count)
    if n <= 1:
        return [0]
    return sorted(set(int(round(x)) for x in np.linspace(0, frame_count - 1, n)))


def analyze_video(args_tuple: tuple[int, dict[str, Any], int]) -> dict[str, Any]:
    idx, row, max_frames = args_tuple
    episode_id = str(row.get("episode_id", f"row_{idx}"))
    video_path = str(row.get("video_640x480_path") or row.get("video") or row.get("raw_video_path") or "")
    first_frame_path = str(row.get("first_frame_320x240_path") or "")
    out: dict[str, Any] = {
        "row_index": idx,
        "episode_id": episode_id,
        "video_path": video_path,
        "first_frame_path": first_frame_path,
        "video_readable": False,
        "qc_status": "reject",
        "qc_reason": "video_missing",
    }

    action_qc = validate_action_joint14(row)
    out.update({
        "action_joint14_valid": action_qc["action_joint14_valid"],
        "action_has_nan": action_qc["action_has_nan"],
        "action_has_inf": action_qc["action_has_inf"],
    })

    if not video_path or not Path(video_path).exists():
        out.update({"deterministic_hard_fail": True, "hard_fail_reason": "video_missing;" + action_qc.get("action_reason", ""), "heuristic_candidate_labels": "", "rule_qc_context": "{}"})
        return out

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        out.update({"qc_reason": "video_unreadable", "deterministic_hard_fail": True, "hard_fail_reason": "video_unreadable;" + action_qc.get("action_reason", ""), "heuristic_candidate_labels": "", "rule_qc_context": "{}"})
        return out

    fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_indices(frame_count, max_frames)

    frames = []
    read_fail = 0
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            read_fail += 1
            continue
        frames.append(frame)
    cap.release()

    if not frames:
        out.update({"fps": fps, "width": width, "height": height, "frame_count": frame_count, "qc_reason": "no_decodable_sample_frames", "deterministic_hard_fail": True, "hard_fail_reason": "no_decodable_sample_frames;" + action_qc.get("action_reason", ""), "heuristic_candidate_labels": "", "rule_qc_context": "{}"})
        return out

    brightness = []
    contrast = []
    sharpness = []
    blockiness = []
    rgb_means = []
    visible = defaultdict(list)
    bad_frames = 0
    motion_scores = []
    invisible_motion = []

    prev_gray = None
    prev_contact = 0.0
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = frame[:, :, ::-1]
        b = float(gray.mean())
        c = float(gray.std())
        brightness.append(b)
        contrast.append(c)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        blockiness.append(blockiness_score(gray))
        rgb_means.append(rgb.reshape(-1, 3).mean(axis=0))
        if b < 8 or b > 248 or c < 2:
            bad_frames += 1
        fg = frame_foreground_metrics(frame)
        for k, v in fg.items():
            visible[k].append(v)
        if prev_gray is not None:
            diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
            motion_scores.append(diff)
            contact_pair = max(prev_contact, fg["contact_visible"], fg["end_effector_visible"])
            invisible_motion.append(float(diff > 8.0 and contact_pair < 0.5))
        prev_gray = gray
        prev_contact = max(fg["contact_visible"], fg["end_effector_visible"])

    rgb_arr = np.vstack(rgb_means) if rgb_means else np.zeros((1, 3), dtype=np.float32)
    if len(rgb_arr) >= 2:
        color_steps = np.linalg.norm(np.diff(rgb_arr, axis=0), axis=1)
        color_shift = float(np.max(color_steps))
    else:
        color_shift = 0.0

    temporal_flicker = 0.0
    if len(brightness) >= 2:
        temporal_flicker = float(np.max(np.abs(np.diff(np.array(brightness, dtype=np.float32)))))

    bad_frame_ratio = float(bad_frames / max(len(frames), 1))
    arm_visible_ratio = float(np.mean([(l or r) for l, r in zip(visible["left_visible"], visible["right_visible"])])) if visible else 0.0
    left_ratio = float(np.mean(visible["left_visible"])) if visible else 0.0
    right_ratio = float(np.mean(visible["right_visible"])) if visible else 0.0
    ee_ratio = float(np.mean(visible["end_effector_visible"])) if visible else 0.0
    contact_ratio = float(np.mean(visible["contact_visible"])) if visible else 0.0
    motion_score = float(np.mean(motion_scores)) if motion_scores else 0.0
    obj_motion_no_contact = float(np.mean(invisible_motion)) if invisible_motion else 0.0

    hard_reasons: list[str] = []
    heuristic_labels: list[str] = []

    if read_fail > max(1, len(indices) // 4):
        hard_reasons.append("many_sample_frames_failed")
    if bad_frame_ratio > 0.20:
        hard_reasons.append("black_or_bad_frames")
    if width and height and (width, height) != (640, 480):
        hard_reasons.append("resolution_invalid")
    if fps and abs(fps - 24.0) > 2.5:
        hard_reasons.append("fps_invalid")
    if not action_qc["action_joint14_valid"]:
        hard_reasons.append(action_qc.get("action_reason", "action_invalid"))
    if action_qc["action_has_nan"]:
        hard_reasons.append("action_nan")
    if action_qc["action_has_inf"]:
        hard_reasons.append("action_inf")

    if color_shift > 70 and np.std(brightness) > 18:
        heuristic_labels.append("strong_color_or_exposure_jump")
    elif color_shift > 40:
        heuristic_labels.append("color_shift")
    if temporal_flicker > 55:
        heuristic_labels.append("strong_temporal_flicker")
    elif temporal_flicker > 30:
        heuristic_labels.append("temporal_flicker")
    if float(np.mean(blockiness)) > 32 and float(np.mean(sharpness)) < 45:
        heuristic_labels.append("severe_compression_artifacts")
    elif float(np.mean(blockiness)) > 18:
        heuristic_labels.append("compression_artifacts")
    if obj_motion_no_contact > 0.55 and motion_score > 10 and contact_ratio < 0.25:
        heuristic_labels.append("object_motion_without_visible_contact")
    elif obj_motion_no_contact > 0.30 and motion_score > 8:
        heuristic_labels.append("possible_motion_without_contact")
    if arm_visible_ratio < 0.10 and motion_score > 8:
        heuristic_labels.append("arm_visibility_low")
    if ee_ratio < 0.10 and motion_score > 8:
        heuristic_labels.append("end_effector_visibility_low")

    deterministic_hard_fail = bool(hard_reasons)
    if deterministic_hard_fail:
        status = "reject"
        reason = ";".join(hard_reasons)
    elif heuristic_labels:
        status = "warn"
        reason = ";".join(heuristic_labels)
    else:
        status = "pass"
        reason = "ok"
    rule_context = {
        "brightness_mean": float(np.mean(brightness)),
        "brightness_temporal_std": float(np.std(brightness)),
        "contrast_mean": float(np.mean(contrast)),
        "color_shift_score": color_shift,
        "temporal_flicker_score": temporal_flicker,
        "compression_artifact_score": float(np.mean(blockiness)),
        "sharpness_laplacian": float(np.mean(sharpness)),
        "arm_visible_ratio": arm_visible_ratio,
        "end_effector_visible_ratio": ee_ratio,
        "contact_region_visible_ratio": contact_ratio,
        "object_motion_without_visible_contact_score": obj_motion_no_contact,
        "motion_score": motion_score,
    }

    out.update({
        "video_readable": True,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "brightness_mean": float(np.mean(brightness)),
        "brightness_temporal_std": float(np.std(brightness)),
        "contrast_mean": float(np.mean(contrast)),
        "color_shift_score": color_shift,
        "temporal_flicker_score": temporal_flicker,
        "compression_artifact_score": float(np.mean(blockiness)),
        "sharpness_laplacian": float(np.mean(sharpness)),
        "arm_visible_ratio": arm_visible_ratio,
        "left_gripper_visible_ratio": left_ratio,
        "right_gripper_visible_ratio": right_ratio,
        "end_effector_visible_ratio": ee_ratio,
        "contact_region_visible_ratio": contact_ratio,
        "object_motion_without_visible_contact_score": obj_motion_no_contact,
        "bad_frame_ratio": bad_frame_ratio,
        "motion_score": motion_score,
        "deterministic_hard_fail": deterministic_hard_fail,
        "hard_fail_reason": ";".join(hard_reasons),
        "action_joint14_valid": action_qc["action_joint14_valid"],
        "action_has_nan": action_qc["action_has_nan"],
        "action_has_inf": action_qc["action_has_inf"],
        "heuristic_candidate_labels": ";".join(heuristic_labels),
        "rule_qc_context": json.dumps(rule_context, ensure_ascii=False),
        "qc_status": status,
        "qc_reason": reason,
    })
    return out


def make_contact_sheet(df: pd.DataFrame, out_path: Path, title: str, n: int = 36) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        img = Image.new("RGB", (640, 160), "white")
        d = ImageDraw.Draw(img)
        d.text((20, 60), f"{title}: no samples", fill=(0, 0, 0))
        img.save(out_path, quality=92)
        return
    sample = df.sample(min(n, len(df)), random_state=17) if len(df) > n else df
    thumb_w, thumb_h = 160, 120
    label_h = 34
    cols = 6
    rows = int(math.ceil(len(sample) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, (_, row) in enumerate(sample.iterrows()):
        x = (i % cols) * thumb_w
        y = 30 + (i // cols) * (thumb_h + label_h)
        img = None
        fp = str(row.get("first_frame_path") or row.get("first_frame_320x240_path") or "")
        vp = str(row.get("video_path") or row.get("video_640x480_path") or "")
        if fp and Path(fp).exists():
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                img = None
        if img is None and vp and Path(vp).exists():
            cap = cv2.VideoCapture(vp)
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                img = Image.fromarray(frame[:, :, ::-1])
        if img is None:
            img = Image.new("RGB", (thumb_w, thumb_h), (230, 230, 230))
        img.thumbnail((thumb_w, thumb_h))
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        canvas.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        sheet.paste(canvas, (x, y))
        label = f"{row.get('episode_id','')}\n{row.get('qc_reason','')}"
        draw.text((x + 3, y + thumb_h + 2), label[:70], fill=(0, 0, 0), font=font)
    sheet.save(out_path, quality=92)


def write_report(out_dir: Path, manifest_path: Path, full: pd.DataFrame, qc: pd.DataFrame) -> None:
    counts = qc["qc_status"].value_counts(dropna=False).to_dict()
    reasons = Counter()
    for reason in qc["qc_reason"].fillna(""):
        for part in str(reason).split(";"):
            if part:
                reasons[part] += 1
    lines = []
    lines.append("# WorldArena v0.1 Video QC Report")
    lines.append("")
    lines.append(f"Manifest: `{manifest_path}`")
    lines.append(f"Episodes checked: `{len(qc)}`")
    lines.append("")
    lines.append("## QC Status")
    lines.append("")
    for key in ["pass", "warn", "reject"]:
        lines.append(f"- `{key}`: `{counts.get(key, 0)}`")
    lines.append("")
    lines.append("## Top Reasons")
    lines.append("")
    for reason, count in reasons.most_common(20):
        lines.append(f"- `{reason}`: `{count}`")
    lines.append("")
    lines.append("## Metric Summary")
    lines.append("")
    metrics = [
        "brightness_mean", "brightness_temporal_std", "contrast_mean", "color_shift_score",
        "temporal_flicker_score", "compression_artifact_score", "sharpness_laplacian",
        "arm_visible_ratio", "end_effector_visible_ratio", "contact_region_visible_ratio",
        "object_motion_without_visible_contact_score", "bad_frame_ratio", "motion_score",
    ]
    for m in metrics:
        if m in qc:
            s = qc[m].dropna()
            if len(s):
                lines.append(f"- `{m}`: mean={s.mean():.4f}, p50={s.quantile(0.50):.4f}, p95={s.quantile(0.95):.4f}, max={s.max():.4f}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- White backgrounds, partially out-of-frame robot arms, and light render grain are treated as target-domain style and are not reject reasons by themselves.")
    lines.append("- Arm/gripper/contact visibility uses image-processing proxies, not a learned detector. Warn/reject samples should be manually spot-checked with the contact sheets.")
    lines.append("- Reject is reserved for deterministic hard failures: unreadable videos, bad/black frames, invalid fps/resolution, invalid joint14 action shape, or NaN/Inf action. Visual heuristic issues are warning/context labels for VLM review.")
    (out_dir / "qc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="WorldArena v0.1 generated video QC")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-sampled-frames", type=int, default=12)
    ap.add_argument("--workers", type=int, default=min(8, max(1, cpu_count() // 2)))
    args = ap.parse_args()

    manifest_path = resolve_manifest(args.manifest)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "contact_sheets").mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(manifest_path).reset_index(drop=True)
    rows = [(i, manifest.iloc[i].to_dict(), args.max_sampled_frames) for i in range(len(manifest))]

    if args.workers <= 1:
        qc_rows = [analyze_video(r) for r in rows]
    else:
        with Pool(processes=args.workers) as pool:
            qc_rows = list(pool.imap_unordered(analyze_video, rows, chunksize=8))
    qc = pd.DataFrame(qc_rows).sort_values("row_index").reset_index(drop=True)

    full = manifest.copy()
    for field in QC_FIELDS:
        full[field] = qc[field].values
    qc.to_csv(out_dir / "qc_scores.csv", index=False)
    full[full["qc_status"] == "pass"].to_parquet(out_dir / "episode_manifest_qc_pass.parquet", index=False)
    full[full["qc_status"] == "warn"].to_parquet(out_dir / "episode_manifest_qc_warn.parquet", index=False)
    full[full["qc_status"] == "reject"].to_parquet(out_dir / "episode_manifest_qc_reject.parquet", index=False)

    for status in ["pass", "warn", "reject"]:
        make_contact_sheet(
            full[full["qc_status"] == status],
            out_dir / "contact_sheets" / f"qc_{status}_samples.jpg",
            f"QC {status} samples",
        )
    write_report(out_dir, manifest_path, full, qc)
    counts = full["qc_status"].value_counts().to_dict()
    print(f"manifest={manifest_path}")
    print(f"out={out_dir}")
    print(f"counts={counts}")


if __name__ == "__main__":
    main()
