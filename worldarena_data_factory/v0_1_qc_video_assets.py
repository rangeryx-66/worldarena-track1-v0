#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

QC_FIELDS = [
    "video_readable",
    "fps",
    "width",
    "height",
    "frame_count",
    "brightness_mean",
    "brightness_temporal_std",
    "contrast_mean",
    "color_shift_score",
    "temporal_flicker_score",  # compatibility: now background-only scores
    "global_content_rgb_shift_score",
    "global_content_brightness_step",
    "foreground_area_change_score",
    "background_luma_flicker_score",
    "background_color_shift_score",
    "background_changed_area_ratio",
    "background_color_changed_area_ratio",
    "background_delta_sign_consistency",
    "background_color_direction_consistency",
    "background_mask_reliable",
    "background_reliable_pair_ratio",
    "illumination_flicker_candidate",
    "true_color_shift_candidate",
    "compression_artifact_score",
    "sharpness_laplacian",
    "robot_visible_ratio",
    "robot_near_object_ratio",
    "arm_visible_ratio",
    "left_gripper_visible_ratio",
    "right_gripper_visible_ratio",
    "end_effector_visible_ratio",
    "contact_region_visible_ratio",
    "object_motion_without_visible_contact_score",
    "object_like_motion_ratio",
    "object_motion_event_count",
    "bad_frame_ratio",
    "motion_score",
    "deterministic_hard_fail",
    "hard_fail_reason",
    "action_joint14_valid",
    "action_has_nan",
    "action_has_inf",
    "heuristic_candidate_labels",
    "rule_qc_context",
    "qc_status",
    "qc_reason",
]

NEGATIVE_LABELS = {
    "background_flicker_candidate",
    "true_color_shift_candidate",
    "possible_object_motion_without_contact",
    "contact_visibility_borderline",
    "robot_visibility_borderline",
    "severe_compression_artifacts",
    "bad_or_black_frames",
}
DPO_LABELS = {"possible_object_motion_without_contact"}


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
    path = str(
        row.get("action_joint14_norm_path") or row.get("action_joint14_raw_path") or ""
    )
    out = {
        "action_joint14_valid": False,
        "action_has_nan": False,
        "action_has_inf": False,
        "action_reason": "action_missing",
    }
    if not path or not Path(path).exists():
        return out
    try:
        arr = np.load(path, mmap_mode="r")
        shape = tuple(arr.shape)
        out["action_has_nan"] = bool(np.isnan(arr).any())
        out["action_has_inf"] = bool(np.isinf(arr).any())
        valid = (
            len(shape) == 2
            and shape[1] == 14
            and shape[0] >= 60
            and not out["action_has_nan"]
            and not out["action_has_inf"]
        )
        out["action_joint14_valid"] = bool(valid)
        out["action_reason"] = (
            "ok" if valid else f"invalid_action_shape_or_values:{shape}"
        )
    except Exception as exc:
        out["action_reason"] = f"action_load_error:{type(exc).__name__}"
    return out


def sample_dense_indices(
    frame_count: int, fps: float, target_fps: float = 6.0, max_frames: int = 96
) -> list[int]:
    if frame_count <= 0:
        return []
    fps = fps if fps and fps > 0 else 24.0
    step = max(1, int(round(fps / max(target_fps, 0.1))))
    idx = list(range(0, frame_count, step))
    if len(idx) > max_frames:
        idx = sorted(
            set(int(round(x)) for x in np.linspace(0, len(idx) - 1, max_frames))
        )
        source = list(range(0, frame_count, step))
        return [source[i] for i in idx]
    return idx or [0]


def sample_sheet_indices(frame_count: int, num_frames: int = 16) -> list[int]:
    if frame_count <= 0:
        return []
    n = min(num_frames, frame_count)
    if n <= 1:
        return [0]
    return sorted(set(int(round(x)) for x in np.linspace(0, frame_count - 1, n)))


def resize_for_metrics(frame_bgr: np.ndarray, max_width: int = 320) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w <= max_width:
        return frame_bgr
    scale = max_width / float(w)
    return cv2.resize(
        frame_bgr,
        (max_width, max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def clean_mask(mask: np.ndarray, k: int = 3, min_area: int = 12) -> np.ndarray:
    m = mask.astype(np.uint8)
    if k > 1:
        kernel = np.ones((k, k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m, dtype=bool)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out |= labels == i
    return out


def compute_frame_masks(
    frame_bgr: np.ndarray, prev_frame_bgr: np.ndarray | None = None
) -> dict[str, Any]:
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    edges = cv2.Canny(gray, 45, 120) > 0
    edge_d = cv2.dilate(
        edges.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
    ).astype(bool)

    motion_mask = np.zeros((h, w), dtype=bool)
    if prev_frame_bgr is not None:
        prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        motion_mask = clean_mask(diff > 12, k=3, min_area=18)
        motion_mask = cv2.dilate(
            motion_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
        ).astype(bool)

    high_v_low_s = (v > 145) & (s < 58)
    very_white = (v > 185) & (s < 75)
    dark = v < 105
    robot_seed = (dark & (s < 135)) | ((v < 135) & edge_d & (s < 90))
    robot_mask = clean_mask(robot_seed, k=3, min_area=max(18, (h * w) // 2500))

    saturated = (s > 55) & (v > 55)
    non_white_non_dark = (v < 205) & (v > 70) & (s > 25)
    object_seed = (saturated | non_white_non_dark) & ~robot_mask
    object_like_mask = clean_mask(object_seed, k=3, min_area=max(12, (h * w) // 5000))

    bg_seed = (high_v_low_s | very_white) & ~edge_d & ~robot_mask & ~object_like_mask
    # Do not subtract motion from background. True exposure flicker makes the
    # whole background "move" in pixel space; excluding motion here would hide
    # exactly the artifact we need to detect. Foreground/edge/object masks carry
    # the responsibility for excluding robot/object content.
    background_mask = clean_mask(bg_seed, k=5, min_area=max(64, (h * w) // 300))
    bg_frac = float(background_mask.mean())
    foreground_mask = clean_mask(
        (robot_mask | object_like_mask | edge_d | motion_mask) & ~background_mask,
        k=3,
        min_area=12,
    )
    reliability = bool(bg_frac > 0.18 and background_mask.sum() > 1200)
    return {
        "background_mask": background_mask,
        "foreground_mask": foreground_mask,
        "robot_mask": robot_mask,
        "object_like_mask": object_like_mask,
        "motion_mask": motion_mask,
        "mask_reliability": reliability,
        "background_fraction": bg_frac,
        "foreground_fraction": float(foreground_mask.mean()),
        "robot_fraction": float(robot_mask.mean()),
        "object_fraction": float(object_like_mask.mean()),
        "lab": lab,
        "gray": gray,
    }


def robust_max(values: list[float], q: float = 1.0) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    if q >= 1.0:
        return float(max(vals))
    return float(np.quantile(vals, q))


def compute_global_content_metrics(
    frames: list[np.ndarray], masks: list[dict[str, Any]]
) -> dict[str, float]:
    if not frames:
        return {
            "global_content_rgb_shift_score": 0.0,
            "global_content_brightness_step": 0.0,
            "foreground_area_change_score": 0.0,
        }
    rgb_means = []
    luma = []
    fg = []
    for frame, m in zip(frames, masks):
        rgb = frame[:, :, ::-1]
        gray = m["gray"]
        rgb_means.append(rgb.reshape(-1, 3).mean(axis=0))
        luma.append(float(gray.mean()))
        fg.append(float(m["foreground_fraction"]))
    rgb_arr = np.vstack(rgb_means) if rgb_means else np.zeros((1, 3), dtype=np.float32)
    rgb_step = (
        float(np.max(np.linalg.norm(np.diff(rgb_arr, axis=0), axis=1)))
        if len(rgb_arr) >= 2
        else 0.0
    )
    luma_step = (
        float(np.max(np.abs(np.diff(np.asarray(luma, dtype=np.float32)))))
        if len(luma) >= 2
        else 0.0
    )
    fg_step = (
        float(np.max(np.abs(np.diff(np.asarray(fg, dtype=np.float32)))))
        if len(fg) >= 2
        else 0.0
    )
    return {
        "global_content_rgb_shift_score": rgb_step,
        "global_content_brightness_step": luma_step,
        "foreground_area_change_score": fg_step,
    }


def compute_background_flicker_metrics(
    frames: list[np.ndarray], masks: list[dict[str, Any]]
) -> dict[str, Any]:
    deltas = []
    changed = []
    sign_cons = []
    fg_delta = []
    reliable_pairs = 0
    pair_count = max(0, len(frames) - 1)
    for i in range(1, len(frames)):
        prev = masks[i - 1]
        cur = masks[i]
        common_bg = prev["background_mask"] & cur["background_mask"]
        common_frac = float(common_bg.mean())
        if not (
            prev["mask_reliability"]
            and cur["mask_reliability"]
            and common_frac > 0.12
            and common_bg.sum() > 900
        ):
            continue
        l0 = prev["lab"][:, :, 0].astype(np.float32)
        l1 = cur["lab"][:, :, 0].astype(np.float32)
        d = l1[common_bg] - l0[common_bg]
        abs_d = np.abs(d)
        med = float(np.median(d))
        area = float(np.mean(abs_d > 8.0))
        if np.any(abs_d > 8.0):
            pos = float(np.mean(d[abs_d > 8.0] > 0))
            sign = max(pos, 1.0 - pos)
        else:
            sign = 0.0
        deltas.append(abs(med))
        changed.append(area)
        sign_cons.append(sign)
        fg_delta.append(
            abs(float(cur["foreground_fraction"]) - float(prev["foreground_fraction"]))
        )
        reliable_pairs += 1
    reliable_ratio = float(reliable_pairs / max(pair_count, 1))
    score = robust_max(deltas)
    changed_score = robust_max(changed)
    sign_score = robust_max(sign_cons)
    fg_score = robust_max(fg_delta)
    bg_reliable = reliable_pairs >= max(2, int(0.35 * max(pair_count, 1)))
    candidate = bool(
        bg_reliable
        and changed_score > 0.55
        and score > 8.0
        and sign_score > 0.75
        and fg_score < 0.10
    )
    return {
        "background_luma_flicker_score": score,
        "background_changed_area_ratio": changed_score,
        "background_delta_sign_consistency": sign_score,
        "background_mask_reliable": bg_reliable,
        "background_reliable_pair_ratio": reliable_ratio,
        "foreground_area_delta_during_bg_change": fg_score,
        "illumination_flicker_candidate": candidate,
    }


def compute_background_color_shift_metrics(
    frames: list[np.ndarray], masks: list[dict[str, Any]]
) -> dict[str, Any]:
    scores = []
    changed = []
    direction_cons = []
    reliable_pairs = 0
    pair_count = max(0, len(frames) - 1)
    for i in range(1, len(frames)):
        prev = masks[i - 1]
        cur = masks[i]
        common_bg = prev["background_mask"] & cur["background_mask"]
        common_frac = float(common_bg.mean())
        if not (
            prev["mask_reliability"]
            and cur["mask_reliability"]
            and common_frac > 0.12
            and common_bg.sum() > 900
        ):
            continue
        ab0 = prev["lab"][:, :, 1:3].astype(np.float32)
        ab1 = cur["lab"][:, :, 1:3].astype(np.float32)
        d = ab1[common_bg] - ab0[common_bg]
        med = np.median(d, axis=0)
        norm = float(np.linalg.norm(med))
        pix_norm = np.linalg.norm(d, axis=1)
        area = float(np.mean(pix_norm > 6.0))
        if np.any(pix_norm > 6.0) and norm > 1e-3:
            unit = med / (norm + 1e-6)
            proj = d[pix_norm > 6.0] @ unit
            direction = float(max(np.mean(proj > 0), np.mean(proj < 0)))
        else:
            direction = 0.0
        scores.append(norm)
        changed.append(area)
        direction_cons.append(direction)
        reliable_pairs += 1
    reliable_ratio = float(reliable_pairs / max(pair_count, 1))
    score = robust_max(scores)
    changed_score = robust_max(changed)
    direction_score = robust_max(direction_cons)
    bg_reliable = reliable_pairs >= max(2, int(0.35 * max(pair_count, 1)))
    candidate = bool(
        bg_reliable and score > 6.0 and changed_score > 0.45 and direction_score > 0.70
    )
    return {
        "background_color_shift_score": score,
        "background_color_changed_area_ratio": changed_score,
        "background_color_direction_consistency": direction_score,
        "background_color_reliable_pair_ratio": reliable_ratio,
        "true_color_shift_candidate": candidate,
    }


def mask_near(a: np.ndarray, b: np.ndarray, radius: int = 12) -> bool:
    if not a.any() or not b.any():
        return False
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    bd = cv2.dilate(b.astype(np.uint8), kernel, iterations=1).astype(bool)
    return bool((a & bd).any())


def compute_contact_motion_metrics(
    frames: list[np.ndarray], masks: list[dict[str, Any]]
) -> dict[str, Any]:
    robot_visible = []
    robot_near_object = []
    contact_visible = []
    object_motion_events = 0
    no_contact_events = 0
    sustained_no_contact = 0
    current_run = 0
    motion_vals = []
    for i, m in enumerate(masks):
        robot = m["robot_mask"]
        obj = m["object_like_mask"]
        robot_visible.append(float(robot.mean() > 0.006))
        near = mask_near(robot, obj, radius=14)
        robot_near_object.append(float(near))
        contact_visible.append(float(near or bool((robot & obj).any())))
        if i == 0:
            continue
        motion = m["motion_mask"]
        prev_obj = masks[i - 1]["object_like_mask"]
        obj_region = cv2.dilate(
            (obj | prev_obj).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1
        ).astype(bool)
        object_motion = (
            motion
            & obj_region
            & ~cv2.dilate(
                robot.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
            ).astype(bool)
        )
        object_motion_frac = float(object_motion.mean())
        motion_vals.append(float(motion.mean()))
        if object_motion_frac > 0.0012 and object_motion.sum() > 28:
            object_motion_events += 1
            near_motion = mask_near(
                object_motion, robot | masks[i - 1]["robot_mask"], radius=18
            )
            if not near_motion:
                no_contact_events += 1
                current_run += 1
                sustained_no_contact = max(sustained_no_contact, current_run)
            else:
                current_run = 0
        else:
            current_run = 0
    event_ratio = (
        float(no_contact_events / max(object_motion_events, 1))
        if object_motion_events
        else 0.0
    )
    object_motion_ratio = (
        float(object_motion_events / max(len(frames) - 1, 1))
        if len(frames) > 1
        else 0.0
    )
    score = (
        event_ratio if object_motion_events >= 2 and sustained_no_contact >= 2 else 0.0
    )
    return {
        "robot_visible_ratio": float(np.mean(robot_visible)) if robot_visible else 0.0,
        "robot_near_object_ratio": (
            float(np.mean(robot_near_object)) if robot_near_object else 0.0
        ),
        "contact_region_visible_ratio": (
            float(np.mean(contact_visible)) if contact_visible else 0.0
        ),
        "object_motion_without_visible_contact_score": score,
        "object_like_motion_ratio": object_motion_ratio,
        "object_motion_event_count": int(object_motion_events),
        "object_motion_no_contact_event_count": int(no_contact_events),
        "object_motion_no_contact_sustained_frames": int(sustained_no_contact),
        "motion_score": float(np.mean(motion_vals)) if motion_vals else 0.0,
    }


def blockiness_score(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    if g.shape[0] < 16 or g.shape[1] < 16:
        return 0.0
    v_boundary = np.abs(g[:, 8::8] - g[:, 7:-1:8]).mean() if g.shape[1] > 16 else 0.0
    h_boundary = np.abs(g[8::8, :] - g[7:-1:8, :]).mean() if g.shape[0] > 16 else 0.0
    v_inner = np.abs(g[:, 4::8] - g[:, 3:-1:8]).mean() if g.shape[1] > 16 else 0.0
    h_inner = np.abs(g[4::8, :] - g[3:-1:8, :]).mean() if g.shape[0] > 16 else 0.0
    return float(
        max(0.0, ((v_boundary + h_boundary) * 0.5) - ((v_inner + h_inner) * 0.5))
    )


def read_video_frames(
    video_path: str, indices: list[int], resize_width: int = 320
) -> tuple[list[np.ndarray], int]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    read_fail = 0
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            read_fail += 1
            continue
        frames.append(resize_for_metrics(frame, resize_width))
    cap.release()
    return frames, read_fail


def analyze_video(
    args_tuple: tuple[int, dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    idx, row, cfg = args_tuple
    dense_target_fps = float(cfg.get("dense_target_fps", 6.0))
    max_dense_frames = int(cfg.get("max_dense_frames", 96))
    disable_v2_masks = bool(cfg.get("disable_v2_masks", False))
    episode_id = str(row.get("episode_id", f"row_{idx}"))
    video_path = str(
        row.get("video_640x480_path")
        or row.get("video")
        or row.get("raw_video_path")
        or ""
    )
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
    out.update(
        {
            "action_joint14_valid": action_qc["action_joint14_valid"],
            "action_has_nan": action_qc["action_has_nan"],
            "action_has_inf": action_qc["action_has_inf"],
        }
    )
    if not video_path or not Path(video_path).exists():
        out.update(
            {
                "deterministic_hard_fail": True,
                "hard_fail_reason": "video_missing;"
                + action_qc.get("action_reason", ""),
                "heuristic_candidate_labels": "",
                "rule_qc_context": "{}",
            }
        )
        return out

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        out.update(
            {
                "qc_reason": "video_unreadable",
                "deterministic_hard_fail": True,
                "hard_fail_reason": "video_unreadable;"
                + action_qc.get("action_reason", ""),
                "heuristic_candidate_labels": "",
                "rule_qc_context": "{}",
            }
        )
        return out
    fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    dense_indices = sample_dense_indices(
        frame_count, fps, dense_target_fps, max_dense_frames
    )
    frames, read_fail = read_video_frames(video_path, dense_indices, resize_width=320)
    if not frames:
        out.update(
            {
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": frame_count,
                "qc_reason": "no_decodable_sample_frames",
                "deterministic_hard_fail": True,
                "hard_fail_reason": "no_decodable_sample_frames;"
                + action_qc.get("action_reason", ""),
                "heuristic_candidate_labels": "",
                "rule_qc_context": "{}",
            }
        )
        return out

    brightness = []
    contrast = []
    sharpness = []
    blockiness = []
    bad_frames = 0
    masks = []
    prev = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        b = float(gray.mean())
        c = float(gray.std())
        brightness.append(b)
        contrast.append(c)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        blockiness.append(blockiness_score(gray))
        if b < 8 or b > 248 or c < 2:
            bad_frames += 1
        masks.append(compute_frame_masks(frame, prev))
        prev = frame

    if disable_v2_masks:
        for m in masks:
            m["background_mask"][:] = False
            m["mask_reliability"] = False

    global_metrics = compute_global_content_metrics(frames, masks)
    flicker_metrics = compute_background_flicker_metrics(frames, masks)
    color_metrics = compute_background_color_shift_metrics(frames, masks)
    contact_metrics = compute_contact_motion_metrics(frames, masks)

    bad_frame_ratio = float(bad_frames / max(len(frames), 1))
    hard_reasons: list[str] = []
    labels: list[str] = []
    if read_fail > max(2, len(dense_indices) // 4):
        hard_reasons.append("many_sample_frames_failed")
    if bad_frame_ratio > 0.30:
        hard_reasons.append("black_or_bad_frames")
    elif bad_frame_ratio > 0.08:
        labels.append("bad_or_black_frames")
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

    if flicker_metrics["illumination_flicker_candidate"]:
        labels.append("background_flicker_candidate")
    if color_metrics["true_color_shift_candidate"]:
        labels.append("true_color_shift_candidate")
    high_global_rgb = global_metrics["global_content_rgb_shift_score"] > 35
    high_global_luma = global_metrics["global_content_brightness_step"] > 18
    fg_changed = global_metrics["foreground_area_change_score"] > 0.045
    bg_stable = (
        flicker_metrics["background_luma_flicker_score"] < 6
        and color_metrics["background_color_shift_score"] < 5
    )
    if (high_global_rgb or high_global_luma) and bg_stable:
        labels.append("content_motion_not_flicker")
        if fg_changed:
            labels.append("foreground_entry_caused_global_shift")
    if contact_metrics["object_motion_without_visible_contact_score"] > 0.70:
        labels.append("possible_object_motion_without_contact")
    elif (
        contact_metrics["object_like_motion_ratio"] > 0.12
        and contact_metrics["contact_region_visible_ratio"] < 0.12
    ):
        labels.append("contact_visibility_borderline")
    if (
        contact_metrics["robot_visible_ratio"] < 0.08
        and contact_metrics["object_like_motion_ratio"] > 0.08
    ):
        labels.append("robot_visibility_borderline")
    if float(np.mean(blockiness)) > 45 and float(np.mean(sharpness)) < 25:
        labels.append("severe_compression_artifacts")

    deterministic_hard_fail = bool(hard_reasons)
    negative_labels = [x for x in labels if x in NEGATIVE_LABELS]
    if deterministic_hard_fail:
        status = "reject"
        reason = ";".join(hard_reasons)
    elif (
        any(x in DPO_LABELS for x in labels)
        and contact_metrics["object_motion_without_visible_contact_score"] > 0.80
    ):
        status = "dpo_loser_candidate"
        reason = ";".join(labels)
    elif negative_labels:
        status = "warn"
        reason = ";".join(labels)
    else:
        status = "pass"
        reason = ";".join(labels) if labels else "ok"

    robot_visible_ratio = contact_metrics["robot_visible_ratio"]
    left_ratio = (
        float(
            np.mean(
                [
                    m["robot_mask"][:, : m["robot_mask"].shape[1] // 2].mean() > 0.004
                    for m in masks
                ]
            )
        )
        if masks
        else 0.0
    )
    right_ratio = (
        float(
            np.mean(
                [
                    m["robot_mask"][:, m["robot_mask"].shape[1] // 2 :].mean() > 0.004
                    for m in masks
                ]
            )
        )
        if masks
        else 0.0
    )
    rule_context = {
        "hard_fail": deterministic_hard_fail,
        "global_content_change": {
            "global_content_rgb_shift_score": global_metrics[
                "global_content_rgb_shift_score"
            ],
            "global_content_brightness_step": global_metrics[
                "global_content_brightness_step"
            ],
            "foreground_area_change_score": global_metrics[
                "foreground_area_change_score"
            ],
        },
        "background_artifact_metrics": {
            "background_luma_flicker_score": flicker_metrics[
                "background_luma_flicker_score"
            ],
            "background_color_shift_score": color_metrics[
                "background_color_shift_score"
            ],
            "background_changed_area_ratio": flicker_metrics[
                "background_changed_area_ratio"
            ],
            "background_color_changed_area_ratio": color_metrics[
                "background_color_changed_area_ratio"
            ],
            "background_delta_sign_consistency": flicker_metrics[
                "background_delta_sign_consistency"
            ],
            "background_color_direction_consistency": color_metrics[
                "background_color_direction_consistency"
            ],
        },
        "contact_motion_metrics": contact_metrics,
        "visibility_metrics": {
            "robot_visible_ratio": robot_visible_ratio,
            "robot_near_object_ratio": contact_metrics["robot_near_object_ratio"],
            "contact_region_visible_ratio": contact_metrics[
                "contact_region_visible_ratio"
            ],
        },
        "metric_reliability": {
            "background_mask_reliable": flicker_metrics["background_mask_reliable"],
            "background_reliable_pair_ratio": flicker_metrics[
                "background_reliable_pair_ratio"
            ],
            "background_color_reliable_pair_ratio": color_metrics[
                "background_color_reliable_pair_ratio"
            ],
        },
        "notes": [
            "Global RGB/luma shifts may be caused by robot entering the frame and are not treated as flicker unless background metrics confirm it.",
            "color_shift_score and temporal_flicker_score are compatibility aliases for background-only metrics in qc-version v2.",
        ],
    }
    out.update(
        {
            "video_readable": True,
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "brightness_mean": float(np.mean(brightness)),
            "brightness_temporal_std": float(np.std(brightness)),
            "contrast_mean": float(np.mean(contrast)),
            "color_shift_score": color_metrics["background_color_shift_score"],
            "temporal_flicker_score": flicker_metrics["background_luma_flicker_score"],
            "global_content_rgb_shift_score": global_metrics[
                "global_content_rgb_shift_score"
            ],
            "global_content_brightness_step": global_metrics[
                "global_content_brightness_step"
            ],
            "foreground_area_change_score": global_metrics[
                "foreground_area_change_score"
            ],
            "background_luma_flicker_score": flicker_metrics[
                "background_luma_flicker_score"
            ],
            "background_color_shift_score": color_metrics[
                "background_color_shift_score"
            ],
            "background_changed_area_ratio": flicker_metrics[
                "background_changed_area_ratio"
            ],
            "background_color_changed_area_ratio": color_metrics[
                "background_color_changed_area_ratio"
            ],
            "background_delta_sign_consistency": flicker_metrics[
                "background_delta_sign_consistency"
            ],
            "background_color_direction_consistency": color_metrics[
                "background_color_direction_consistency"
            ],
            "background_mask_reliable": flicker_metrics["background_mask_reliable"],
            "background_reliable_pair_ratio": flicker_metrics[
                "background_reliable_pair_ratio"
            ],
            "illumination_flicker_candidate": flicker_metrics[
                "illumination_flicker_candidate"
            ],
            "true_color_shift_candidate": color_metrics["true_color_shift_candidate"],
            "compression_artifact_score": float(np.mean(blockiness)),
            "sharpness_laplacian": float(np.mean(sharpness)),
            "robot_visible_ratio": robot_visible_ratio,
            "robot_near_object_ratio": contact_metrics["robot_near_object_ratio"],
            "arm_visible_ratio": robot_visible_ratio,
            "left_gripper_visible_ratio": left_ratio,
            "right_gripper_visible_ratio": right_ratio,
            "end_effector_visible_ratio": robot_visible_ratio,
            "contact_region_visible_ratio": contact_metrics[
                "contact_region_visible_ratio"
            ],
            "object_motion_without_visible_contact_score": contact_metrics[
                "object_motion_without_visible_contact_score"
            ],
            "object_like_motion_ratio": contact_metrics["object_like_motion_ratio"],
            "object_motion_event_count": contact_metrics["object_motion_event_count"],
            "bad_frame_ratio": bad_frame_ratio,
            "motion_score": contact_metrics["motion_score"],
            "deterministic_hard_fail": deterministic_hard_fail,
            "hard_fail_reason": ";".join(hard_reasons),
            "action_joint14_valid": action_qc["action_joint14_valid"],
            "action_has_nan": action_qc["action_has_nan"],
            "action_has_inf": action_qc["action_has_inf"],
            "heuristic_candidate_labels": ";".join(labels),
            "rule_qc_context": json.dumps(rule_context, ensure_ascii=False),
            "qc_status": status,
            "qc_reason": reason,
        }
    )
    return out


def make_contact_sheet(
    df: pd.DataFrame, out_path: Path, title: str, n: int = 36
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        img = Image.new("RGB", (640, 160), "white")
        ImageDraw.Draw(img).text((20, 60), f"{title}: no samples", fill=(0, 0, 0))
        img.save(out_path, quality=92)
        return
    sample = df.sample(min(n, len(df)), random_state=17) if len(df) > n else df
    thumb_w, thumb_h = 160, 120
    label_h = 38
    cols = 6
    rows = int(math.ceil(len(sample) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    font = ImageFont.load_default()
    for i, (_, row) in enumerate(sample.iterrows()):
        x = (i % cols) * thumb_w
        y = 30 + (i // cols) * (thumb_h + label_h)
        img = None
        fp = str(
            row.get("first_frame_path") or row.get("first_frame_320x240_path") or ""
        )
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
        draw.text((x + 3, y + thumb_h + 2), label[:82], fill=(0, 0, 0), font=font)
    sheet.save(out_path, quality=92)


def write_report(
    out_dir: Path,
    manifest_path: Path,
    full: pd.DataFrame,
    qc: pd.DataFrame,
    version: str,
) -> None:
    counts = qc["qc_status"].value_counts(dropna=False).to_dict()
    reasons = Counter()
    for reason in qc["qc_reason"].fillna(""):
        for part in str(reason).split(";"):
            if part:
                reasons[part] += 1
    lines = [
        f"# WorldArena Video QC Report ({version})",
        "",
        f"Manifest: `{manifest_path}`",
        f"Episodes checked: `{len(qc)}`",
        "",
        "## QC Status",
        "",
    ]
    for key in ["pass", "warn", "dpo_loser_candidate", "reject"]:
        lines.append(f"- `{key}`: `{counts.get(key, 0)}`")
    lines += ["", "## Top Reasons", ""]
    for reason, count in reasons.most_common(30):
        lines.append(f"- `{reason}`: `{count}`")
    lines += ["", "## Metric Summary", ""]
    metrics = [
        "global_content_rgb_shift_score",
        "global_content_brightness_step",
        "foreground_area_change_score",
        "background_luma_flicker_score",
        "background_color_shift_score",
        "background_changed_area_ratio",
        "background_delta_sign_consistency",
        "robot_visible_ratio",
        "robot_near_object_ratio",
        "contact_region_visible_ratio",
        "object_motion_without_visible_contact_score",
        "object_like_motion_ratio",
        "bad_frame_ratio",
        "motion_score",
    ]
    for m in metrics:
        if m in qc:
            s = pd.to_numeric(qc[m], errors="coerce").dropna()
            if len(s):
                lines.append(
                    f"- `{m}`: mean={s.mean():.4f}, p50={s.quantile(0.50):.4f}, p95={s.quantile(0.95):.4f}, max={s.max():.4f}"
                )
    lines += [
        "",
        "## Field Semantics",
        "",
        "- `global_content_rgb_shift_score` and `global_content_brightness_step` measure whole-image content change. They are not flicker/color-shift evidence by themselves.",
        "- `color_shift_score` is now a compatibility alias for `background_color_shift_score`.",
        "- `temporal_flicker_score` is now a compatibility alias for `background_luma_flicker_score`.",
        "- Flicker/color shift candidates require reliable shared background masks and large, direction-consistent background changes.",
        "- Robot foreground entry, gripper close-ups, and object motion should appear as `content_motion_not_flicker` or `foreground_entry_caused_global_shift`, not temporal flicker/color shift.",
        "- Deterministic hard fail is the only direct rule reject path. Visual heuristics produce pass/warn/dpo_loser_candidate for VLM or human review.",
    ]
    (out_dir / ("qc_report_v2.md" if version == "v2" else "qc_report.md")).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_outputs(
    out_dir: Path,
    manifest_path: Path,
    manifest: pd.DataFrame,
    qc: pd.DataFrame,
    version: str,
) -> None:
    full = manifest.copy()
    for field in QC_FIELDS:
        full[field] = qc[field].values if field in qc else None
    suffix = "_v2" if version == "v2" else ""
    qc.to_csv(out_dir / f"qc_scores{suffix}.csv", index=False)
    if version == "v2":
        qc.to_csv(out_dir / "qc_scores.csv", index=False)
    full[full["qc_status"] == "pass"].to_parquet(
        out_dir / f"episode_manifest_qc_pass{suffix}.parquet", index=False
    )
    full[full["qc_status"] == "warn"].to_parquet(
        out_dir / f"episode_manifest_qc_warn{suffix}.parquet", index=False
    )
    full[full["qc_status"] == "reject"].to_parquet(
        out_dir / f"episode_manifest_qc_reject{suffix}.parquet", index=False
    )
    full[full["qc_status"] == "dpo_loser_candidate"].to_csv(
        out_dir / "dpo_loser_candidates_rule_v2.csv", index=False
    )
    cs = out_dir / "contact_sheets"
    cs.mkdir(parents=True, exist_ok=True)
    filters = {
        "background_flicker_candidate": full[
            full["heuristic_candidate_labels"]
            .fillna("")
            .str.contains("background_flicker_candidate", regex=False)
        ],
        "true_color_shift_candidate": full[
            full["heuristic_candidate_labels"]
            .fillna("")
            .str.contains("true_color_shift_candidate", regex=False)
        ],
        "foreground_entry_caused_global_shift": full[
            full["heuristic_candidate_labels"]
            .fillna("")
            .str.contains("foreground_entry_caused_global_shift", regex=False)
        ],
        "object_motion_without_contact_candidate": full[
            full["heuristic_candidate_labels"]
            .fillna("")
            .str.contains("possible_object_motion_without_contact", regex=False)
        ],
        "contact_visibility_borderline": full[
            full["heuristic_candidate_labels"]
            .fillna("")
            .str.contains("contact_visibility_borderline", regex=False)
        ],
        "pass_high_global_shift_but_background_stable": full[
            (full["qc_status"] == "pass")
            & (
                (full["global_content_rgb_shift_score"] > 35)
                | (full["global_content_brightness_step"] > 18)
            )
            & (full["background_luma_flicker_score"] < 6)
            & (full["background_color_shift_score"] < 5)
        ],
        "qc_pass_samples": full[full["qc_status"] == "pass"],
        "qc_warn_samples": full[full["qc_status"] == "warn"],
        "qc_reject_samples": full[full["qc_status"] == "reject"],
    }
    for name, df in filters.items():
        make_contact_sheet(df, cs / f"{name}.jpg", name)
    write_report(out_dir, manifest_path, full, qc, version)


def synthetic_foreground_entry_frames(n: int = 24) -> list[np.ndarray]:
    frames = []
    for i in range(n):
        img = np.full((120, 160, 3), 235, dtype=np.uint8)
        cv2.rectangle(img, (20, 70), (145, 88), (238, 238, 238), -1)
        x = int(-85 + i * 11)
        cv2.rectangle(img, (x, 20), (x + 88, 112), (20, 20, 20), -1)
        cv2.rectangle(img, (90, 78), (125, 86), (45, 140, 230), -1)
        frames.append(img)
    return frames


def synthetic_exposure_jump_frames(n: int = 24) -> list[np.ndarray]:
    frames = []
    for i in range(n):
        base = 210 if i < n // 2 else 245
        img = np.full((120, 160, 3), base, dtype=np.uint8)
        cv2.rectangle(img, (25, 40), (70, 90), (35, 35, 35), -1)
        cv2.rectangle(img, (95, 78), (128, 86), (40, 130, 220), -1)
        frames.append(img)
    return frames


def compute_metrics_for_frames(frames: list[np.ndarray]) -> dict[str, Any]:
    resized = [resize_for_metrics(f, 320) for f in frames]
    masks = []
    prev = None
    for f in resized:
        masks.append(compute_frame_masks(f, prev))
        prev = f
    out = {}
    out.update(compute_global_content_metrics(resized, masks))
    out.update(compute_background_flicker_metrics(resized, masks))
    out.update(compute_background_color_shift_metrics(resized, masks))
    return out


def run_regression_tests() -> None:
    fg = compute_metrics_for_frames(synthetic_foreground_entry_frames())
    assert fg["global_content_rgb_shift_score"] > 15, fg
    assert fg["global_content_brightness_step"] > 3, fg
    assert fg["background_luma_flicker_score"] < 6, fg
    assert not fg["illumination_flicker_candidate"], fg
    exp = compute_metrics_for_frames(synthetic_exposure_jump_frames())
    assert exp["background_luma_flicker_score"] > 8, exp
    assert exp["background_changed_area_ratio"] > 0.55, exp
    assert exp["illumination_flicker_candidate"], exp
    print("regression tests passed")


def progress_iter(iterator, total: int, enabled: bool = True, interval: int = 25):
    if not enabled:
        yield from iterator
        return
    try:
        from tqdm import tqdm

        yield from tqdm(iterator, total=total, dynamic_ncols=True, desc="video-qc")
        return
    except Exception:
        pass
    start = time.time()
    last = 0
    for i, item in enumerate(iterator, 1):
        yield item
        if i == total or i - last >= max(1, interval):
            elapsed = max(1e-6, time.time() - start)
            rate = i / elapsed
            eta = (total - i) / max(rate, 1e-6)
            print(
                f"video-qc {i}/{total} ({i/total:.1%}) rate={rate:.2f}/s eta={eta/60:.1f}min",
                flush=True,
            )
            last = i


def main() -> None:
    ap = argparse.ArgumentParser(description="WorldArena v0.1/v0.2 generated video QC")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--max-sampled-frames",
        type=int,
        default=16,
        help="Legacy alias; v2 uses dense sampling.",
    )
    ap.add_argument("--workers", type=int, default=min(8, max(1, cpu_count() // 2)))
    ap.add_argument("--qc-version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--dense-target-fps", type=float, default=6.0)
    ap.add_argument("--max-dense-frames", type=int, default=96)
    ap.add_argument("--disable-v2-masks", action="store_true")
    ap.add_argument("--save-debug-frames", action="store_true")
    ap.add_argument("--save-debug-masks", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--progress-interval", type=int, default=25)
    ap.add_argument("--run-regression-tests", action="store_true")
    args = ap.parse_args()
    if args.run_regression_tests:
        run_regression_tests()
        return
    if not args.manifest or not args.out:
        ap.error(
            "--manifest and --out are required unless --run-regression-tests is used"
        )
    manifest_path = resolve_manifest(args.manifest)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(manifest_path).reset_index(drop=True)
    cfg = {
        "dense_target_fps": args.dense_target_fps,
        "max_dense_frames": args.max_dense_frames,
        "disable_v2_masks": args.disable_v2_masks,
        "save_debug_frames": args.save_debug_frames,
        "save_debug_masks": args.save_debug_masks,
        "out_dir": str(out_dir),
    }
    rows = [(i, manifest.iloc[i].to_dict(), cfg) for i in range(len(manifest))]
    progress_enabled = not args.no_progress
    if args.workers <= 1:
        qc_rows = list(
            progress_iter(
                (analyze_video(r) for r in rows),
                len(rows),
                progress_enabled,
                args.progress_interval,
            )
        )
    else:
        with Pool(processes=args.workers) as pool:
            qc_rows = list(
                progress_iter(
                    pool.imap_unordered(analyze_video, rows, chunksize=8),
                    len(rows),
                    progress_enabled,
                    args.progress_interval,
                )
            )
    qc = pd.DataFrame(qc_rows).sort_values("row_index").reset_index(drop=True)
    write_outputs(out_dir, manifest_path, manifest, qc, args.qc_version)
    counts = qc["qc_status"].value_counts().to_dict()
    print(f"manifest={manifest_path}")
    print(f"out={out_dir}")
    print(f"qc_version={args.qc_version}")
    print(f"counts={counts}")


if __name__ == "__main__":
    main()
