"""Action and HDF5 trajectory statistics for WorldArena."""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ACTION_FIELDS = {
    "joint_vector": "/joint_action/vector",
    "left_arm": "/joint_action/left_arm",
    "right_arm": "/joint_action/right_arm",
    "left_gripper": "/joint_action/left_gripper",
    "right_gripper": "/joint_action/right_gripper",
    "left_endpose": "/endpose/left_endpose",
    "right_endpose": "/endpose/right_endpose",
    "left_endpose_gripper": "/endpose/left_gripper",
    "right_endpose_gripper": "/endpose/right_gripper",
}

EPISODE_FIELDS = [
    "split", "episode_id", "task_name", "hdf5_path", "T", "action_dim", "has_nan", "has_inf",
    "joint_action_min", "joint_action_max", "joint_action_mean_l2", "joint_delta_mean_l2",
    "joint_delta_max_l2", "joint_delta_p95_l2", "left_arm_motion_l2_sum", "right_arm_motion_l2_sum",
    "left_arm_active_ratio", "right_arm_active_ratio", "bimanual_active_ratio", "dominant_arm",
    "left_gripper_min", "left_gripper_max", "left_gripper_mean", "left_gripper_std",
    "right_gripper_min", "right_gripper_max", "right_gripper_mean", "right_gripper_std",
    "left_gripper_transition_count", "right_gripper_transition_count", "pause_ratio", "spike_ratio",
    "estimated_motion_segments", "action_complexity_score", "left_quat_norm_mean", "left_quat_norm_std",
    "left_quat_norm_max_abs_dev", "left_quat_norm_close_ratio", "right_quat_norm_mean",
    "right_quat_norm_std", "right_quat_norm_max_abs_dev", "right_quat_norm_close_ratio",
]

DIM_FIELDS = [
    "dim", "global_min", "p01", "p05", "mean", "std", "p50", "p95", "p99", "global_max",
    "zero_or_constant_ratio", "nan_count", "inf_count",
]

TRANSITION_FIELDS = [
    "split", "episode_id", "T", "transition_count", "delta_l2_min", "delta_l2_mean", "delta_l2_std",
    "delta_l2_p50", "delta_l2_p90", "delta_l2_p95", "delta_l2_p99", "delta_l2_max",
    "pause_transition_count", "spike_transition_count", "pause_ratio", "spike_ratio",
    "left_gripper_transition_count", "right_gripper_transition_count", "bimanual_active_ratio",
    "estimated_motion_segments",
]

OUTLIER_FIELDS = [
    "split", "episode_id", "hdf5_path", "reason", "T", "joint_delta_max_l2", "spike_ratio",
    "action_complexity_score", "left_quat_norm_max_abs_dev", "right_quat_norm_max_abs_dev",
]


def maybe_import_h5py():
    try:
        import h5py  # type: ignore
    except Exception as exc:
        return None, exc
    return h5py, None


def read_episode_level(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(root: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    return path if path.is_absolute() else root / path


def safe_array(handle: Any, name: str) -> np.ndarray | None:
    try:
        if name not in handle:
            return None
        return np.asarray(handle[name][()])
    except Exception:
        return None


def finite_flat(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    return flat[np.isfinite(flat)]


def safe_percentile(values: np.ndarray, q: float) -> float:
    finite = finite_flat(values)
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def safe_stat(values: np.ndarray, fn: str) -> float:
    finite = finite_flat(values)
    if finite.size == 0:
        return float("nan")
    if fn == "min":
        return float(np.min(finite))
    if fn == "max":
        return float(np.max(finite))
    if fn == "mean":
        return float(np.mean(finite))
    if fn == "std":
        return float(np.std(finite))
    raise ValueError(fn)


def row_l2(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return np.linalg.norm(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), axis=1)


def delta_l2(values: np.ndarray) -> np.ndarray:
    if values is None or len(values) < 2:
        return np.asarray([], dtype=float)
    return row_l2(np.diff(values, axis=0))


def estimate_binary_threshold(values: np.ndarray) -> float:
    finite = finite_flat(values)
    if finite.size == 0:
        return 0.5
    unique = np.unique(np.round(finite, 6))
    if unique.size <= 8:
        return float((np.min(unique) + np.max(unique)) / 2.0)
    p05 = float(np.percentile(finite, 5))
    p95 = float(np.percentile(finite, 95))
    return float((p05 + p95) / 2.0)


def gripper_transition_count(values: np.ndarray | None, threshold: float) -> int:
    if values is None or len(values) < 2:
        return 0
    finite = np.nan_to_num(np.asarray(values).reshape(-1), nan=threshold, posinf=threshold, neginf=threshold)
    states = finite > threshold
    return int(np.count_nonzero(states[1:] != states[:-1]))


def active_ratio(delta: np.ndarray, threshold: float) -> float:
    if delta.size == 0:
        return 0.0
    return float(np.mean(delta > threshold))


def count_motion_segments(delta: np.ndarray, threshold: float) -> int:
    if delta.size == 0:
        return 0
    active = delta > threshold
    if active.size == 0:
        return 0
    starts = active & np.concatenate([[True], ~active[:-1]])
    return int(np.count_nonzero(starts))


def quaternion_stats(endpose: np.ndarray | None) -> dict[str, float]:
    if endpose is None or endpose.ndim != 2 or endpose.shape[1] < 7:
        return {"mean": float("nan"), "std": float("nan"), "max_abs_dev": float("nan"), "close_ratio": float("nan")}
    quat = np.asarray(endpose[:, 3:7], dtype=float)
    norms = np.linalg.norm(np.nan_to_num(quat, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
    if norms.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "max_abs_dev": float("nan"), "close_ratio": float("nan")}
    dev = np.abs(norms - 1.0)
    return {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "max_abs_dev": float(np.max(dev)),
        "close_ratio": float(np.mean(dev <= 0.05)),
    }


def round_float(value: Any, ndigits: int = 6) -> Any:
    try:
        f = float(value)
    except Exception:
        return value
    if math.isnan(f) or math.isinf(f):
        return ""
    return round(f, ndigits)


def load_hdf5_arrays(h5py: Any, path: Path) -> dict[str, np.ndarray | None]:
    with h5py.File(path, "r") as handle:
        return {key: safe_array(handle, field) for key, field in ACTION_FIELDS.items()}


def build_dim_stats(all_vectors: list[np.ndarray], episode_constant_counts: np.ndarray, nan_counts: np.ndarray, inf_counts: np.ndarray, episode_count: int) -> list[dict[str, Any]]:
    if all_vectors:
        joined = np.concatenate(all_vectors, axis=0)
    else:
        joined = np.empty((0, 14), dtype=float)
    rows = []
    for dim in range(14):
        values = joined[:, dim] if joined.size else np.asarray([], dtype=float)
        finite = values[np.isfinite(values)] if values.size else np.asarray([], dtype=float)
        rows.append({
            "dim": dim,
            "global_min": round_float(np.min(finite) if finite.size else float("nan")),
            "p01": round_float(np.percentile(finite, 1) if finite.size else float("nan")),
            "p05": round_float(np.percentile(finite, 5) if finite.size else float("nan")),
            "mean": round_float(np.mean(finite) if finite.size else float("nan")),
            "std": round_float(np.std(finite) if finite.size else float("nan")),
            "p50": round_float(np.percentile(finite, 50) if finite.size else float("nan")),
            "p95": round_float(np.percentile(finite, 95) if finite.size else float("nan")),
            "p99": round_float(np.percentile(finite, 99) if finite.size else float("nan")),
            "global_max": round_float(np.max(finite) if finite.size else float("nan")),
            "zero_or_constant_ratio": round_float(episode_constant_counts[dim] / max(episode_count, 1)),
            "nan_count": int(nan_counts[dim]),
            "inf_count": int(inf_counts[dim]),
        })
    return rows


def episode_stats(row: dict[str, str], arrays: dict[str, np.ndarray | None], root: Path, thresholds: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    vector = arrays["joint_vector"]
    if vector is None:
        raise ValueError("missing /joint_action/vector")
    vector = np.asarray(vector, dtype=float)
    T = int(vector.shape[0]) if vector.ndim >= 1 else 0
    action_dim = int(vector.shape[1]) if vector.ndim == 2 else 0
    delta = delta_l2(vector)
    left_delta = delta_l2(arrays["left_arm"] if arrays["left_arm"] is not None else vector[:, :6])
    right_delta = delta_l2(arrays["right_arm"] if arrays["right_arm"] is not None else vector[:, 7:13])

    active_thr = thresholds["arm_active_threshold"]
    pause_thr = thresholds["pause_threshold"]
    spike_thr = thresholds["spike_threshold"]
    left_active = left_delta > active_thr if left_delta.size else np.asarray([], dtype=bool)
    right_active = right_delta > active_thr if right_delta.size else np.asarray([], dtype=bool)
    aligned = min(left_active.size, right_active.size)
    bimanual_ratio = float(np.mean(left_active[:aligned] & right_active[:aligned])) if aligned else 0.0

    left_motion = float(np.sum(left_delta)) if left_delta.size else 0.0
    right_motion = float(np.sum(right_delta)) if right_delta.size else 0.0
    if left_motion > right_motion * 1.15:
        dominant = "left"
    elif right_motion > left_motion * 1.15:
        dominant = "right"
    else:
        dominant = "balanced"

    left_gripper = arrays["left_gripper"]
    right_gripper = arrays["right_gripper"]
    left_q = quaternion_stats(arrays["left_endpose"])
    right_q = quaternion_stats(arrays["right_endpose"])

    pause_ratio = float(np.mean(delta <= pause_thr)) if delta.size else 0.0
    spike_ratio = float(np.mean(delta >= spike_thr)) if delta.size else 0.0
    estimated_segments = count_motion_segments(delta, pause_thr)
    complexity = (
        min(float(np.mean(delta)) / max(thresholds["delta_mean_ref"], 1e-9), 3.0)
        + min(float(np.percentile(delta, 95)) / max(thresholds["delta_p95_ref"], 1e-9), 3.0) if delta.size else 0.0
    )
    complexity += bimanual_ratio * 1.5
    complexity += min((gripper_transition_count(left_gripper, thresholds["left_gripper_threshold"]) + gripper_transition_count(right_gripper, thresholds["right_gripper_threshold"])) / 6.0, 1.5)
    complexity += min(estimated_segments / 8.0, 1.0)
    complexity = min(complexity, 10.0)

    episode_row = {
        "split": row["dataset"],
        "episode_id": int(row["episode_id"]),
        "task_name": row.get("task_name", ""),
        "hdf5_path": row.get("hdf5_path", ""),
        "T": T,
        "action_dim": action_dim,
        "has_nan": bool(np.isnan(vector).any()),
        "has_inf": bool(np.isinf(vector).any()),
        "joint_action_min": round_float(safe_stat(vector, "min")),
        "joint_action_max": round_float(safe_stat(vector, "max")),
        "joint_action_mean_l2": round_float(np.mean(row_l2(vector)) if T else float("nan")),
        "joint_delta_mean_l2": round_float(np.mean(delta) if delta.size else float("nan")),
        "joint_delta_max_l2": round_float(np.max(delta) if delta.size else float("nan")),
        "joint_delta_p95_l2": round_float(np.percentile(delta, 95) if delta.size else float("nan")),
        "left_arm_motion_l2_sum": round_float(left_motion),
        "right_arm_motion_l2_sum": round_float(right_motion),
        "left_arm_active_ratio": round_float(active_ratio(left_delta, active_thr)),
        "right_arm_active_ratio": round_float(active_ratio(right_delta, active_thr)),
        "bimanual_active_ratio": round_float(bimanual_ratio),
        "dominant_arm": dominant,
        "left_gripper_min": round_float(safe_stat(left_gripper, "min") if left_gripper is not None else float("nan")),
        "left_gripper_max": round_float(safe_stat(left_gripper, "max") if left_gripper is not None else float("nan")),
        "left_gripper_mean": round_float(safe_stat(left_gripper, "mean") if left_gripper is not None else float("nan")),
        "left_gripper_std": round_float(safe_stat(left_gripper, "std") if left_gripper is not None else float("nan")),
        "right_gripper_min": round_float(safe_stat(right_gripper, "min") if right_gripper is not None else float("nan")),
        "right_gripper_max": round_float(safe_stat(right_gripper, "max") if right_gripper is not None else float("nan")),
        "right_gripper_mean": round_float(safe_stat(right_gripper, "mean") if right_gripper is not None else float("nan")),
        "right_gripper_std": round_float(safe_stat(right_gripper, "std") if right_gripper is not None else float("nan")),
        "left_gripper_transition_count": gripper_transition_count(left_gripper, thresholds["left_gripper_threshold"]),
        "right_gripper_transition_count": gripper_transition_count(right_gripper, thresholds["right_gripper_threshold"]),
        "pause_ratio": round_float(pause_ratio),
        "spike_ratio": round_float(spike_ratio),
        "estimated_motion_segments": estimated_segments,
        "action_complexity_score": round_float(complexity),
        "left_quat_norm_mean": round_float(left_q["mean"]),
        "left_quat_norm_std": round_float(left_q["std"]),
        "left_quat_norm_max_abs_dev": round_float(left_q["max_abs_dev"]),
        "left_quat_norm_close_ratio": round_float(left_q["close_ratio"]),
        "right_quat_norm_mean": round_float(right_q["mean"]),
        "right_quat_norm_std": round_float(right_q["std"]),
        "right_quat_norm_max_abs_dev": round_float(right_q["max_abs_dev"]),
        "right_quat_norm_close_ratio": round_float(right_q["close_ratio"]),
    }
    transition_row = {
        "split": row["dataset"],
        "episode_id": int(row["episode_id"]),
        "T": T,
        "transition_count": max(T - 1, 0),
        "delta_l2_min": round_float(np.min(delta) if delta.size else float("nan")),
        "delta_l2_mean": episode_row["joint_delta_mean_l2"],
        "delta_l2_std": round_float(np.std(delta) if delta.size else float("nan")),
        "delta_l2_p50": round_float(np.percentile(delta, 50) if delta.size else float("nan")),
        "delta_l2_p90": round_float(np.percentile(delta, 90) if delta.size else float("nan")),
        "delta_l2_p95": episode_row["joint_delta_p95_l2"],
        "delta_l2_p99": round_float(np.percentile(delta, 99) if delta.size else float("nan")),
        "delta_l2_max": episode_row["joint_delta_max_l2"],
        "pause_transition_count": int(np.count_nonzero(delta <= pause_thr)) if delta.size else 0,
        "spike_transition_count": int(np.count_nonzero(delta >= spike_thr)) if delta.size else 0,
        "pause_ratio": episode_row["pause_ratio"],
        "spike_ratio": episode_row["spike_ratio"],
        "left_gripper_transition_count": episode_row["left_gripper_transition_count"],
        "right_gripper_transition_count": episode_row["right_gripper_transition_count"],
        "bimanual_active_ratio": episode_row["bimanual_active_ratio"],
        "estimated_motion_segments": estimated_segments,
    }
    return episode_row, transition_row


def make_action_plots(out_dir: Path, episode_rows: list[dict[str, Any]], dim_rows: list[dict[str, Any]], all_vectors: list[np.ndarray], all_delta_l2: list[np.ndarray], grippers: dict[str, list[np.ndarray]]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save_hist(values: list[float], filename: str, title: str, xlabel: str, bins: int = 50):
        path = plots_dir / filename
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values if values else [0], bins=bins, color="#2563eb", alpha=0.82)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    save_hist([float(r["T"]) for r in episode_rows], "trajectory_length_distribution.png", "Trajectory Length Distribution", "T")

    if all_vectors:
        joined = np.concatenate(all_vectors, axis=0)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.boxplot([joined[:, i][np.isfinite(joined[:, i])] for i in range(joined.shape[1])], showfliers=False)
        ax.set_title("Joint Action Dim Boxplot")
        ax.set_xlabel("joint_action dim")
        ax.set_ylabel("value")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = plots_dir / "joint_action_dim_boxplot.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    deltas = np.concatenate(all_delta_l2) if all_delta_l2 else np.asarray([], dtype=float)
    save_hist(deltas[np.isfinite(deltas)].tolist(), "joint_delta_l2_distribution.png", "Joint Delta L2 Distribution", "||a[t+1]-a[t]||2")

    for side in ("left", "right"):
        vals = np.concatenate(grippers[side]) if grippers.get(side) else np.asarray([], dtype=float)
        vals = vals[np.isfinite(vals)]
        save_hist(vals.tolist(), f"gripper_hist_{side}.png", f"{side.title()} Gripper Histogram", "gripper value")

    transitions = [float(r["left_gripper_transition_count"]) + float(r["right_gripper_transition_count"]) for r in episode_rows]
    save_hist(transitions, "gripper_transition_distribution.png", "Gripper Transition Distribution", "left+right transition count", bins=40)
    save_hist([float(r["bimanual_active_ratio"]) for r in episode_rows], "bimanual_activity_distribution.png", "Bimanual Activity Distribution", "bimanual active ratio")
    save_hist([float(r["action_complexity_score"]) for r in episode_rows], "action_complexity_distribution.png", "Action Complexity Distribution", "complexity score")
    return paths


def build_outliers(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not episode_rows:
        return []
    T_p99 = np.percentile([float(r["T"]) for r in episode_rows], 99)
    delta_p99 = np.percentile([float(r["joint_delta_max_l2"] or 0) for r in episode_rows], 99)
    complexity_p99 = np.percentile([float(r["action_complexity_score"] or 0) for r in episode_rows], 99)
    out = []
    for row in episode_rows:
        reasons = []
        if row["has_nan"]:
            reasons.append("has_nan")
        if row["has_inf"]:
            reasons.append("has_inf")
        if float(row["T"]) >= T_p99:
            reasons.append("trajectory_length_p99")
        if float(row["joint_delta_max_l2"] or 0) >= delta_p99:
            reasons.append("delta_max_p99")
        if float(row["action_complexity_score"] or 0) >= complexity_p99:
            reasons.append("complexity_p99")
        for qkey in ("left_quat_norm_max_abs_dev", "right_quat_norm_max_abs_dev"):
            value = row.get(qkey, "")
            if value != "" and float(value) > 0.1:
                reasons.append(qkey + ">0.1")
        if reasons:
            out.append({
                "split": row["split"],
                "episode_id": row["episode_id"],
                "hdf5_path": row["hdf5_path"],
                "reason": ";".join(reasons),
                "T": row["T"],
                "joint_delta_max_l2": row["joint_delta_max_l2"],
                "spike_ratio": row["spike_ratio"],
                "action_complexity_score": row["action_complexity_score"],
                "left_quat_norm_max_abs_dev": row["left_quat_norm_max_abs_dev"],
                "right_quat_norm_max_abs_dev": row["right_quat_norm_max_abs_dev"],
            })
    return out


def abot_a2v_recommendation(summary: dict[str, Any]) -> dict[str, str]:
    quat_close = min(summary.get("left_quat_close_mean", 0.0), summary.get("right_quat_close_mean", 0.0))
    complexity_mean = summary.get("action_complexity_mean", 0.0)
    text = []
    if quat_close >= 0.98:
        text.append("End-effector quaternion norms are mostly valid, so ee16 is usable as an auxiliary representation.")
    else:
        text.append("End-effector quaternion norms show deviations; inspect ee16 before using it as the only action signal.")
    if complexity_mean >= 4.0:
        text.append("Motion complexity is non-trivial; joint14+ee16 is recommended for richer A2V conditioning if model capacity allows.")
        primary = "joint14+ee16"
    else:
        text.append("joint14 is compact and directly matches /joint_action/vector; use it as the primary baseline.")
        primary = "joint14"
    text.append("Compare joint14, ee16, and joint14+ee16 in ablations; use joint14 as the minimum reliable representation.")
    return {"recommended_primary": primary, "recommendation": " ".join(text)}


def unavailable_outputs(out_dir: Path, error: Exception) -> dict[str, Any]:
    write_csv(out_dir / "action_stats_episode.csv", [], EPISODE_FIELDS)
    write_csv(out_dir / "action_stats_dim.csv", [], DIM_FIELDS)
    write_csv(out_dir / "action_transition_stats.csv", [], TRANSITION_FIELDS)
    write_csv(out_dir / "action_outlier_episodes.csv", [], OUTLIER_FIELDS)
    return {
        "available": False,
        "error": f"h5py is not available: {error}",
        "outputs": {
            "action_stats_episode_csv": "action_stats_episode.csv",
            "action_stats_dim_csv": "action_stats_dim.csv",
            "action_transition_stats_csv": "action_transition_stats.csv",
            "action_outlier_episodes_csv": "action_outlier_episodes.csv",
        },
    }


def compute_action_statistics(root: Path, out_dir: Path, episode_csv: Path, logger: logging.Logger) -> dict[str, Any]:
    h5py, import_error = maybe_import_h5py()
    if h5py is None:
        logger.warning("Skipping full action stats because h5py is unavailable: %s", import_error)
        return unavailable_outputs(out_dir, import_error)

    rows = read_episode_level(episode_csv)
    all_vectors: list[np.ndarray] = []
    all_delta_l2: list[np.ndarray] = []
    grippers: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    episode_constant_counts = np.zeros(14, dtype=int)
    nan_counts = np.zeros(14, dtype=int)
    inf_counts = np.zeros(14, dtype=int)
    load_cache: list[tuple[dict[str, str], dict[str, np.ndarray | None]]] = []
    errors = []

    logger.info("Reading HDF5 actions for global stats from %s", episode_csv)
    for row in rows:
        path = resolve_path(root, row.get("hdf5_path", ""))
        try:
            arrays = load_hdf5_arrays(h5py, path)
            vector = arrays["joint_vector"]
            if vector is None:
                raise ValueError("missing /joint_action/vector")
            vector = np.asarray(vector, dtype=float)
            if vector.ndim != 2 or vector.shape[1] != 14:
                raise ValueError(f"expected /joint_action/vector shape (T,14), got {vector.shape}")
            load_cache.append((row, arrays))
            all_vectors.append(vector)
            d = delta_l2(vector)
            if d.size:
                all_delta_l2.append(d)
            for dim in range(14):
                values = vector[:, dim]
                nan_counts[dim] += int(np.isnan(values).sum())
                inf_counts[dim] += int(np.isinf(values).sum())
                finite = values[np.isfinite(values)]
                if finite.size == 0 or np.max(finite) - np.min(finite) <= 1e-8 or np.max(np.abs(finite)) <= 1e-8:
                    episode_constant_counts[dim] += 1
            if arrays["left_gripper"] is not None:
                grippers["left"].append(np.asarray(arrays["left_gripper"], dtype=float).reshape(-1))
            if arrays["right_gripper"] is not None:
                grippers["right"].append(np.asarray(arrays["right_gripper"], dtype=float).reshape(-1))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            logger.warning("Failed reading %s: %s", path, exc)

    left_gripper_values = np.concatenate(grippers["left"]) if grippers["left"] else np.asarray([], dtype=float)
    right_gripper_values = np.concatenate(grippers["right"]) if grippers["right"] else np.asarray([], dtype=float)
    all_delta = np.concatenate(all_delta_l2) if all_delta_l2 else np.asarray([], dtype=float)
    finite_delta = all_delta[np.isfinite(all_delta)] if all_delta.size else np.asarray([], dtype=float)
    thresholds = {
        "left_gripper_threshold": estimate_binary_threshold(left_gripper_values),
        "right_gripper_threshold": estimate_binary_threshold(right_gripper_values),
        "pause_threshold": float(np.percentile(finite_delta, 10)) if finite_delta.size else 1e-6,
        "arm_active_threshold": float(np.percentile(finite_delta, 25)) if finite_delta.size else 1e-4,
        "spike_threshold": float(np.percentile(finite_delta, 99)) if finite_delta.size else 1.0,
        "delta_mean_ref": float(np.mean(finite_delta)) if finite_delta.size else 1.0,
        "delta_p95_ref": float(np.percentile(finite_delta, 95)) if finite_delta.size else 1.0,
    }

    episode_rows = []
    transition_rows = []
    for row, arrays in load_cache:
        ep_row, tr_row = episode_stats(row, arrays, root, thresholds)
        episode_rows.append(ep_row)
        transition_rows.append(tr_row)

    dim_rows = build_dim_stats(all_vectors, episode_constant_counts, nan_counts, inf_counts, len(load_cache))
    outlier_rows = build_outliers(episode_rows)
    plot_paths = make_action_plots(out_dir, episode_rows, dim_rows, all_vectors, all_delta_l2, grippers)

    write_csv(out_dir / "action_stats_episode.csv", episode_rows, EPISODE_FIELDS)
    write_csv(out_dir / "action_stats_dim.csv", dim_rows, DIM_FIELDS)
    write_csv(out_dir / "action_transition_stats.csv", transition_rows, TRANSITION_FIELDS)
    write_csv(out_dir / "action_outlier_episodes.csv", outlier_rows, OUTLIER_FIELDS)

    by_split = defaultdict(int)
    dominant = Counter()
    for row in episode_rows:
        by_split[row["split"]] += 1
        dominant[row["dominant_arm"]] += 1
    left_close = [float(row["left_quat_norm_close_ratio"] or 0) for row in episode_rows]
    right_close = [float(row["right_quat_norm_close_ratio"] or 0) for row in episode_rows]
    complexity = [float(row["action_complexity_score"] or 0) for row in episode_rows]
    summary = {
        "available": True,
        "episode_count": len(episode_rows),
        "by_split": dict(by_split),
        "thresholds": {key: round_float(value) for key, value in thresholds.items()},
        "T": {
            "min": int(min(float(row["T"]) for row in episode_rows)) if episode_rows else None,
            "median": round_float(np.median([float(row["T"]) for row in episode_rows])) if episode_rows else None,
            "mean": round_float(np.mean([float(row["T"]) for row in episode_rows])) if episode_rows else None,
            "p95": round_float(np.percentile([float(row["T"]) for row in episode_rows], 95)) if episode_rows else None,
            "max": int(max(float(row["T"]) for row in episode_rows)) if episode_rows else None,
        },
        "dominant_arm": dict(dominant.most_common()),
        "left_quat_close_mean": round_float(np.mean(left_close) if left_close else float("nan")),
        "right_quat_close_mean": round_float(np.mean(right_close) if right_close else float("nan")),
        "action_complexity_mean": round_float(np.mean(complexity) if complexity else float("nan")),
        "outlier_count": len(outlier_rows),
        "plots": plot_paths,
        "errors": errors[:50],
    }
    summary["abot_a2v_recommendation"] = abot_a2v_recommendation(summary)
    logger.info("Wrote action statistics CSVs and %d action plots", len(plot_paths))
    return summary
