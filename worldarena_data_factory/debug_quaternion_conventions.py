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


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported manifest: {path}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def matrix_from_json_value(obj: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for key in ["matrix", "intrinsic", "extrinsic", "K", "camera_matrix", "value"]:
            if key in obj:
                mat = matrix_from_json_value(obj[key], shape)
                if mat is not None:
                    return mat
        if {"fx", "fy"}.issubset(obj.keys()) and shape == (3, 3):
            return np.asarray(
                [
                    [float(obj["fx"]), 0.0, float(obj.get("ppx", obj.get("cx", 0.0)))],
                    [0.0, float(obj["fy"]), float(obj.get("ppy", obj.get("cy", 0.0)))],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        if {"rotation_matrix", "translation_vector"}.issubset(obj.keys()) and shape == (4, 4):
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = np.asarray(obj["rotation_matrix"], dtype=np.float32)
            mat[:3, 3] = np.asarray(obj["translation_vector"], dtype=np.float32).reshape(3)
            return mat
    try:
        arr = np.asarray(obj, dtype=np.float32)
        if arr.shape == shape:
            return arr
    except Exception:
        return None
    return None


def load_camera(row: pd.Series, n_frames: int, height: int, width: int):
    import torch

    intrinsic_path = Path(safe_str(row.get("intrinsic_path")))
    extrinsic_path = Path(safe_str(row.get("extrinsic_path")))
    K = matrix_from_json_value(load_json(intrinsic_path), (3, 3))
    extrinsic_obj = load_json(extrinsic_path)
    E = None
    if isinstance(extrinsic_obj, list):
        mats = [matrix_from_json_value(x, (4, 4)) for x in extrinsic_obj]
        mats = [x for x in mats if x is not None]
        if mats:
            E = np.stack(mats, axis=0)
    else:
        E = matrix_from_json_value(extrinsic_obj, (4, 4))
    if K is None or E is None:
        fx = fy = width / (2.0 * np.tan(np.radians(30.0)))
        K = np.asarray([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]], dtype=np.float32)
        E = np.repeat(np.eye(4, dtype=np.float32)[None], n_frames, axis=0)
        source = "fallback"
    else:
        source = "camera_runtime_verified"
    E = np.asarray(E, dtype=np.float32)
    if E.ndim == 2:
        E = np.repeat(E[None], n_frames, axis=0)
    if E.shape[0] < n_frames:
        E = np.concatenate([E, np.repeat(E[-1:], n_frames - E.shape[0], axis=0)], axis=0)
    elif E.shape[0] > n_frames:
        E = E[:n_frames]
    return torch.as_tensor(K, dtype=torch.float32), torch.as_tensor(E, dtype=torch.float32), source


def sample_manifest(df: pd.DataFrame, n: int, seed: int, episode_ids: str = "") -> pd.DataFrame:
    if episode_ids:
        ids = {x.strip() for x in episode_ids.split(",") if x.strip()}
        return df[df["episode_id"].astype(str).isin(ids)].copy()
    if len(df) <= n:
        return df.copy()
    selected = []
    if "task_family" in df.columns:
        grouped = {safe_str(k): g for k, g in df.groupby("task_family", dropna=False)}
        for family in PRIORITY_TASK_FAMILIES:
            group = grouped.get(family)
            if group is not None and not group.empty:
                selected.append(group.sample(1, random_state=seed + len(selected)).index[0])
                if len(selected) >= n:
                    break
    if len(selected) < n:
        rest = df.drop(index=selected, errors="ignore")
        selected.extend(rest.sample(n - len(selected), random_state=seed).index.tolist())
    return df.loc[selected[:n]].copy()


def import_abot_action_utils(abot_root: Path):
    for path in [abot_root / "inference", abot_root]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from diffsynth.utils.action_utils import (  # type: ignore
        get_vace_traj_maps_with_scaled_intrinsic,
        simple_radius_gen_func,
    )
    return get_vace_traj_maps_with_scaled_intrinsic, simple_radius_gen_func


def video_meta(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    meta = {"readable": False, "frame_count": 0, "fps": 24.0, "width": 0, "height": 0}
    if cap.isOpened():
        meta = {
            "readable": True,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 24.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
    cap.release()
    return meta


def resample_action_to_video(action: np.ndarray, n_frames: int) -> np.ndarray:
    if len(action) == n_frames:
        return action
    idx = np.linspace(0, max(len(action) - 1, 0), n_frames).round().astype(int)
    return action[idx]


def convert_quat_order(action: np.ndarray, quat_order: str) -> np.ndarray:
    # ABot expects xyzw in columns 3:7 and 11:15. If source is wxyz, convert wxyz -> xyzw.
    out = action.copy()
    if quat_order == "xyzw":
        return out
    if quat_order != "wxyz":
        raise ValueError(quat_order)
    for start in [3, 11]:
        q = out[:, start : start + 4].copy()
        out[:, start : start + 4] = q[:, [1, 2, 3, 0]]
    return out


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def quat_angular_delta_xyzw(q: np.ndarray) -> np.ndarray:
    if len(q) < 2:
        return np.zeros((0,), dtype=np.float32)
    q = q.astype(np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    dots = np.abs(np.sum(q[1:] * q[:-1], axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    return (2.0 * np.arccos(dots)).astype(np.float32)


def rotation_metrics(raw_action: np.ndarray, quat_order: str) -> dict[str, float]:
    act = convert_quat_order(raw_action, quat_order)
    vals = []
    for start in [3, 11]:
        vals.append(quat_angular_delta_xyzw(act[:, start : start + 4]))
    allv = np.concatenate([x for x in vals if len(x)], axis=0) if any(len(x) for x in vals) else np.zeros((0,), dtype=np.float32)
    if len(allv) == 0:
        return {"quat_delta_mean": 0.0, "quat_delta_p95": 0.0, "quat_jump_count": 0}
    return {
        "quat_delta_mean": float(np.mean(allv)),
        "quat_delta_p95": float(np.quantile(allv, 0.95)),
        "quat_jump_count": int((allv > 1.0).sum()),
    }


def generate_maps(action: np.ndarray, quat_order: str, row: pd.Series, abot_root: Path, n_frames: int, height: int, width: int, z_offset: float):
    import torch

    gen, radius_func = import_abot_action_utils(abot_root)
    pose = convert_quat_order(resample_action_to_video(action, n_frames), quat_order)
    K, extrinsics, camera_source = load_camera(row, n_frames, height, width)
    c2w = extrinsics.unsqueeze(0)
    w2c = torch.linalg.inv(c2w)
    cond = gen(
        torch.as_tensor(pose, dtype=torch.float32),
        w2c,
        c2w,
        K.unsqueeze(0),
        (height, width),
        (height, width),
        radius_gen_func=radius_func,
        ee_local_z_offset=z_offset,
    )
    rgb = cond[:3, 0].detach().cpu().numpy()
    maps = np.transpose(rgb, (1, 2, 3, 0))
    return np.clip(maps * 255.0, 0, 255).astype(np.uint8), camera_source


def action_map_stats(maps: np.ndarray) -> dict[str, float]:
    diff = np.max(np.abs(maps.astype(np.int16) - 50), axis=-1)
    mask = diff > 10
    empty = mask.reshape(mask.shape[0], -1).sum(axis=1) < 5
    return {
        "nonzero_ratio": float(mask.mean()),
        "out_of_frame_ratio": float(empty.mean()),
    }


def put_label(frame: np.ndarray, lines: list[str]) -> None:
    y = 22
    for line in lines:
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18


def write_videos_and_sheet(row: pd.Series, video_path: Path, maps_wxyz: np.ndarray, maps_xyzw: np.ndarray, out: Path, args: argparse.Namespace) -> tuple[str, str, str, str]:
    eid = safe_str(row.get("episode_id"))
    out_w = out / "wxyz_overlays" / f"{eid}.mp4"
    out_x = out / "xyzw_overlays" / f"{eid}.mp4"
    out_s = out / "side_by_side" / f"{eid}.mp4"
    out_c = out / "contact_sheets" / f"{eid}.jpg"
    for p in [out_w, out_x, out_s, out_c]:
        p.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or args.fps or 24.0)
    writer_w = cv2.VideoWriter(str(out_w), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.width, args.height))
    writer_x = cv2.VideoWriter(str(out_x), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.width, args.height))
    writer_s = cv2.VideoWriter(str(out_s), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.width * 3, args.height))
    sheet_items = []
    sheet_indices = set(np.linspace(0, max(n_video - 1, 0), min(12, max(n_video, 1))).round().astype(int).tolist())
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        bgr = cv2.resize(bgr, (args.width, args.height), interpolation=cv2.INTER_AREA)
        mi = min(int(round(idx / max(n_video - 1, 1) * max(len(maps_wxyz) - 1, 0))), len(maps_wxyz) - 1)
        mw = cv2.cvtColor(cv2.resize(maps_wxyz[mi], (args.width, args.height)), cv2.COLOR_RGB2BGR)
        mx = cv2.cvtColor(cv2.resize(maps_xyzw[mi], (args.width, args.height)), cv2.COLOR_RGB2BGR)
        ow = cv2.addWeighted(bgr, 1.0 - args.alpha, mw, args.alpha, 0.0)
        ox = cv2.addWeighted(bgr, 1.0 - args.alpha, mx, args.alpha, 0.0)
        base = bgr.copy()
        put_label(base, [f"{eid} RGB", f"frame={idx} action_t={mi}", safe_str(row.get("task_family"))])
        put_label(ow, ["quat=wxyz", f"frame={idx} action_t={mi}"])
        put_label(ox, ["quat=xyzw", f"frame={idx} action_t={mi}"])
        writer_w.write(ow)
        writer_x.write(ox)
        sbs = np.concatenate([base, ow, ox], axis=1)
        writer_s.write(sbs)
        if idx in sheet_indices:
            sheet_items.append(cv2.cvtColor(sbs, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release(); writer_w.release(); writer_x.release(); writer_s.release()
    make_contact_sheet(sheet_items, out_c, f"{eid}: RGB | wxyz overlay | xyzw overlay")
    return str(out_w), str(out_x), str(out_s), str(out_c)


def make_contact_sheet(frames_rgb: list[np.ndarray], out_path: Path, title: str) -> None:
    thumb_w, thumb_h = 960, 240
    rows = max(1, len(frames_rgb))
    sheet = Image.new("RGB", (thumb_w, rows * thumb_h + 30), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    y = 30
    for i, arr in enumerate(frames_rgb):
        im = Image.fromarray(arr).convert("RGB")
        im.thumbnail((thumb_w, thumb_h))
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        canvas.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
        sheet.paste(canvas, (0, y))
        draw.text((4, y + 4), f"sample {i}", fill=(0, 0, 0), font=font)
        y += thumb_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode_id", "task_family", "video_path", "wxyz_overlay", "xyzw_overlay",
        "suggested_quat_order", "human_rotation_label",
        "side_by_side", "contact_sheet", "wxyz_quat_delta_p95", "xyzw_quat_delta_p95",
        "wxyz_quat_jump_count", "xyzw_quat_jump_count", "wxyz_out_of_frame_ratio", "xyzw_out_of_frame_ratio",
        "wxyz_nonzero_ratio", "xyzw_nonzero_ratio",
    ]
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--abot-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--episode-ids", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--ee-local-z-offset", type=float, default=0.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = sample_manifest(read_table(args.manifest).reset_index(drop=True), args.num_samples, args.seed, args.episode_ids).reset_index(drop=True)
    rows, errors = [], []
    for i, row in df.iterrows():
        eid = safe_str(row.get("episode_id"))
        print(f"[{i + 1}/{len(df)}] {eid}", flush=True)
        try:
            video_path = Path(safe_str(row.get("video_640x480_path") or row.get("video_path")))
            action_path = Path(safe_str(row.get("action_ee16_raw_path") or row.get("action_ee16_path")))
            action = np.load(action_path).astype(np.float32)
            meta = video_meta(video_path)
            if not meta["readable"]:
                raise ValueError(f"unreadable video: {video_path}")
            n_frames = int(meta["frame_count"])
            maps_w, camera_source = generate_maps(action, "wxyz", row, args.abot_root, n_frames, args.height, args.width, args.ee_local_z_offset)
            maps_x, _ = generate_maps(action, "xyzw", row, args.abot_root, n_frames, args.height, args.width, args.ee_local_z_offset)
            ow, ox, sbs, sheet = write_videos_and_sheet(row, video_path, maps_w, maps_x, args.out, args)
            mw = rotation_metrics(action, "wxyz")
            mx = rotation_metrics(action, "xyzw")
            sw = action_map_stats(maps_w)
            sx = action_map_stats(maps_x)
            # Conservative suggestion: use continuity only as a weak prior; keep human label empty.
            if mw["quat_delta_p95"] + 1e-6 < mx["quat_delta_p95"] * 0.8:
                suggested = "wxyz"
            elif mx["quat_delta_p95"] + 1e-6 < mw["quat_delta_p95"] * 0.8:
                suggested = "xyzw"
            else:
                suggested = "manual_review"
            rows.append({
                "episode_id": eid,
                "task_family": safe_str(row.get("task_family")),
                "video_path": str(video_path),
                "wxyz_overlay": ow,
                "xyzw_overlay": ox,
                "suggested_quat_order": suggested,
                "human_rotation_label": "",
                "side_by_side": sbs,
                "contact_sheet": sheet,
                "camera_source": camera_source,
                "ee_local_z_offset": args.ee_local_z_offset,
                "wxyz_quat_delta_mean": mw["quat_delta_mean"],
                "wxyz_quat_delta_p95": mw["quat_delta_p95"],
                "wxyz_quat_jump_count": mw["quat_jump_count"],
                "xyzw_quat_delta_mean": mx["quat_delta_mean"],
                "xyzw_quat_delta_p95": mx["quat_delta_p95"],
                "xyzw_quat_jump_count": mx["quat_jump_count"],
                "wxyz_out_of_frame_ratio": sw["out_of_frame_ratio"],
                "xyzw_out_of_frame_ratio": sx["out_of_frame_ratio"],
                "wxyz_nonzero_ratio": sw["nonzero_ratio"],
                "xyzw_nonzero_ratio": sx["nonzero_ratio"],
            })
        except Exception as exc:
            errors.append({"episode_id": eid, "error": repr(exc)})
            print(f"[ERROR] {eid}: {exc}", flush=True)
    write_csv(args.out / "rotation_convention_review.csv", rows)
    write_csv(args.out / "errors.csv", errors)
    df_rows = pd.DataFrame(rows)
    counts = df_rows["suggested_quat_order"].value_counts().to_dict() if not df_rows.empty else {}
    report = [
        "# Rotation Convention Sanity Report", "",
        f"Samples processed: {len(rows)}", f"Errors: {len(errors)}", "",
        "## Convention", "",
        "- camera/action convention: head_camera + raw K + inverse(extrinsic_cv)",
        f"- ee_local_z_offset: `{args.ee_local_z_offset}`", "",
        "## Weak Automated Continuity Prior", "",
        f"Suggested order counts from quaternion continuity only: `{json.dumps(counts, ensure_ascii=False)}`", "",
        "This is not a final decision. Use the contact sheets and side-by-side videos for human judgement.", "",
    ]
    if not df_rows.empty:
        summary = df_rows[[
            "suggested_quat_order", "wxyz_quat_delta_p95", "xyzw_quat_delta_p95",
            "wxyz_quat_jump_count", "xyzw_quat_jump_count", "wxyz_out_of_frame_ratio", "xyzw_out_of_frame_ratio"
        ]].describe(include="all")
        report.extend(["## Metric Summary", "", summary.to_markdown(), ""])
    report.extend([
        "## Manual Questions", "",
        "- 哪种 quaternion order 的方向轴更稳定：看 `wxyz_quat_delta_p95` / `xyzw_quat_delta_p95`，但最终以视频中轴方向是否稳定为准。",
        "- 哪种更接近 gripper/tool frame：看 side-by-side 中旋转轴是否沿着夹爪/工具视觉方向，而不是只看位置。",
        "- 是否存在方向轴突然翻转或跳变：重点检查 contact sheets 的连续帧。",
        "- 是否存在左右臂方向明显反掉：看左右两臂颜色/轴在接触阶段是否分别落在对应夹爪上。",
        "- 位置是否两者都对但 orientation 不同：这是预期可能出现的情况，本脚本就是为了区分这一点。", "",
        "## Outputs", "",
        f"- review csv: `{args.out / 'rotation_convention_review.csv'}`",
        f"- side-by-side videos: `{args.out / 'side_by_side'}`",
        f"- contact sheets: `{args.out / 'contact_sheets'}`",
        f"- wxyz overlays: `{args.out / 'wxyz_overlays'}`",
        f"- xyzw overlays: `{args.out / 'xyzw_overlays'}`",
    ])
    (args.out / "rotation_convention_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(args.out / "rotation_convention_report.md")
    print(args.out / "rotation_convention_review.csv")


if __name__ == "__main__":
    main()
