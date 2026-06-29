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
import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

VARIANT_NAMES = [
    "extrinsic_cv_direct",
    "inverse_extrinsic_cv",
    "cam2world_gl_direct",
    "inverse_cam2world_gl",
    "extrinsic_cv_direct_glcv",
    "cam2world_gl_direct_glcv",
]
CAMERA_NAMES = ["head_camera", "front_camera"]
Z_OFFSETS = [0.0, 0.10, 0.23]


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    if "task_family" not in df.columns:
        return df.sample(n, random_state=seed).copy()
    selected = []
    groups = list(df.groupby("task_family", dropna=False))
    for _, group in groups:
        if len(selected) >= n:
            break
        selected.append(group.sample(1, random_state=seed + len(selected)).index[0])
    if len(selected) < n:
        rest = df.drop(index=selected, errors="ignore")
        selected.extend(rest.sample(n - len(selected), random_state=seed).index.tolist())
    return df.loc[selected[:n]].copy()


def decode_hdf5_rgb(value: Any) -> np.ndarray | None:
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        out = arr[..., :3]
        return out.astype(np.uint8) if out.dtype != np.uint8 else out
    if arr.ndim == 1 and arr.dtype.kind in ("S", "O", "V", "U"):
        value = arr[0]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, np.void):
        value = bytes(value)
    if isinstance(value, str):
        value = value.encode("latin1")
    if isinstance(value, bytes):
        value = value.rstrip(b"\0")
        buf = np.frombuffer(value, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def read_video_first_frame(path: Path) -> tuple[np.ndarray | None, dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    meta = {"readable": False, "width": 0, "height": 0, "fps": 0.0, "frame_count": 0}
    if not cap.isOpened():
        cap.release()
        return None, meta
    meta.update(
        {
            "readable": True,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        }
    )
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return None, meta
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), meta


def simple_ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    ssim = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2) + 1e-6
    )
    return float(np.clip(ssim.mean(), -1.0, 1.0))


def compare_frames(video: np.ndarray, camera: np.ndarray | None, size=(640, 480)) -> dict[str, float]:
    if video is None or camera is None:
        return {"l1": float("inf"), "ssim": -1.0}
    v = cv2.resize(video, size, interpolation=cv2.INTER_AREA)
    c = cv2.resize(camera, size, interpolation=cv2.INTER_AREA)
    return {"l1": float(np.mean(np.abs(v.astype(np.float32) - c.astype(np.float32))) / 255.0), "ssim": simple_ssim_gray(v, c)}


def read_hdf5_camera_first_frames(hdf5_path: Path) -> dict[str, np.ndarray | None]:
    out = {}
    with h5py.File(hdf5_path, "r") as f:
        for cam in CAMERA_NAMES:
            key = f"/observation/{cam}/rgb"
            if key in f:
                out[cam] = decode_hdf5_rgb(f[key][0])
            else:
                out[cam] = None
    return out


def read_camera_arrays(hdf5_path: Path, camera_name: str) -> dict[str, np.ndarray]:
    base = f"/observation/{camera_name}"
    with h5py.File(hdf5_path, "r") as f:
        return {
            "intrinsic_cv": np.asarray(f[f"{base}/intrinsic_cv"]).astype(np.float32),
            "extrinsic_cv": np.asarray(f[f"{base}/extrinsic_cv"]).astype(np.float32),
            "cam2world_gl": np.asarray(f[f"{base}/cam2world_gl"]).astype(np.float32),
        }


def as_4x4(mats: np.ndarray) -> np.ndarray:
    mats = np.asarray(mats, dtype=np.float32)
    if mats.ndim == 2:
        mats = mats[None]
    if mats.shape[-2:] == (4, 4):
        return mats.copy()
    if mats.shape[-2:] == (3, 4):
        out = np.tile(np.eye(4, dtype=np.float32), (mats.shape[0], 1, 1))
        out[:, :3, :4] = mats
        return out
    raise ValueError(f"bad extrinsic shape: {mats.shape}")


def make_c2w(arrs: dict[str, np.ndarray], variant: str) -> np.ndarray:
    cv_to_gl = np.diag([1, -1, -1, 1]).astype(np.float32)
    ext_cv = as_4x4(arrs["extrinsic_cv"])
    c2w_gl = as_4x4(arrs["cam2world_gl"])
    if variant == "extrinsic_cv_direct":
        return ext_cv
    if variant == "inverse_extrinsic_cv":
        return np.linalg.inv(ext_cv)
    if variant == "cam2world_gl_direct":
        return c2w_gl
    if variant == "inverse_cam2world_gl":
        return np.linalg.inv(c2w_gl)
    if variant == "extrinsic_cv_direct_glcv":
        # Treat extrinsic_cv as w2c_cv; convert to a GL-like w2c then invert for c2w.
        w2c = np.einsum("ij,tjk->tik", cv_to_gl, ext_cv)
        return np.linalg.inv(w2c)
    if variant == "cam2world_gl_direct_glcv":
        # Convert a GL camera-to-world matrix to CV-style camera-to-world.
        return np.einsum("tij,jk->tik", c2w_gl, cv_to_gl)
    raise ValueError(variant)


def resample_action(action: np.ndarray, n_frames: int) -> np.ndarray:
    if len(action) == n_frames:
        return action
    idx = np.linspace(0, max(len(action) - 1, 0), n_frames).round().astype(int)
    return action[idx]


def quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    x, y, z, w = q
    n = max(float(np.dot(q, q)), 1e-8)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.asarray(
        [
            [1 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1 - xx - yy],
        ],
        dtype=np.float32,
    )


def pose_to_mat(xyz_quat: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = quat_xyzw_to_rot(xyz_quat[3:7])
    mat[:3, 3] = xyz_quat[:3]
    return mat


def project_point(pt_cam: np.ndarray, K: np.ndarray) -> tuple[int, int] | None:
    z = float(pt_cam[2])
    if z <= 1e-5:
        return None
    uvw = K @ pt_cam[:3]
    return int(round(uvw[0] / z)), int(round(uvw[1] / z))


def generate_debug_maps(action: np.ndarray, K: np.ndarray, c2w: np.ndarray, n_frames: int, height: int, width: int, z_offset: float) -> np.ndarray:
    pose = resample_action(action, n_frames)
    if c2w.shape[0] < n_frames:
        c2w = np.concatenate([c2w, np.repeat(c2w[-1:], n_frames - c2w.shape[0], axis=0)], axis=0)
    elif c2w.shape[0] > n_frames:
        c2w = c2w[:n_frames]
    w2c = np.linalg.inv(c2w)
    maps = []
    prev_l = prev_r = None
    local_offset = np.asarray([0.0, 0.0, z_offset, 1.0], dtype=np.float32)
    for t in range(n_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8) + 50
        lmat = pose_to_mat(pose[t, 0:7])
        rmat = pose_to_mat(pose[t, 8:15])
        pts = []
        for mat in [lmat, rmat]:
            p_world = mat @ local_offset
            p_cam = w2c[t] @ p_world
            pts.append(project_point(p_cam, K))
        for idx, pt in enumerate(pts):
            if pt is None:
                continue
            color = (60, 240, 80) if idx == 0 else (240, 80, 80)
            prev = prev_l if idx == 0 else prev_r
            if 0 <= pt[0] < width and 0 <= pt[1] < height:
                if prev is not None and 0 <= prev[0] < width and 0 <= prev[1] < height:
                    cv2.line(img, prev, pt, color, 4, cv2.LINE_AA)
                cv2.circle(img, pt, 7, color, -1, cv2.LINE_AA)
            if idx == 0:
                prev_l = pt
            else:
                prev_r = pt
        maps.append(img)
    return np.stack(maps, axis=0)


def action_map_stats(maps: np.ndarray) -> dict[str, float]:
    diff = np.max(np.abs(maps.astype(np.int16) - 50), axis=-1)
    mask = diff > 10
    nz = np.argwhere(mask)
    empty = mask.reshape(mask.shape[0], -1).sum(axis=1) < 5
    if len(nz) == 0:
        return {
            "action_map_nonzero_ratio": 0.0,
            "out_of_frame_ratio": 1.0,
            "bbox_area_ratio": 0.0,
            "bbox_center_x": -1.0,
            "bbox_center_y": -1.0,
            "distance_to_image_center": 1.0,
            "empty_map_count": int(empty.sum()),
            "corner_map_count": 0,
            "bbox_xmin": -1,
            "bbox_ymin": -1,
            "bbox_xmax": -1,
            "bbox_ymax": -1,
        }
    _, ys, xs = nz[:, 0], nz[:, 1], nz[:, 2]
    xmin, xmax, ymin, ymax = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    h, w = mask.shape[1:]
    area = ((xmax - xmin + 1) * (ymax - ymin + 1)) / float(w * h)
    cx, cy = (xmin + xmax) / 2.0 / w, (ymin + ymax) / 2.0 / h
    dist = float(math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / math.sqrt(0.5))
    frame_centers = []
    corner_count = 0
    for t in range(mask.shape[0]):
        pts = np.argwhere(mask[t])
        if len(pts) < 5:
            continue
        yy, xx = pts[:, 0], pts[:, 1]
        fcx, fcy = float(xx.mean()) / w, float(yy.mean()) / h
        frame_centers.append((fcx, fcy))
        if (fcx < 0.15 or fcx > 0.85) and (fcy < 0.15 or fcy > 0.85):
            corner_count += 1
    return {
        "action_map_nonzero_ratio": float(mask.mean()),
        "out_of_frame_ratio": float(empty.mean()),
        "bbox_area_ratio": float(area),
        "bbox_center_x": float(cx),
        "bbox_center_y": float(cy),
        "distance_to_image_center": dist,
        "empty_map_count": int(empty.sum()),
        "corner_map_count": int(corner_count),
        "bbox_xmin": xmin,
        "bbox_ymin": ymin,
        "bbox_xmax": xmax,
        "bbox_ymax": ymax,
    }


def overlay(rgb: np.ndarray, amap: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(rgb, 1.0 - alpha, amap, alpha, 0.0)


def put_text(img: np.ndarray, lines: list[str]) -> None:
    y = 18
    for line in lines:
        cv2.putText(img, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        y += 15


def make_grid_sheet(video_frame: np.ndarray, maps_by_variant: list[tuple[dict[str, Any], np.ndarray]], out_path: Path, title: str, alpha: float) -> None:
    thumb_w, thumb_h = 220, 165
    cols = 6
    header = 44
    rows = int(math.ceil(len(maps_by_variant) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h + header), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    v = cv2.resize(video_frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    for i, (meta, amap) in enumerate(maps_by_variant):
        a = cv2.resize(amap, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
        ov = overlay(v, a, alpha)
        put_text(ov, [meta["camera_name"], meta["variant"], f"z={meta['z_offset']}"])
        im = Image.fromarray(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
        x = (i % cols) * thumb_w
        y = header + (i // cols) * thumb_h
        sheet.paste(im, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def read_video_sample_frames(path: Path, n: int, width: int, height: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], {"readable": False}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    idxs = np.linspace(0, max(total - 1, 0), max(1, min(n, total))).round().astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if ok and bgr is not None:
            frames.append(cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA))
    cap.release()
    return frames, {"readable": True, "frame_count": total, "fps": fps}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--max-video-frames", type=int, default=96)
    ap.add_argument("--make-side-by-side-compare", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    ensure_dir(out)
    df = sample_rows(read_table(Path(args.manifest)), args.num_samples, args.seed)
    match_rows, score_rows, errors = [], [], []

    for idx, row in df.reset_index(drop=True).iterrows():
        eid = safe_str(row.get("episode_id")) or f"sample_{idx:03d}"
        print(f"[{idx + 1}/{len(df)}] {eid}", flush=True)
        try:
            video_path = Path(safe_str(row.get("video_640x480_path") or row.get("video_path")))
            hdf5_path = Path(safe_str(row.get("raw_hdf5_path")))
            action_path = Path(safe_str(row.get("action_ee16_raw_path") or row.get("action_ee16_path")))
            video_first, vmeta = read_video_first_frame(video_path)
            if video_first is None:
                raise ValueError(f"unreadable video: {video_path}")
            hframes = read_hdf5_camera_first_frames(hdf5_path)
            cam_scores = {cam: compare_frames(video_first, hframes.get(cam), (args.width, args.height)) for cam in CAMERA_NAMES}
            best_cam = min(CAMERA_NAMES, key=lambda c: cam_scores[c]["l1"])
            match_rows.append(
                {
                    "episode_id": eid,
                    "video_path": str(video_path),
                    "raw_hdf5_path": str(hdf5_path),
                    "best_camera_by_l1": best_cam,
                    "head_l1": cam_scores["head_camera"]["l1"],
                    "front_l1": cam_scores["front_camera"]["l1"],
                    "head_ssim": cam_scores["head_camera"]["ssim"],
                    "front_ssim": cam_scores["front_camera"]["ssim"],
                }
            )

            action = np.load(action_path).astype(np.float32)
            frames_bgr, meta = read_video_sample_frames(video_path, args.max_video_frames, args.width, args.height)
            n_frames = len(frames_bgr)
            if n_frames == 0:
                raise ValueError("no decodable sampled video frames")
            video_frame_bgr = frames_bgr[min(len(frames_bgr) // 2, len(frames_bgr) - 1)]
            maps_for_sheet = []
            for camera_name in CAMERA_NAMES:
                arrs = read_camera_arrays(hdf5_path, camera_name)
                K_raw = arrs["intrinsic_cv"][0]
                for intrinsic_mode in ["raw", "scaled"]:
                    if intrinsic_mode == "scaled":
                        # The exported video is 640x480. HDF5 head_camera is usually already 640x480;
                        # front_camera is usually 320x240 and must be scaled for fair comparison.
                        src_w = max(float(K_raw[0, 2] * 2.0), 1.0)
                        src_h = max(float(K_raw[1, 2] * 2.0), 1.0)
                        K = K_raw.copy()
                        K[0, :] *= float(args.width) / src_w
                        K[1, :] *= float(args.height) / src_h
                    else:
                        K = K_raw.copy()
                    for variant in VARIANT_NAMES:
                        c2w = make_c2w(arrs, variant)
                        for z in Z_OFFSETS:
                            maps = generate_debug_maps(action, K, c2w, n_frames, args.height, args.width, z)
                            st = action_map_stats(maps)
                            meta_row = {
                                "episode_id": eid,
                                "task_family": safe_str(row.get("task_family")),
                                "robotwin_task_name": safe_str(row.get("robotwin_task_name")),
                                "camera_name": camera_name,
                                "intrinsic_mode": intrinsic_mode,
                                "variant": variant,
                                "z_offset": z,
                                "video_best_camera_by_l1": best_cam,
                                **st,
                            }
                            score_rows.append(meta_row)
                            # Use middle frame for the grid cell.
                            maps_for_sheet.append((meta_row, cv2.cvtColor(maps[len(maps)//2], cv2.COLOR_RGB2BGR)))
            sheet_path = out / "convention_grid" / "contact_sheets" / f"{eid}.jpg"
            make_grid_sheet(video_frame_bgr, maps_for_sheet, sheet_path, f"{eid}: overlay grid (RGB + debug action map)", args.alpha)
        except Exception as exc:
            errors.append({"episode_id": eid, "error": str(exc)})

    write_csv(out / "video_camera_match_scores.csv", match_rows)
    write_csv(out / "convention_grid_scores.csv", score_rows)
    write_csv(out / "errors.csv", errors)

    match_df = pd.DataFrame(match_rows)
    scores_df = pd.DataFrame(score_rows)
    best_counts = match_df["best_camera_by_l1"].value_counts().to_dict() if not match_df.empty else {}
    if not scores_df.empty:
        grouped = scores_df.groupby(["camera_name", "intrinsic_mode", "variant", "z_offset"], dropna=False).agg(
            out_of_frame_ratio=("out_of_frame_ratio", "mean"),
            nonzero=("action_map_nonzero_ratio", "mean"),
            center_dist=("distance_to_image_center", "mean"),
            empty=("empty_map_count", "mean"),
            corner=("corner_map_count", "mean"),
        ).reset_index()
        grouped["rank_score"] = grouped["out_of_frame_ratio"] + grouped["center_dist"] + 0.1 * grouped["corner"] - grouped["nonzero"]
        top = grouped.sort_values("rank_score").head(12)
    else:
        top = pd.DataFrame()

    video_report = [
        "# Video Camera Match Report", "",
        f"Samples processed: {len(match_rows)}", "",
        f"Best camera by first-frame L1 counts: `{json.dumps(best_counts, ensure_ascii=False)}`", "",
        "Lower L1 and higher SSIM means the exported video first frame is closer to that HDF5 camera RGB.", "",
    ]
    if not match_df.empty:
        video_report.append(match_df.to_markdown(index=False))
    (out / "video_camera_match_report.md").write_text("\n".join(video_report), encoding="utf-8")

    conv_report = [
        "# Convention Grid Report", "",
        f"Samples processed: {len(match_rows)}", "",
        "This is a diagnostic grid only. It does not modify manifests or training data.",
        "The grid maps reimplement ABot/VACE projection geometry so z_offset can be enumerated; official ABot code hardcodes z_offset=0.23.", "",
        "## Best Camera Match", "",
        f"Best camera counts: `{json.dumps(best_counts, ensure_ascii=False)}`", "",
        "## Top Convention Candidates By Heuristic Rank", "",
    ]
    if not top.empty:
        conv_report.append(top.to_markdown(index=False))
    conv_report.extend([
        "", "## Required Answers", "",
        "1. Exported video camera: decide from `video_camera_match_scores.csv`; most samples should indicate head or front.",
        "2. Converter assumption: if `inverse_extrinsic_cv` ranks poorly and another convention is clearly better, the current converter assumption is suspect.",
        "3. Best overlay convention: use the table above plus manual contact sheets, not metrics alone.",
        "4. z_offset=0.23: compare rows with z_offset 0.00/0.10/0.23; if 0.23 creates more out-of-frame maps, RoboTwin may need a different end-effector offset.",
        "5. WorldArena no-camera case: if no verified camera exists, use calibrated fallback only after matching it to the exported-video convention; do not mix arbitrary fallback with HDF5 verified camera.", "",
        f"Contact sheets: `{out / 'convention_grid/contact_sheets'}`", "",
    ])
    (out / "convention_grid_report.md").write_text("\n".join(conv_report), encoding="utf-8")
    print(out / "video_camera_match_report.md")
    print(out / "convention_grid_report.md")


if __name__ == "__main__":
    main()
