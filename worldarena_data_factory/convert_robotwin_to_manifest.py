#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ensure_dirs,
    is_v0_training_embodiment,
    normalize_embodiment,
    read_csv,
    write_csv,
    write_json,
    write_table,
)


def video_meta(path: Path) -> dict:
    try:
        import cv2
    except Exception:
        return {"readable": False, "reason": "cv2_unavailable"}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"readable": False, "reason": "open_failed"}
    ok, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if not ok or frame is None:
        return {"readable": False, "reason": "first_frame_failed"}
    return {
        "readable": True,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps if fps > 0 else 0,
        "area": width * height,
    }


def find_raw_video(hdf5_path: Path) -> Path | None:
    stem = hdf5_path.stem
    root = hdf5_path.parent.parent
    candidates = [
        root / "video" / f"{stem}.mp4",
        root / "video" / "videos" / f"{stem}.mp4",
    ]
    candidates.extend(root.glob(f"**/{stem}.mp4"))
    candidates.extend(
        p
        for p in root.glob("**/*.mp4")
        if any(token in str(p).lower() for token in ["head", "camera", "video"])
    )
    unique = []
    seen = set()
    for path in candidates:
        if path.exists() and path not in seen:
            unique.append(path)
            seen.add(path)
    scored = []
    for path in unique:
        meta = video_meta(path)
        if not meta.get("readable"):
            continue
        if meta.get("frame_count", 0) < 12:
            continue
        scored.append((meta.get("area", 0), meta.get("duration", 0), path))
    if not scored:
        return None
    return sorted(scored, reverse=True)[0][2]


def read_first_frames(video_path: Path, n: int = 12) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def write_contact_sheet(frames: list[np.ndarray], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tw, th, lh, cols = 160, 120, 18, 4
    rows = max(1, int(np.ceil(len(frames) / cols)))
    sheet = Image.new("RGB", (cols * tw, rows * (th + lh) + 24), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((6, 6), title, fill=(0, 0, 0), font=font)
    for i, arr in enumerate(frames):
        im = Image.fromarray(arr).convert("RGB")
        im.thumbnail((tw, th))
        x = (i % cols) * tw
        y = 24 + (i // cols) * (th + lh)
        canvas = Image.new("RGB", (tw, th), "white")
        canvas.paste(im, ((tw - im.width) // 2, (th - im.height) // 2))
        sheet.paste(canvas, (x, y))
        draw.text((x + 4, y + 3), f"frame {i}", fill=(0, 0, 0), font=font)
    sheet.save(path, quality=92)


def visual_sanity(frames: list[np.ndarray]) -> tuple[str, str, dict]:
    if not frames:
        return "FAIL", "no_decodable_frames", {}
    arr = (
        np.stack(
            [np.asarray(Image.fromarray(f).resize((160, 120))) for f in frames]
        ).astype(np.float32)
        / 255.0
    )
    gray = arr.mean(axis=-1)
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    saturation = (mx - mn) / np.maximum(mx, 1e-6)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    dark_frac = float((gray < 0.28).mean())
    saturated_frac = float((saturation > 0.22).mean())
    very_white = brightness > 0.94 and contrast < 0.025
    very_black = brightness < 0.05
    no_foreground = dark_frac < 0.003 and saturated_frac < 0.003 and contrast < 0.03
    metrics = {
        "brightness": brightness,
        "contrast": contrast,
        "dark_frac": dark_frac,
        "saturated_frac": saturated_frac,
    }
    if very_white:
        return "FAIL", "almost_all_white", metrics
    if very_black:
        return "FAIL", "almost_all_black", metrics
    if no_foreground:
        return "FAIL", "almost_no_foreground", metrics
    return "PASS", "ok", metrics


def make_video_and_frame(raw_video: Path, epdir: Path) -> tuple[str, str, list[str]]:
    out_video = epdir / "observation.mp4"
    out_frame = epdir / "first_frame.png"
    flags = []
    if not raw_video or not raw_video.exists():
        return "", "", ["raw_video_missing"]
    if not out_video.exists():
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(raw_video),
                    "-vf",
                    "scale=640:480,fps=24",
                    "-pix_fmt",
                    "yuv420p",
                    str(out_video),
                ],
                check=True,
            )
        except Exception:
            flags.append("video_export_failed")
    if not out_frame.exists() and out_video.exists():
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(out_video),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=320:240",
                    str(out_frame),
                ],
                check=True,
            )
        except Exception:
            flags.append("first_frame_export_failed")
    return (
        str(out_video) if out_video.exists() else "",
        str(out_frame) if out_frame.exists() else "",
        flags,
    )


def reject_row(hdf5_path, job, reason):
    return {
        "raw_hdf5_path": str(hdf5_path) if hdf5_path else "",
        "job_id": job.get("job_id", ""),
        "embodiment": job.get("embodiment", ""),
        "reason": reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/root/autodl-tmp/worldarena_data_factory_v0")
    parser.add_argument("--robotwin-root")
    parser.add_argument("--jobs-csv")
    parser.add_argument("--include-secondary-embodiment", action="store_true")
    parser.add_argument("--secondary-embodiment", default="piper")
    args = parser.parse_args()

    out = Path(args.out)
    ensure_dirs(out)
    jobs = read_csv(
        Path(args.jobs_csv)
        if args.jobs_csv
        else out / "manifests" / "robotwin_collection_jobs.csv"
    )
    rows = []
    rejected = []
    try:
        import h5py
    except Exception as exc:
        raise SystemExit(f"h5py required: {exc}")

    for job in jobs:
        embodiment = normalize_embodiment(job.get("embodiment", ""))
        if not is_v0_training_embodiment(
            embodiment,
            "aloha-agilex",
            args.secondary_embodiment,
            out,
            args.include_secondary_embodiment,
        ):
            rejected.append(
                reject_row("", job, f"wrong_embodiment_for_v0:{embodiment}")
            )
            continue
        raw_dir = Path(job["output_dir"]) / "data"
        if not raw_dir.exists():
            continue
        for hdf5_path in sorted(raw_dir.glob("episode*.hdf5")):
            try:
                with h5py.File(hdf5_path, "r") as f:
                    if "/joint_action/vector" not in f:
                        raise ValueError(
                            "rejected_action_schema: missing /joint_action/vector"
                        )
                    action = np.asarray(f["/joint_action/vector"])
                    T = action.shape[0] if action.ndim else 0
                    required = [
                        "/joint_action/left_arm",
                        "/joint_action/right_arm",
                        "/joint_action/left_gripper",
                        "/joint_action/right_gripper",
                    ]
                    missing = [x for x in required if x not in f]
                    if action.ndim != 2 or action.shape[1] != 14:
                        raise ValueError(
                            f"rejected_action_schema: joint14 shape {action.shape}"
                        )
                    if missing:
                        raise ValueError(
                            "rejected_action_schema: missing " + ",".join(missing)
                        )
                    if T < 60:
                        raise ValueError(f"rejected_action_schema: T<60: {T}")
                    if not np.isfinite(action).all():
                        raise ValueError("rejected_action_schema: NaN_or_Inf")
                    left_endpose = (
                        np.asarray(f["/endpose/left_endpose"])
                        if "/endpose/left_endpose" in f
                        else np.zeros((T, 7))
                    )
                    right_endpose = (
                        np.asarray(f["/endpose/right_endpose"])
                        if "/endpose/right_endpose" in f
                        else np.zeros((T, 7))
                    )
                    left_gripper = np.asarray(f["/joint_action/left_gripper"]).reshape(
                        T, 1
                    )
                    right_gripper = np.asarray(
                        f["/joint_action/right_gripper"]
                    ).reshape(T, 1)
                    if left_endpose.shape[0] != T or right_endpose.shape[0] != T:
                        raise ValueError(
                            "rejected_action_schema: endpose length mismatch"
                        )

                raw_video = find_raw_video(hdf5_path)
                if raw_video is None:
                    rejected.append(
                        reject_row(hdf5_path, job, "raw_video_discovery_failed")
                    )
                    continue
                meta = video_meta(raw_video)
                if not meta.get("readable"):
                    rejected.append(
                        reject_row(
                            hdf5_path, job, f"raw_video_unreadable:{meta.get('reason')}"
                        )
                    )
                    continue

                episode_id = f"rt_{len(rows):06d}"
                epdir = out / "episodes" / episode_id
                epdir.mkdir(parents=True, exist_ok=True)
                video_path, first_frame_path, media_flags = make_video_and_frame(
                    raw_video, epdir
                )
                if not video_path or not first_frame_path:
                    rejected.append(
                        reject_row(hdf5_path, job, "video_or_first_frame_export_failed")
                    )
                    continue

                frames = read_first_frames(Path(video_path), 12)
                contact_sheet = epdir / "quick_contact_sheet.jpg"
                write_contact_sheet(frames, contact_sheet, episode_id)
                visual_status, visual_reason, visual_metrics = visual_sanity(frames)
                visual_ok = visual_status == "PASS"

                np.save(epdir / "action_joint14_raw.npy", action)
                ee = np.concatenate(
                    [left_endpose, left_gripper, right_endpose, right_gripper], axis=1
                )
                np.save(epdir / "action_ee16.npy", ee)
                np.save(
                    epdir / "action_joint14_ee16.npy",
                    np.concatenate([action, ee], axis=1),
                )
                write_json(epdir / "camera_intrinsic.json", {"fallback": True})
                write_json(epdir / "camera_extrinsic.json", {"fallback": True})
                write_json(epdir / "meta.json", job)
                write_json(
                    epdir / "visual_sanity.json",
                    {
                        "status": visual_status,
                        "reason": visual_reason,
                        "metrics": visual_metrics,
                    },
                )

                qflags = [
                    "camera_fallback",
                    "dual_arm_joint14_valid",
                    "aloha_agilex_domain",
                ] + media_flags
                qflags.append(
                    "visual_sanity_pass" if visual_ok else "visual_sanity_failed"
                )
                rows.append(
                    {
                        "episode_id": episode_id,
                        "source": "robotwin",
                        "robotwin_task_name": job["robotwin_task_name"],
                        "task_family": job["task_family"],
                        "config_name": job["task_config"],
                        "embodiment": embodiment,
                        "seed": "",
                        "success": visual_ok,
                        "success_source": "hdf5_only_unverified",
                        "visual_sanity_status": visual_status,
                        "visual_sanity_reason": visual_reason,
                        "accepted_for_sft": visual_ok,
                        "accepted_for_a2v": visual_ok,
                        "raw_hdf5_path": str(hdf5_path),
                        "raw_video_path": str(raw_video),
                        "video_640x480_path": video_path,
                        "first_frame_320x240_path": first_frame_path,
                        "quick_contact_sheet_path": str(contact_sheet),
                        "action_joint14_raw_path": str(
                            epdir / "action_joint14_raw.npy"
                        ),
                        "action_joint14_norm_path": "",
                        "action_ee16_raw_path": str(epdir / "action_ee16.npy"),
                        "action_joint14_ee16_raw_path": str(
                            epdir / "action_joint14_ee16.npy"
                        ),
                        "intrinsic_path": str(epdir / "camera_intrinsic.json"),
                        "extrinsic_path": str(epdir / "camera_extrinsic.json"),
                        "T": T,
                        "fps": 24,
                        "dominant_arm": "",
                        "gripper_transition_count": "",
                        "action_complexity_score": "",
                        "prompt_short": job["robotwin_task_name"],
                        "prompt_worldarena_style": f"In a fixed robotic workspace, perform {job['robotwin_task_name'].replace('_', ' ')}.",
                        "prompt_long_caption": f"An Aloha-AgileX dual-arm robot attempts {job['robotwin_task_name'].replace('_', ' ')}.",
                        "quality_flags": ";".join(qflags),
                        "split": "train",
                    }
                )
            except Exception as exc:
                rejected.append(reject_row(hdf5_path, job, str(exc)))

    actual, mode = write_table(out / "manifests" / "episode_manifest.parquet", rows)
    write_csv(out / "rejected" / "convert_rejected.csv", rejected)
    write_csv(out / "rejected" / "rejected_episodes.csv", rejected)
    print(f"episode manifest rows={len(rows)} actual={actual} mode={mode}")


if __name__ == "__main__":
    main()
