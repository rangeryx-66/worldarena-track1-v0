#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ensure_dirs,
    is_v0_training_embodiment,
    read_csv,
    read_table,
    write_csv,
    write_table,
)

OBJECT_HINTS = {
    "basket",
    "bottle",
    "bottles",
    "block",
    "blocks",
    "bowl",
    "bowls",
    "cup",
    "microwave",
    "laptop",
    "qrcode",
    "stapler",
    "bell",
    "pot",
    "mug",
    "hammer",
    "roller",
    "can",
    "cans",
    "plate",
    "phone",
    "shoe",
    "object",
}
TARGET_HINTS = {
    "basket",
    "bin",
    "bigbin",
    "cabinet",
    "plate",
    "stand",
    "pad",
    "dustbin",
    "box",
}


def truthy(value) -> bool:
    return str(value).lower() == "true"


def task_semantics(task_name: str) -> tuple[str, str, str]:
    tokens = [x for x in task_name.replace("-", "_").split("_") if x]
    verbs = [tokens[0]] if tokens else []
    if len(tokens) > 1 and tokens[0] in {
        "place",
        "pick",
        "put",
        "move",
        "open",
        "click",
        "press",
        "stack",
        "scan",
        "shake",
        "rotate",
        "lift",
        "dump",
        "handover",
        "hanging",
        "grab",
        "beat",
        "stamp",
        "turn",
        "adjust",
    }:
        verbs = [tokens[0]]
    objects = [t for t in tokens if t in OBJECT_HINTS]
    targets = [t for t in tokens if t in TARGET_HINTS]
    return (
        ";".join(sorted(set(verbs))),
        ";".join(sorted(set(objects))),
        ";".join(sorted(set(targets))),
    )


def add_robotwin_rows(rows, manifest_rows, out, include_secondary, secondary):
    for row in manifest_rows:
        if not truthy(row.get("accepted_for_a2v", row.get("success", ""))):
            continue
        if "dual_arm_joint14_valid" not in str(row.get("quality_flags", "")):
            continue
        if not is_v0_training_embodiment(
            row.get("embodiment", ""), "aloha-agilex", secondary, out, include_secondary
        ):
            continue
        verbs, objects, targets = task_semantics(row.get("robotwin_task_name", ""))
        rows.append(
            {
                "library_id": f"rt_{row.get('episode_id', '')}",
                "source": "robotwin",
                "split": row.get("split", "train"),
                "episode_id": row.get("episode_id", ""),
                "robotwin_task_name": row.get("robotwin_task_name", ""),
                "worldarena_episode_id_if_any": "",
                "task_family": row.get("task_family", ""),
                "main_verbs": verbs,
                "main_objects": objects,
                "receptacles_or_targets": targets,
                "spatial_relations": "",
                "prompt_short": row.get("prompt_short", ""),
                "prompt_worldarena_style": row.get("prompt_worldarena_style", ""),
                "action_joint14_path": row.get("action_joint14_raw_path", ""),
                "action_ee16_path": row.get("action_ee16_raw_path", ""),
                "action_joint14_ee16_path": row.get("action_joint14_ee16_raw_path", ""),
                "video_path": row.get("video_640x480_path", ""),
                "first_frame_path": row.get("first_frame_320x240_path", ""),
                "T": row.get("T", ""),
                "dominant_arm": row.get("dominant_arm", ""),
                "action_complexity_score": row.get("action_complexity_score", ""),
                "success": True,
                "quality_score": 1.0,
                "embodiment": row.get("embodiment", "aloha-agilex"),
                "inference_only": False,
                "training_allowed": True,
            }
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worldarena-root", default="/root/autodl-tmp/worldarena_testset"
    )
    parser.add_argument(
        "--analysis", default="/root/autodl-tmp/worldarena_testset/analysis_v2"
    )
    parser.add_argument("--out", default="/root/autodl-tmp/worldarena_data_factory_v0")
    parser.add_argument("--episode-manifest")
    parser.add_argument("--include-secondary-embodiment", action="store_true")
    parser.add_argument("--secondary-embodiment", default="piper")
    args = parser.parse_args()

    root = Path(args.worldarena_root)
    analysis = Path(args.analysis)
    out = Path(args.out)
    ensure_dirs(out)
    try:
        import h5py
    except Exception as exc:
        raise SystemExit(f"h5py required: {exc}")

    rows = []
    rejected = []
    manifest_path = (
        Path(args.episode_manifest)
        if args.episode_manifest
        else out / "manifests" / "episode_manifest.parquet"
    )
    add_robotwin_rows(
        rows,
        read_table(manifest_path),
        out,
        args.include_secondary_embodiment,
        args.secondary_embodiment,
    )

    semantics = read_csv(analysis / "prompt_semantics.csv")
    semantics_index = {
        (r["split"], r["episode_id"], r["prompt_set"]): r for r in semantics
    }
    action_stats = read_csv(analysis / "action_stats_episode.csv")
    for row in action_stats:
        split = row["split"]
        episode_id = row["episode_id"]
        hdf5_path = root / row["hdf5_path"]
        epdir = out / "worldarena_actions" / split / f"episode{episode_id}"
        epdir.mkdir(parents=True, exist_ok=True)
        try:
            with h5py.File(hdf5_path, "r") as f:
                action = np.asarray(f["/joint_action/vector"])
                T = action.shape[0]
                if action.ndim != 2 or action.shape[1] != 14:
                    raise ValueError(
                        f"WorldArena joint14 shape mismatch: {action.shape}"
                    )
                left_endpose = np.asarray(f["/endpose/left_endpose"])
                right_endpose = np.asarray(f["/endpose/right_endpose"])
                left_gripper = np.asarray(f["/joint_action/left_gripper"]).reshape(T, 1)
                right_gripper = np.asarray(f["/joint_action/right_gripper"]).reshape(
                    T, 1
                )
            ee = np.concatenate(
                [left_endpose, left_gripper, right_endpose, right_gripper], axis=1
            )
            np.save(epdir / "action_joint14.npy", action)
            np.save(epdir / "action_ee16.npy", ee)
            np.save(
                epdir / "action_joint14_ee16.npy", np.concatenate([action, ee], axis=1)
            )
            sem = semantics_index.get((split, episode_id, "base"), {})
            rows.append(
                {
                    "library_id": f"wa_{split}_{episode_id}",
                    "source": "worldarena",
                    "split": split,
                    "episode_id": episode_id,
                    "robotwin_task_name": "",
                    "worldarena_episode_id_if_any": episode_id,
                    "task_family": sem.get("task_family", ""),
                    "main_verbs": sem.get("main_verbs", ""),
                    "main_objects": sem.get("main_objects", ""),
                    "receptacles_or_targets": sem.get("receptacles_or_targets", ""),
                    "spatial_relations": sem.get("spatial_relations", ""),
                    "prompt_short": sem.get("prefix_removed_prompt", ""),
                    "prompt_worldarena_style": sem.get("raw_prompt", ""),
                    "action_joint14_path": str(epdir / "action_joint14.npy"),
                    "action_ee16_path": str(epdir / "action_ee16.npy"),
                    "action_joint14_ee16_path": str(epdir / "action_joint14_ee16.npy"),
                    "video_path": "",
                    "first_frame_path": str(
                        root
                        / split
                        / "first_frame"
                        / "fixed_scene_task"
                        / f"episode{episode_id}.png"
                    ),
                    "T": T,
                    "dominant_arm": row.get("dominant_arm", ""),
                    "action_complexity_score": row.get("action_complexity_score", ""),
                    "success": True,
                    "quality_score": 1.0,
                    "embodiment": "aloha-agilex",
                    "inference_only": True,
                    "training_allowed": False,
                }
            )
        except Exception as exc:
            rejected.append({"hdf5_path": str(hdf5_path), "reason": str(exc)})

    actual, mode = write_table(out / "manifests" / "action_library.parquet", rows)
    write_csv(out / "rejected" / "action_library_rejected.csv", rejected)
    print(f"{len(rows)} rows {actual} {mode}")


if __name__ == "__main__":
    main()
