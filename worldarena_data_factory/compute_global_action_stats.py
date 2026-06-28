#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from utils import ensure_dirs, read_table, write_json, write_table  # noqa: E402


def truthy(value) -> bool:
    return str(value).lower() == "true"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/root/autodl-tmp/worldarena_data_factory_v0")
    parser.add_argument("--manifest")
    parser.add_argument("--accepted-field", default="accepted_for_a2v")
    parser.add_argument("--update-manifest", action="store_true", default=True)
    args = parser.parse_args()

    out = Path(args.out)
    ensure_dirs(out)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else out / "manifests" / "episode_manifest.parquet"
    )
    rows = read_table(manifest_path)
    accepted = [
        r
        for r in rows
        if truthy(r.get(args.accepted_field, ""))
        and str(r.get("split", "train")) == "train"
        and r.get("action_joint14_raw_path")
    ]
    if not accepted:
        raise SystemExit("no accepted train episodes with action_joint14_raw_path")

    arrays = []
    for row in accepted:
        arr = np.load(row["action_joint14_raw_path"]).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 14:
            raise ValueError(
                f"bad joint14 shape for {row.get('episode_id')}: {arr.shape}"
            )
        arrays.append(arr)
    all_actions = np.concatenate(arrays, axis=0)
    p01 = np.percentile(all_actions, 1, axis=0)
    p99 = np.percentile(all_actions, 99, axis=0)
    clipped = np.clip(all_actions, p01, p99)
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0) + 1e-6

    config = {
        "schema": "joint14",
        "source": "accepted_robotwin_train_global",
        "accepted_field": args.accepted_field,
        "episode_count": len(accepted),
        "frame_count": int(all_actions.shape[0]),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "p01": p01.tolist(),
        "p99": p99.tolist(),
    }
    write_json(out / "manifests" / "action_normalization_config.json", config)

    by_path = {row["action_joint14_raw_path"]: row for row in accepted}
    for raw_path, row in by_path.items():
        arr = np.load(raw_path).astype(np.float32)
        norm = (np.clip(arr, p01, p99) - mean) / std
        norm_path = Path(raw_path).with_name("action_joint14_norm.npy")
        np.save(norm_path, norm.astype(np.float32))
        row["action_joint14_norm_path"] = str(norm_path)

    if args.update_manifest:
        for row in rows:
            accepted_row = by_path.get(row.get("action_joint14_raw_path", ""))
            if accepted_row is not None:
                row["action_joint14_norm_path"] = accepted_row[
                    "action_joint14_norm_path"
                ]
        actual, mode = write_table(manifest_path, rows)
        print(f"updated manifest {actual} mode={mode}")
    print(out / "manifests" / "action_normalization_config.json")


if __name__ == "__main__":
    main()
