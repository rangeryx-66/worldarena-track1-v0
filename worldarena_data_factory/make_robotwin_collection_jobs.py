#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    TASK_CANDIDATES,
    TASK_QUOTAS,
    available_robotwin_tasks,
    detect_robotwin_root,
    embodiment_plan,
    ensure_dirs,
    write_csv,
)

FIELDS = [
    "job_id",
    "task_family",
    "robotwin_task_name",
    "base_task_config",
    "task_config",
    "embodiment",
    "expected_action_dim",
    "action_schema_required",
    "is_dual_arm_required",
    "target_success",
    "max_attempts",
    "gpu_id",
    "output_dir",
]


def safe(embodiment: str) -> str:
    return embodiment.replace("-", "_").replace("+", "_")


def config_counts(quota: int, enable_mild: bool, enable_hard: bool) -> dict[str, int]:
    if not enable_mild and not enable_hard:
        return {"wa_clean_fixed": quota}
    if enable_mild and enable_hard:
        clean = round(quota * 0.70)
        mild = round(quota * 0.20)
        return {
            "wa_clean_fixed": clean,
            "wa_mild_random": mild,
            "wa_hard_success": quota - clean - mild,
        }
    if enable_mild:
        clean = round(quota * 0.80)
        return {"wa_clean_fixed": clean, "wa_mild_random": quota - clean}
    clean = round(quota * 0.90)
    return {"wa_clean_fixed": clean, "wa_hard_success": quota - clean}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/root/autodl-tmp/worldarena_data_factory_v0")
    parser.add_argument("--robotwin-root")
    parser.add_argument("--enable-mild-random", action="store_true")
    parser.add_argument("--enable-hard-random", action="store_true")
    parser.add_argument("--main-embodiment", default="aloha-agilex")
    parser.add_argument("--secondary-embodiment", default="piper")
    parser.add_argument("--include-secondary-embodiment", action="store_true")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--target-success-per-job", type=int, default=5)
    parser.add_argument("--max-attempts-multiplier", type=int, default=6)
    args = parser.parse_args()

    out = Path(args.out)
    ensure_dirs(out)
    robotwin_root = detect_robotwin_root(args.robotwin_root)
    tasks = set(available_robotwin_tasks(robotwin_root))
    all_tasks = sorted(tasks)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    gpu_cycle = itertools.cycle(gpus or ["0"])
    emb_slots = embodiment_plan(
        sum(TASK_QUOTAS.values()),
        out,
        args.main_embodiment,
        args.secondary_embodiment,
        args.include_secondary_embodiment,
    )

    slot_idx = 0
    rows = []
    missing = []
    job_index = 0
    for family, quota in TASK_QUOTAS.items():
        candidates = TASK_CANDIDATES.get(family) or all_tasks
        candidates = [task for task in candidates if not tasks or task in tasks]
        if not candidates:
            missing.append(
                {
                    "task_family": family,
                    "requested_candidates": ";".join(TASK_CANDIDATES.get(family, [])),
                }
            )
            slot_idx += quota
            continue

        for config_name, count in config_counts(
            quota, args.enable_mild_random, args.enable_hard_random
        ).items():
            remaining = count
            while remaining > 0:
                embodiment = emb_slots[slot_idx]
                same_run = 0
                while (
                    slot_idx + same_run < len(emb_slots)
                    and emb_slots[slot_idx + same_run] == embodiment
                    and same_run < remaining
                    and same_run < args.target_success_per_job
                ):
                    same_run += 1
                target_success = max(1, same_run)
                job_id = f"job_{job_index:05d}"
                task_name = candidates[job_index % len(candidates)]
                base_task_config = f"{config_name}__{safe(embodiment)}"
                task_config = f"{base_task_config}__{job_id}"
                rows.append(
                    {
                        "job_id": job_id,
                        "task_family": family,
                        "robotwin_task_name": task_name,
                        "base_task_config": base_task_config,
                        "task_config": task_config,
                        "embodiment": embodiment,
                        "expected_action_dim": 14,
                        "action_schema_required": "joint14",
                        "is_dual_arm_required": True,
                        "target_success": target_success,
                        "max_attempts": target_success * args.max_attempts_multiplier,
                        "gpu_id": next(gpu_cycle),
                        "output_dir": str(
                            out / "robotwin_raw" / task_name / task_config
                        ),
                    }
                )
                job_index += 1
                remaining -= target_success
                slot_idx += target_success

    write_csv(out / "manifests" / "robotwin_collection_jobs.csv", rows, FIELDS)
    write_csv(
        out / "manifests" / "missing_tasks.csv",
        missing,
        ["task_family", "requested_candidates"],
    )
    print(f"wrote {len(rows)} jobs to {out}/manifests/robotwin_collection_jobs.csv")


if __name__ == "__main__":
    main()
