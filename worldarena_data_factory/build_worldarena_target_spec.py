#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from make_robotwin_collection_jobs import config_counts  # noqa: E402
from utils import (  # noqa: E402
    TASK_QUOTAS,
    embodiment_plan,
    embodiment_weights,
    ensure_dirs,
    write_json,
    write_yaml,
)


def ratios_for_quota(
    quota: int, enable_mild: bool, enable_hard: bool
) -> dict[str, float]:
    counts = config_counts(quota, enable_mild, enable_hard)
    return {name: count / quota for name, count in counts.items() if quota}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/root/autodl-tmp/worldarena_data_factory_v0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--enable-mild-random", action="store_true")
    parser.add_argument("--enable-hard-random", action="store_true")
    parser.add_argument("--include-secondary-embodiment", action="store_true")
    parser.add_argument("--secondary-embodiment", default="piper")
    parser.add_argument("--main-embodiment", default="aloha-agilex")
    args = parser.parse_args()

    out = Path(args.out)
    ensure_dirs(out)
    random.seed(args.seed)
    total = sum(TASK_QUOTAS.values())
    embodiment_slots = embodiment_plan(
        total,
        out,
        args.main_embodiment,
        args.secondary_embodiment,
        args.include_secondary_embodiment,
    )
    weights = [
        {"embodiment": e, "ratio": r}
        for e, r in embodiment_weights(
            out,
            args.main_embodiment,
            args.secondary_embodiment,
            args.include_secondary_embodiment,
        )
    ]
    spec = {
        "version": "v0_clean",
        "target_successful_episodes": 1500,
        "task_family_quotas": TASK_QUOTAS,
        "config_ratios": ratios_for_quota(
            100, args.enable_mild_random, args.enable_hard_random
        ),
        "target_domain": "RoboTwin2 Clean-50 / WorldArena Track1 Aloha-AgileX dual-arm gripper manipulation",
        "action_schema_required": "joint14",
        "expected_action_dim": 14,
        "is_dual_arm_required": True,
        "embodiment_strategy": {
            "main_embodiment": "aloha-agilex",
            "default_weights": [{"embodiment": "aloha-agilex", "ratio": 1.0}],
            "include_secondary_embodiment": bool(args.include_secondary_embodiment),
            "weights": weights,
            "plan_counts": {},
            "note": "v0_clean defaults to 100% Aloha-AgileX and wa_clean_fixed.",
        },
        "jobs": [],
    }
    i = 0
    for family, quota in TASK_QUOTAS.items():
        for config_name, count in config_counts(
            quota, args.enable_mild_random, args.enable_hard_random
        ).items():
            for _ in range(count):
                embodiment = embodiment_slots[i]
                spec["jobs"].append(
                    {
                        "target_id": f"target_{i:04d}",
                        "task_family": family,
                        "config_name": config_name,
                        "embodiment": embodiment,
                        "target_success": 1,
                        "expected_action_dim": 14,
                        "action_schema_required": "joint14",
                        "is_dual_arm_required": True,
                    }
                )
                counts = spec["embodiment_strategy"]["plan_counts"]
                counts[embodiment] = counts.get(embodiment, 0) + 1
                i += 1
    write_yaml(out / "manifests" / "worldarena_target_spec.yaml", spec)
    write_json(out / "manifests" / "worldarena_target_spec.json", spec)
    print(out / "manifests" / "worldarena_target_spec.yaml")


if __name__ == "__main__":
    main()
