#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from create_robotwin_configs import (
    cfg,
    enabled_config_names,
    load_template,
)  # noqa: E402
from utils import (  # noqa: E402
    PROBE_TASKS,
    TASK_CANDIDATES,
    available_robotwin_tasks,
    detect_robotwin_root,
    ensure_dirs,
    write_csv,
    write_json,
    write_yaml,
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

TASK_FAMILY = {
    "pick_dual_bottles": "lifting",
    "place_object_basket": "object_to_container",
    "stack_blocks_two": "stacking",
    "click_bell": "button_press_click",
    "open_microwave": "articulated_open_close",
}
for _family, _tasks in TASK_CANDIDATES.items():
    for _task in _tasks:
        TASK_FAMILY.setdefault(_task, _family)


def safe(value: str) -> str:
    return value.replace("-", "_").replace("+", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="/root/autodl-tmp/worldarena_data_factory_v0_clean_probe"
    )
    parser.add_argument("--robotwin-root")
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--enable-mild-random", action="store_true")
    parser.add_argument("--enable-hard-random", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--tasks",
        default=",".join(PROBE_TASKS),
        help="Comma-separated RoboTwin tasks to probe. Default is the standard small probe set.",
    )
    parser.add_argument(
        "--head-camera-type",
        default="Large_D435",
        help="Use Large_D435 by default to render native 640x480 clips for QC.",
    )
    parser.add_argument("--rt-spp", type=int, default=256)
    parser.add_argument("--rt-path-depth", type=int, default=8)
    parser.add_argument("--rt-denoiser", default="optix")
    args = parser.parse_args()

    out = Path(args.out)
    ensure_dirs(out)
    robotwin_root = detect_robotwin_root(args.robotwin_root)
    template = load_template(robotwin_root)
    configs = enabled_config_names(args.enable_mild_random, args.enable_hard_random)
    written_configs = []
    for name in configs:
        cfg_name = f"{name}__aloha_agilex"
        path = out / "configs_to_apply" / f"{cfg_name}.yml"
        write_yaml(
            path,
            cfg(
                name,
                "aloha-agilex",
                template,
                args.head_camera_type,
                args.rt_spp,
                args.rt_path_depth,
                args.rt_denoiser or None,
            ),
        )
        written_configs.append(str(path))
        if args.apply and robotwin_root:
            shutil.copy2(path, robotwin_root / "task_config" / f"{cfg_name}.yml")

    available = set(available_robotwin_tasks(robotwin_root))
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()] or ["0"]
    probe_tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    rows = []
    missing = []
    job_id = 0
    for config_name in configs:
        for task in probe_tasks:
            if available and task not in available:
                missing.append({"task": task, "reason": "not_found_in_robotwin"})
                continue
            jid = f"probe_{job_id:05d}"
            base_config = f"{config_name}__aloha_agilex"
            task_config = f"{base_config}__{jid}"
            rows.append(
                {
                    "job_id": jid,
                    "task_family": TASK_FAMILY.get(task, "coverage_unknown"),
                    "robotwin_task_name": task,
                    "base_task_config": base_config,
                    "task_config": task_config,
                    "embodiment": "aloha-agilex",
                    "expected_action_dim": 14,
                    "action_schema_required": "joint14",
                    "is_dual_arm_required": True,
                    "target_success": args.episodes_per_task,
                    "max_attempts": args.episodes_per_task * 4,
                    "gpu_id": gpus[job_id % len(gpus)],
                    "output_dir": str(out / "robotwin_raw" / task / task_config),
                }
            )
            job_id += 1

    jobs_path = out / "manifests" / "probe_collection_jobs.csv"
    write_csv(jobs_path, rows, FIELDS)
    write_csv(
        out / "manifests" / "probe_missing_tasks.csv", missing, ["task", "reason"]
    )
    report = {
        "probe_pass": False,
        "reason": "jobs_generated_only_not_collected",
        "configs": configs,
        "episodes_per_task": args.episodes_per_task,
        "tasks": probe_tasks,
        "head_camera_type": args.head_camera_type,
        "rt_spp": args.rt_spp,
        "rt_path_depth": args.rt_path_depth,
        "rt_denoiser": args.rt_denoiser or None,
        "jobs_csv": str(jobs_path),
        "written_configs": written_configs,
        "next_step": "Run run_robotwin_jobs.py with --jobs-csv probe_collection_jobs.csv, convert, inspect contact sheets, then set probe_pass true only after manual approval.",
    }
    write_json(out / "probe_report.json", report)
    write_json(out / "manifests" / "probe_report.json", report)
    md = [
        "# v0 Clean Probe Report",
        "",
        "probe_pass: false",
        "reason: jobs generated only; collection not run by this script",
        f"jobs: {len(rows)}",
        f"jobs_csv: `{jobs_path}`",
        "",
        "Run the probe jobs first, inspect contact sheets, then only mark probe_pass=true after the camera/framing is approved.",
    ]
    (out / "probe_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(jobs_path)
    print(out / "probe_report.json")


if __name__ == "__main__":
    main()
