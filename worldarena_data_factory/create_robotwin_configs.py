#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import shutil
import sys
from numbers import Number
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    DUAL_ARM_CANDIDATE_EMBODIMENTS,
    detect_robotwin_root,
    embodiment_weights,
    ensure_dirs,
    normalize_embodiment,
    write_yaml,
)

NUMERIC_LIMITS = {
    "random_head_camera_dis": 0.03,
    "random_table_height": 0.03,
    "crazy_random_light_rate": 0.03,
}


def load_template(robotwin_root: Path | None):
    template = None
    if robotwin_root:
        for name in ["demo_clean.yml", "_config_template.yml"]:
            path = robotwin_root / "task_config" / name
            if not path.exists():
                continue
            try:
                import yaml

                with path.open("r", encoding="utf-8") as f:
                    template = yaml.safe_load(f) or {}
                break
            except Exception:
                template = None
    if template is None:
        template = {
            "render_freq": 0,
            "episode_num": 5,
            "use_seed": False,
            "save_freq": 15,
            "embodiment": ["aloha-agilex"],
            "language_num": 100,
            "domain_randomization": {},
            "camera": {
                "head_camera_type": "Large_D435",
                "wrist_camera_type": "D435",
                "collect_head_camera": True,
                "collect_wrist_camera": False,
            },
            "data_type": {
                "rgb": True,
                "third_view": False,
                "depth": False,
                "pointcloud": False,
                "observer": False,
                "endpose": True,
                "qpos": True,
                "mesh_segmentation": False,
                "actor_segmentation": False,
            },
            "pcd_down_sample_num": 1024,
            "pcd_crop": True,
            "save_path": "./data",
            "clear_cache_freq": 5,
            "collect_data": True,
            "eval_video_log": True,
        }
    return template


def validate_config_values(config: dict) -> None:
    randomization = config.get("domain_randomization") or {}
    for key, limit in NUMERIC_LIMITS.items():
        value = randomization.get(key, 0.0)
        if isinstance(value, bool) or not isinstance(value, Number):
            raise ValueError(
                f"{key} must be numeric, not {type(value).__name__}: {value!r}"
            )
        if value < 0 or value > limit:
            raise ValueError(f"{key} out of v0 safe range: {value} > {limit}")


def domain_randomization(name: str) -> dict:
    if name == "wa_clean_fixed":
        return {
            "cluttered_table": False,
            "random_background": False,
            "clean_background_rate": 1.0,
            "random_light": False,
            "crazy_random_light_rate": 0.0,
            "random_table_height": 0.0,
            "random_head_camera_dis": 0.0,
        }
    if name == "wa_mild_random":
        return {
            "cluttered_table": True,
            "random_background": True,
            "clean_background_rate": 0.7,
            "random_light": True,
            "crazy_random_light_rate": 0.03,
            "random_table_height": 0.02,
            "random_head_camera_dis": 0.0,
        }
    if name == "wa_hard_success":
        return {
            "cluttered_table": True,
            "random_background": True,
            "clean_background_rate": 0.35,
            "random_light": True,
            "crazy_random_light_rate": 0.03,
            "random_table_height": 0.03,
            "random_head_camera_dis": 0.0,
        }
    raise ValueError(f"unknown config name: {name}")


def cfg(
    name: str,
    embodiment: str,
    template: dict,
    head_camera_type: str = "Large_D435",
) -> dict:
    base = copy.deepcopy(template)
    base["episode_num"] = 5
    base["use_seed"] = False
    base["collect_data"] = True
    base["save_path"] = "./data"
    base["render_freq"] = 0
    base["clear_cache_freq"] = 5
    base["save_freq"] = base.get("save_freq", 15) or 15
    base["language_num"] = base.get("language_num", 100) or 100
    base["embodiment"] = [embodiment]
    base["camera"] = base.get("camera") or {}
    base["camera"].update(
        {
            "head_camera_type": head_camera_type,
            "collect_head_camera": True,
            "wrist_camera_type": "D435",
            "collect_wrist_camera": False,
        }
    )
    base["data_type"] = base.get("data_type") or {}
    base["data_type"].update(
        {
            "rgb": True,
            "third_view": False,
            "depth": False,
            "pointcloud": False,
            "observer": False,
            "endpose": True,
            "qpos": True,
            "mesh_segmentation": False,
            "actor_segmentation": False,
        }
    )
    base["domain_randomization"] = domain_randomization(name)
    base["worldarena_v0_constraints"] = {
        "target_domain": "RoboTwin2 Clean-50 Aloha-AgileX dual-arm gripper",
        "expected_action_dim": 14,
        "action_schema_required": "joint14",
        "is_dual_arm_required": True,
    }
    validate_config_values(base)
    return base


def enabled_config_names(enable_mild: bool, enable_hard: bool) -> list[str]:
    names = ["wa_clean_fixed"]
    if enable_mild:
        names.append("wa_mild_random")
    if enable_hard:
        names.append("wa_hard_success")
    return names


def safe(embodiment: str) -> str:
    return embodiment.replace("-", "_").replace("+", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/root/autodl-tmp/worldarena_data_factory_v0")
    parser.add_argument("--robotwin-root")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--enable-mild-random", action="store_true")
    parser.add_argument("--enable-hard-random", action="store_true")
    parser.add_argument(
        "--include-secondary-embodiment",
        action="store_true",
        help="Optional ablation configs. Default formal configs are aloha-agilex only.",
    )
    parser.add_argument("--secondary-embodiment", default="piper")
    parser.add_argument("--write-probe-configs", action="store_true")
    parser.add_argument("--embodiment")
    parser.add_argument("--main-embodiment", default="aloha-agilex")
    parser.add_argument(
        "--head-camera-type",
        default="Large_D435",
        help="Use Large_D435 by default so v0 renders native 640x480 instead of upscaling D435 320x240.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    ensure_dirs(out)
    robotwin_root = detect_robotwin_root(args.robotwin_root)
    template = load_template(robotwin_root)
    written = []
    embodiments = [
        e
        for e, _ in embodiment_weights(
            out,
            args.main_embodiment,
            args.secondary_embodiment,
            args.include_secondary_embodiment,
        )
    ]

    for name in enabled_config_names(args.enable_mild_random, args.enable_hard_random):
        for embodiment in embodiments:
            cfg_name = f"{name}__{safe(embodiment)}"
            path = out / "configs_to_apply" / f"{cfg_name}.yml"
            write_yaml(path, cfg(name, embodiment, template, args.head_camera_type))
            written.append(str(path))
            if args.apply and robotwin_root:
                shutil.copy2(path, robotwin_root / "task_config" / f"{cfg_name}.yml")

    if args.write_probe_configs:
        for embodiment in [
            normalize_embodiment(e) for e in DUAL_ARM_CANDIDATE_EMBODIMENTS
        ]:
            cfg_name = f"wa_probe_dual_arm__{safe(embodiment)}"
            path = out / "configs_to_apply" / f"{cfg_name}.yml"
            write_yaml(
                path, cfg("wa_clean_fixed", embodiment, template, args.head_camera_type)
            )
            written.append(str(path))
            if args.apply and robotwin_root:
                shutil.copy2(path, robotwin_root / "task_config" / f"{cfg_name}.yml")

    readme = out / "configs_to_apply" / "README_apply_configs.md"
    readme.write_text(
        "Formal v0 defaults to wa_clean_fixed only. Mild/hard random configs are "
        "explicit opt-in and use numeric safe camera/table randomization values.\n",
        encoding="utf-8",
    )
    print("\n".join(written))


if __name__ == "__main__":
    main()
