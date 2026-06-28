from __future__ import annotations
import csv, json, os, re, subprocess, sys, math, random, shutil
from pathlib import Path
from typing import Any

TASK_QUOTAS = {
    "pick_place": 180,
    "object_to_container": 170,
    "button_press_click": 150,
    "articulated_open_close": 130,
    "tool_use": 120,
    "stacking": 120,
    "lifting": 100,
    "scanning_qrcode": 80,
    "shaking": 70,
    "rotation_orientation": 70,
    "dumping_pouring": 60,
    "ranking_arrangement": 60,
    "handover": 50,
    "hanging": 40,
    "coverage_unknown": 100,
}
CONFIG_RATIOS = {"wa_clean_fixed": 1.0}
V0_MAIN_EMBODIMENT = "aloha-agilex"
V0_OPTIONAL_SECONDARY_EMBODIMENTS = {"piper", "ARX-X5"}
DUAL_ARM_CANDIDATE_EMBODIMENTS = ["aloha-agilex", "piper", "ARX-X5", "ur5-wsg"]
V0_DEFAULT_EMBODIMENTS = [("aloha-agilex", 1.0)]
V0_SECONDARY_RATIO = 0.05
EMBODIMENT_ALIASES = {
    "ur5-wsg": "ur5-wsg",
    "UR5-Wsg": "ur5-wsg",
    "UR5-WSG": "ur5-wsg",
    "arx-x5": "ARX-X5",
    "ARX-X5": "ARX-X5",
    "aloha-agilex": "aloha-agilex",
    "piper": "piper",
    "franka-panda": "franka-panda",
}
TASK_CANDIDATES = {
    "pick_place": [
        "place_a2b_left",
        "place_a2b_right",
        "place_empty_cup",
        "place_object_stand",
        "place_object_scale",
        "place_phone_stand",
        "place_shoe",
        "move_stapler_pad",
        "move_pillbottle_pad",
    ],
    "object_to_container": [
        "place_object_basket",
        "place_can_basket",
        "place_cans_plasticbox",
        "put_bottles_dustbin",
        "put_object_cabinet",
        "place_container_plate",
    ],
    "button_press_click": [
        "click_alarmclock",
        "click_bell",
        "press_stapler",
        "turn_switch",
        "stamp_seal",
    ],
    "articulated_open_close": ["open_laptop", "open_microwave", "put_object_cabinet"],
    "tool_use": [
        "beat_block_hammer",
        "press_stapler",
        "stamp_seal",
        "grab_roller",
        "scan_object",
    ],
    "stacking": [
        "stack_blocks_two",
        "stack_blocks_three",
        "stack_bowls_two",
        "stack_bowls_three",
    ],
    "lifting": ["lift_pot", "pick_diverse_bottles", "pick_dual_bottles", "grab_roller"],
    "scanning_qrcode": ["scan_object", "rotate_qrcode"],
    "shaking": ["shake_bottle", "shake_bottle_horizontally"],
    "rotation_orientation": [
        "adjust_bottle",
        "rotate_qrcode",
        "blocks_ranking_rgb",
        "blocks_ranking_size",
    ],
    "dumping_pouring": ["dump_bin_bigbin"],
    "ranking_arrangement": ["blocks_ranking_rgb", "blocks_ranking_size"],
    "handover": ["handover_block", "handover_mic"],
    "hanging": ["hanging_mug"],
    "coverage_unknown": [],
}
PROBE_TASKS = [
    "pick_dual_bottles",
    "place_object_basket",
    "stack_blocks_two",
    "click_bell",
    "open_microwave",
]
POLICY_TO_MODE = {
    "SAME_ACTION_OK": "action_driven_original_hdf5",
    "ACTION_MAYBE_OK": "action_driven_original_or_retrieved_hdf5",
    "TARGET_CHANGED": "action_retrieval_or_robotwin_replan",
    "VERB_CHANGED": "action_retrieval_or_robotwin_replan",
    "AMBIGUOUS": "second_stage_review_or_text_fallback",
}


def ensure_dirs(out: Path):
    for d in [
        "manifests",
        "logs",
        "configs_to_apply",
        "episodes",
        "sft_worldarena_style",
        "a2v_worldarena_joint14",
        "a2v_worldarena_ee16",
        "a2v_worldarena_joint14_ee16",
        "dpo_prompt_action",
        "inference_manifests",
        "rejected",
        "contact_sheets",
        "embodiment_probe",
    ]:
        (out / d).mkdir(parents=True, exist_ok=True)


def normalize_embodiment(name: str | None) -> str | None:
    if not name:
        return None
    return EMBODIMENT_ALIASES.get(
        name, EMBODIMENT_ALIASES.get(name.strip(), name.strip())
    )


def detect_robotwin_root(explicit: str | None = None) -> Path | None:
    cands = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get("ROBOTWIN_ROOT"):
        cands.append(Path(os.environ["ROBOTWIN_ROOT"]))
    cands += [Path("/root/autodl-tmp/RoboTwin"), Path.cwd() / "RoboTwin"]
    for p in cands:
        if (p / "collect_data.sh").exists() and (
            p / "script" / "collect_data.py"
        ).exists():
            return p.resolve()
    return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def simple_yaml(obj, indent=0):
    sp = "  " * indent
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                out.append(f"{sp}{k}:\n" + simple_yaml(v, indent + 1))
            else:
                out.append(f"{sp}{k}: {json.dumps(v) if isinstance(v,str) else v}")
        return "\n".join(out)
    if isinstance(obj, list):
        out = []
        for v in obj:
            if isinstance(v, (dict, list)):
                out.append(f"{sp}-\n" + simple_yaml(v, indent + 1))
            else:
                out.append(f"{sp}- {json.dumps(v) if isinstance(v,str) else v}")
        return "\n".join(out)
    return sp + str(obj)


def write_yaml(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(simple_yaml(data) + "\n", encoding="utf-8")


def available_robotwin_tasks(root: Path | None) -> list[str]:
    if not root:
        return []
    tasks = set()
    for base in [root / "envs", root / "description" / "task_instruction"]:
        if base.exists():
            for p in base.glob("*.py"):
                if not p.name.startswith("_") and p.stem != "__init__":
                    tasks.add(p.stem)
            for p in base.glob("*.json"):
                tasks.add(p.stem)
    return sorted(tasks)


def weighted_cycle(items):
    expanded = []
    for name, ratio in items:
        expanded += [name] * max(1, int(ratio * 100))
    while True:
        for x in expanded:
            yield x


def read_probe_recommendation(out: Path) -> tuple[str | None, str | None]:
    data = (
        read_json(out / "embodiment_probe" / "embodiment_probe_summary.json", {}) or {}
    )
    rec = data.get("recommendation", {})
    return normalize_embodiment(rec.get("main_embodiment")), normalize_embodiment(
        rec.get("secondary_embodiment")
    )


def embodiment_weights(
    out: Path | None = None,
    main: str | None = None,
    secondary: str | None = None,
    include_secondary: bool = False,
):
    # v0_smoke intentionally targets RoboTwin2 Clean-50 / WorldArena Track1's
    # Aloha-AgileX dual-arm gripper domain. The probe remains optional diagnostics
    # and does not affect formal v0 collection unless an explicit secondary flag is used.
    primary = V0_MAIN_EMBODIMENT
    if include_secondary:
        sec = normalize_embodiment(
            secondary or os.environ.get("WORLD_ARENA_SECONDARY_EMBODIMENT") or "piper"
        )
        if sec not in V0_OPTIONAL_SECONDARY_EMBODIMENTS:
            sec = "piper"
        return [(primary, 1.0 - V0_SECONDARY_RATIO), (sec, V0_SECONDARY_RATIO)]
    return [(primary, 1.0)]


def embodiment_plan(
    n: int,
    out: Path | None = None,
    main: str | None = None,
    secondary: str | None = None,
    include_secondary: bool = False,
):
    gen = weighted_cycle(embodiment_weights(out, main, secondary, include_secondary))
    return [next(gen) for _ in range(n)]


def read_table(path: Path) -> list[dict[str, Any]]:
    if path.exists() and path.suffix == ".csv":
        return read_csv(path)
    if path.exists() and path.suffix == ".parquet":
        try:
            import pandas as pd

            return pd.read_parquet(path).fillna("").to_dict("records")
        except Exception:
            pass
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return read_csv(csv_path)
    fb = path.with_suffix(path.suffix + ".fallback.json")
    if fb.exists():
        try:
            actual = Path(read_json(fb, {}).get("actual", ""))
            if actual.exists():
                return read_csv(actual)
        except Exception:
            pass
    return []


def write_table(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        import pyarrow  # noqa

        pd.DataFrame(rows).to_parquet(path, index=False)
        return str(path), "parquet"
    except Exception as e:
        csv_path = path.with_suffix(".csv")
        write_csv(csv_path, rows)
        write_json(
            path.with_suffix(path.suffix + ".fallback.json"),
            {"requested": str(path), "actual": str(csv_path), "reason": str(e)},
        )
        return str(csv_path), "csv_fallback"


def episode_num_from_name(s: str):
    m = re.search(r"episode(\d+)", s or "")
    return int(m.group(1)) if m else None


def log_error(out: Path, name: str, rows: list[dict[str, Any]]):
    if rows:
        write_csv(out / "rejected" / name, rows)


def copy_or_link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def is_v0_training_embodiment(
    embodiment: str,
    main: str | None = None,
    secondary: str | None = None,
    out: Path | None = None,
    include_secondary: bool = False,
) -> bool:
    emb = normalize_embodiment(embodiment)
    allowed = {
        x for x, _ in embodiment_weights(out, main, secondary, include_secondary)
    }
    return emb in allowed and emb != "franka-panda"
