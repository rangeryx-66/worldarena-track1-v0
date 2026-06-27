#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

DECISIONS = ["PASS", "WARN_KEEP", "REJECT", "DPO_LOSER_CANDIDATE", "NEED_HUMAN_REVIEW"]
SCORE_KEYS = [
    "domain_match", "robot_visibility", "gripper_contact_visibility", "object_visibility",
    "physical_plausibility", "temporal_consistency", "visual_quality", "task_relevance",
    "sft_positive_suitability", "a2v_positive_suitability", "dpo_loser_suitability",
]
FLAG_KEYS = [
    "arm_partially_out_of_frame_but_ok", "critical_contact_invisible", "object_moves_without_contact",
    "severe_flicker_or_exposure_jump", "severe_noise_or_compression", "wrong_robot_or_missing_robot",
    "prompt_action_mismatch", "possible_left_right_swap",
]


def resolve_manifest(path: Path) -> Path:
    if path.exists():
        return path
    alt = path.parent / "manifests" / path.name
    if alt.exists():
        return alt
    if path.name == "episode_manifest.parquet":
        alt = path / "manifests" / "episode_manifest.parquet"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"manifest not found: {path}")


def read_table(path: Path) -> pd.DataFrame:
    path = resolve_manifest(path) if path.name == "episode_manifest.parquet" or path.is_dir() else path
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path}")


def write_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported output format: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def normalize_vlm_result(obj: dict[str, Any]) -> dict[str, Any]:
    decision = str(obj.get("overall_decision", "NEED_HUMAN_REVIEW")).strip().upper()
    if decision not in DECISIONS:
        decision = "NEED_HUMAN_REVIEW"
    scores = obj.get("scores") if isinstance(obj.get("scores"), dict) else {}
    flags = obj.get("flags") if isinstance(obj.get("flags"), dict) else {}
    rec = obj.get("recommended_use") if isinstance(obj.get("recommended_use"), dict) else {}
    evidence = obj.get("evidence") if isinstance(obj.get("evidence"), list) else []
    return {
        "overall_decision": decision,
        "confidence": max(0.0, min(1.0, safe_float(obj.get("confidence"), 0.0))),
        "scores": {k: int(max(0, min(2, safe_float(scores.get(k), 1)))) for k in SCORE_KEYS},
        "flags": {k: bool(flags.get(k, False)) for k in FLAG_KEYS},
        "evidence": [str(x)[:240] for x in evidence[:6]],
        "recommended_use": {
            "use_for_sft": bool(rec.get("use_for_sft", False)),
            "use_for_a2v": bool(rec.get("use_for_a2v", False)),
            "use_for_dpo_winner": bool(rec.get("use_for_dpo_winner", False)),
            "use_for_dpo_loser": bool(rec.get("use_for_dpo_loser", False)),
        },
    }


def strict_json_from_text(text: str) -> dict[str, Any]:
    try:
        return normalize_vlm_result(json.loads(text))
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object found")
    return normalize_vlm_result(json.loads(m.group(0)))


def mock_vlm_from_rule(rule: dict[str, Any]) -> dict[str, Any]:
    hard = str(rule.get("deterministic_hard_fail", "")).lower() == "true" or bool(rule.get("deterministic_hard_fail", False))
    labels = str(rule.get("heuristic_candidate_labels", ""))
    status = str(rule.get("qc_status", "pass")).lower()
    if hard:
        decision, conf = "REJECT", 0.95
    elif "possible_motion_without_contact" in labels:
        decision, conf = "DPO_LOSER_CANDIDATE", 0.72
    elif status == "pass":
        decision, conf = "PASS", 0.82
    elif status == "reject":
        decision, conf = "WARN_KEEP", 0.65
    else:
        decision, conf = "WARN_KEEP", 0.68
    visual_quality = 1 if ("color_shift" in labels or "temporal_flicker" in labels) else 2
    scores = {k: 1 for k in SCORE_KEYS}
    scores.update({
        "domain_match": 2, "robot_visibility": 2, "object_visibility": 2,
        "physical_plausibility": 1 if decision == "DPO_LOSER_CANDIDATE" else 2,
        "temporal_consistency": visual_quality, "visual_quality": visual_quality,
        "sft_positive_suitability": 2 if decision == "PASS" else 1,
        "a2v_positive_suitability": 2 if decision == "PASS" else 1,
        "dpo_loser_suitability": 2 if decision == "DPO_LOSER_CANDIDATE" else 0,
    })
    flags = {k: False for k in FLAG_KEYS}
    flags["severe_flicker_or_exposure_jump"] = "strong_color_or_exposure_jump" in labels
    flags["object_moves_without_contact"] = "possible_motion_without_contact" in labels
    return normalize_vlm_result({
        "overall_decision": decision,
        "confidence": conf,
        "scores": scores,
        "flags": flags,
        "evidence": ["dummy backend decision from rule QC context", labels or "no heuristic labels"],
        "recommended_use": {
            "use_for_sft": decision == "PASS",
            "use_for_a2v": decision in {"PASS", "WARN_KEEP"},
            "use_for_dpo_winner": decision == "PASS",
            "use_for_dpo_loser": decision == "DPO_LOSER_CANDIDATE",
        },
    })


def make_contact_sheet(df: pd.DataFrame, out_path: Path, title: str, n: int = 36) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h, label_h, cols = 160, 120, 36, 6
    if df.empty:
        img = Image.new("RGB", (640, 160), "white")
        ImageDraw.Draw(img).text((20, 60), f"{title}: no samples", fill=(0, 0, 0))
        img.save(out_path, quality=92)
        return
    sample = df.sample(min(n, len(df)), random_state=17) if len(df) > n else df
    rows = int(math.ceil(len(sample) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    for i, (_, row) in enumerate(sample.iterrows()):
        x = (i % cols) * thumb_w
        y = 30 + (i // cols) * (thumb_h + label_h)
        img = None
        fp = str(row.get("first_frame_320x240_path") or row.get("first_frame_path") or "")
        if fp and Path(fp).exists():
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                img = None
        if img is None:
            img = Image.new("RGB", (thumb_w, thumb_h), (230, 230, 230))
        img.thumbnail((thumb_w, thumb_h))
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        canvas.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        sheet.paste(canvas, (x, y))
        label = f"{row.get('episode_id','')} {row.get('final_decision', row.get('qc_status',''))}\n{row.get('final_reason', row.get('qc_reason',''))}"
        draw.text((x + 3, y + thumb_h + 2), label[:90], fill=(0, 0, 0), font=font)
    sheet.save(out_path, quality=92)
