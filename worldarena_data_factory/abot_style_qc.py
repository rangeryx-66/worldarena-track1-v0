#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parent))
from v0_1_vlm_qc import Qwen3VLBackend

ABOT_QC_FIELDS = [
    "episode_id", "task_family", "robotwin_task_name", "video_640x480_path", "action_joint14_raw_path",
    "hard_filter_pass", "hard_fail_reason", "fps", "width", "height", "frame_count", "T",
    "motion_score", "motion_active_ratio", "visual_motion_energy", "static_clip", "over_motion_clip",
    "clip_temporal_coherence_score", "dino_temporal_coherence_score", "temporal_coherence_score",
    "scene_cut_score", "duplicate_frame_ratio", "identity_jump_candidate",
    "vlm_domain_score", "vlm_task_progress_score", "vlm_physics_score", "vlm_semantic_consistency_score",
    "vlm_decision", "vlm_confidence", "vlm_evidence", "vlm_raw_output_path",
    "action_motion_energy", "gripper_transition_count", "action_video_consistency_score",
    "final_qc_status", "final_reason", "recommended_for_sft", "recommended_for_a2v", "recommended_for_dpo_loser",
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


def read_manifest(path: Path) -> pd.DataFrame:
    path = resolve_manifest(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported manifest format: {path}")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def json_extract(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object found in VLM output")
    return json.loads(m.group(0))


def normalize_vlm(obj: dict[str, Any]) -> dict[str, Any]:
    decision = str(obj.get("vlm_decision") or obj.get("decision") or "WARN").upper()
    if decision not in {"PASS", "WARN", "REJECT", "DPO_LOSER"}:
        decision = "WARN"
    def score(name: str, default: int = 1) -> int:
        try:
            return int(max(0, min(2, float(obj.get(name, default)))))
        except Exception:
            return default
    evidence = obj.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    return {
        "vlm_decision": decision,
        "vlm_confidence": clamp01(safe_float(obj.get("confidence"), 0.0)),
        "vlm_domain_score": score("vlm_domain_score"),
        "vlm_task_progress_score": score("vlm_task_progress_score"),
        "vlm_physics_score": score("vlm_physics_score"),
        "vlm_semantic_consistency_score": score("vlm_semantic_consistency_score"),
        "vlm_evidence": "; ".join(str(x)[:220] for x in evidence[:6]),
    }


class TemporalEmbeddingBackend:
    def __init__(self, cache_dir: Path, clip_model: str, dino_model: str, local_files_only: bool = False):
        self.cache_dir = cache_dir
        self.clip_model_id = clip_model
        self.dino_model_id = dino_model
        self.local_files_only = local_files_only
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
        except Exception as exc:
            raise RuntimeError("CLIP/DINO temporal coherence requires torch + transformers in this Python env") from exc
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model, cache_dir=str(cache_dir), local_files_only=local_files_only)
            self.clip_model = CLIPModel.from_pretrained(clip_model, cache_dir=str(cache_dir), local_files_only=local_files_only).to(self.device).eval()
            self.dino_processor = AutoImageProcessor.from_pretrained(dino_model, cache_dir=str(cache_dir), local_files_only=local_files_only)
            self.dino_model = AutoModel.from_pretrained(dino_model, cache_dir=str(cache_dir), local_files_only=local_files_only).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load/download CLIP or DINO. If network is needed, run: cd /root && source ./proxyon.sh, "
                f"then rerun. cache_dir={cache_dir} clip={clip_model} dino={dino_model}"
            ) from exc

    def embed_clip(self, pil_images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        outs = []
        with self.torch.no_grad():
            for i in range(0, len(pil_images), batch_size):
                batch = pil_images[i:i + batch_size]
                inputs = self.clip_processor(images=batch, return_tensors="pt").to(self.device)
                feats = self.clip_model.get_image_features(**inputs)
                if not hasattr(feats, "norm"):
                    if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                        feats = feats.image_embeds
                    elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                        feats = feats.pooler_output
                    elif hasattr(feats, "last_hidden_state"):
                        feats = feats.last_hidden_state[:, 0]
                    elif isinstance(feats, (tuple, list)):
                        feats = feats[0]
                    else:
                        raise TypeError(f"unsupported CLIP feature output type: {type(feats).__name__}")
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                outs.append(feats.detach().cpu().float().numpy())
        return np.vstack(outs) if outs else np.zeros((0, 1), dtype=np.float32)

    def embed_dino(self, pil_images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        outs = []
        with self.torch.no_grad():
            for i in range(0, len(pil_images), batch_size):
                batch = pil_images[i:i + batch_size]
                inputs = self.dino_processor(images=batch, return_tensors="pt").to(self.device)
                output = self.dino_model(**inputs)
                if hasattr(output, "pooler_output") and output.pooler_output is not None:
                    feats = output.pooler_output
                else:
                    feats = output.last_hidden_state[:, 0]
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                outs.append(feats.detach().cpu().float().numpy())
        return np.vstack(outs) if outs else np.zeros((0, 1), dtype=np.float32)

    def temporal_metrics(self, pil_images: list[Image.Image], gray_frames: list[np.ndarray]) -> dict[str, Any]:
        if len(pil_images) < 2:
            return {
                "clip_temporal_coherence_score": 0.0,
                "dino_temporal_coherence_score": 0.0,
                "temporal_coherence_score": 0.0,
                "scene_cut_score": 1.0,
                "duplicate_frame_ratio": 1.0,
                "identity_jump_candidate": True,
            }
        clip = self.embed_clip(pil_images)
        dino = self.embed_dino(pil_images)
        clip_adj = np.sum(clip[:-1] * clip[1:], axis=1) if len(clip) > 1 else np.asarray([0.0])
        dino_adj = np.sum(dino[:-1] * dino[1:], axis=1) if len(dino) > 1 else np.asarray([0.0])
        clip_coh = float(np.mean(clip_adj))
        dino_coh = float(np.mean(dino_adj))
        # CLIP is more robust to normal robot/object motion; DINO is sensitive to
        # structure/identity jumps. Combine them so close-up manipulation does not
        # become a false scene cut from DINO alone.
        combined_adj = 0.65 * clip_adj + 0.35 * dino_adj
        scene_cut = float(1.0 - np.min(combined_adj))
        dup = []
        for a, b in zip(gray_frames[:-1], gray_frames[1:]):
            if a.shape != b.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]))
            mad = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
            dup.append(float(mad < 1.0))
        duplicate_ratio = float(np.mean(dup)) if dup else 0.0
        temporal = float(0.5 * clip_coh + 0.5 * dino_coh)
        identity_jump = bool(scene_cut > 0.55 and np.min(clip_adj) < 0.72 and np.min(dino_adj) < 0.45)
        return {
            "clip_temporal_coherence_score": clip_coh,
            "dino_temporal_coherence_score": dino_coh,
            "temporal_coherence_score": temporal,
            "scene_cut_score": scene_cut,
            "duplicate_frame_ratio": duplicate_ratio,
            "identity_jump_candidate": identity_jump,
        }


def sample_indices(frame_count: int, fps: float, sample_fps: float) -> list[int]:
    if frame_count <= 0:
        return []
    fps = fps if fps > 0 else 24.0
    step = max(1, int(round(fps / max(sample_fps, 0.1))))
    return list(range(0, frame_count, step)) or [0]


def vlm_indices(frame_count: int, num_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    n = min(num_frames, frame_count)
    return sorted(set(int(round(x)) for x in np.linspace(0, frame_count - 1, n)))


def read_video_sample(video_path: Path, sample_fps: float) -> tuple[dict[str, Any], list[np.ndarray], list[Image.Image], list[np.ndarray], int]:
    meta = {"video_readable": False, "fps": 0.0, "width": 0, "height": 0, "frame_count": 0, "read_fail": 0}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return meta, [], [], [], 0
    fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    meta.update({"video_readable": True, "fps": fps, "width": width, "height": height, "frame_count": frame_count})
    frames_bgr = []
    pil_images = []
    gray_frames = []
    read_fail = 0
    for idx in sample_indices(frame_count, fps, sample_fps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            read_fail += 1
            continue
        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        frames_bgr.append(small)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb))
        gray_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    cap.release()
    meta["read_fail"] = read_fail
    return meta, frames_bgr, pil_images, gray_frames, read_fail


def compute_motion_metrics(frames_bgr: list[np.ndarray]) -> dict[str, Any]:
    if len(frames_bgr) < 2:
        return {"motion_score": 0.0, "visual_motion_energy": 0.0, "motion_active_ratio": 0.0, "static_clip": True, "over_motion_clip": False}
    means = []
    p95s = []
    active = []
    prev = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    for frame in frames_bgr[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        except Exception:
            mag = cv2.absdiff(prev, gray).astype(np.float32) / 8.0
        mean = float(np.mean(mag))
        p95 = float(np.quantile(mag, 0.95))
        means.append(mean)
        p95s.append(p95)
        active.append(float(mean > 0.25 or p95 > 1.2))
        prev = gray
    motion_score = float(np.mean(means)) if means else 0.0
    visual_energy = float(np.mean(p95s)) if p95s else 0.0
    active_ratio = float(np.mean(active)) if active else 0.0
    static_clip = bool(active_ratio < 0.03 and visual_energy < 0.25)
    over_motion = bool(active_ratio > 0.90 and visual_energy > 18.0)
    return {
        "motion_score": motion_score,
        "visual_motion_energy": visual_energy,
        "motion_active_ratio": active_ratio,
        "static_clip": static_clip,
        "over_motion_clip": over_motion,
    }


def validate_action(row: pd.Series) -> tuple[dict[str, Any], np.ndarray | None]:
    path = Path(str(row.get("action_joint14_raw_path") or row.get("action_joint14_norm_path") or ""))
    result = {
        "action_valid": False,
        "action_reason": "action_missing",
        "action_motion_energy": 0.0,
        "gripper_transition_count": 0,
        "T": int(safe_float(row.get("T"), 0)),
    }
    if not path.exists():
        return result, None
    try:
        arr = np.load(path)
        result["T"] = int(arr.shape[0]) if len(arr.shape) >= 1 else 0
        has_nan = bool(np.isnan(arr).any())
        has_inf = bool(np.isinf(arr).any())
        if len(arr.shape) != 2 or arr.shape[1] != 14 or arr.shape[0] < 60:
            result["action_reason"] = f"invalid_action_shape:{arr.shape}"
            return result, arr
        if has_nan or has_inf:
            result["action_reason"] = "action_nan_or_inf"
            return result, arr
        delta = np.diff(arr.astype(np.float32), axis=0)
        arm_delta = np.concatenate([delta[:, :6], delta[:, 7:13]], axis=1) if len(delta) else np.zeros((0, 12), dtype=np.float32)
        result["action_motion_energy"] = float(np.mean(np.linalg.norm(arm_delta, axis=1))) if len(arm_delta) else 0.0
        grip = arr[:, [6, 13]]
        result["gripper_transition_count"] = int(np.sum(np.abs(np.diff((grip > 0.5).astype(np.int8), axis=0))))
        result["action_valid"] = True
        result["action_reason"] = "ok"
        return result, arr
    except Exception as exc:
        result["action_reason"] = f"action_load_error:{type(exc).__name__}"
        return result, None


def action_video_consistency(action_energy: float, visual_energy: float) -> float:
    # Joint deltas and optical-flow magnitudes live on very different scales.
    # These monotonic maps make normal Aloha manipulation (moderate joint deltas,
    # large gripper/object foreground flow) compare as consistent instead of
    # punishing close-up motion.
    a = clamp01(math.log1p(max(action_energy, 0.0) * 100.0) / math.log1p(8.0))
    v = clamp01(math.log1p(max(visual_energy, 0.0) / 4.0) / math.log1p(4.0))
    return clamp01(1.0 - abs(a - v))


def draw_contact_sheet(video_path: Path, out_path: Path, num_frames: int) -> list[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        img = Image.new("RGB", (640, 160), "white")
        ImageDraw.Draw(img).text((20, 70), "unreadable video", fill=(0, 0, 0))
        img.save(out_path, quality=92)
        return [out_path]
    fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 24.0) or 24.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    for idx in vlm_indices(count, num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append((idx, idx / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    cols = 4
    thumb_w, thumb_h, label_h = 240, 180, 24
    rows = max(1, int(math.ceil(len(frames) / cols)))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h) + 28), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), video_path.name, fill=(0, 0, 0), font=font)
    for i, (idx, ts, arr) in enumerate(frames):
        im = Image.fromarray(arr)
        im.thumbnail((thumb_w, thumb_h))
        x = (i % cols) * thumb_w
        y = 28 + (i // cols) * (thumb_h + label_h)
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        canvas.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
        sheet.paste(canvas, (x, y))
        draw.text((x + 4, y + 4), f"frame {idx} / {ts:.2f}s", fill=(0, 0, 0), font=font)
    sheet.save(out_path, quality=92)
    return [out_path]


def build_vlm_prompt(row: pd.Series) -> str:
    return f"""
You are judging whether a robot manipulation clip is suitable as a positive SFT/A2V training sample for ABot-PhysWorld.
This is WorldArena/RoboTwin-style fixed tabletop manipulation. White background, mild simulator render grain, and robot arms partially outside the camera are normal.
Do not use or infer any old rule-QC labels such as color_shift, flicker, or arm_visibility_low. Judge the contact sheet directly.

Episode:
episode_id: {row.get('episode_id')}
task_family: {row.get('task_family')}
robotwin_task_name: {row.get('robotwin_task_name')}
prompt: {row.get('prompt_worldarena_style') or row.get('prompt_short')}

Check only these points:
1. Is this a fixed tabletop robotic manipulation scene?
2. Is it visually consistent with an Aloha/AgileX-style dual-arm gripper setup?
3. Is a target object visible?
4. Is there meaningful task progress across time?
5. Is the clip broadly consistent with the prompt/task family?
6. Are there obvious physics/semantic failures: object disappears, tunnels, floats, teleports, or moves without visible/plausible contact?

Return strict JSON only:
{{
  "vlm_decision": "PASS|WARN|REJECT|DPO_LOSER",
  "confidence": 0.0,
  "vlm_domain_score": 0,
  "vlm_task_progress_score": 0,
  "vlm_physics_score": 0,
  "vlm_semantic_consistency_score": 0,
  "evidence": ["short evidence"]
}}
Scores are 0 bad/absent, 1 acceptable or uncertain, 2 good.
""".strip()


class DummyVLM:
    def generate(self, prompt: str, image_paths: list[Path], video_path: Path | None = None) -> str:
        return json.dumps({
            "vlm_decision": "PASS",
            "confidence": 0.75,
            "vlm_domain_score": 2,
            "vlm_task_progress_score": 1,
            "vlm_physics_score": 1,
            "vlm_semantic_consistency_score": 1,
            "evidence": ["dummy backend: flow/temporal/action metrics decide final status"],
        })


def run_vlm(backend: Any, prompt: str, image_paths: list[Path], raw_path: Path, episode_id: str) -> dict[str, Any]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    last = ""
    for attempt in range(1, 4):
        try:
            text = backend.generate(prompt, image_paths)
            last = text
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(f"\n--- attempt {attempt} ---\n{text}\n")
            return normalize_vlm(json_extract(text))
        except Exception as exc:
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(f"\n--- parse/generate error attempt {attempt} ---\n{type(exc).__name__}: {exc}\n{last[:1500]}\n")
            prompt = "Return only valid JSON matching the requested schema. Use WARN if uncertain. Previous output:\n" + last[:1500]
    return normalize_vlm({
        "vlm_decision": "WARN",
        "confidence": 0.0,
        "vlm_domain_score": 1,
        "vlm_task_progress_score": 1,
        "vlm_physics_score": 1,
        "vlm_semantic_consistency_score": 1,
        "evidence": ["VLM failed to produce parseable JSON"],
    })


def final_decision(row: dict[str, Any]) -> tuple[str, str]:
    reasons = []
    if not row["hard_filter_pass"]:
        return "REJECT", row["hard_fail_reason"]
    if row["static_clip"]:
        reasons.append("static_clip")
    if row["over_motion_clip"]:
        reasons.append("over_motion_clip")
    if row["duplicate_frame_ratio"] > 0.65:
        reasons.append("large_duplicate_frame_ratio")
    if row["scene_cut_score"] > 0.55 or row["identity_jump_candidate"]:
        reasons.append("temporal_identity_jump_or_scene_cut")
    if row["vlm_decision"] == "REJECT" and row["vlm_confidence"] >= 0.65:
        reasons.append("vlm_reject")
    if row["vlm_decision"] == "DPO_LOSER" and row["vlm_confidence"] >= 0.60:
        return "DPO_LOSER", "vlm_dpo_loser;" + ";".join(reasons)
    if row["vlm_task_progress_score"] == 0 or row["vlm_physics_score"] == 0 or row["vlm_semantic_consistency_score"] == 0:
        reasons.append("vlm_zero_task_physics_or_semantic_score")
    if row["action_video_consistency_score"] < 0.35:
        reasons.append("low_action_video_consistency")
    if row["action_motion_energy"] > 0.04 and row["visual_motion_energy"] < 0.20:
        reasons.append("action_large_video_static")
    if row["gripper_transition_count"] > 12 and row["vlm_task_progress_score"] <= 1:
        reasons.append("many_gripper_transitions_low_task_progress")
    if row["visual_motion_energy"] > 2.0 and row["action_motion_energy"] < 0.003:
        return "DPO_LOSER", "visual_motion_large_action_weak"
    severe = {"static_clip", "over_motion_clip", "temporal_identity_jump_or_scene_cut", "vlm_reject", "vlm_zero_task_physics_or_semantic_score"}
    if any(r in severe for r in reasons):
        return "REJECT", ";".join(reasons)
    if reasons:
        return "WARN", ";".join(reasons)
    if row["vlm_decision"] == "WARN" or row["vlm_confidence"] < 0.55:
        return "WARN", "vlm_warn_or_low_confidence"
    return "PASS", "ok"


def existing_scores(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame(columns=ABOT_QC_FIELDS)


def append_score(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ABOT_QC_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in ABOT_QC_FIELDS})


def make_sample_sheet(df: pd.DataFrame, out_path: Path, title: str, n: int = 36) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        img = Image.new("RGB", (640, 160), "white")
        ImageDraw.Draw(img).text((20, 70), f"{title}: no samples", fill=(0, 0, 0))
        img.save(out_path, quality=92)
        return
    sample = df.sample(min(n, len(df)), random_state=23) if len(df) > n else df
    cols, tw, th, lh = 6, 160, 120, 38
    rows = int(math.ceil(len(sample) / cols))
    sheet = Image.new("RGB", (cols * tw, rows * (th + lh) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    for i, (_, row) in enumerate(sample.iterrows()):
        x = (i % cols) * tw
        y = 30 + (i // cols) * (th + lh)
        img = None
        fp = str(row.get("first_frame_320x240_path") or "")
        if fp and Path(fp).exists():
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                img = None
        if img is None:
            img = Image.new("RGB", (tw, th), (230, 230, 230))
        img.thumbnail((tw, th))
        canvas = Image.new("RGB", (tw, th), "white")
        canvas.paste(img, ((tw - img.width) // 2, (th - img.height) // 2))
        sheet.paste(canvas, (x, y))
        draw.text((x + 3, y + th + 2), f"{row.get('episode_id','')}\n{row.get('final_reason','')[:60]}", fill=(0, 0, 0), font=font)
    sheet.save(out_path, quality=92)


def write_report(out: Path, scores: pd.DataFrame) -> None:
    reasons = Counter()
    for reason in scores["final_reason"].fillna(""):
        for part in str(reason).split(";"):
            if part:
                reasons[part] += 1
    counts = scores["final_qc_status"].value_counts(dropna=False).to_dict() if not scores.empty else {}
    step_counts = {
        "hard_filter_reject": int((scores["hard_filter_pass"] == False).sum()) if "hard_filter_pass" in scores else 0,
        "static_clip": int(scores["static_clip"].sum()) if "static_clip" in scores else 0,
        "over_motion_clip": int(scores["over_motion_clip"].sum()) if "over_motion_clip" in scores else 0,
        "identity_jump_candidate": int(scores["identity_jump_candidate"].sum()) if "identity_jump_candidate" in scores else 0,
        "vlm_reject": int((scores["vlm_decision"] == "REJECT").sum()) if "vlm_decision" in scores else 0,
        "vlm_dpo_loser": int((scores["vlm_decision"] == "DPO_LOSER").sum()) if "vlm_decision" in scores else 0,
        "low_action_video_consistency": int((scores["action_video_consistency_score"] < 0.35).sum()) if "action_video_consistency_score" in scores else 0,
    }
    lines = [
        "# ABot-PhysWorld Style QC Report", "",
        f"Episodes checked: `{len(scores)}`", "",
        "## Final QC Status", "",
    ]
    for k in ["PASS", "WARN", "REJECT", "DPO_LOSER"]:
        lines.append(f"- `{k}`: `{counts.get(k, 0)}`")
    lines += ["", "## Step Filter Counts", ""]
    for k, v in step_counts.items():
        lines.append(f"- `{k}`: `{v}`")
    lines += ["", "## Top Reject/Warn/DPO Reasons", ""]
    for reason, count in reasons.most_common(30):
        lines.append(f"- `{reason}`: `{count}`")
    lines += [
        "", "## Training Recommendations", "",
        "- SFT: use `episode_manifest_abot_qc_pass.parquet` only.",
        "- A2V: use PASS and manually reviewed/low-weight WARN only.",
        "- DPO loser bank: use `dpo_loser_candidates_abot_qc.csv`.",
        "- This QC intentionally does not use old v2 visual heuristic status or labels as training decisions.",
    ]
    (out / "abot_qc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_outputs(out: Path, manifest: pd.DataFrame, scores: pd.DataFrame) -> None:
    scores = scores.drop_duplicates("episode_id", keep="last")
    scores.to_csv(out / "abot_qc_scores.csv", index=False)
    full = manifest.merge(scores, on=["episode_id", "task_family", "robotwin_task_name", "video_640x480_path", "action_joint14_raw_path"], how="inner")
    full[full["final_qc_status"] == "PASS"].to_parquet(out / "episode_manifest_abot_qc_pass.parquet", index=False)
    full[full["final_qc_status"] == "WARN"].to_parquet(out / "episode_manifest_abot_qc_warn.parquet", index=False)
    full[full["final_qc_status"] == "REJECT"].to_parquet(out / "episode_manifest_abot_qc_reject.parquet", index=False)
    full[full["final_qc_status"] == "DPO_LOSER"].to_csv(out / "dpo_loser_candidates_abot_qc.csv", index=False)
    cs = out / "contact_sheets"
    make_sample_sheet(full[full["final_qc_status"] == "PASS"], cs / "pass_samples.jpg", "ABot QC PASS samples")
    make_sample_sheet(full[full["final_qc_status"] == "WARN"], cs / "warn_samples.jpg", "ABot QC WARN samples")
    make_sample_sheet(full[full["final_qc_status"] == "REJECT"], cs / "reject_samples.jpg", "ABot QC REJECT samples")
    review = full[full["final_qc_status"].isin(["WARN", "DPO_LOSER"])]
    make_sample_sheet(review, cs / "vlm_review_samples.jpg", "ABot QC VLM/manual review samples")
    write_report(out, scores)


def progress_iter(iterator, total: int, interval: int = 10):
    try:
        from tqdm import tqdm
        yield from tqdm(iterator, total=total, desc="abot-qc", dynamic_ncols=True)
        return
    except Exception:
        pass
    start = time.time()
    last = 0
    for i, item in enumerate(iterator, 1):
        yield item
        if i == total or i - last >= interval:
            elapsed = max(1e-6, time.time() - start)
            rate = i / elapsed
            eta = (total - i) / max(rate, 1e-6)
            print(f"abot-qc {i}/{total} ({i/total:.1%}) rate={rate:.3f}/s eta={eta/60:.1f}min", flush=True)
            last = i


def process_episode(row: pd.Series, args: argparse.Namespace, temporal: TemporalEmbeddingBackend, vlm_backend: Any, out: Path) -> dict[str, Any]:
    episode_id = str(row.get("episode_id"))
    video_path = Path(str(row.get("video_640x480_path") or ""))
    action_info, _ = validate_action(row)
    meta, frames_bgr, pil_images, gray_frames, read_fail = read_video_sample(video_path, args.sample_fps)
    hard_reasons = []
    if not video_path.exists():
        hard_reasons.append("video_missing")
    elif not meta["video_readable"]:
        hard_reasons.append("video_unreadable")
    if meta["video_readable"]:
        if abs(float(meta["fps"]) - 24.0) > 2.5:
            hard_reasons.append("fps_invalid")
        if (int(meta["width"]), int(meta["height"])) != (640, 480):
            hard_reasons.append("resolution_invalid")
        if int(meta["frame_count"]) < 24:
            hard_reasons.append("frame_count_too_short")
        if not frames_bgr:
            hard_reasons.append("no_decodable_sample_frames")
        if read_fail > max(2, len(sample_indices(int(meta["frame_count"]), float(meta["fps"]), args.sample_fps)) // 4):
            hard_reasons.append("many_sample_frames_failed")
    if not action_info["action_valid"]:
        hard_reasons.append(action_info["action_reason"])
    if action_info["T"] and meta["frame_count"]:
        ratio = float(meta["frame_count"]) / max(float(action_info["T"]), 1.0)
        if ratio < 0.1 or ratio > 10.0:
            hard_reasons.append("action_video_length_impossible_to_align")

    motion = compute_motion_metrics(frames_bgr)
    if frames_bgr and action_info["action_valid"]:
        temporal_metrics = temporal.temporal_metrics(pil_images, gray_frames)
    else:
        temporal_metrics = {
            "clip_temporal_coherence_score": 0.0,
            "dino_temporal_coherence_score": 0.0,
            "temporal_coherence_score": 0.0,
            "scene_cut_score": 1.0,
            "duplicate_frame_ratio": 1.0,
            "identity_jump_candidate": True,
        }

    sheet_path = out / "vlm_inputs" / f"{episode_id}.jpg"
    image_paths = draw_contact_sheet(video_path, sheet_path, args.num_vlm_frames)
    raw_vlm_path = out / "raw_vlm_outputs" / f"{episode_id}.txt"
    if args.backend == "dummy" or args.dry_run:
        vlm = normalize_vlm(json.loads(DummyVLM().generate("", image_paths)))
    else:
        vlm = run_vlm(vlm_backend, build_vlm_prompt(row), image_paths, raw_vlm_path, episode_id)
    if args.backend == "dummy" or args.dry_run:
        raw_vlm_path.parent.mkdir(parents=True, exist_ok=True)
        raw_vlm_path.write_text(json.dumps(vlm, ensure_ascii=False, indent=2), encoding="utf-8")

    consistency = action_video_consistency(action_info["action_motion_energy"], motion["visual_motion_energy"])
    result = {
        "episode_id": episode_id,
        "task_family": row.get("task_family", ""),
        "robotwin_task_name": row.get("robotwin_task_name", ""),
        "video_640x480_path": str(video_path),
        "action_joint14_raw_path": row.get("action_joint14_raw_path", ""),
        "hard_filter_pass": not bool(hard_reasons),
        "hard_fail_reason": ";".join(hard_reasons),
        "fps": meta.get("fps", 0.0),
        "width": meta.get("width", 0),
        "height": meta.get("height", 0),
        "frame_count": meta.get("frame_count", 0),
        "T": action_info["T"],
        **motion,
        **temporal_metrics,
        **vlm,
        "vlm_raw_output_path": str(raw_vlm_path),
        "action_motion_energy": action_info["action_motion_energy"],
        "gripper_transition_count": action_info["gripper_transition_count"],
        "action_video_consistency_score": consistency,
    }
    status, reason = final_decision(result)
    result.update({
        "final_qc_status": status,
        "final_reason": reason,
        "recommended_for_sft": status == "PASS",
        "recommended_for_a2v": status in {"PASS", "WARN"},
        "recommended_for_dpo_loser": status == "DPO_LOSER",
    })
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="ABot-PhysWorld style manipulation clip QC")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--num-vlm-frames", type=int, default=12)
    ap.add_argument("--backend", choices=["qwen3-vl", "dummy"], default="qwen3-vl")
    ap.add_argument("--model-path", default="/root/autodl-tmp/qwen3vl8b")
    ap.add_argument("--model-cache-dir", default="/root/autodl-tmp/model_cache/abot_style_qc")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--dino-model", default="facebook/dinov2-small")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--progress-interval", type=int, default=10)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "contact_sheets").mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest).reset_index(drop=True)
    if args.max_samples:
        manifest = manifest.head(args.max_samples).copy()
    scores_path = out / "abot_qc_scores.csv"
    existing = existing_scores(scores_path) if args.resume else pd.DataFrame(columns=ABOT_QC_FIELDS)
    done = set(existing["episode_id"].astype(str).tolist()) if not existing.empty else set()

    temporal = TemporalEmbeddingBackend(Path(args.model_cache_dir), args.clip_model, args.dino_model, args.local_files_only)
    if args.backend == "dummy" or args.dry_run:
        vlm_backend = DummyVLM()
    else:
        vlm_backend = Qwen3VLBackend(args.model_path, mode="contact_sheet")

    rows = [row for _, row in manifest.iterrows() if str(row.get("episode_id")) not in done]
    for row in progress_iter(rows, len(rows), args.progress_interval):
        result = process_episode(row, args, temporal, vlm_backend, out)
        append_score(scores_path, result)
        with (out / "vlm_qc_results.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: result.get(k) for k in ["episode_id", "vlm_decision", "vlm_confidence", "vlm_domain_score", "vlm_task_progress_score", "vlm_physics_score", "vlm_semantic_consistency_score", "vlm_evidence", "vlm_raw_output_path"]}, ensure_ascii=False) + "\n")

    scores = existing_scores(scores_path)
    # Keep only selected manifest rows when max_samples is used, otherwise all completed rows.
    if args.max_samples:
        allowed = set(manifest["episode_id"].astype(str))
        scores = scores[scores["episode_id"].astype(str).isin(allowed)]
    finalize_outputs(out, manifest, scores)
    print(f"wrote {out / 'abot_qc_scores.csv'} rows={len(scores)}")
    print(scores["final_qc_status"].value_counts(dropna=False).to_dict() if not scores.empty else {})


if __name__ == "__main__":
    main()
