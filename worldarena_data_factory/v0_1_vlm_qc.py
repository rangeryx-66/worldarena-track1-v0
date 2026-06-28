#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parent))
from v0_1_vlm_qc_common import (
    DECISIONS,
    append_jsonl,
    mock_vlm_from_rule,
    normalize_vlm_result,
    read_jsonl,
    read_table,
    strict_json_from_text,
)


def load_prompt_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"prompt config not found: {path}")
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data
    except Exception as exc:
        raise RuntimeError(f"failed to read YAML prompt config {path}: {exc}") from exc


class BaseVLMBackend:
    def generate(
        self, prompt: str, image_paths: list[Path], video_path: Path | None = None
    ) -> str:
        raise NotImplementedError


class DummyBackend(BaseVLMBackend):
    def __init__(self, rule_by_episode: dict[str, dict[str, Any]]):
        self.rule_by_episode = rule_by_episode

    def generate(
        self, prompt: str, image_paths: list[Path], video_path: Path | None = None
    ) -> str:
        episode_id = "unknown"
        for line in prompt.splitlines():
            if line.startswith("episode_id:"):
                episode_id = line.split(":", 1)[1].strip()
                break
        result = mock_vlm_from_rule(self.rule_by_episode.get(episode_id, {}))
        return json.dumps(result, ensure_ascii=False)


class Qwen3VLBackend(BaseVLMBackend):
    def __init__(self, model_path: str, mode: str = "contact_sheet"):
        self.model_path = Path(model_path).expanduser()
        self.mode = mode
        if mode == "video":
            raise RuntimeError(
                "Qwen3-VL video mode is not enabled in this pipeline; use --mode contact_sheet for stable QC."
            )
        if not self.model_path.exists():
            raise FileNotFoundError(f"Qwen3-VL model path not found: {self.model_path}")
        try:
            import torch
            from transformers import AutoProcessor

            try:
                from transformers import Qwen3VLForConditionalGeneration as ModelCls
            except Exception:
                try:
                    from transformers import AutoModelForImageTextToText as ModelCls
                except Exception:
                    from transformers import AutoModelForCausalLM as ModelCls
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as exc:
                raise RuntimeError(
                    "missing qwen_vl_utils; install with: python -m pip install qwen-vl-utils"
                ) from exc
        except Exception as exc:
            raise RuntimeError(
                "failed to import Qwen3-VL dependencies. Recommended env: "
                "/root/autodl-tmp/conda/envs/fantasyworld/bin/python -m pip install qwen-vl-utils pyarrow"
            ) from exc
        self.torch = torch
        self.AutoProcessor = AutoProcessor
        self.process_vision_info = process_vision_info
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = ModelCls.from_pretrained(
            str(self.model_path), torch_dtype=dtype, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(str(self.model_path))

    def generate(
        self, prompt: str, image_paths: list[Path], video_path: Path | None = None
    ) -> str:
        content = []
        for p in image_paths:
            content.append({"type": "image", "image": str(p)})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=768, do_sample=False
            )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def read_frame_at(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_labeled_sheet(
    frames: list[tuple[int, float, np.ndarray]],
    out_path: Path,
    title: str,
    cols: int = 4,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        img = Image.new("RGB", (640, 160), "white")
        ImageDraw.Draw(img).text(
            (20, 70), f"{title}: no readable frames", fill=(0, 0, 0)
        )
        img.save(out_path, quality=92)
        return
    thumb_w, thumb_h, label_h = 240, 180, 24
    rows = int(math.ceil(len(frames) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h) + 28), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    for i, (idx, ts, arr) in enumerate(frames):
        img = Image.fromarray(arr).convert("RGB")
        img.thumbnail((thumb_w, thumb_h))
        x = (i % cols) * thumb_w
        y = 28 + (i // cols) * (thumb_h + label_h)
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        canvas.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        sheet.paste(canvas, (x, y))
        draw.text((x + 4, y + 4), f"frame {idx} / {ts:.2f}s", fill=(0, 0, 0), font=font)
    sheet.save(out_path, quality=92)


def make_overview_sheet(video_path: Path, out_path: Path, num_frames: int) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 24.0) or 24.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count <= 0:
        cap.release()
        return False
    indices = np.linspace(0, max(0, count - 1), num_frames).astype(int).tolist()
    frames = []
    for idx in indices:
        arr = read_frame_at(cap, idx)
        if arr is not None:
            frames.append((idx, idx / fps, arr))
    cap.release()
    draw_labeled_sheet(frames, out_path, f"overview {video_path.name}")
    return bool(frames)


def motion_peak_indices(video_path: Path, num_frames: int) -> list[int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count <= 2:
        cap.release()
        return []
    sample_count = min(80, count - 1)
    sample_indices = np.linspace(1, count - 1, sample_count).astype(int).tolist()
    prev_gray = None
    scores: list[tuple[float, int]] = []
    for idx in sample_indices:
        arr = read_frame_at(cap, idx)
        if arr is None:
            continue
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            scores.append((float(diff.mean()), idx))
        prev_gray = gray
    cap.release()
    if not scores:
        return []
    scores.sort(reverse=True)
    selected: set[int] = set()
    radius = max(1, count // max(32, num_frames * 4))
    for _, peak in scores[: max(4, num_frames)]:
        for off in (-radius, 0, radius):
            selected.add(min(count - 1, max(0, peak + off)))
        if len(selected) >= num_frames:
            break
    return sorted(selected)[:num_frames]


def make_motion_sheet(video_path: Path, out_path: Path, num_frames: int) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 24.0) or 24.0
    indices = motion_peak_indices(video_path, num_frames)
    if not indices:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = (
            np.linspace(0, max(0, count - 1), num_frames).astype(int).tolist()
            if count > 0
            else []
        )
    frames = []
    for idx in indices:
        arr = read_frame_at(cap, idx)
        if arr is not None:
            frames.append((idx, idx / fps, arr))
    cap.release()
    draw_labeled_sheet(frames, out_path, f"motion peaks {video_path.name}")
    return bool(frames)


def find_overlay_asset(row: pd.Series) -> Path | None:
    episode_dir = Path(str(row.get("video_640x480_path") or "")).parent
    for name in [
        "condition_overlay.jpg",
        "condition_map.jpg",
        "trajectory_overlay.jpg",
        "ee16_overlay.jpg",
    ]:
        p = episode_dir / name
        if p.exists():
            return p
    return None


def make_vlm_inputs(row: pd.Series, out: Path, num_frames: int) -> list[Path]:
    episode_id = str(row.get("episode_id"))
    video = Path(str(row.get("video_640x480_path") or row.get("video_path") or ""))
    image_paths: list[Path] = []
    overview = out / "vlm_inputs" / "overview" / f"{episode_id}.jpg"
    motion = out / "vlm_inputs" / "motion" / f"{episode_id}.jpg"
    if not overview.exists():
        make_overview_sheet(video, overview, num_frames)
    if overview.exists():
        image_paths.append(overview)
    if not motion.exists():
        make_motion_sheet(video, motion, num_frames)
    if motion.exists():
        image_paths.append(motion)
    overlay_src = find_overlay_asset(row)
    if overlay_src:
        overlay = out / "vlm_inputs" / "overlay" / f"{episode_id}.jpg"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        if not overlay.exists():
            Image.open(overlay_src).convert("RGB").save(overlay, quality=92)
        image_paths.append(overlay)
    return image_paths


def parse_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    text = str(value)
    if not text or text.lower() == "nan":
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return [x for x in text.replace(";", ",").split(",") if x.strip()]


def build_prompt(
    row: pd.Series,
    rule: dict[str, Any],
    cfg: dict[str, Any],
    include_rule_heuristics: bool = False,
) -> str:
    task_family = str(row.get("task_family") or "unknown")
    task_specific = cfg.get("task_specific_checklists", {}).get(task_family, [])
    parts = [
        cfg.get("system_context", ""),
        "Common checklist:",
        "\n".join(f"- {x}" for x in cfg.get("common_checklist", [])),
        "Negative checklist:",
        "\n".join(f"- {x}" for x in cfg.get("negative_checklist", [])),
        f"Task-specific checklist for {task_family}:",
        "\n".join(f"- {x}" for x in task_specific),
        "Return strict JSON with this schema:",
        json.dumps(cfg.get("output_schema", {}), ensure_ascii=False),
        cfg.get("score_meaning", ""),
        "Episode context:",
        f"episode_id: {row.get('episode_id')}",
        f"task_family: {task_family}",
        f"robotwin_task_name: {row.get('robotwin_task_name', '')}",
        f"prompt_worldarena_style: {row.get('prompt_worldarena_style', '')}",
        f"deterministic_hard_fail: {rule.get('deterministic_hard_fail', False)}",
        f"hard_fail_reason: {rule.get('hard_fail_reason', '')}",
    ]
    if include_rule_heuristics:
        parts.extend(
            [
                "Optional rule-QC soft heuristics. Treat these as non-authoritative triage hints only; judge the images directly.",
                f"rule_qc_status: {rule.get('qc_status', '')}",
                f"rule_qc_reason: {rule.get('qc_reason', '')}",
                f"heuristic_candidate_labels: {parse_labels(rule.get('heuristic_candidate_labels', ''))}",
                f"rule_qc_context: {rule.get('rule_qc_context', '')}",
            ]
        )
    else:
        parts.append(
            "Do not infer quality from hidden rule-QC scores. Judge the contact sheets directly. "
            "White background, mild simulator render grain, and partially out-of-frame arms are normal WorldArena style and are not rejection reasons by themselves."
        )
    parts.append(
        "Important: output JSON only. No markdown, no code fence, no prose outside JSON."
    )
    return "\n\n".join(str(x) for x in parts if str(x).strip())


def select_rows(
    manifest: pd.DataFrame, rule_qc: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    df = manifest.copy()
    if "episode_id" not in df.columns:
        raise ValueError("manifest must contain episode_id")
    rule = rule_qc.copy()
    if not rule.empty and "episode_id" in rule.columns:
        keep_cols = [
            c for c in rule.columns if c not in df.columns or c == "episode_id"
        ]
        df = df.merge(rule[keep_cols], on="episode_id", how="left")
    if args.run_all:
        selected = df
    else:
        status = (
            df.get("qc_status", pd.Series(["pass"] * len(df)))
            .fillna("pass")
            .astype(str)
            .str.lower()
        )
        labels = (
            df.get("heuristic_candidate_labels", pd.Series([""] * len(df)))
            .fillna("")
            .astype(str)
        )
        hard = (
            df.get("deterministic_hard_fail", pd.Series([False] * len(df)))
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        borderline = status.isin(["warn", "reject"]) | labels.ne("") | hard
        selected = df[borderline] if args.only_borderline else df[borderline]
        rng = random.Random(17)
        pass_rows = df[status.eq("pass") & ~borderline]
        reject_rows = df[status.eq("reject") & ~hard]
        add_parts = []
        if args.sample_pass_ratio > 0 and not pass_rows.empty:
            add_parts.append(
                pass_rows.sample(
                    max(1, int(len(pass_rows) * args.sample_pass_ratio)),
                    random_state=rng.randrange(10**9),
                )
            )
        if args.sample_reject_ratio > 0 and not reject_rows.empty:
            add_parts.append(
                reject_rows.sample(
                    max(1, int(len(reject_rows) * args.sample_reject_ratio)),
                    random_state=rng.randrange(10**9),
                )
            )
        if add_parts:
            selected = pd.concat([selected] + add_parts).drop_duplicates("episode_id")
    if args.max_samples:
        selected = selected.head(args.max_samples)
    return selected.reset_index(drop=True)


def existing_episode_ids(results_path: Path) -> set[str]:
    done = set()
    for row in read_jsonl(results_path):
        ep = str(row.get("episode_id", ""))
        if ep:
            done.add(ep)
    return done


def write_parse_error(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        fields = ["episode_id", "attempt", "error", "raw_output_path"]
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def run_backend_with_retries(
    backend: BaseVLMBackend,
    prompt: str,
    image_paths: list[Path],
    raw_path: Path,
    parse_error_path: Path,
    episode_id: str,
) -> dict[str, Any]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    current_prompt = prompt
    last_text = ""
    for attempt in range(1, 4):
        try:
            text = backend.generate(current_prompt, image_paths)
            last_text = text
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(f"\n--- attempt {attempt} ---\n{text}\n")
            return strict_json_from_text(text)
        except Exception as exc:
            errors.append(str(exc))
            write_parse_error(
                parse_error_path,
                {
                    "episode_id": episode_id,
                    "attempt": attempt,
                    "error": str(exc),
                    "raw_output_path": str(raw_path),
                },
            )
            current_prompt = (
                "Return ONLY valid JSON matching the requested QC schema. No markdown. "
                "Use NEED_HUMAN_REVIEW if uncertain. Previous invalid output follows:\n"
                + last_text[:2000]
            )
    return normalize_vlm_result(
        {
            "overall_decision": "NEED_HUMAN_REVIEW",
            "confidence": 0.0,
            "evidence": ["VLM output failed strict JSON parsing after retries"]
            + errors[:2],
            "recommended_use": {
                "use_for_sft": False,
                "use_for_a2v": False,
                "use_for_dpo_winner": False,
                "use_for_dpo_loser": False,
            },
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="WorldArena v0.1 VLM-assisted video QC")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rule-qc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3-vl")
    ap.add_argument("--model-path", default="/root/autodl-tmp/qwen3vl8b")
    ap.add_argument("--backend", choices=["auto", "qwen3-vl", "dummy"], default="auto")
    ap.add_argument(
        "--mode", choices=["contact_sheet", "video"], default="contact_sheet"
    )
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--only-borderline", action="store_true")
    ap.add_argument("--sample-pass-ratio", type=float, default=0.0)
    ap.add_argument("--sample-reject-ratio", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-samples", type=int)
    ap.add_argument(
        "--prompt-config",
        default=str(Path(__file__).resolve().parent / "configs" / "vlm_qc_prompt.yaml"),
    )
    ap.add_argument(
        "--include-rule-heuristics",
        action="store_true",
        help="Include non-authoritative soft rule QC metrics in the VLM prompt. Default is blind visual review.",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_table(Path(args.manifest))
    rule_qc = (
        pd.read_csv(args.rule_qc) if Path(args.rule_qc).exists() else pd.DataFrame()
    )
    cfg = load_prompt_config(Path(args.prompt_config))
    rule_by_episode = (
        {str(r.get("episode_id")): r for r in rule_qc.fillna("").to_dict("records")}
        if not rule_qc.empty
        else {}
    )
    selected = select_rows(manifest, rule_qc, args)
    selected.to_csv(out / "vlm_selected_episodes.csv", index=False)

    backend_name = (
        "dummy"
        if args.dry_run or args.backend == "dummy"
        else (args.backend if args.backend != "auto" else args.model)
    )
    if backend_name == "dummy":
        backend: BaseVLMBackend = DummyBackend(rule_by_episode)
    elif backend_name == "qwen3-vl":
        backend = Qwen3VLBackend(args.model_path, args.mode)
    else:
        raise ValueError(f"unsupported backend: {backend_name}")

    results_path = out / "vlm_qc_results.jsonl"
    parse_error_path = out / "vlm_parse_errors.csv"
    done = existing_episode_ids(results_path) if args.resume else set()
    processed = 0
    start = time.time()
    for _, row in selected.iterrows():
        episode_id = str(row.get("episode_id"))
        if args.resume and episode_id in done:
            continue
        image_paths = make_vlm_inputs(row, out, args.num_frames)
        rule = rule_by_episode.get(episode_id, {})
        prompt = build_prompt(
            row, rule, cfg, include_rule_heuristics=args.include_rule_heuristics
        )
        raw_path = out / "raw_vlm_outputs" / f"{episode_id}.txt"
        if raw_path.exists() and not args.resume:
            raw_path.unlink()
        result = run_backend_with_retries(
            backend, prompt, image_paths, raw_path, parse_error_path, episode_id
        )
        record = {
            "episode_id": episode_id,
            "task_family": row.get("task_family", ""),
            "prompt_worldarena_style": row.get("prompt_worldarena_style", ""),
            "image_paths": [str(p) for p in image_paths],
            "raw_output_path": str(raw_path),
            "backend": backend_name,
            "mode": args.mode,
            **result,
        }
        append_jsonl(results_path, record)
        processed += 1
        if processed % max(1, args.batch_size) == 0:
            elapsed = max(1e-6, time.time() - start)
            print(
                f"processed={processed} total_selected={len(selected)} rate={processed/elapsed:.3f}/s episode={episode_id}",
                flush=True,
            )
    print(f"wrote {results_path} processed={processed} selected={len(selected)}")


if __name__ == "__main__":
    main()
