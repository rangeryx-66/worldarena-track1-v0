#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parent))
from utils import ensure_dirs, read_jsonl, read_table, write_json  # noqa: E402


def exists(path) -> bool:
    return bool(path) and Path(path).exists()


def truthy(value) -> bool:
    return str(value).lower() == "true"


def make_sheet(
    items: list[tuple[str, Path]], out_path: Path, title: str, cols: int = 5
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tw, th, lh = 180, 135, 42
    rows = max(1, (len(items) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * tw, rows * (th + lh) + 28), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    for i, (label, path) in enumerate(items):
        x = (i % cols) * tw
        y = 28 + (i // cols) * (th + lh)
        try:
            im = Image.open(path).convert("RGB")
            im.thumbnail((tw, th))
            canvas = Image.new("RGB", (tw, th), "white")
            canvas.paste(im, ((tw - im.width) // 2, (th - im.height) // 2))
            sheet.paste(canvas, (x, y))
        except Exception:
            draw.rectangle([x, y, x + tw - 1, y + th - 1], outline=(255, 0, 0))
            draw.text((x + 5, y + 55), "missing/bad", fill=(255, 0, 0), font=font)
        draw.text((x + 3, y + th + 2), label[:48], fill=(0, 0, 0), font=font)
    sheet.save(out_path, quality=92)


def metadata_check(path: Path, base_path: Path | None = None) -> dict:
    rows = read_jsonl(path)
    missing = 0
    for row in rows:
        for key in [
            "video",
            "action_path",
            "intrinsic_path",
            "extrinsic_path",
            "winner_video",
            "loser_video",
        ]:
            value = row.get(key)
            if not value:
                continue
            p = Path(value)
            if key == "video" and base_path and not p.is_absolute():
                p = base_path / p
            if not p.exists():
                missing += 1
    return {"rows": len(rows), "missing_paths": missing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/root/autodl-tmp/worldarena_data_factory_v0")
    parser.add_argument("--preview-seed", type=int, default=7)
    args = parser.parse_args()

    out = Path(args.out)
    ensure_dirs(out)
    report = {"checks": {}, "errors": []}
    episodes = read_table(out / "manifests" / "episode_manifest.parquet")
    total = len(episodes)
    visual_pass = sum(
        str(r.get("visual_sanity_status", "")).upper() == "PASS" for r in episodes
    )
    accepted_sft = sum(truthy(r.get("accepted_for_sft", "")) for r in episodes)
    accepted_a2v = sum(truthy(r.get("accepted_for_a2v", "")) for r in episodes)
    camera_fallback = sum(
        "camera_fallback" in str(r.get("quality_flags", "")) for r in episodes
    )
    raw_video_fail = sum(
        not r.get("raw_video_path") or not exists(r.get("raw_video_path"))
        for r in episodes
    )
    visual_rate = visual_pass / total if total else 0.0
    stats_path = out / "manifests" / "action_normalization_config.json"
    block_training = visual_rate < 0.9

    report["checks"].update(
        {
            "episode_manifest_rows": total,
            "visual_sanity_pass_count": visual_pass,
            "visual_sanity_pass_rate": visual_rate,
            "accepted_for_sft_count": accepted_sft,
            "accepted_for_a2v_count": accepted_a2v,
            "camera_fallback_count": camera_fallback,
            "raw_video_discovery_failure_count": raw_video_fail,
            "action_global_stats_exists": stats_path.exists(),
            "BLOCK_TRAINING": block_training,
        }
    )

    for rel in [
        "sft_worldarena_style/metadata.jsonl",
        "a2v_worldarena_joint14/metadata.jsonl",
        "a2v_worldarena_ee16/metadata.jsonl",
        "a2v_worldarena_joint14_ee16/metadata.jsonl",
        "dpo_prompt_action/dpo_pairs.jsonl",
    ]:
        report["checks"][rel] = metadata_check(out / rel, out)

    for rel, expected in [
        ("inference_manifests/worldarena_val_base_a2v.jsonl", 500),
        ("inference_manifests/worldarena_val_1_retrieval.jsonl", 500),
        ("inference_manifests/worldarena_val_2_retrieval.jsonl", 500),
        ("inference_manifests/worldarena_test_base_a2v.jsonl", 1000),
        ("inference_manifests/worldarena_test_1_retrieval.jsonl", 1000),
        ("inference_manifests/worldarena_test_2_retrieval.jsonl", 1000),
    ]:
        rows = read_jsonl(out / rel)
        report["checks"][rel] = {
            "rows": len(rows),
            "expected": expected,
            "covered": len(rows) == expected,
        }

    sheet_dir = out / "contact_sheets" / "validator"
    by_config = {}
    for row in episodes:
        config = row.get("config_name", "unknown") or "unknown"
        first_frame = Path(str(row.get("first_frame_320x240_path", "")))
        if first_frame.exists():
            by_config.setdefault(config, []).append(
                (row.get("episode_id", ""), first_frame)
            )
    for config, items in sorted(by_config.items())[:50]:
        make_sheet(items[:20], sheet_dir / f"config_{config}.jpg", config)
    rng = random.Random(args.preview_seed)
    preview = []
    valid_frames = [
        (r.get("episode_id", ""), Path(str(r.get("first_frame_320x240_path", ""))))
        for r in episodes
    ]
    valid_frames = [(eid, p) for eid, p in valid_frames if p.exists()]
    for eid, path in rng.sample(valid_frames, min(30, len(valid_frames))):
        preview.append((eid, path))
    make_sheet(
        preview,
        sheet_dir / "random_30_video_preview.jpg",
        "random 30 first-frame preview",
        cols=6,
    )

    write_json(out / "v0_dataset_report.json", report)
    md = [
        "# WorldArena Data Factory v0 Report",
        "",
        f"Episode manifest rows: {total}",
        f"Accepted for SFT: {accepted_sft}",
        f"Accepted for A2V: {accepted_a2v}",
        f"Visual sanity pass rate: {visual_rate:.3f}",
        f"BLOCK_TRAINING={str(block_training).lower()}",
        "",
    ]
    for key, value in report["checks"].items():
        md.append(f"- `{key}`: `{value}`")
    (out / "v0_dataset_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out / "v0_dataset_report.md")


if __name__ == "__main__":
    main()
