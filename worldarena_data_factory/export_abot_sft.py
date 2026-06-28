#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, random, os

sys.path.append(str(Path(__file__).resolve().parent))
from utils import ensure_dirs, read_table, write_jsonl, is_v0_training_embodiment


def truthy(x):
    return str(x).lower() == "true"


def load_rows(
    out: Path, manifest_arg: str | None, qc_source: str, vlm_qc_dir: str | None
):
    if qc_source == "vlm":
        qc_dir = Path(vlm_qc_dir) if vlm_qc_dir else out / "v0_1_qc_vlm"
        path = qc_dir / "vlm_qc_scores.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"VLM QC scores not found: {path}. Run merge_qc_with_vlm.py first."
            )
        return read_table(path)
    manifest_path = (
        Path(manifest_arg)
        if manifest_arg
        else out / "manifests" / "episode_manifest.parquet"
    )
    return read_table(manifest_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/worldarena_data_factory_v0")
    ap.add_argument("--episode-manifest-csv")
    ap.add_argument("--real-replay-manifest")
    ap.add_argument("--include-real-replay", action="store_true")
    ap.add_argument("--main-embodiment")
    ap.add_argument("--secondary-embodiment", default="piper")
    ap.add_argument("--include-secondary-embodiment", action="store_true")
    ap.add_argument("--qc-source", choices=["none", "vlm"], default="none")
    ap.add_argument("--vlm-qc-dir")
    args = ap.parse_args()
    out = Path(args.out)
    ensure_dirs(out)
    rows = load_rows(out, args.episode_manifest_csv, args.qc_source, args.vlm_qc_dir)
    out_dir = out / (
        "sft_worldarena_style_v0_1_vlm"
        if args.qc_source == "vlm"
        else "sft_worldarena_style"
    )
    data = []
    rng = random.Random(7)
    for r in rows:
        if str(
            r.get("accepted_for_sft", r.get("success", ""))
        ).lower() != "true" or not r.get("video_640x480_path"):
            continue
        if not is_v0_training_embodiment(
            r.get("embodiment", ""),
            "aloha-agilex",
            args.secondary_embodiment,
            out,
            args.include_secondary_embodiment,
        ):
            continue
        if "dual_arm_joint14_valid" not in str(r.get("quality_flags", "")):
            continue
        if (
            args.qc_source == "vlm"
            and str(r.get("final_decision", "")).upper() != "PASS"
        ):
            continue
        x = rng.random()
        prompt = (
            r.get("prompt_short")
            if x < 0.6
            else (
                r.get("prompt_worldarena_style")
                if x < 0.85
                else r.get("prompt_long_caption")
            )
        )
        video = os.path.relpath(str(r["video_640x480_path"]), str(out))
        data.append({"video": video, "prompt": prompt})
    # Real replay remains excluded unless explicitly requested. v0.1 QC export intentionally keeps it out by default.
    write_jsonl(out_dir / "metadata.jsonl", data)
    print(len(data))


if __name__ == "__main__":
    main()
