#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, json, csv

sys.path.append(str(Path(__file__).resolve().parent))
from utils import (
    ensure_dirs,
    read_table,
    write_jsonl,
    write_json,
    is_v0_training_embodiment,
)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["episode_id"])
        w.writeheader()
        w.writerows(rows)


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
    return read_table(
        Path(manifest_arg)
        if manifest_arg
        else out / "manifests" / "episode_manifest.parquet"
    )


def base_episode_ok(r, out, secondary_embodiment, include_secondary):
    if str(
        r.get("accepted_for_a2v", r.get("success", ""))
    ).lower() != "true" or not r.get("video_640x480_path"):
        return False
    if not is_v0_training_embodiment(
        r.get("embodiment", ""),
        "aloha-agilex",
        secondary_embodiment,
        out,
        include_secondary,
    ):
        return False
    return "dual_arm_joint14_valid" in str(r.get("quality_flags", ""))


def build_row(r, out, field):
    flags = r.get("quality_flags", "")
    intr = r.get("intrinsic_path") or str(out / "fallback_camera_intrinsic.json")
    ext = r.get("extrinsic_path") or str(out / "fallback_camera_extrinsic.json")
    return {
        "video": r["video_640x480_path"],
        "prompt": r.get("prompt_worldarena_style") or r.get("prompt_short"),
        "action_path": r[field],
        "intrinsic_path": intr,
        "extrinsic_path": ext,
        "original_size": [480, 640],
        "quality_flags": flags,
    }


def export(rows, out, name, field, secondary_embodiment=None, include_secondary=False):
    data = []
    for r in rows:
        if not base_episode_ok(
            r, out, secondary_embodiment, include_secondary
        ) or not r.get(field):
            continue
        data.append(build_row(r, out, field))
    write_jsonl(out / name / "metadata.jsonl", data)
    return len(data)


def export_vlm(rows, out, secondary_embodiment=None, include_secondary=False):
    data = []
    dpo_loser = []
    for r in rows:
        if not base_episode_ok(r, out, secondary_embodiment, include_secondary):
            continue
        final_decision = str(r.get("final_decision", "")).upper()
        a2v_score = int(float(r.get("score_a2v_positive_suitability") or 0))
        if r.get("action_ee16_raw_path") and (
            final_decision == "PASS"
            or (final_decision == "WARN_KEEP" and a2v_score >= 1)
        ):
            data.append(build_row(r, out, "action_ee16_raw_path"))
        if final_decision == "DPO_LOSER_CANDIDATE":
            dpo_loser.append(
                {
                    "episode_id": r.get("episode_id", ""),
                    "video": r.get("video_640x480_path", ""),
                    "prompt": r.get("prompt_worldarena_style") or r.get("prompt_short"),
                    "final_decision": final_decision,
                    "final_reason": r.get("final_reason", ""),
                    "vlm_decision": r.get("vlm_decision", ""),
                    "vlm_confidence": r.get("vlm_confidence", ""),
                    "vlm_evidence": r.get("vlm_evidence", ""),
                }
            )
    write_jsonl(out / "a2v_worldarena_ee16_v0_1_vlm" / "metadata.jsonl", data)
    write_csv(out / "dpo_loser_bank_v0_1_vlm.csv", dpo_loser)
    return len(data), len(dpo_loser)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/worldarena_data_factory_v0")
    ap.add_argument("--episode-manifest-csv")
    ap.add_argument("--main-embodiment")
    ap.add_argument("--secondary-embodiment", default="piper")
    ap.add_argument("--include-secondary-embodiment", action="store_true")
    ap.add_argument("--qc-source", choices=["none", "vlm"], default="none")
    ap.add_argument("--vlm-qc-dir")
    args = ap.parse_args()
    out = Path(args.out)
    ensure_dirs(out)
    write_json(out / "fallback_camera_intrinsic.json", {"fallback": True})
    write_json(out / "fallback_camera_extrinsic.json", {"fallback": True})
    rows = load_rows(out, args.episode_manifest_csv, args.qc_source, args.vlm_qc_dir)
    if args.qc_source == "vlm":
        n_a2v, n_dpo = export_vlm(
            rows, out, args.secondary_embodiment, args.include_secondary_embodiment
        )
        print(f"a2v_ee16_v0_1_vlm={n_a2v} dpo_loser_bank={n_dpo}")
    else:
        n1 = export(
            rows,
            out,
            "a2v_worldarena_joint14",
            "action_joint14_norm_path",
            args.secondary_embodiment,
            args.include_secondary_embodiment,
        )
        n2 = export(
            rows,
            out,
            "a2v_worldarena_ee16",
            "action_ee16_raw_path",
            args.secondary_embodiment,
            args.include_secondary_embodiment,
        )
        n3 = export(
            rows,
            out,
            "a2v_worldarena_joint14_ee16",
            "action_joint14_ee16_raw_path",
            args.secondary_embodiment,
            args.include_secondary_embodiment,
        )
        print(f"a2v_joint14={n1} a2v_ee16={n2} a2v_joint14_ee16={n3}")


if __name__ == "__main__":
    main()
