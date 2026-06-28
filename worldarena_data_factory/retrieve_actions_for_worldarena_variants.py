#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    POLICY_TO_MODE,
    ensure_dirs,
    read_csv,
    read_table,
    write_json,
    write_jsonl,
)


def toks(value: str) -> set[str]:
    return {x for x in (value or "").split(";") if x}


def score(query: dict, candidate: dict) -> float:
    sc = 0.0
    sc += 4 * (query.get("task_family") == candidate.get("task_family"))
    sc += 3 * bool(
        toks(query.get("main_verbs", "")) & toks(candidate.get("main_verbs", ""))
    )
    sc += 3 * bool(
        toks(query.get("main_objects", "")) & toks(candidate.get("main_objects", ""))
    )
    sc += 2 * bool(
        toks(query.get("receptacles_or_targets", ""))
        & toks(candidate.get("receptacles_or_targets", ""))
    )
    sc += 1 * bool(
        toks(query.get("spatial_relations", ""))
        & toks(candidate.get("spatial_relations", ""))
    )
    try:
        sc -= (
            abs(float(query.get("T", 0) or 0) - float(candidate.get("T", 0) or 0)) / 500
        )
    except Exception:
        pass
    try:
        sc -= max(0.0, 1 - float(candidate.get("quality_score", 1) or 1))
    except Exception:
        pass
    return sc


def best(query: dict, library: list[dict], k: int = 5) -> tuple[list[dict], int]:
    scored = [(score(query, candidate), candidate) for candidate in library]
    positive = [(s, c) for s, c in scored if s > 0]
    ranked = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
    return [{**candidate, "_score": sc} for sc, candidate in ranked], len(positive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worldarena-root", default="/root/autodl-tmp/worldarena_testset"
    )
    parser.add_argument(
        "--analysis", default="/root/autodl-tmp/worldarena_testset/analysis_v2"
    )
    parser.add_argument("--out", default="/root/autodl-tmp/worldarena_data_factory_v0")
    args = parser.parse_args()

    out = Path(args.out)
    analysis = Path(args.analysis)
    ensure_dirs(out)
    library = read_table(out / "manifests" / "action_library.parquet")
    if not library:
        raise SystemExit(
            "action_library has 0 rows; run build_action_library.py and check accepted episodes"
        )

    semantics = read_csv(analysis / "prompt_semantics.csv")
    semantics_index = {
        (r["split"], r["episode_id"], r["prompt_set"]): r for r in semantics
    }
    policies = read_csv(analysis / "prompt_action_policy.csv")
    policy_index = {
        (r["split"], r["episode_id"], r["variant_prompt_set"]): r for r in policies
    }
    episodes = read_csv(analysis / "episode_level.csv")
    report_rows = []

    for split, label, _expected in [
        ("val_dataset", "val", 500),
        ("test_dataset", "test", 1000),
    ]:
        base_rows = []
        v1_rows = []
        v2_rows = []
        for row in [x for x in episodes if x["dataset"] == split]:
            episode_id = row["episode_id"]
            base_sem = semantics_index.get((split, episode_id, "base"), {})
            base_rows.append(
                {
                    "worldarena_episode_id": int(episode_id),
                    "prompt_set": "base",
                    "inference_mode": "action_driven_original_hdf5",
                    "first_frame": row["first_frame_path"],
                    "action_source": "worldarena_original",
                    "action_joint14_path": str(
                        out
                        / "worldarena_actions"
                        / split
                        / f"episode{episode_id}"
                        / "action_joint14.npy"
                    ),
                    "prompt": base_sem.get("raw_prompt", ""),
                }
            )
            for variant, arr in [("variant_1", v1_rows), ("variant_2", v2_rows)]:
                policy_row = policy_index.get((split, episode_id, variant), {})
                sem = semantics_index.get((split, episode_id, variant), {})
                policy = policy_row.get("estimated_action_reuse_policy", "AMBIGUOUS")
                candidates, candidate_count = best({**sem, "T": ""}, library, 5)
                top1 = candidates[0]["_score"] if candidates else 0.0
                report_rows.append(
                    {
                        "split": split,
                        "episode_id": episode_id,
                        "variant": variant,
                        "candidate_count": candidate_count,
                        "top1_score": top1,
                    }
                )
                arr.append(
                    {
                        "worldarena_episode_id": int(episode_id),
                        "prompt_set": variant,
                        "policy": policy,
                        "inference_mode": POLICY_TO_MODE.get(
                            policy, "second_stage_review_or_text_fallback"
                        ),
                        "first_frame": row["first_frame_path"],
                        "original_action_joint14_path": str(
                            out
                            / "worldarena_actions"
                            / split
                            / f"episode{episode_id}"
                            / "action_joint14.npy"
                        ),
                        "retrieved_candidates": [
                            {
                                "library_id": c.get("library_id"),
                                "score": c.get("_score", 0.0),
                                "action_joint14_path": c.get("action_joint14_path"),
                                "source": c.get("source"),
                            }
                            for c in candidates
                        ],
                        "prompt": sem.get("raw_prompt", ""),
                    }
                )
        write_jsonl(
            out / "inference_manifests" / f"worldarena_{label}_base_a2v.jsonl",
            base_rows,
        )
        write_jsonl(
            out / "inference_manifests" / f"worldarena_{label}_1_retrieval.jsonl",
            v1_rows,
        )
        write_jsonl(
            out / "inference_manifests" / f"worldarena_{label}_2_retrieval.jsonl",
            v2_rows,
        )

    counts = [r["candidate_count"] for r in report_rows]
    top_scores = [r["top1_score"] for r in report_rows]
    buckets = {
        "<=0": sum(s <= 0 for s in top_scores),
        "0-3": sum(0 < s <= 3 for s in top_scores),
        "3-6": sum(3 < s <= 6 for s in top_scores),
        ">6": sum(s > 6 for s in top_scores),
    }
    report = {
        "library_rows": len(library),
        "queries": len(report_rows),
        "zero_candidate_count": sum(c == 0 for c in counts),
        "avg_candidate_count": sum(counts) / len(counts) if counts else 0.0,
        "top1_score_distribution": buckets,
    }
    write_json(out / "inference_manifests" / "retrieval_report.json", report)
    print("wrote inference manifests")
    print(report)


if __name__ == "__main__":
    main()
