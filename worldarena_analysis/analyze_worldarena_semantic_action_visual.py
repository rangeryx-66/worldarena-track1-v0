#!/usr/bin/env python3
"""WorldArena semantic/action/visual analysis entry point.

The entry point first performs dataset discovery and episode indexing, then
runs the offline semantic prompt parser and prompt-action reuse policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from action_stats import compute_action_statistics
from cross_analysis import run_cross_analysis
from plotting import make_semantic_plots
from video_stats import run_video_statistics
from visual_stats import run_visual_statistics
from report_writer import (
    append_action_report,
    append_semantic_report,
    append_visual_video_cross_report,
    write_episode_level_csv,
    write_initial_report,
    write_summary_json,
)
from semantic_parser import run_semantic_prompt_analysis
from utils import log_exceptions, setup_logging
from worldarena_io import discover_worldarena, index_dataset_episodes, summarize_discovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze WorldArena val/test semantic, action, and visual data."
    )
    parser.add_argument("--root", type=Path, required=True, help="WorldArena dataset root.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for all generated files.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose console logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir, verbose=args.verbose)

    logger.info("Starting WorldArena analysis v2")
    logger.info("Root: %s", root)
    logger.info("Output: %s", out_dir)

    with log_exceptions(logger, "discovering datasets"):
        discovery = discover_worldarena(root, logger)

    with log_exceptions(logger, "indexing episodes"):
        records = index_dataset_episodes(discovery, logger)

    with log_exceptions(logger, "writing discovery outputs"):
        summary = summarize_discovery(discovery, records, root)
        csv_path = write_episode_level_csv(out_dir, root, records)
        summary_path = write_summary_json(out_dir, summary)
        report_path = write_initial_report(out_dir, summary)

    with log_exceptions(logger, "running semantic prompt analysis"):
        semantic_result = run_semantic_prompt_analysis(root, out_dir, csv_path, logger)
        plot_paths = make_semantic_plots(out_dir)
        report_path = append_semantic_report(
            out_dir,
            semantic_result["semantic_summary"],
            plot_paths,
        )
        summary["semantic_analysis"] = semantic_result["semantic_summary"]
        summary["semantic_outputs"] = {
            "prompt_semantics_csv": "prompt_semantics.csv",
            "prompt_variant_diff_csv": "prompt_variant_diff.csv",
            "prompt_action_policy_csv": "prompt_action_policy.csv",
            "plots": [str(path) for path in plot_paths],
        }
        summary_path = write_summary_json(out_dir, summary)

    with log_exceptions(logger, "running action statistics"):
        action_summary = compute_action_statistics(root, out_dir, csv_path, logger)
        report_path = append_action_report(out_dir, action_summary)
        summary["action_statistics"] = action_summary
        summary["action_outputs"] = {
            "action_stats_episode_csv": "action_stats_episode.csv",
            "action_stats_dim_csv": "action_stats_dim.csv",
            "action_transition_stats_csv": "action_transition_stats.csv",
            "action_outlier_episodes_csv": "action_outlier_episodes.csv",
            "plots": action_summary.get("plots", []),
        }
        summary_path = write_summary_json(out_dir, summary)

    with log_exceptions(logger, "running visual, video, and cross-modal analysis"):
        visual_summary = run_visual_statistics(root, out_dir, csv_path, out_dir / "action_stats_episode.csv", out_dir / "prompt_action_policy.csv", logger)
        video_summary = run_video_statistics(root, out_dir, logger)
        cross_summary = run_cross_analysis(out_dir, logger)
        report_path = append_visual_video_cross_report(out_dir, visual_summary, video_summary, cross_summary)
        summary["visual_statistics"] = visual_summary
        summary["video_statistics"] = video_summary
        summary["cross_analysis"] = cross_summary
        summary_path = write_summary_json(out_dir, summary)

    logger.info("Indexed %d episodes", len(records))
    logger.info("Wrote %s", csv_path)
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
