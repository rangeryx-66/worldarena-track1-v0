"""Report and table writers for WorldArena analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils import write_csv, write_json
from worldarena_io import EpisodeRecord


EPISODE_LEVEL_COLUMNS = [
    "dataset",
    "episode_id",
    "task_name",
    "task_level",
    "modality_task_levels",
    "hdf5_path",
    "first_frame_path",
    "instruction_path",
    "instruction_1_path",
    "instruction_2_path",
    "has_hdf5",
    "has_first_frame",
    "has_instruction",
    "has_instruction_1",
    "has_instruction_2",
]


def write_episode_level_csv(out_dir: Path, root: Path, records: list[EpisodeRecord]) -> Path:
    path = out_dir / "episode_level.csv"
    rows = [record.to_csv_row(root) for record in records]
    write_csv(path, rows, EPISODE_LEVEL_COLUMNS)
    return path


def write_summary_json(out_dir: Path, summary: dict[str, Any]) -> Path:
    path = out_dir / "summary.json"
    write_json(path, summary)
    return path


def write_initial_report(out_dir: Path, summary: dict[str, Any]) -> Path:
    path = out_dir / "report.md"
    lines: list[str] = []
    lines.append("# WorldArena Analysis V2")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This report combines dataset path discovery, semantic prompt analysis, action statistics, visual/video domain statistics, and cross-modal training recommendations for WorldArena val/test splits.")
    lines.append("")
    lines.append("## Dataset Path Schema")
    lines.append("")
    lines.append(f"- Root: `{summary['root']}`")
    lines.append(
        "- Episode files: "
        f"`{summary['path_schema']['dataset_episode_files']}`"
    )
    lines.append(f"- Detected task level: `{summary['path_schema']['detected_task_level']}`")
    lines.append(
        "- Detected task names: "
        + ", ".join(f"`{name}`" for name in summary["path_schema"]["detected_task_names"])
    )
    lines.append("")
    lines.append("### Modalities")
    lines.append("")
    lines.append("| modality | suffix |")
    lines.append("| --- | --- |")
    for modality, suffix in summary["path_schema"]["modalities"].items():
        lines.append(f"| `{modality}` | `{suffix}` |")
    lines.append("")
    lines.append("### Dataset Splits")
    lines.append("")
    lines.append("| dataset | episodes | episode range | task levels | missing counts |")
    lines.append("| --- | ---: | --- | --- | --- |")
    for dataset_name, stats in summary["datasets"].items():
        ep_range = f"{stats['episode_min']}..{stats['episode_max']}"
        task_levels = "<br>".join(f"`{level}`" for level in stats["task_levels"])
        missing = ", ".join(f"{key}={value}" for key, value in stats["missing_counts"].items())
        lines.append(
            f"| `{dataset_name}` | {stats['episode_count']} | {ep_range} | {task_levels} | {missing} |"
        )
    lines.append("")
    lines.append("### Task Level By Modality")
    lines.append("")
    lines.append("| dataset | modality | detected level |")
    lines.append("| --- | --- | --- |")
    for dataset_name, stats in summary["datasets"].items():
        for modality, levels in stats["task_levels_by_modality"].items():
            level_text = "<br>".join(f"`{level}`" for level in levels)
            lines.append(f"| `{dataset_name}` | `{modality}` | {level_text} |")
    lines.append("")
    lines.append("### Example Video Directories")
    lines.append("")
    if summary["example_video_dirs"]:
        for name, relpath in summary["example_video_dirs"].items():
            lines.append(f"- `{name}`: `{relpath}`")
    else:
        lines.append("- None found.")
    if summary["extra_example_video_dirs"]:
        lines.append("")
        lines.append("Extra nested example-like directories were also detected:")
        for name, relpath in summary["extra_example_video_dirs"].items():
            lines.append(f"- `{name}`: `{relpath}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def append_semantic_report(out_dir: Path, semantic_summary: dict[str, Any], plot_paths: list[str]) -> Path:
    """Append semantic/action-policy sections to the existing report.md."""
    path = out_dir / "report.md"
    lines: list[str] = []
    lines.append("")
    lines.append("## Semantic Distribution")
    lines.append("")
    lines.append(f"- Parsed prompt rows: `{semantic_summary.get('semantic_rows', 0)}`")
    lines.append(f"- Policy comparison rows: `{semantic_summary.get('policy_rows', 0)}`")
    lines.append(f"- Average sequence similarity: `{semantic_summary.get('avg_sequence_similarity', 0)}`")
    lines.append("")
    lines.append("### Task Family By Split")
    lines.append("")
    lines.append("| split | task family | count |")
    lines.append("| --- | --- | ---: |")
    for split, family_counts in semantic_summary.get("task_family_by_split", {}).items():
        for family, count in family_counts.items():
            lines.append(f"| `{split}` | `{family}` | {count} |")
    lines.append("")
    lines.append("### Top Verbs")
    lines.append("")
    lines.append("| verb | count |")
    lines.append("| --- | ---: |")
    for verb, count in semantic_summary.get("top_verbs", {}).items():
        lines.append(f"| `{verb}` | {count} |")
    lines.append("")
    lines.append("### Top Objects")
    lines.append("")
    lines.append("| object | count |")
    lines.append("| --- | ---: |")
    for obj, count in semantic_summary.get("top_objects", {}).items():
        lines.append(f"| `{obj}` | {count} |")
    lines.append("")
    lines.append("## Instruction Variant Analysis")
    lines.append("")
    lines.append("### Prompt-Action Reuse Policy")
    lines.append("")
    lines.append("| split | policy | count |")
    lines.append("| --- | --- | ---: |")
    for split, policy_counts in semantic_summary.get("policy_by_split", {}).items():
        for policy, count in policy_counts.items():
            lines.append(f"| `{split}` | `{policy}` | {count} |")
    lines.append("")
    lines.append("Policy labels:")
    lines.append("- `SAME_ACTION_OK`: variant likely preserves the original action.")
    lines.append("- `ACTION_MAYBE_OK`: same broad family, but manual check recommended.")
    lines.append("- `TARGET_CHANGED`: target/receptacle likely changed, so original HDF5 action is risky.")
    lines.append("- `VERB_CHANGED`: main action verb likely changed, so original HDF5 action is risky.")
    lines.append("- `AMBIGUOUS`: rule-based signals conflict or are too weak.")
    lines.append("")
    lines.append("Generated CSV outputs:")
    lines.append("- `prompt_semantics.csv`")
    lines.append("- `prompt_variant_diff.csv`")
    lines.append("- `prompt_action_policy.csv`")
    lines.append("")
    lines.append("Generated plots:")
    for plot_path in plot_paths:
        try:
            rel = Path(plot_path).resolve().relative_to(out_dir.resolve())
            lines.append(f"- `{rel}`")
        except ValueError:
            lines.append(f"- `{plot_path}`")
    if semantic_summary.get("errors"):
        lines.append("")
        lines.append("Semantic parser warnings/errors, truncated:")
        for error in semantic_summary["errors"][:10]:
            lines.append(f"- `{error}`")
    lines.append("")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def append_action_report(out_dir: Path, action_summary: dict[str, Any]) -> Path:
    """Append action statistics section to report.md."""
    path = out_dir / "report.md"
    lines: list[str] = []
    lines.append("")
    lines.append("## Action Statistics")
    lines.append("")
    if not action_summary.get("available"):
        lines.append(f"- Action statistics unavailable: `{action_summary.get('error', 'unknown error')}`")
        lines.append("- Install/use an environment with `h5py` to read HDF5 action fields.")
    else:
        lines.append(f"- Episodes analyzed: `{action_summary.get('episode_count')}`")
        lines.append(f"- By split: `{action_summary.get('by_split')}`")
        lines.append(f"- Trajectory length T: `{action_summary.get('T')}`")
        lines.append(f"- Estimated thresholds: `{action_summary.get('thresholds')}`")
        lines.append(f"- Dominant arm distribution: `{action_summary.get('dominant_arm')}`")
        lines.append(f"- Outlier episodes: `{action_summary.get('outlier_count')}`")
        lines.append(f"- Mean left quaternion close ratio: `{action_summary.get('left_quat_close_mean')}`")
        lines.append(f"- Mean right quaternion close ratio: `{action_summary.get('right_quat_close_mean')}`")
        lines.append(f"- Mean action complexity score: `{action_summary.get('action_complexity_mean')}`")
        rec = action_summary.get("abot_a2v_recommendation", {})
        lines.append("")
        lines.append("### ABot-A2V Action Representation Recommendation")
        lines.append("")
        lines.append(f"- Recommended primary representation: `{rec.get('recommended_primary', 'unknown')}`")
        lines.append(f"- Rationale: {rec.get('recommendation', '')}")
    lines.append("")
    lines.append("Generated CSV outputs:")
    lines.append("- `action_stats_episode.csv`")
    lines.append("- `action_stats_dim.csv`")
    lines.append("- `action_transition_stats.csv`")
    lines.append("- `action_outlier_episodes.csv`")
    if action_summary.get("plots"):
        lines.append("")
        lines.append("Generated action plots:")
        for plot_path in action_summary["plots"]:
            try:
                rel = Path(plot_path).resolve().relative_to(out_dir.resolve())
                lines.append(f"- `{rel}`")
            except ValueError:
                lines.append(f"- `{plot_path}`")
    if action_summary.get("errors"):
        lines.append("")
        lines.append("Action read warnings/errors, truncated:")
        for error in action_summary["errors"][:10]:
            lines.append(f"- `{error}`")
    lines.append("")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def append_visual_video_cross_report(out_dir: Path, visual_summary: dict[str, Any], video_summary: dict[str, Any], cross_summary: dict[str, Any]) -> Path:
    path = out_dir / "report.md"
    lines=[]
    lines += ["", "## Visual Domain Statistics", ""]
    lines.append(f"- Visual episodes analyzed: `{visual_summary.get('episode_count')}`")
    lines.append(f"- Visual clusters: `{visual_summary.get('cluster_count')}`")
    lines.append(f"- Domain gap summary: `{visual_summary.get('visual_domain_gap')}`")
    lines.append(f"- Contact sheets: `{visual_summary.get('contact_sheets_dir')}`")
    lines += ["", "## Video Statistics", ""]
    lines.append(f"- Video rows: `{video_summary.get('video_rows')}`")
    lines.append(f"- Video sets: `{video_summary.get('video_sets')}`")
    lines.append(f"- Unreadable videos: `{video_summary.get('unreadable_total')}`")
    lines += ["", "## Cross-Modal Findings", ""]
    lines.append(f"- Cross-analysis rows: `{cross_summary.get('rows')}`")
    lines.append(f"- DPO candidate counts: `{cross_summary.get('dpo_candidate_counts')}`")
    lines += ["", "## Implications for ABot-PhysWorld", ""]
    lines.append("- Use `joint14` as the primary action representation and export `ee16` plus `joint14+ee16` for ablations.")
    lines.append("- Base prompts should use the original HDF5 action. Variant prompts should follow `prompt_action_policy.csv` and `inference_policy_summary.json`.")
    lines.append("- SFT sampling should stratify by task family, visual cluster, and action complexity using `recommended_sft_weight`.")
    lines.append("- DPO mining should emphasize prompt-action mismatch hard negatives from `TARGET_CHANGED` and `VERB_CHANGED`.")
    lines += ["", "## Files Generated", ""]
    files=['episode_level.csv','summary.json','prompt_semantics.csv','prompt_variant_diff.csv','prompt_action_policy.csv','action_stats_episode.csv','action_stats_dim.csv','action_transition_stats.csv','action_outlier_episodes.csv','visual_stats_episode.csv','visual_cluster_samples.csv','video_stats.csv','visual_domain_gap.json','cross_analysis.csv','training_sampling_plan.json','inference_policy_summary.json','abot_training_recommendations.md']
    for f in files: lines.append(f"- `{f}`")
    lines.append("- `plots/*.png`")
    lines.append("- `contact_sheets/*.jpg`")
    lines += ["", "## Open Questions", ""]
    lines.append("- Should `instruction_1/2` actions be regenerated with RoboTwin planning or retrieved from nearby episodes?")
    lines.append("- What temporal resampling target should ABot-A2V use for long-tail trajectories up to T=1074?")
    lines.append("- Should visually near-duplicate first frames be downweighted or used for controlled prompt/action comparisons?")
    lines.append("- Which representation wins in ablation: `joint14`, `ee16`, or `joint14+ee16`?")
    with path.open('a',encoding='utf-8') as h: h.write('\n'.join(lines)+"\n")
    return path
