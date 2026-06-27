"""Plotting utilities for WorldArena analysis reports."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - depends on local env
    plt = None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def split_semicolon(value: str) -> list[str]:
    return [part for part in value.split(";") if part]


def bar_plot(counter: Counter[str], path: Path, title: str, xlabel: str, ylabel: str, topk: int | None = None) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    path.parent.mkdir(parents=True, exist_ok=True)
    items = counter.most_common(topk) if topk else counter.most_common()
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    if not labels:
        labels = ["none"]
        values = [0]
    width = max(8, min(18, len(labels) * 0.55 + 4))
    fig, ax = plt.subplots(figsize=(width, 5.5))
    ax.bar(range(len(labels)), values, color="#3b82f6")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_semantic_plots(out_dir: Path) -> list[str]:
    if plt is None:
        return []
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    semantics = read_csv(out_dir / "prompt_semantics.csv")
    policies = read_csv(out_dir / "prompt_action_policy.csv")
    paths: list[str] = []

    for split in ("val_dataset", "test_dataset"):
        counter = Counter(row["task_family"] for row in semantics if row["split"] == split)
        path = plots_dir / f"task_family_distribution_{'val' if split == 'val_dataset' else 'test'}.png"
        bar_plot(counter, path, f"Task Family Distribution ({split})", "task family", "prompt count")
        paths.append(str(path))

    verb_counter: Counter[str] = Counter()
    object_counter: Counter[str] = Counter()
    for row in semantics:
        verb_counter.update(split_semicolon(row.get("main_verbs", "")))
        object_counter.update(split_semicolon(row.get("main_objects", "")))

    path = plots_dir / "verb_distribution.png"
    bar_plot(verb_counter, path, "Verb Distribution", "verb", "prompt count", topk=25)
    paths.append(str(path))

    path = plots_dir / "object_topk.png"
    bar_plot(object_counter, path, "Top Objects", "object", "prompt count", topk=30)
    paths.append(str(path))

    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in policies:
        by_split[row["split"]][row["estimated_action_reuse_policy"]] += 1
    for split in ("val_dataset", "test_dataset"):
        path = plots_dir / f"action_reuse_policy_distribution_{'val' if split == 'val_dataset' else 'test'}.png"
        bar_plot(by_split[split], path, f"Action Reuse Policy Distribution ({split})", "policy", "comparison count")
        paths.append(str(path))

    return paths
