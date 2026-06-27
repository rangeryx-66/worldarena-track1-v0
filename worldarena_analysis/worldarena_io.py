"""Dataset discovery and episode indexing for WorldArena val/test splits."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils import file_exists, safe_relpath


DATASET_NAMES = ("val_dataset", "test_dataset")
EXAMPLE_VIDEO_DIR_NAMES = (
    "example_val",
    "example_val_1",
    "example_val_2",
    "example_test",
    "example_test_1",
    "example_test_2",
)
MODALITY_TO_SUFFIX = {
    "data": ".hdf5",
    "first_frame": ".png",
    "instructions": ".json",
    "instructions_1": ".json",
    "instructions_2": ".json",
}
EPISODE_RE = re.compile(r"^episode(\d+)$")


@dataclass
class DatasetDiscovery:
    root: Path
    dataset_dirs: dict[str, Path] = field(default_factory=dict)
    example_video_dirs: dict[str, Path] = field(default_factory=dict)
    extra_example_video_dirs: dict[str, Path] = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    dataset: str
    episode_id: int
    task_name: str
    task_level: str
    hdf5_path: Path | None = None
    first_frame_path: Path | None = None
    instruction_path: Path | None = None
    instruction_1_path: Path | None = None
    instruction_2_path: Path | None = None
    modality_task_levels: dict[str, str] = field(default_factory=dict)

    def to_csv_row(self, root: Path) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "episode_id": self.episode_id,
            "task_name": self.task_name,
            "task_level": self.task_level,
            "modality_task_levels": ";".join(
                f"{modality}={level}" for modality, level in sorted(self.modality_task_levels.items())
            ),
            "hdf5_path": safe_relpath(self.hdf5_path, root),
            "first_frame_path": safe_relpath(self.first_frame_path, root),
            "instruction_path": safe_relpath(self.instruction_path, root),
            "instruction_1_path": safe_relpath(self.instruction_1_path, root),
            "instruction_2_path": safe_relpath(self.instruction_2_path, root),
            "has_hdf5": file_exists(self.hdf5_path),
            "has_first_frame": file_exists(self.first_frame_path),
            "has_instruction": file_exists(self.instruction_path),
            "has_instruction_1": file_exists(self.instruction_1_path),
            "has_instruction_2": file_exists(self.instruction_2_path),
        }


def natural_episode_id(path: Path) -> int | None:
    match = EPISODE_RE.match(path.stem)
    if not match:
        return None
    return int(match.group(1))


def discover_worldarena(root: Path, logger: logging.Logger) -> DatasetDiscovery:
    root = root.resolve()
    discovery = DatasetDiscovery(root=root)
    logger.info("Discovering WorldArena data under %s", root)

    for name in DATASET_NAMES:
        path = root / name
        if path.is_dir():
            discovery.dataset_dirs[name] = path
            logger.info("Found dataset split: %s", path)
        else:
            logger.warning("Missing dataset split: %s", path)

    for name in EXAMPLE_VIDEO_DIR_NAMES:
        direct = root / name
        if direct.is_dir():
            discovery.example_video_dirs[name] = direct
            logger.info("Found example video dir: %s", direct)

    for path in sorted(root.rglob("*")):
        if not path.is_dir() or path.parent == root:
            continue
        if path.name not in EXAMPLE_VIDEO_DIR_NAMES:
            continue
        if path.name in discovery.example_video_dirs and discovery.example_video_dirs[path.name] == path:
            continue
        key = str(path.relative_to(root))
        discovery.extra_example_video_dirs[key] = path
        logger.info("Found extra nested example-like video dir: %s", path)

    return discovery


def infer_task_from_episode_path(modality_root: Path, episode_path: Path) -> tuple[str, str]:
    rel = episode_path.relative_to(modality_root)
    if len(rel.parts) == 1:
        return ".", "none"
    task_name = rel.parts[-2]
    task_level = f"{modality_root.name}/" + "/".join(rel.parts[:-1])
    return task_name, task_level


def index_dataset_episodes(discovery: DatasetDiscovery, logger: logging.Logger) -> list[EpisodeRecord]:
    records: dict[tuple[str, int], EpisodeRecord] = {}
    task_levels: dict[str, set[str]] = {}

    path_fields = {
        "data": "hdf5_path",
        "first_frame": "first_frame_path",
        "instructions": "instruction_path",
        "instructions_1": "instruction_1_path",
        "instructions_2": "instruction_2_path",
    }

    for dataset_name, dataset_root in discovery.dataset_dirs.items():
        task_levels.setdefault(dataset_name, set())
        for modality, suffix in MODALITY_TO_SUFFIX.items():
            modality_root = dataset_root / modality
            if not modality_root.is_dir():
                logger.warning("Missing modality directory: %s", modality_root)
                continue

            for path in sorted(modality_root.rglob(f"*{suffix}")):
                episode_id = natural_episode_id(path)
                if episode_id is None:
                    logger.debug("Skipping non-episode file: %s", path)
                    continue
                task_name, task_level = infer_task_from_episode_path(modality_root, path)
                task_levels[dataset_name].add(task_level)
                key = (dataset_name, episode_id)
                record = records.get(key)
                if record is None:
                    record = EpisodeRecord(
                        dataset=dataset_name,
                        episode_id=episode_id,
                        task_name=task_name,
                        task_level=task_level,
                    )
                    records[key] = record
                elif record.task_name != task_name:
                    logger.warning(
                        "Episode %s/%s has inconsistent task names: %s vs %s",
                        dataset_name,
                        episode_id,
                        record.task_name,
                        task_name,
                    )
                record.modality_task_levels[modality] = task_level
                setattr(record, path_fields[modality], path)

    for dataset_name, levels in sorted(task_levels.items()):
        logger.info("Task path levels for %s: %s", dataset_name, sorted(levels))

    return sorted(records.values(), key=lambda item: (item.dataset, item.episode_id))


def summarize_discovery(discovery: DatasetDiscovery, records: list[EpisodeRecord], root: Path) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset_name in DATASET_NAMES:
        split_records = [row for row in records if row.dataset == dataset_name]
        task_counts: dict[str, int] = {}
        for record in split_records:
            task_counts[record.task_name] = task_counts.get(record.task_name, 0) + 1
        missing_counts = {
            "hdf5": sum(not file_exists(row.hdf5_path) for row in split_records),
            "first_frame": sum(not file_exists(row.first_frame_path) for row in split_records),
            "instruction": sum(not file_exists(row.instruction_path) for row in split_records),
            "instruction_1": sum(not file_exists(row.instruction_1_path) for row in split_records),
            "instruction_2": sum(not file_exists(row.instruction_2_path) for row in split_records),
        }
        observed_task_levels: dict[str, list[str]] = {}
        for row in split_records:
            for modality, level in row.modality_task_levels.items():
                observed_task_levels.setdefault(modality, [])
                if level not in observed_task_levels[modality]:
                    observed_task_levels[modality].append(level)
        episode_ids = [row.episode_id for row in split_records]
        by_dataset[dataset_name] = {
            "dataset_dir": safe_relpath(discovery.dataset_dirs.get(dataset_name), root),
            "episode_count": len(split_records),
            "episode_min": min(episode_ids) if episode_ids else None,
            "episode_max": max(episode_ids) if episode_ids else None,
            "task_counts": task_counts,
            "task_levels": sorted({level for row in split_records for level in row.modality_task_levels.values()}),
            "task_levels_by_modality": {
                modality: sorted(levels) for modality, levels in sorted(observed_task_levels.items())
            },
            "missing_counts": missing_counts,
        }

    return {
        "root": str(root.resolve()),
        "datasets": by_dataset,
        "example_video_dirs": {
            name: safe_relpath(path, root) for name, path in sorted(discovery.example_video_dirs.items())
        },
        "extra_example_video_dirs": {
            name: safe_relpath(path, root) for name, path in sorted(discovery.extra_example_video_dirs.items())
        },
        "path_schema": {
            "dataset_episode_files": "<root>/<val_dataset|test_dataset>/<modality>/<task_name>/episodeN.<ext>",
            "modalities": MODALITY_TO_SUFFIX,
            "detected_task_level": "<modality>/<task_name>",
            "detected_task_names": sorted({row.task_name for row in records}),
        },
    }
