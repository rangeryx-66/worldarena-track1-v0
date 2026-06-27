"""Shared utilities for WorldArena analysis."""

from __future__ import annotations

import csv
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


def setup_logging(out_dir: Path, verbose: bool = False) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("worldarena_analysis")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    logfile = logging.FileHandler(out_dir / "analysis.log", encoding="utf-8")
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)
    return logger


@contextmanager
def log_exceptions(logger: logging.Logger, context: str):
    try:
        yield
    except Exception:
        logger.exception("Failed while %s", context)
        raise


def safe_relpath(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def file_exists(path: Path | None) -> bool:
    return bool(path and path.exists())
