#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

KEYWORDS = [
    "camera",
    "observation",
    "intrinsic",
    "extrinsic",
    "cam2world",
    "world2cam",
    "fov",
    "pose",
    "d435",
    "rgb",
    "depth",
]
TARGET_KEYS = [
    "observation/head_camera/intrinsic_cv",
    "observation/head_camera/extrinsic_cv",
    "observation/head_camera/cam2world_gl",
]


def rel_key(name: str) -> str:
    return name[1:] if name.startswith("/") else name


def is_camera_related(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in KEYWORDS)


def scalar_preview(ds: h5py.Dataset) -> Any:
    try:
        arr = ds[()]
        if isinstance(arr, bytes):
            return arr[:120].decode("utf-8", errors="replace")
        a = np.asarray(arr)
        if a.size == 0:
            return []
        flat = a.reshape(-1)
        vals = flat.tolist() if a.size <= 12 else flat[:12].tolist()

        def clean(v):
            if isinstance(v, bytes):
                return v[:120].decode("utf-8", errors="replace")
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, bytes):
                return v[:120].decode("utf-8", errors="replace")
            return v

        return [clean(v) for v in vals]
    except Exception as exc:
        return f"preview_error:{type(exc).__name__}"


def inspect_file(path: Path) -> dict[str, Any]:
    fields = []
    key_tree = []
    target_presence = {}
    with h5py.File(path, "r") as f:

        def visitor(name, obj):
            indent = "  " * name.count("/")
            if isinstance(obj, h5py.Dataset):
                key_tree.append(f"{indent}{name} shape={obj.shape} dtype={obj.dtype}")
                if is_camera_related(name):
                    fields.append(
                        {
                            "key": rel_key(name),
                            "shape": list(obj.shape),
                            "dtype": str(obj.dtype),
                            "ndim": len(obj.shape),
                            "preview": scalar_preview(obj),
                        }
                    )
            else:
                key_tree.append(f"{indent}{name}/")

        f.visititems(visitor)
        for key in TARGET_KEYS:
            target_presence[key] = key in f
    return {
        "path": str(path),
        "fields": fields,
        "target_presence": target_presence,
        "key_tree": key_tree,
    }


def classify_camera(field_key: str) -> str:
    parts = field_key.split("/")
    for i, p in enumerate(parts):
        if "camera" in p.lower() or "d435" in p.lower():
            return p
    return "unknown"


def inspect_root(root: Path, max_files: int) -> dict[str, Any]:
    files = sorted(root.rglob("episode*.hdf5"))[:max_files]
    results = []
    field_counter = Counter()
    shape_by_key = defaultdict(Counter)
    camera_counter = Counter()
    presence_counter = Counter()
    for path in files:
        try:
            info = inspect_file(path)
            results.append(info)
            for k, present in info["target_presence"].items():
                if present:
                    presence_counter[k] += 1
            for field in info["fields"]:
                key = field["key"]
                field_counter[key] += 1
                shape_by_key[key][tuple(field["shape"])] += 1
                camera_counter[classify_camera(key)] += 1
        except Exception as exc:
            results.append(
                {
                    "path": str(path),
                    "error": repr(exc),
                    "fields": [],
                    "target_presence": {},
                }
            )
    return {
        "root": str(root),
        "num_files_scanned": len(files),
        "files": [str(x) for x in files],
        "results": results,
        "field_counts": dict(field_counter),
        "shape_by_key": {
            k: {str(shape): c for shape, c in v.items()}
            for k, v in shape_by_key.items()
        },
        "camera_counts": dict(camera_counter),
        "target_presence_counts": dict(presence_counter),
    }


def summarize_bool(summary: dict[str, Any], key: str) -> str:
    n = summary["num_files_scanned"]
    c = summary.get("target_presence_counts", {}).get(key, 0)
    return f"{c}/{n}"


def write_outputs(out: Path, primary: dict[str, Any], compare: dict[str, Any] | None):
    out.mkdir(parents=True, exist_ok=True)
    roots = {"primary": primary}
    if compare is not None:
        roots["compare"] = compare
    (out / "camera_field_summary.json").write_text(
        json.dumps(roots, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    examples = {}
    for name, summary in roots.items():
        examples[name] = []
        for res in summary["results"]:
            if res.get("fields"):
                examples[name].append(
                    {"path": res["path"], "fields": res["fields"][:40]}
                )
            if len(examples[name]) >= 5:
                break
    (out / "camera_field_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    tree_lines = []
    for name, summary in roots.items():
        tree_lines.append(f"===== {name}: {summary['root']} =====")
        for res in summary["results"][:5]:
            tree_lines.append(f"--- {res.get('path')} ---")
            tree_lines.extend(res.get("key_tree", [])[:400])
    (out / "hdf5_key_tree_samples.txt").write_text(
        "\n".join(tree_lines) + "\n", encoding="utf-8"
    )

    lines = ["# HDF5 Camera Field Report", ""]
    for name, summary in roots.items():
        lines += [
            f"## {name}",
            "",
            f"Root: `{summary['root']}`",
            f"Files scanned: {summary['num_files_scanned']}",
            "",
            "### Required/Expected Fields",
            "",
        ]
        for key in TARGET_KEYS:
            lines.append(f"- `{key}`: {summarize_bool(summary, key)}")
        lines += ["", "### Camera-like Groups", ""]
        for cam, count in sorted(summary.get("camera_counts", {}).items()):
            lines.append(f"- `{cam}`: {count} camera-related dataset hits")
        lines += ["", "### Field Shapes", ""]
        for key, shapes in sorted(summary.get("shape_by_key", {}).items()):
            if not is_camera_related(key):
                continue
            shape_text = ", ".join(f"{s}: {c}" for s, c in shapes.items())
            lines.append(f"- `{key}`: {shape_text}")
        lines.append("")
    if compare is not None:
        lines += [
            "## Self-collected vs WorldArena Field Consistency",
            "",
            "See `camera_field_summary.json` for exact key sets and counts. Fields are considered consistent if required head_camera camera matrices appear in both roots with compatible shapes.",
            "",
        ]
        pkeys = set(primary.get("field_counts", {}))
        ckeys = set(compare.get("field_counts", {}))
        for key in TARGET_KEYS:
            lines.append(
                f"- `{key}`: primary={'yes' if key in pkeys else 'no'}, compare={'yes' if key in ckeys else 'no'}"
            )
    (out / "camera_field_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5-dir", required=True, type=Path)
    ap.add_argument("--compare-hdf5-dir", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-files", type=int, default=50)
    args = ap.parse_args()
    primary = inspect_root(args.hdf5_dir, args.max_files)
    compare = (
        inspect_root(args.compare_hdf5_dir, args.max_files)
        if args.compare_hdf5_dir
        else None
    )
    write_outputs(args.out, primary, compare)
    print(args.out / "camera_field_report.md")
    print(args.out / "camera_field_summary.json")


if __name__ == "__main__":
    main()
