#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, sys, subprocess, math, random, json
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent))
from utils import (
    ensure_dirs,
    detect_robotwin_root,
    PROBE_TASKS,
    DUAL_ARM_CANDIDATE_EMBODIMENTS,
    normalize_embodiment,
    write_csv,
    write_json,
    read_csv,
)

FIELDS = [
    "probe_id",
    "task_name",
    "embodiment",
    "target_success",
    "gpu_id",
    "task_config",
    "command",
    "status",
    "success_count",
    "action_shape_distribution",
    "can_export_joint14",
    "first_frame_path",
    "brightness_gap",
    "edge_density_gap",
    "image_entropy_gap",
    "feature_distance",
    "output_dir",
]


def image_stats(path: Path):
    if not path.exists():
        return None
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    gray = arr.mean(2)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    ent = float(-(hist * np.log2(hist)).sum())
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    return {
        "brightness": float(gray.mean()),
        "edge_density": float((mag > mag.mean() + mag.std()).mean()),
        "entropy": ent,
    }


def worldarena_reference(root: Path):
    vals = []
    for p in list(
        (root / "val_dataset" / "first_frame" / "fixed_scene_task").glob("episode*.png")
    )[:80]:
        st = image_stats(p)
        if st:
            vals.append(st)
    if not vals:
        return {"brightness": 0, "edge_density": 0, "entropy": 0}
    return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}


def collect_outputs(out_dir: Path):
    h5s = list(out_dir.rglob("episode*.hdf5"))
    frames = list(out_dir.rglob("*.png")) + list(out_dir.rglob("*.jpg"))
    shapes = Counter()
    ok = 0
    try:
        import h5py
    except Exception:
        h5py = None
    for h in h5s:
        try:
            with h5py.File(h, "r") as f:
                if "/joint_action/vector" in f:
                    sh = tuple(f["/joint_action/vector"].shape)
                    shapes[str(sh)] += 1
                    if (
                        len(sh) == 2
                        and sh[1] == 14
                        and "/joint_action/left_arm" in f
                        and "/joint_action/right_arm" in f
                        and "/joint_action/left_gripper" in f
                        and "/joint_action/right_gripper" in f
                    ):
                        ok += 1
        except Exception:
            pass
    return len(h5s), dict(shapes), ok, (frames[0] if frames else None)


def make_sheet(rows, out: Path):
    imgs = []
    for r in rows:
        p = Path(r.get("first_frame_path") or "")
        if str(p) not in ("", ".") and p.is_file():
            imgs.append((r, p))
    if not imgs:
        return
    cols = 5
    tw, th = 160, 120
    lh = 34
    sheet = Image.new(
        "RGB", (cols * tw, math.ceil(len(imgs) / cols) * (th + lh) + 24), "white"
    )
    d = ImageDraw.Draw(sheet)
    d.text((5, 5), out.stem, fill=(0, 0, 0))
    for i, (r, p) in enumerate(imgs):
        im = Image.open(p).convert("RGB")
        im.thumbnail((tw, th))
        x = (i % cols) * tw
        y = 24 + (i // cols) * (th + lh)
        sheet.paste(im, (x, y))
        d.text((x + 3, y + th), f"{r['task_name']}\n{r['status']}", fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)


def recommend(rows):
    scores = []
    for emb in sorted(set(r["embodiment"] for r in rows)):
        part = [r for r in rows if r["embodiment"] == emb]
        success = sum(int(r.get("success_count") or 0) for r in part)
        can = sum(str(r.get("can_export_joint14")).lower() == "true" for r in part)
        fd = []
        for r in part:
            try:
                fd.append(float(r.get("feature_distance") or 999))
            except Exception:
                pass
        scores.append(
            (success * 10 + can * 5 - (sum(fd) / len(fd) if fd else 999), emb)
        )
    scores = sorted(scores, reverse=True)
    if not scores or all(s <= 0 for s, _ in scores):
        return {
            "main_embodiment": "aloha-agilex",
            "secondary_embodiment": "piper",
            "ranking": [{"embodiment": e, "score": s} for s, e in scores],
            "fallback_reason": "no successful joint14 probe data; using v0 dual-arm default",
        }
    return {
        "main_embodiment": scores[0][1],
        "secondary_embodiment": scores[1][1] if len(scores) > 1 else "piper",
        "ranking": [{"embodiment": e, "score": s} for s, e in scores],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/worldarena_data_factory_v0")
    ap.add_argument("--worldarena-root", default="/root/autodl-tmp/worldarena_testset")
    ap.add_argument("--robotwin-root")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--task-config", default="wa_probe_dual_arm")
    args = ap.parse_args()
    out = Path(args.out)
    ensure_dirs(out)
    root = detect_robotwin_root(args.robotwin_root)
    wa = Path(args.worldarena_root)
    ref = worldarena_reference(wa)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()] or ["0"]
    rows = []
    commands = []
    pid = 0
    for emb in DUAL_ARM_CANDIDATE_EMBODIMENTS:
        emb = normalize_embodiment(emb)
        for task in PROBE_TASKS:
            od = out / "embodiment_probe" / emb / task
            task_cfg = f"{args.task_config}__{emb.replace('-', '_').replace('+', '_')}"
            cmd = [
                (
                    str(root / "collect_data.sh")
                    if root
                    else "ROBOTWIN_ROOT/collect_data.sh"
                ),
                task,
                task_cfg,
                gpus[pid % len(gpus)],
            ]
            row = {
                "probe_id": f"probe_{pid:03d}",
                "task_name": task,
                "embodiment": emb,
                "target_success": 5,
                "gpu_id": gpus[pid % len(gpus)],
                "task_config": task_cfg,
                "command": " ".join(cmd),
                "status": "dry_run",
                "success_count": 0,
                "action_shape_distribution": "{}",
                "can_export_joint14": "unknown_not_executed",
                "first_frame_path": "",
                "brightness_gap": "",
                "edge_density_gap": "",
                "image_entropy_gap": "",
                "feature_distance": "",
                "output_dir": str(od),
            }
            if args.execute and root:
                log = out / "logs" / f"embodiment_probe_{emb}_{task}.log"
                log.parent.mkdir(exist_ok=True)
                with log.open("a", encoding="utf-8") as f:
                    subprocess.run(cmd, cwd=root, stdout=f, stderr=subprocess.STDOUT)
                n, sh, ok, frame = collect_outputs(od)
                row.update(
                    status="executed",
                    success_count=n,
                    action_shape_distribution=json.dumps(sh),
                    can_export_joint14=ok > 0,
                    first_frame_path=str(frame or ""),
                )
                st = image_stats(frame) if frame else None
                if st:
                    bg = st["brightness"] - ref["brightness"]
                    eg = st["edge_density"] - ref["edge_density"]
                    hg = st["entropy"] - ref["entropy"]
                    fd = math.sqrt(bg * bg + 10000 * eg * eg + hg * hg)
                    row.update(
                        brightness_gap=round(bg, 4),
                        edge_density_gap=round(eg, 6),
                        image_entropy_gap=round(hg, 4),
                        feature_distance=round(fd, 4),
                    )
            rows.append(row)
            pid += 1
    write_csv(out / "embodiment_probe" / "embodiment_probe_manifest.csv", rows, FIELDS)
    for emb in sorted(set(r["embodiment"] for r in rows)):
        make_sheet(
            [r for r in rows if r["embodiment"] == emb],
            out / "contact_sheets" / f"embodiment_probe_{emb}.jpg",
        )
    rec = recommend(rows)
    write_json(
        out / "embodiment_probe" / "embodiment_probe_summary.json",
        {"worldarena_reference_visual_stats": ref, "recommendation": rec, "rows": rows},
    )
    md = [
        "# Embodiment Probe Report",
        "",
        "WorldArena target domain is dual-arm gripper manipulation. Probe candidates are evaluated for joint14 export and rough visual similarity.",
    ]
    if args.dry_run and not args.execute:
        md += [
            "",
            "Probe was generated in dry-run mode. No trajectories were collected, so joint14 exportability and visual similarity are unknown until rerun with `--execute`.",
        ]
    if rec.get("fallback_reason"):
        md += ["", f"Fallback reason: {rec['fallback_reason']}"]
    md += [
        "",
        f"Recommended main embodiment: `{rec['main_embodiment']}`",
        f"Recommended secondary embodiment: `{rec['secondary_embodiment']}`",
        "",
        "| embodiment | success count | action shapes | can export joint14 | mean feature distance | contact sheet |",
        "| --- | ---: | --- | --- | ---: | --- |",
    ]
    for emb in sorted(set(r["embodiment"] for r in rows)):
        part = [r for r in rows if r["embodiment"] == emb]
        succ = sum(int(r["success_count"]) for r in part)
        shapes = Counter(r["action_shape_distribution"] for r in part)
        can = any(str(r["can_export_joint14"]).lower() == "true" for r in part)
        fds = [float(r["feature_distance"]) for r in part if str(r["feature_distance"])]

        if succ == 0 and not fds:
            can_cell = "unknown_not_executed"
        else:
            can_cell = str(can)
        md.append(
            f"| `{emb}` | {succ} | `{dict(shapes)}` | {can_cell} | {round(sum(fds)/len(fds),4) if fds else ''} | `contact_sheets/embodiment_probe_{emb}.jpg` |"
        )
    (out / "embodiment_probe" / "embodiment_probe_report.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    for r in rows[:20]:
        print(r["command"])
    print(f"wrote {out/'embodiment_probe'/'embodiment_probe_manifest.csv'}")


if __name__ == "__main__":
    main()
