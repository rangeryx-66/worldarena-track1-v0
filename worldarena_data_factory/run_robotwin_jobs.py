#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from create_robotwin_configs import validate_config_values  # noqa: E402
from utils import (  # noqa: E402
    available_robotwin_tasks,
    detect_robotwin_root,
    ensure_dirs,
    read_csv,
    read_json,
    write_csv,
)

SUMMARY_FIELDS = [
    "job_id",
    "rc",
    "seconds",
    "expected_episodes",
    "hdf5_count",
    "video_count",
    "readable_video_count",
    "config_path",
    "config_numeric_sanity_passed",
    "attempts_used",
    "log",
]


def hdf5_count(job: dict) -> int:
    data_dir = Path(job["output_dir"]) / "data"
    return len(list(data_dir.glob("episode*.hdf5"))) if data_dir.exists() else 0


def video_files(job: dict) -> list[Path]:
    root = Path(job["output_dir"])
    return sorted(root.glob("**/*.mp4")) if root.exists() else []


def readable_video_count(job: dict) -> int:
    try:
        import cv2
    except Exception:
        return 0
    count = 0
    for path in video_files(job):
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok and frame is not None:
            count += 1
    return count


def target_count(job: dict) -> int:
    return int(job.get("target_success") or 0)


def capped_success(job: dict) -> int:
    return min(hdf5_count(job), target_count(job))


def aggregate(jobs: list[dict]) -> tuple[int, int, int, int]:
    total = sum(target_count(j) for j in jobs)
    done = sum(capped_success(j) for j in jobs)
    completed = sum(1 for j in jobs if capped_success(j) >= target_count(j))
    return done, total, completed, len(jobs)


def fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def bar(done: int, total: int, width: int = 30) -> str:
    ratio = 0 if total <= 0 else min(1.0, done / total)
    fill = int(ratio * width)
    return "[" + "#" * fill + "." * (width - fill) + f"] {ratio * 100:5.1f}%"


def cmd_for(root: Path, job: dict) -> list[str]:
    return [
        str(root / "collect_data.sh"),
        job["robotwin_task_name"],
        job["task_config"],
        str(job["gpu_id"]),
    ]


def ensure_job_config(root: Path, out: Path, job: dict) -> tuple[Path, bool]:
    import yaml

    task_config = job["task_config"]
    config_dst = root / "task_config" / f"{task_config}.yml"
    base = job.get("base_task_config") or task_config.rsplit("__job_", 1)[0]
    candidates = [
        out / "configs_to_apply" / f"{base}.yml",
        root / "task_config" / f"{base}.yml",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if not src:
        checked = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"base task config not found for {task_config}: checked {checked}"
        )
    with src.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    validate_config_values(data)
    data["episode_num"] = target_count(job)
    data["save_path"] = str(out / "robotwin_raw")
    data.setdefault("worldarena_v0_constraints", {})
    data["worldarena_v0_constraints"].update(
        {
            "job_id": job["job_id"],
            "base_task_config": base,
            "expected_action_dim": 14,
            "action_schema_required": "joint14",
            "is_dual_arm_required": True,
        }
    )
    validate_config_values(data)
    config_dst.parent.mkdir(parents=True, exist_ok=True)
    with config_dst.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return config_dst, True


def run_one(root: Path, out: Path, job: dict) -> dict:
    log = out / "logs" / f"collect_{job['job_id']}.log"
    log.parent.mkdir(exist_ok=True)
    start = time.time()
    attempts_used = 0
    config_path = ""
    config_ok = False
    rc = 0
    max_attempts = max(1, int(job.get("max_attempts") or target_count(job) or 1))
    try:
        cfg_path, config_ok = ensure_job_config(root, out, job)
        config_path = str(cfg_path)
        while hdf5_count(job) < target_count(job) and attempts_used < max_attempts:
            attempts_used += 1
            with log.open("a", encoding="utf-8") as f:
                f.write("\n===== WORLD_ARENA_JOB_ATTEMPT =====\n")
                f.write(f"ATTEMPT: {attempts_used}/{max_attempts}\n")
                f.write("JOB: " + str(job) + "\n")
                f.write("CONFIG: " + config_path + "\n")
                f.write("CMD: " + " ".join(cmd_for(root, job)) + "\n")
                f.flush()
                rc = subprocess.run(
                    cmd_for(root, job), cwd=root, stdout=f, stderr=subprocess.STDOUT
                ).returncode
            if rc != 0:
                break
    except Exception as exc:
        with log.open("a", encoding="utf-8") as f:
            f.write("RUNNER_ERROR: " + repr(exc) + "\n")
        rc = 997

    got = hdf5_count(job)
    target = target_count(job)
    if got < target and rc == 0:
        rc = 996
        with log.open("a", encoding="utf-8") as f:
            f.write(
                f"RUNNER_INCOMPLETE: got {got}/{target} hdf5 files; "
                "marking job failed despite shell rc=0\n"
            )
    return {
        "job_id": job["job_id"],
        "rc": rc,
        "seconds": round(time.time() - start, 3),
        "expected_episodes": target,
        "hdf5_count": got,
        "video_count": len(video_files(job)),
        "readable_video_count": readable_video_count(job),
        "config_path": config_path,
        "config_numeric_sanity_passed": str(config_ok).lower(),
        "attempts_used": attempts_used,
        "log": str(log),
    }


def probe_passed(out: Path) -> bool:
    for path in [out / "probe_report.json", out / "manifests" / "probe_report.json"]:
        data = read_json(path, {}) or {}
        if data.get("probe_pass") is True:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/root/autodl-tmp/worldarena_data_factory_v0")
    parser.add_argument("--robotwin-root")
    parser.add_argument("--jobs-csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-gpus", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow formal collection without a passing probe report.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    ensure_dirs(out)
    root = detect_robotwin_root(args.robotwin_root)
    if not root:
        raise SystemExit("RoboTwin root not found; pass --robotwin-root")

    jobs_path = (
        Path(args.jobs_csv)
        if args.jobs_csv
        else out / "manifests" / "robotwin_collection_jobs.csv"
    )
    jobs = read_csv(jobs_path)
    tasks = set(available_robotwin_tasks(root))
    missing = []
    todo = []
    for job in jobs:
        if job["robotwin_task_name"] not in tasks:
            missing.append(job)
            continue
        if args.resume and hdf5_count(job) >= target_count(job):
            continue
        todo.append(job)
    write_csv(out / "manifests" / "missing_tasks.csv", missing)

    for job in todo[:20]:
        print(" ".join(cmd_for(root, job)))
    done, total, completed, total_jobs = aggregate(jobs)
    print(f"{len(todo)} jobs ready; missing={len(missing)}")
    print(
        f"progress {bar(done, total)} episodes {done}/{total} jobs_done {completed}/{total_jobs}"
    )
    print(
        "note: job-specific task_config files are generated automatically on --execute"
    )

    if args.dry_run or not args.execute:
        return
    if not args.force and not probe_passed(out):
        raise SystemExit(
            "Probe report missing or probe_pass=false. Run probe_collection.py first, "
            "or pass --force to override."
        )

    start = time.time()
    finished = 0
    failed = 0
    summaries = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_parallel_gpus
    ) as executor:
        futures = {executor.submit(run_one, root, out, job): job for job in todo}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            summaries.append(result)
            finished += 1
            if int(result["rc"]) != 0:
                failed += 1
            done, total, completed, total_jobs = aggregate(jobs)
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 1 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"{bar(done, total)} episodes {done}/{total} jobs_done {completed}/{total_jobs} "
                f"run_done {finished}/{len(todo)} failed {failed} elapsed {fmt_time(elapsed)} "
                f"eta {fmt_time(eta)} last {result['job_id']} rc={result['rc']} "
                f"hdf5={result['hdf5_count']}/{result['expected_episodes']} "
                f"videos={result['readable_video_count']}/{result['video_count']} log={result['log']}",
                flush=True,
            )
    write_csv(
        out / "manifests" / "collection_job_summary.csv", summaries, SUMMARY_FIELDS
    )


if __name__ == "__main__":
    main()
