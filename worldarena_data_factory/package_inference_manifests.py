#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, tarfile

sys.path.append(str(Path(__file__).resolve().parent))
from utils import ensure_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/worldarena_data_factory_v0")
    args = ap.parse_args()
    out = Path(args.out)
    ensure_dirs(out)
    pkg = out / "inference_manifests_worldarena_v0.tar.gz"
    with tarfile.open(pkg, "w:gz") as t:
        for p in (out / "inference_manifests").glob("*.jsonl"):
            t.add(p, arcname=f"inference_manifests/{p.name}")
    print(pkg)


if __name__ == "__main__":
    main()
