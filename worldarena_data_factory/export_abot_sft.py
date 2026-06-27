#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, random
sys.path.append(str(Path(__file__).resolve().parent))
from utils import ensure_dirs, read_csv, read_table, write_jsonl, is_v0_training_embodiment

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='/root/autodl-tmp/worldarena_data_factory_v0'); ap.add_argument('--episode-manifest-csv'); ap.add_argument('--real-replay-manifest'); ap.add_argument('--include-real-replay',action='store_true'); ap.add_argument('--main-embodiment'); ap.add_argument('--secondary-embodiment',default='piper'); ap.add_argument('--include-secondary-embodiment',action='store_true'); args=ap.parse_args(); out=Path(args.out); ensure_dirs(out)
    manifest_path=Path(args.episode_manifest_csv) if args.episode_manifest_csv else out/'manifests'/'episode_manifest.parquet'; rows=read_table(manifest_path); data=[]; rng=random.Random(7)
    for r in rows:
        if str(r.get('success')).lower()!='true' or not r.get('video_640x480_path'): continue
        if not is_v0_training_embodiment(r.get('embodiment',''), 'aloha-agilex', args.secondary_embodiment, out, args.include_secondary_embodiment): continue
        if 'dual_arm_joint14_valid' not in r.get('quality_flags',''): continue
        x=rng.random(); prompt=r.get('prompt_short') if x<.6 else (r.get('prompt_worldarena_style') if x<.85 else r.get('prompt_long_caption'))
        data.append({'video':r['video_640x480_path'],'prompt':prompt})
    # Real replay is intentionally excluded unless --include-real-replay is passed.
    write_jsonl(out/'sft_worldarena_style'/'metadata.jsonl',data); print(len(data))
if __name__=='__main__': main()
