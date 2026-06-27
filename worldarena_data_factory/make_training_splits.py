#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, random
sys.path.append(str(Path(__file__).resolve().parent))
from utils import ensure_dirs, read_csv, write_csv

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='/root/autodl-tmp/worldarena_data_factory_v0'); ap.add_argument('--train-ratio',type=float,default=.9); args=ap.parse_args(); out=Path(args.out); ensure_dirs(out)
    rows=read_csv(out/'manifests'/'episode_manifest.csv'); random.Random(7).shuffle(rows); n=int(len(rows)*args.train_ratio)
    write_csv(out/'manifests'/'train_split.csv',rows[:n]); write_csv(out/'manifests'/'val_split.csv',rows[n:]); print(n,len(rows)-n)
if __name__=='__main__': main()
