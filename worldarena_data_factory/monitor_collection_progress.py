#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, csv, os, sys, time
from collections import Counter, defaultdict
sys.path.append(str(Path(__file__).resolve().parent))
from utils import read_csv


def success_count(job):
    p=Path(job['output_dir'])/'data'
    return len(list(p.glob('episode*.hdf5'))) if p.exists() else 0

def target_count(job): return int(job.get('target_success') or 0)

def capped(job): return min(success_count(job), target_count(job))

def bar(done,total,width=42):
    ratio=0 if total<=0 else min(1.0,done/total)
    fill=int(ratio*width)
    return '['+'#'*fill+'.'*(width-fill)+f'] {ratio*100:5.1f}%'

def fmt_time(sec):
    sec=max(0,int(sec)); h=sec//3600; m=(sec%3600)//60; s=sec%60
    return f'{h:02d}:{m:02d}:{s:02d}'

def latest_log(out: Path):
    logs=list((out/'logs').glob('collect_*.log'))
    if not logs: return None
    return max(logs, key=lambda p: p.stat().st_mtime)

def render(out: Path, jobs):
    total=sum(target_count(j) for j in jobs)
    done=sum(capped(j) for j in jobs)
    complete_jobs=sum(1 for j in jobs if capped(j)>=target_count(j))
    partial_jobs=sum(1 for j in jobs if 0<capped(j)<target_count(j))
    empty_jobs=sum(1 for j in jobs if capped(j)==0)
    fam_done=defaultdict(int); fam_total=defaultdict(int)
    cfg_done=defaultdict(int); cfg_total=defaultdict(int)
    for j in jobs:
        fam=j.get('task_family','unknown'); cfg=(j.get('base_task_config') or j.get('task_config','')).split('__job_')[0]
        fam_done[fam]+=capped(j); fam_total[fam]+=target_count(j)
        cfg_done[cfg]+=capped(j); cfg_total[cfg]+=target_count(j)
    lines=[]
    lines.append('WorldArena v0 Aloha-AgileX collection progress')
    lines.append(bar(done,total)+f' episodes {done}/{total}')
    lines.append(f'jobs complete={complete_jobs}/{len(jobs)} partial={partial_jobs} empty={empty_jobs}')
    lines.append('')
    lines.append('By config:')
    for cfg in sorted(cfg_total):
        lines.append(f'  {cfg:32s} {cfg_done[cfg]:4d}/{cfg_total[cfg]:4d} {bar(cfg_done[cfg],cfg_total[cfg],20)}')
    lines.append('')
    lines.append('By task family:')
    for fam in sorted(fam_total):
        lines.append(f'  {fam:28s} {fam_done[fam]:4d}/{fam_total[fam]:4d} {bar(fam_done[fam],fam_total[fam],20)}')
    log=latest_log(out)
    if log:
        age=time.time()-log.stat().st_mtime
        lines.append('')
        lines.append(f'latest log: {log} updated {fmt_time(age)} ago')
        try:
            tail=log.read_text(encoding='utf-8',errors='ignore').splitlines()[-5:]
            lines.append('latest log tail:')
            lines += ['  '+x[-160:] for x in tail]
        except Exception as e:
            lines.append(f'could not read latest log: {e}')
    return '\n'.join(lines)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='/root/autodl-tmp/worldarena_data_factory_v0')
    ap.add_argument('--jobs-csv')
    ap.add_argument('--watch',action='store_true')
    ap.add_argument('--interval',type=float,default=10)
    args=ap.parse_args()
    out=Path(args.out)
    jobs=read_csv(Path(args.jobs_csv) if args.jobs_csv else out/'manifests'/'robotwin_collection_jobs.csv')
    if not jobs: raise SystemExit('no jobs found')
    while True:
        text=render(out,jobs)
        if args.watch:
            print('\033[2J\033[H'+text, flush=True)
            time.sleep(args.interval)
        else:
            print(text)
            return
if __name__=='__main__': main()
