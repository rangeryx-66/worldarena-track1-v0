#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

FIELDS = [
    'sample_id','sample_group','episode_id','video_path','overview_sheet','motion_peak_sheet','action_peak_sheet','first_frame_path',
    'task_family','robotwin_task_name','prompt_short','T','action_complexity_score','dominant_arm',
    'current_qc_status','current_qc_reason','current_qc_labels','risk_summary',
    'human_label','human_reason','human_confidence','notes','annotated_at','annotator'
]


def resolve_manifest(path: Path) -> Path:
    if path.exists():
        return path
    alt = path.parent / 'manifests' / path.name
    if alt.exists():
        return alt
    raise FileNotFoundError(path)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    return pd.DataFrame()


def safe_str(x: Any) -> str:
    if x is None:
        return ''
    if isinstance(x, float) and math.isnan(x):
        return ''
    return str(x)


def load_current_qc(dataset_root: Path) -> pd.DataFrame:
    sources = []
    specs = [
        ('abot', dataset_root/'abot_style_qc'/'abot_qc_scores.csv'),
        ('vlm', dataset_root/'v0_1_qc_vlm'/'vlm_qc_scores.csv'),
        ('rule_v2', dataset_root/'v0_1_qc'/'qc_scores_v2.csv'),
        ('rule_v1', dataset_root/'v0_1_qc'/'qc_scores.csv'),
    ]
    for name, path in specs:
        df = read_table(path)
        if df.empty or 'episode_id' not in df.columns:
            continue
        df = df.copy()
        df['_qc_source'] = name
        sources.append(df)
    if not sources:
        return pd.DataFrame(columns=['episode_id','current_qc_status','current_qc_reason','current_qc_labels','risk_summary'])
    all_ids = sorted(set().union(*[set(x['episode_id'].astype(str)) for x in sources]))
    rows = []
    by_source = {s['_qc_source'].iloc[0]: s.set_index(s['episode_id'].astype(str), drop=False) for s in sources}
    for eid in all_ids:
        status = reason = labels = ''
        risk = []
        for src in ['abot','vlm','rule_v2','rule_v1']:
            df = by_source.get(src)
            if df is None or eid not in df.index:
                continue
            r = df.loc[eid]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[-1]
            if not status:
                if src == 'abot':
                    status = safe_str(r.get('final_qc_status'))
                    reason = safe_str(r.get('final_reason'))
                    labels = safe_str(r.get('vlm_decision'))
                elif src == 'vlm':
                    status = safe_str(r.get('final_decision'))
                    reason = safe_str(r.get('final_reason'))
                    labels = safe_str(r.get('vlm_decision'))
                else:
                    status = safe_str(r.get('qc_status'))
                    reason = safe_str(r.get('qc_reason'))
                    labels = safe_str(r.get('heuristic_candidate_labels'))
            bits = []
            for c in ['final_qc_status','final_decision','qc_status','final_reason','qc_reason','vlm_decision','vlm_evidence','hard_fail_reason']:
                v = safe_str(r.get(c))
                if v:
                    bits.append(f'{c}={v[:140]}')
            for c in ['static_clip','over_motion_clip','action_video_consistency_score','motion_score','visual_motion_energy','robot_visible_ratio','contact_region_visible_ratio','object_motion_without_visible_contact_score']:
                if c in r and safe_str(r.get(c)):
                    bits.append(f'{c}={r.get(c)}')
            if bits:
                risk.append(f'{src}: ' + '; '.join(bits[:7]))
        rows.append({'episode_id':eid,'current_qc_status':status or 'unknown','current_qc_reason':reason,'current_qc_labels':labels,'risk_summary':' | '.join(risk)[:1600]})
    return pd.DataFrame(rows)


def sample_stratified(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.head(0)
    rng = np.random.default_rng(seed)
    work = df.copy()
    work['_T_bin'] = pd.qcut(pd.to_numeric(work['T'], errors='coerce').fillna(0).rank(method='first'), q=min(4, len(work)), labels=False, duplicates='drop')
    work['_C_bin'] = pd.qcut(pd.to_numeric(work['action_complexity_score'], errors='coerce').fillna(0).rank(method='first'), q=min(4, len(work)), labels=False, duplicates='drop')
    groups = list(work.groupby(['task_family','_T_bin','_C_bin'], dropna=False))
    selected = []
    random.Random(seed).shuffle(groups)
    while groups and len(selected) < min(n, len(work)):
        new_groups = []
        for _, g in groups:
            remaining = g.drop(index=selected, errors='ignore')
            if remaining.empty:
                continue
            selected.append(remaining.sample(1, random_state=int(rng.integers(0, 1_000_000))).index[0])
            if len(selected) >= min(n, len(work)):
                break
            new_groups.append((None, remaining.iloc[1:]))
        groups = [(i,g) for i,g in new_groups if not g.empty]
    if len(selected) < min(n, len(work)):
        rest = work.drop(index=selected).sample(min(n-len(selected), len(work)-len(selected)), random_state=seed).index.tolist()
        selected += rest
    return work.loc[selected].drop(columns=['_T_bin','_C_bin'], errors='ignore')


def hard_score(df: pd.DataFrame) -> pd.Series:
    status = df.get('current_qc_status', pd.Series(['unknown']*len(df))).fillna('').astype(str).str.upper()
    score = status.isin(['WARN','REJECT','DPO_LOSER','DPO_LOSER_CANDIDATE','NEED_HUMAN_REVIEW']).astype(float) * 5
    T = pd.to_numeric(df.get('T', 0), errors='coerce').fillna(0)
    C = pd.to_numeric(df.get('action_complexity_score', 0), errors='coerce').fillna(0)
    score += T.rank(pct=True).fillna(0) * 1.2
    score += C.rank(pct=True).fillna(0) * 1.5
    risk = df.get('risk_summary', pd.Series(['']*len(df))).fillna('').astype(str).str.lower()
    for token, w in [('static_clip',2.0),('over_motion',2.0),('low_action',1.5),('hard_fail',2.0),('contact_region_visible_ratio=0',1.0),('robot_visible_ratio=0',1.0)]:
        score += risk.str.contains(token, regex=False).astype(float) * w
    return score


def read_frame(cap: cv2.VideoCapture, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_sheet(frames: list[tuple[int,float,np.ndarray]], out_path: Path, title: str, cols: int = 4):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tw, th, lh = 240, 180, 24
    if not frames:
        img = Image.new('RGB', (640, 160), 'white')
        ImageDraw.Draw(img).text((20,70), f'{title}: no frames', fill=(0,0,0))
        img.save(out_path, quality=92)
        return
    rows = int(math.ceil(len(frames)/cols))
    sheet = Image.new('RGB', (cols*tw, rows*(th+lh)+28), 'white')
    d = ImageDraw.Draw(sheet); font = ImageFont.load_default(); d.text((8,8), title, fill=(0,0,0), font=font)
    for i,(idx,ts,arr) in enumerate(frames):
        im = Image.fromarray(arr).convert('RGB'); im.thumbnail((tw, th))
        x=(i%cols)*tw; y=28+(i//cols)*(th+lh)
        canvas=Image.new('RGB',(tw,th),'white'); canvas.paste(im,((tw-im.width)//2,(th-im.height)//2)); sheet.paste(canvas,(x,y))
        d.text((x+4,y+4),f'frame {idx} / {ts:.2f}s',fill=(0,0,0),font=font)
    sheet.save(out_path, quality=92)


def make_overview(video: Path, out_path: Path, n: int = 12):
    if out_path.exists(): return
    cap=cv2.VideoCapture(str(video)); frames=[]
    if cap.isOpened():
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 24); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for idx in np.linspace(0, max(0,count-1), min(n,max(1,count))).astype(int).tolist() if count else []:
            f=read_frame(cap, idx)
            if f is not None: frames.append((idx, idx/fps, f))
    cap.release(); draw_sheet(frames,out_path,'overview')


def make_motion_peak(video: Path, out_path: Path, n: int = 12):
    if out_path.exists(): return
    cap=cv2.VideoCapture(str(video)); frames=[]; peaks=[]
    if cap.isOpened():
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 24); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample=np.linspace(1,max(1,count-1),min(80,max(1,count-1))).astype(int).tolist() if count>1 else []
        prev=None
        for idx in sample:
            f=read_frame(cap, idx)
            if f is None: continue
            g=cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            if prev is not None: peaks.append((float(cv2.absdiff(g,prev).mean()),idx))
            prev=g
        chosen=[]
        for _,idx in sorted(peaks, reverse=True)[:n]:
            chosen.append(idx)
        if not chosen and count:
            chosen=np.linspace(0,count-1,min(n,count)).astype(int).tolist()
        for idx in sorted(set(chosen))[:n]:
            f=read_frame(cap,idx)
            if f is not None: frames.append((idx,idx/fps,f))
    cap.release(); draw_sheet(frames,out_path,'motion peaks')


def action_peak_indices(row: pd.Series, n: int = 12) -> list[int]:
    path = Path(safe_str(row.get('action_joint14_raw_path') or row.get('action_joint14_norm_path')))
    frame_count = int(pd.to_numeric(row.get('frame_count', 0), errors='coerce') or 0)
    T = int(pd.to_numeric(row.get('T', 0), errors='coerce') or 0)
    if not path.exists() or T <= 1 or frame_count <= 0:
        return []
    try:
        arr=np.load(path)
        d=np.linalg.norm(np.diff(arr[:,:14].astype(np.float32),axis=0),axis=1)
        if len(d)==0: return []
        peaks=np.argsort(-d)[:n]
        return sorted(set(int(round((p/max(T-1,1))*(frame_count-1))) for p in peaks))[:n]
    except Exception:
        return []


def make_action_peak(row: pd.Series, out_path: Path, n: int = 12):
    if out_path.exists(): return
    video=Path(safe_str(row.get('video_640x480_path') or row.get('video_path')))
    cap=cv2.VideoCapture(str(video)); frames=[]
    if cap.isOpened():
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 24)
        for idx in action_peak_indices(row, n):
            f=read_frame(cap,idx)
            if f is not None: frames.append((idx,idx/fps,f))
    cap.release(); draw_sheet(frames,out_path,'action peaks')


def link_video(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists(): return str(src)
    if dst.exists(): return str(dst)
    try:
        os.symlink(src, dst)
        return str(dst)
    except Exception:
        return str(src)


def write_errors(path: Path, rows: list[dict[str,Any]]):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists=path.exists()
    with path.open('a', encoding='utf-8', newline='') as f:
        fields=['episode_id','stage','error']
        w=csv.DictWriter(f, fieldnames=fields)
        if not exists: w.writeheader()
        w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--n-total', type=int, default=500)
    ap.add_argument('--random-n', type=int, default=200)
    ap.add_argument('--hard-n', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-samples', type=int)
    ap.add_argument('--force-regenerate', action='store_true')
    args=ap.parse_args()
    out=args.out; out.mkdir(parents=True, exist_ok=True)
    csv_path=out/'manual_qc_500.csv'
    if csv_path.exists() and not args.force_regenerate:
        print(csv_path)
        print(f'exists rows={len(pd.read_csv(csv_path))}; use --force-regenerate to resample')
        return
    dataset_root=out.parent
    manifest=read_table(resolve_manifest(args.manifest)).reset_index(drop=True)
    qc=load_current_qc(dataset_root)
    df=manifest.merge(qc, on='episode_id', how='left')
    df['current_qc_status']=df['current_qc_status'].fillna('unknown')
    df['current_qc_reason']=df['current_qc_reason'].fillna('')
    df['current_qc_labels']=df['current_qc_labels'].fillna('')
    df['risk_summary']=df['risk_summary'].fillna('')
    n_total=args.max_samples or args.n_total
    random_n=min(args.random_n, n_total)
    hard_n=max(0, min(args.hard_n, n_total-random_n))
    random_part=sample_stratified(df, random_n, args.seed).copy(); random_part['sample_group']='random_eval'
    rest=df.drop(index=random_part.index, errors='ignore').copy()
    rest['_hard_score']=hard_score(rest)
    hard_part=rest.sort_values('_hard_score', ascending=False).head(hard_n).drop(columns=['_hard_score'], errors='ignore').copy(); hard_part['sample_group']='hard_dev'
    selected=pd.concat([random_part, hard_part], ignore_index=True)
    if len(selected) < n_total:
        fill=df.drop(index=selected.index, errors='ignore').sample(min(n_total-len(selected), len(df)-len(selected)), random_state=args.seed).copy(); fill['sample_group']='random_eval'; selected=pd.concat([selected, fill], ignore_index=True)
    selected=selected.head(n_total).reset_index(drop=True)
    errors=[]; rows=[]
    for i,row in selected.iterrows():
        eid=safe_str(row.get('episode_id'))
        try:
            video=Path(safe_str(row.get('video_640x480_path')))
            overview=out/'contact_sheets'/'overview'/f'{eid}.jpg'
            motion=out/'contact_sheets'/'motion_peak'/f'{eid}.jpg'
            action=out/'contact_sheets'/'action_peak'/f'{eid}.jpg'
            make_overview(video, overview)
            make_motion_peak(video, motion)
            make_action_peak(row, action)
            vpath=link_video(video, out/'videos'/f'{eid}.mp4')
            rows.append({
                'sample_id':f'mqc_{i:06d}', 'sample_group':row.get('sample_group',''), 'episode_id':eid,
                'video_path':vpath, 'overview_sheet':str(overview), 'motion_peak_sheet':str(motion), 'action_peak_sheet':str(action),
                'first_frame_path':safe_str(row.get('first_frame_320x240_path')), 'task_family':safe_str(row.get('task_family')),
                'robotwin_task_name':safe_str(row.get('robotwin_task_name')), 'prompt_short':safe_str(row.get('prompt_short') or row.get('prompt_worldarena_style')),
                'T':safe_str(row.get('T')), 'action_complexity_score':safe_str(row.get('action_complexity_score')), 'dominant_arm':safe_str(row.get('dominant_arm')),
                'current_qc_status':safe_str(row.get('current_qc_status')), 'current_qc_reason':safe_str(row.get('current_qc_reason')),
                'current_qc_labels':safe_str(row.get('current_qc_labels')), 'risk_summary':safe_str(row.get('risk_summary')),
                'human_label':'', 'human_reason':'', 'human_confidence':'', 'notes':'', 'annotated_at':'', 'annotator':'',
            })
        except Exception as e:
            errors.append({'episode_id':eid,'stage':'prepare','error':f'{type(e).__name__}: {e}'})
    pd.DataFrame(rows, columns=FIELDS).to_csv(csv_path, index=False)
    write_errors(out/'errors.csv', errors)
    print(f'wrote {len(rows)} rows to {csv_path}; errors={len(errors)}')

if __name__=='__main__':
    main()
