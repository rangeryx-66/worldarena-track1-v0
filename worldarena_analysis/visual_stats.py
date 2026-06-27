"""Visual first-frame statistics and contact-sheet generation."""
from __future__ import annotations

import csv, json, math, random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VISUAL_FIELDS = ["split","episode_id","task_name","first_frame_path","width","height","mode","mean_rgb","std_rgb","brightness_mean","contrast","sharpness_laplacian","saturation_mean","edge_density","average_hash","duplicate_or_near_duplicate_group","image_entropy","visual_cluster_id"]


def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open('r', encoding='utf-8', newline='') as f: return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict[str,Any]], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def rel_or_abs(root: Path, s: str) -> Path:
    p=Path(s); return p if p.is_absolute() else root/p

def entropy(gray: np.ndarray) -> float:
    hist,_=np.histogram(gray.ravel(), bins=256, range=(0,255), density=True)
    hist=hist[hist>0]
    return float(-(hist*np.log2(hist)).sum())

def avg_hash(img: Image.Image) -> str:
    g=img.convert('L').resize((8,8), Image.Resampling.BILINEAR)
    a=np.asarray(g, dtype=float); bits=a>a.mean()
    return ''.join('1' if b else '0' for b in bits.ravel())

def hamming(a: str, b: str) -> int:
    return sum(x!=y for x,y in zip(a,b))

def laplacian_var(gray: np.ndarray) -> float:
    try:
        import cv2
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        gy,gx=np.gradient(gray.astype(float)); return float((gx*gx+gy*gy).var())

def edge_density(gray: np.ndarray) -> float:
    try:
        import cv2
        edges=cv2.Canny(gray.astype(np.uint8), 80, 160)
        return float((edges>0).mean())
    except Exception:
        gy,gx=np.gradient(gray.astype(float)); mag=np.sqrt(gx*gx+gy*gy)
        return float((mag>mag.mean()+mag.std()).mean())

def saturation_mean(rgb: np.ndarray) -> float:
    arr=rgb.astype(float)/255.0
    mx=arr.max(axis=2); mn=arr.min(axis=2)
    sat=np.where(mx==0,0,(mx-mn)/mx)
    return float(sat.mean())

def feature_row(root: Path, row: dict[str,str]) -> tuple[dict[str,Any], list[float]]:
    p=rel_or_abs(root,row['first_frame_path'])
    img=Image.open(p).convert('RGB')
    rgb=np.asarray(img, dtype=np.float32)
    gray=np.asarray(img.convert('L'), dtype=np.float32)
    mean=rgb.reshape(-1,3).mean(axis=0); std=rgb.reshape(-1,3).std(axis=0)
    bright=float(gray.mean()); contrast=float(gray.std())
    sharp=laplacian_var(gray); edge=edge_density(gray); sat=saturation_mean(rgb)
    ent=entropy(gray); ah=avg_hash(img)
    out={
        'split':row['dataset'],'episode_id':int(row['episode_id']),'task_name':row.get('task_name',''),
        'first_frame_path':row['first_frame_path'],'width':img.width,'height':img.height,'mode':'RGB',
        'mean_rgb':';'.join(f'{x:.4f}' for x in mean),'std_rgb':';'.join(f'{x:.4f}' for x in std),
        'brightness_mean':round(bright,6),'contrast':round(contrast,6),'sharpness_laplacian':round(sharp,6),
        'saturation_mean':round(sat,6),'edge_density':round(edge,6),'average_hash':ah,
        'duplicate_or_near_duplicate_group':'','image_entropy':round(ent,6),'visual_cluster_id':-1,
    }
    feat=[bright,contrast,sharp,sat,edge,ent,*mean.tolist(),*std.tolist()]
    return out, feat

def assign_duplicate_groups(rows: list[dict[str,Any]]) -> None:
    groups=[]
    for r in rows:
        placed=False
        for gid,rep in groups:
            if hamming(r['average_hash'], rep) <= 4:
                r['duplicate_or_near_duplicate_group']=f'hash_group_{gid}'; placed=True; break
        if not placed:
            gid=len(groups); groups.append((gid,r['average_hash'])); r['duplicate_or_near_duplicate_group']=f'hash_group_{gid}'

def kmeans_labels(features: np.ndarray, k: int=12) -> np.ndarray:
    if len(features)==0: return np.asarray([], dtype=int)
    k=min(k,len(features))
    try:
        from sklearn.cluster import KMeans
        x=(features-features.mean(axis=0))/(features.std(axis=0)+1e-6)
        return KMeans(n_clusters=k, random_state=7, n_init=10).fit_predict(x)
    except Exception:
        rng=np.random.default_rng(7); x=(features-features.mean(axis=0))/(features.std(axis=0)+1e-6)
        centers=x[rng.choice(len(x), size=k, replace=False)]
        labels=np.zeros(len(x), dtype=int)
        for _ in range(30):
            labels=((x[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
            for i in range(k):
                if np.any(labels==i): centers[i]=x[labels==i].mean(axis=0)
        return labels

def make_sheet(root: Path, out: Path, rows: list[dict[str,Any]], title: str, max_n: int=24):
    rows=rows[:max_n]
    if not rows: return
    thumb_w,thumb_h=160,120; label_h=18; cols=4; rows_n=math.ceil(len(rows)/cols)
    sheet=Image.new('RGB',(cols*thumb_w, rows_n*(thumb_h+label_h)+28),'white')
    d=ImageDraw.Draw(sheet); d.text((6,6), title, fill=(0,0,0))
    for i,r in enumerate(rows):
        p=rel_or_abs(root,r['first_frame_path']); img=Image.open(p).convert('RGB'); img.thumbnail((thumb_w,thumb_h))
        x=(i%cols)*thumb_w; y=28+(i//cols)*(thumb_h+label_h)
        sheet.paste(img,(x,y)); d.text((x+4,y+thumb_h),f"{r['split'].replace('_dataset','')}:{r['episode_id']}", fill=(0,0,0))
    out.parent.mkdir(parents=True, exist_ok=True); sheet.save(out, quality=90)

def make_visual_plots(out_dir: Path, rows: list[dict[str,Any]]) -> list[str]:
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except Exception: return []
    plots=out_dir/'plots'; plots.mkdir(exist_ok=True); paths=[]
    def hist(key, fn, title):
        vals=[float(r[key]) for r in rows]
        fig,ax=plt.subplots(figsize=(8,5)); ax.hist(vals,bins=50,color='#16a34a',alpha=.85); ax.set_title(title); ax.set_xlabel(key); ax.set_ylabel('count'); ax.grid(axis='y',alpha=.25); fig.tight_layout(); p=plots/fn; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    for key,fn,title in [('brightness_mean','visual_brightness_distribution.png','Brightness Distribution'),('edge_density','visual_edge_density_distribution.png','Edge Density Distribution'),('sharpness_laplacian','visual_sharpness_distribution.png','Sharpness Distribution'),('image_entropy','visual_entropy_distribution.png','Entropy Distribution')]: hist(key,fn,title)
    c=Counter(str(r['visual_cluster_id']) for r in rows); fig,ax=plt.subplots(figsize=(8,5)); labels=list(c); vals=[c[x] for x in labels]; ax.bar(labels,vals,color='#16a34a'); ax.set_title('Visual Cluster Distribution'); ax.set_xlabel('cluster'); ax.set_ylabel('episode count'); fig.tight_layout(); p=plots/'visual_cluster_distribution.png'; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    return paths

def run_visual_statistics(root: Path, out_dir: Path, episode_csv: Path, action_csv: Path|None, policy_csv: Path|None, logger) -> dict[str,Any]:
    ep=read_csv(episode_csv); rows=[]; feats=[]; errors=[]
    for r in ep:
        try:
            row,feat=feature_row(root,r); rows.append(row); feats.append(feat)
        except Exception as e: errors.append(f"{r.get('first_frame_path')}: {e}")
    if feats:
        labels=kmeans_labels(np.asarray(feats,dtype=float),12)
        for r,l in zip(rows,labels): r['visual_cluster_id']=int(l)
    assign_duplicate_groups(rows)
    write_csv(out_dir/'visual_stats_episode.csv', rows, VISUAL_FIELDS)
    # cluster samples
    sample_rows=[]
    for cid in sorted({r['visual_cluster_id'] for r in rows}):
        members=[r for r in rows if r['visual_cluster_id']==cid]
        members=sorted(members, key=lambda x:(x['split'], int(x['episode_id'])))[:12]
        for r in members: sample_rows.append({'visual_cluster_id':cid,'split':r['split'],'episode_id':r['episode_id'],'first_frame_path':r['first_frame_path'],'brightness_mean':r['brightness_mean'],'edge_density':r['edge_density']})
    write_csv(out_dir/'visual_cluster_samples.csv', sample_rows, ['visual_cluster_id','split','episode_id','first_frame_path','brightness_mean','edge_density'])
    # sheets
    sheets=out_dir/'contact_sheets'; sheets.mkdir(exist_ok=True)
    rng=random.Random(7)
    for split in ['val_dataset','test_dataset']:
        part=[r for r in rows if r['split']==split]; rng.shuffle(part); make_sheet(root,sheets/f'random_{split}.jpg',part,f'Random {split}')
    for cid in sorted({r['visual_cluster_id'] for r in rows}): make_sheet(root,sheets/f'cluster_{cid}.jpg',[r for r in rows if r['visual_cluster_id']==cid],f'Visual cluster {cid}')
    for key in ['brightness_mean','edge_density']:
        sr=sorted(rows,key=lambda r:float(r[key])); make_sheet(root,sheets/f'low_{key}.jpg',sr,f'Low {key}'); make_sheet(root,sheets/f'high_{key}.jpg',list(reversed(sr)),f'High {key}')
    if action_csv and Path(action_csv).exists():
        ar=read_csv(Path(action_csv)); top=sorted(ar,key=lambda r:float(r.get('action_complexity_score') or 0), reverse=True)[:24]
        idx={(r['split'],r['episode_id']):r for r in rows}; make_sheet(root,sheets/'high_action_complexity.jpg',[idx[(r['split'],r['episode_id'])] for r in top if (r['split'],r['episode_id']) in idx], 'High action complexity')
    if policy_csv and Path(policy_csv).exists():
        pr=read_csv(Path(policy_csv)); idx={(r['split'],str(r['episode_id'])):r for r in rows}
        for pol in sorted({r['estimated_action_reuse_policy'] for r in pr}):
            sel=[]
            for p in [x for x in pr if x['estimated_action_reuse_policy']==pol][:24]:
                k=(p['split'],p['episode_id'])
                if k in idx: sel.append(idx[k])
            make_sheet(root,sheets/f'policy_{pol}.jpg',sel,f'Policy {pol}')
    plots=make_visual_plots(out_dir,rows)
    by_split=defaultdict(list)
    for r in rows: by_split[r['split']].append(r)
    domain={}
    for split,part in by_split.items():
        domain[split]={k:float(np.mean([float(r[k]) for r in part])) for k in ['brightness_mean','contrast','sharpness_laplacian','saturation_mean','edge_density','image_entropy']}
    if 'val_dataset' in domain and 'test_dataset' in domain:
        domain['test_minus_val']={k:domain['test_dataset'][k]-domain['val_dataset'][k] for k in domain['val_dataset']}
    (out_dir/'visual_domain_gap.json').write_text(json.dumps({'visual_domain_stats':domain,'errors':errors[:50]},indent=2),encoding='utf-8')
    return {'available':True,'episode_count':len(rows),'cluster_count':len(set(r['visual_cluster_id'] for r in rows)) if rows else 0,'visual_domain_gap':domain,'plots':plots,'contact_sheets_dir':str(sheets),'errors':errors[:50]}
