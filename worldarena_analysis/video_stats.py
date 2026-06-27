"""Example video statistics for WorldArena submission directories."""
from __future__ import annotations
import csv, re, json
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

VIDEO_DIRS=['example_val','example_val_1','example_val_2','example_test','example_test_1','example_test_2']
FIELDS=['video_set','split','prompt_set','episode_id','video_path','file_readable','width','height','fps','frame_count','duration','first_frame_similarity_to_dataset_first_frame','sampled_frame_motion_score','bad_video_reason']
EP_RE=re.compile(r'episode(\d+)')

def write_csv(path: Path, rows: list[dict[str,Any]], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def video_meta(name: str):
    split='val_dataset' if 'val' in name else 'test_dataset'
    prompt='base'
    if name.endswith('_1'): prompt='variant_1'
    if name.endswith('_2'): prompt='variant_2'
    return split,prompt

def first_frame_path(root: Path, split: str, episode: int) -> Path:
    return root/split/'first_frame'/'fixed_scene_task'/f'episode{episode}.png'

def similarity(frame: np.ndarray, image_path: Path) -> float:
    if not image_path.exists(): return float('nan')
    img=np.asarray(Image.open(image_path).convert('RGB').resize((frame.shape[1],frame.shape[0])),dtype=np.float32)
    fr=frame.astype(np.float32)
    # frame from cv2 is BGR
    fr=fr[:,:,::-1]
    mad=float(np.mean(np.abs(fr-img)))/255.0
    return round(1.0-mad,6)

def inspect_video(path: Path, dataset_first: Path) -> dict[str,Any]:
    try:
        import cv2
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {'file_readable':False,'bad_video_reason':'cv2 could not open video'}
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 0); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok,first=cap.read()
        if not ok or first is None:
            cap.release(); return {'file_readable':False,'width':width,'height':height,'fps':fps,'frame_count':count,'bad_video_reason':'could not read first frame'}
        sim=similarity(first,dataset_first)
        sample_count=min(16,max(count,1)); indices=np.linspace(0,max(count-1,0),sample_count,dtype=int) if count else np.arange(16)
        prev=None; motions=[]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok,frame=cap.read()
            if not ok or frame is None: continue
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None and prev.shape==gray.shape: motions.append(float(np.mean(np.abs(gray-prev)))/255.0)
            prev=gray
        cap.release()
        duration=float(count/fps) if fps>0 and count>0 else float('nan')
        return {'file_readable':True,'width':width,'height':height,'fps':round(fps,6),'frame_count':count,'duration':round(duration,6),'first_frame_similarity_to_dataset_first_frame':sim,'sampled_frame_motion_score':round(float(np.mean(motions)) if motions else 0.0,6),'bad_video_reason':''}
    except Exception as e:
        return {'file_readable':False,'bad_video_reason':str(e)}

def run_video_statistics(root: Path, out_dir: Path, logger) -> dict[str,Any]:
    rows=[]; counts={}
    for name in VIDEO_DIRS:
        d=root/name; split,prompt=video_meta(name); readable=0; total=0
        if not d.is_dir(): continue
        for p in sorted(d.glob('*.mp4'), key=lambda x:int(EP_RE.search(x.stem).group(1)) if EP_RE.search(x.stem) else 10**9):
            m=EP_RE.search(p.stem)
            if not m: continue
            ep=int(m.group(1)); total+=1
            res=inspect_video(p, first_frame_path(root,split,ep)); readable += int(bool(res.get('file_readable')))
            row={'video_set':name,'split':split,'prompt_set':prompt,'episode_id':ep,'video_path':str(p.relative_to(root))}
            row.update({k:res.get(k,'') for k in FIELDS if k not in row})
            rows.append(row)
        counts[name]={'total':total,'readable':readable,'unreadable':total-readable}
        logger.info('Video set %s: %s', name, counts[name])
    write_csv(out_dir/'video_stats.csv', rows, FIELDS)
    return {'available':True,'video_rows':len(rows),'video_sets':counts,'unreadable_total':sum(v['unreadable'] for v in counts.values())}
