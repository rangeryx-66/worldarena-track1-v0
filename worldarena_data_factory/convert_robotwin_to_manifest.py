#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, subprocess, shutil
import numpy as np
sys.path.append(str(Path(__file__).resolve().parent))
from utils import ensure_dirs, read_csv, write_json, write_table, write_csv, normalize_embodiment, is_v0_training_embodiment


def make_video_and_frame(raw_video: Path, epdir: Path):
    out_video = epdir / 'observation.mp4'
    out_frame = epdir / 'first_frame.png'
    flags = []
    if raw_video.exists():
        if not out_video.exists():
            try:
                subprocess.run([
                    'ffmpeg','-y','-loglevel','error','-i',str(raw_video),
                    '-vf','scale=640:480,fps=24','-pix_fmt','yuv420p',str(out_video)
                ], check=True)
            except Exception:
                try:
                    shutil.copy2(raw_video, out_video)
                    flags.append('video_copy_fallback')
                except Exception:
                    flags.append('video_export_failed')
        if not out_frame.exists():
            try:
                subprocess.run([
                    'ffmpeg','-y','-loglevel','error','-i',str(raw_video),
                    '-frames:v','1','-vf','scale=320:240',str(out_frame)
                ], check=True)
            except Exception:
                flags.append('first_frame_export_failed')
    else:
        flags.append('video_missing')
    return str(out_video) if out_video.exists() else '', str(out_frame) if out_frame.exists() else '', flags


def reject_row(h, j, reason):
    return {'raw_hdf5_path':str(h) if h else '', 'job_id':j.get('job_id',''), 'embodiment':j.get('embodiment',''), 'reason':reason, 'dpo_loser_allowed':str('rejected_action_schema' in reason or 'wrong_embodiment' in reason).lower()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='/root/autodl-tmp/worldarena_data_factory_v0')
    ap.add_argument('--robotwin-root')
    ap.add_argument('--jobs-csv')
    ap.add_argument('--include-secondary-embodiment',action='store_true',help='Optional ablation: allow <=5% Piper/ARX-X5 positives. Default converter accepts only aloha-agilex.')
    ap.add_argument('--secondary-embodiment',default='piper')
    args=ap.parse_args()
    out=Path(args.out); ensure_dirs(out); jobs=read_csv(Path(args.jobs_csv) if args.jobs_csv else out/'manifests'/'robotwin_collection_jobs.csv')
    rows=[]; rejected=[]
    try: import h5py
    except Exception as e: h5py=None; rejected.append({'reason':'h5py_unavailable','detail':str(e)})
    for j in jobs:
        emb=normalize_embodiment(j.get('embodiment',''))
        if not is_v0_training_embodiment(emb, 'aloha-agilex', args.secondary_embodiment, out, args.include_secondary_embodiment):
            rejected.append(reject_row('', j, f'wrong_embodiment_for_v0:{emb}; expected aloha-agilex'))
            continue
        raw=Path(j['output_dir'])/'data'
        if not raw.exists(): continue
        for h in sorted(raw.glob('episode*.hdf5')):
            try:
                with h5py.File(h,'r') as f:
                    if '/joint_action/vector' not in f: raise ValueError('rejected_action_schema: missing /joint_action/vector')
                    a=np.asarray(f['/joint_action/vector']); T=a.shape[0] if a.ndim else 0
                    required=['/joint_action/left_arm','/joint_action/right_arm','/joint_action/left_gripper','/joint_action/right_gripper']
                    missing=[x for x in required if x not in f]
                    if a.ndim!=2 or a.shape[1]!=14: raise ValueError(f'rejected_action_schema: joint14 shape {a.shape}')
                    if missing: raise ValueError('rejected_action_schema: missing '+','.join(missing))
                    if T < 60: raise ValueError(f'rejected_action_schema: T<{60}: {T}')
                    if not np.isfinite(a).all(): raise ValueError('rejected_action_schema: NaN_or_Inf')
                    le=np.asarray(f['/endpose/left_endpose']) if '/endpose/left_endpose' in f else np.zeros((T,7))
                    re=np.asarray(f['/endpose/right_endpose']) if '/endpose/right_endpose' in f else np.zeros((T,7))
                    lg=np.asarray(f['/joint_action/left_gripper']).reshape(T,1); rg=np.asarray(f['/joint_action/right_gripper']).reshape(T,1)
                    if le.shape[0]!=T or re.shape[0]!=T: raise ValueError('rejected_action_schema: endpose length mismatch')
                eid=f"rt_{len(rows):06d}"; epdir=out/'episodes'/eid; epdir.mkdir(parents=True,exist_ok=True)
                np.save(epdir/'action_joint14.npy',a)
                mean=a.mean(0); std=a.std(0)+1e-6
                np.save(epdir/'action_joint14_norm.npy',np.clip((a-mean)/std,-5,5))
                ee=np.concatenate([le,lg,re,rg],axis=1)
                np.save(epdir/'action_ee16.npy',ee); np.save(epdir/'action_joint14_ee16.npy',np.concatenate([a,ee],axis=1))
                write_json(epdir/'camera_intrinsic.json',{'fallback':True}); write_json(epdir/'camera_extrinsic.json',{'fallback':True}); write_json(epdir/'meta.json',j)
                raw_video = h.parent.parent / 'video' / (h.stem + '.mp4')
                video_path, first_frame_path, media_flags = make_video_and_frame(raw_video, epdir)
                qflags = ['camera_fallback','dual_arm_joint14_valid','aloha_agilex_domain'] + media_flags
                rows.append({'episode_id':eid,'source':'robotwin','robotwin_task_name':j['robotwin_task_name'],'task_family':j['task_family'],'config_name':j['task_config'],'embodiment':emb,'seed':'','success':True,'raw_hdf5_path':str(h),'raw_video_path':str(raw_video) if raw_video.exists() else '','video_640x480_path':video_path,'first_frame_320x240_path':first_frame_path,'action_joint14_raw_path':str(epdir/'action_joint14.npy'),'action_joint14_norm_path':str(epdir/'action_joint14_norm.npy'),'action_ee16_raw_path':str(epdir/'action_ee16.npy'),'action_joint14_ee16_raw_path':str(epdir/'action_joint14_ee16.npy'),'intrinsic_path':str(epdir/'camera_intrinsic.json'),'extrinsic_path':str(epdir/'camera_extrinsic.json'),'T':T,'fps':24,'dominant_arm':'','gripper_transition_count':'','action_complexity_score':'','prompt_short':j['robotwin_task_name'],'prompt_worldarena_style':f"In a fixed robotic workspace, perform {j['robotwin_task_name'].replace('_',' ')}.",'prompt_long_caption':f"An Aloha-AgileX dual-arm robot successfully completes {j['robotwin_task_name'].replace('_',' ')}.",'quality_flags':';'.join(qflags),'split':'train'})
            except Exception as e:
                rejected.append(reject_row(h, j, str(e)))
    actual,mode=write_table(out/'manifests'/'episode_manifest.parquet',rows)
    write_json(out/'manifests'/'action_normalization_config.json',{'note':'per-episode placeholder unless train data collected; recompute globally from RoboTwin train only before final training','episode_count':len(rows),'training_embodiment':'aloha-agilex','action_schema':'joint14'})
    write_csv(out/'rejected'/'convert_rejected.csv',rejected); write_csv(out/'rejected'/'rejected_episodes.csv',rejected)
    print(f'episode manifest rows={len(rows)} actual={actual} mode={mode}')
if __name__=='__main__': main()
