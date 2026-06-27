#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
import numpy as np
sys.path.append(str(Path(__file__).resolve().parent))
from utils import ensure_dirs, read_csv, read_table, write_table, write_csv, is_v0_training_embodiment


def add_robotwin_rows(rows, manifest_rows, out, include_secondary, secondary):
    for r in manifest_rows:
        if str(r.get('success')).lower()!='true': continue
        if 'dual_arm_joint14_valid' not in r.get('quality_flags',''): continue
        if not is_v0_training_embodiment(r.get('embodiment',''), 'aloha-agilex', secondary, out, include_secondary): continue
        rows.append({'library_id':f"rt_{r.get('episode_id','')}",'source':'robotwin','split':r.get('split','train'),'episode_id':r.get('episode_id',''),
                     'robotwin_task_name':r.get('robotwin_task_name',''),'worldarena_episode_id_if_any':'','task_family':r.get('task_family',''),
                     'main_verbs':'','main_objects':'','receptacles_or_targets':'','spatial_relations':'',
                     'prompt_short':r.get('prompt_short',''),'prompt_worldarena_style':r.get('prompt_worldarena_style',''),
                     'action_joint14_path':r.get('action_joint14_raw_path',''),'action_ee16_path':r.get('action_ee16_raw_path',''),'action_joint14_ee16_path':r.get('action_joint14_ee16_raw_path',''),
                     'video_path':r.get('video_640x480_path',''),'first_frame_path':r.get('first_frame_320x240_path',''),
                     'T':r.get('T',''),'dominant_arm':r.get('dominant_arm',''),'action_complexity_score':r.get('action_complexity_score',''),
                     'success':True,'quality_score':1.0,'embodiment':r.get('embodiment','aloha-agilex'),'inference_only':False,'training_allowed':True})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--worldarena-root',default='/root/autodl-tmp/worldarena_testset')
    ap.add_argument('--analysis',default='/root/autodl-tmp/worldarena_testset/analysis_v2')
    ap.add_argument('--out',default='/root/autodl-tmp/worldarena_data_factory_v0')
    ap.add_argument('--episode-manifest-csv')
    ap.add_argument('--include-secondary-embodiment',action='store_true',help='Optional ablation: include <=5% secondary embodiment RoboTwin rows. Default library training rows are aloha-agilex only.')
    ap.add_argument('--secondary-embodiment',default='piper')
    args=ap.parse_args()
    root=Path(args.worldarena_root); analysis=Path(args.analysis); out=Path(args.out); ensure_dirs(out)
    try: import h5py
    except Exception as e: raise SystemExit(f'h5py required: {e}')
    rows=[]; rej=[]
    manifest_path=Path(args.episode_manifest_csv) if args.episode_manifest_csv else out/'manifests'/'episode_manifest.parquet'
    add_robotwin_rows(rows, read_table(manifest_path), out, args.include_secondary_embodiment, args.secondary_embodiment)
    sem=read_csv(analysis/'prompt_semantics.csv'); sem_i={(r['split'],r['episode_id'],r['prompt_set']):r for r in sem}; act=read_csv(analysis/'action_stats_episode.csv')
    for r in act:
        split=r['split']; ep=r['episode_id']; h=root/r['hdf5_path']; epdir=out/'worldarena_actions'/split/f'episode{ep}'; epdir.mkdir(parents=True,exist_ok=True)
        try:
            with h5py.File(h,'r') as f:
                a=np.asarray(f['/joint_action/vector']); T=a.shape[0]
                if a.ndim!=2 or a.shape[1]!=14: raise ValueError(f'WorldArena joint14 shape mismatch: {a.shape}')
                le=np.asarray(f['/endpose/left_endpose']); re=np.asarray(f['/endpose/right_endpose']); lg=np.asarray(f['/joint_action/left_gripper']).reshape(T,1); rg=np.asarray(f['/joint_action/right_gripper']).reshape(T,1)
            ee=np.concatenate([le,lg,re,rg],axis=1); np.save(epdir/'action_joint14.npy',a); np.save(epdir/'action_ee16.npy',ee); np.save(epdir/'action_joint14_ee16.npy',np.concatenate([a,ee],axis=1))
            s=sem_i.get((split,ep,'base'),{})
            rows.append({'library_id':f'wa_{split}_{ep}','source':'worldarena','split':split,'episode_id':ep,'robotwin_task_name':'','worldarena_episode_id_if_any':ep,'task_family':s.get('task_family',''),'main_verbs':s.get('main_verbs',''),'main_objects':s.get('main_objects',''),'receptacles_or_targets':s.get('receptacles_or_targets',''),'spatial_relations':s.get('spatial_relations',''),'prompt_short':s.get('prefix_removed_prompt',''),'prompt_worldarena_style':s.get('raw_prompt',''),'action_joint14_path':str(epdir/'action_joint14.npy'),'action_ee16_path':str(epdir/'action_ee16.npy'),'action_joint14_ee16_path':str(epdir/'action_joint14_ee16.npy'),'video_path':'','first_frame_path':str(root/split/'first_frame'/'fixed_scene_task'/f'episode{ep}.png'),'T':T,'dominant_arm':r.get('dominant_arm',''),'action_complexity_score':r.get('action_complexity_score',''),'success':True,'quality_score':1.0,'embodiment':'aloha-agilex','inference_only':True,'training_allowed':False})
        except Exception as e:
            rej.append({'hdf5_path':str(h),'reason':str(e)})
    actual,mode=write_table(out/'manifests'/'action_library.parquet',rows)
    write_csv(out/'rejected'/'action_library_rejected.csv',rej)
    print(f'{len(rows)} rows {actual} {mode}')
if __name__=='__main__': main()
