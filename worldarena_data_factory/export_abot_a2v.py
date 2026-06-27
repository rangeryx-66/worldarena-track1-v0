#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, json
sys.path.append(str(Path(__file__).resolve().parent))
from utils import ensure_dirs, read_csv, read_table, write_jsonl, write_json, is_v0_training_embodiment

def export(rows,out,name,field,main_embodiment=None,secondary_embodiment=None,include_secondary=False):
    data=[]
    for r in rows:
        if str(r.get('success')).lower()!='true' or not r.get('video_640x480_path') or not r.get(field): continue
        if not is_v0_training_embodiment(r.get('embodiment',''), main_embodiment, secondary_embodiment, out, include_secondary): continue
        if 'dual_arm_joint14_valid' not in r.get('quality_flags',''): continue
        flags=r.get('quality_flags','')
        intr=r.get('intrinsic_path') or str(out/'fallback_camera_intrinsic.json'); ext=r.get('extrinsic_path') or str(out/'fallback_camera_extrinsic.json')
        data.append({'video':r['video_640x480_path'],'prompt':r.get('prompt_worldarena_style') or r.get('prompt_short'),'action_path':r[field],'intrinsic_path':intr,'extrinsic_path':ext,'original_size':[480,640],'quality_flags':flags})
    write_jsonl(out/name/'metadata.jsonl',data)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='/root/autodl-tmp/worldarena_data_factory_v0'); ap.add_argument('--episode-manifest-csv'); ap.add_argument('--main-embodiment'); ap.add_argument('--secondary-embodiment',default='piper'); ap.add_argument('--include-secondary-embodiment',action='store_true'); args=ap.parse_args(); out=Path(args.out); ensure_dirs(out)
    write_json(out/'fallback_camera_intrinsic.json',{'fallback':True}); write_json(out/'fallback_camera_extrinsic.json',{'fallback':True})
    rows=read_table(Path(args.episode_manifest_csv) if args.episode_manifest_csv else out/'manifests'/'episode_manifest.parquet')
    export(rows,out,'a2v_worldarena_joint14','action_joint14_norm_path','aloha-agilex',args.secondary_embodiment,args.include_secondary_embodiment); export(rows,out,'a2v_worldarena_ee16','action_ee16_raw_path','aloha-agilex',args.secondary_embodiment,args.include_secondary_embodiment); export(rows,out,'a2v_worldarena_joint14_ee16','action_joint14_ee16_raw_path','aloha-agilex',args.secondary_embodiment,args.include_secondary_embodiment)
    print('done')
if __name__=='__main__': main()
