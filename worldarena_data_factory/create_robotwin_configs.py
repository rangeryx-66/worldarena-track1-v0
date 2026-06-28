#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, shutil, copy
sys.path.append(str(Path(__file__).resolve().parent))
from utils import ensure_dirs, detect_robotwin_root, write_yaml, normalize_embodiment, DUAL_ARM_CANDIDATE_EMBODIMENTS, embodiment_weights


def load_template(robotwin_root: Path|None):
    template = None
    if robotwin_root:
        for name in ['demo_clean.yml', '_config_template.yml']:
            p = robotwin_root / 'task_config' / name
            if p.exists():
                try:
                    import yaml
                    with p.open('r', encoding='utf-8') as f:
                        template = yaml.safe_load(f) or {}
                    break
                except Exception:
                    template = None
    if template is None:
        template = {
            'render_freq': 0,
            'episode_num': 5,
            'use_seed': False,
            'save_freq': 15,
            'embodiment': ['aloha-agilex'],
            'language_num': 100,
            'domain_randomization': {},
            'camera': {'head_camera_type': 'D435', 'wrist_camera_type': 'D435', 'collect_head_camera': True, 'collect_wrist_camera': False},
            'data_type': {'rgb': True, 'third_view': False, 'depth': False, 'pointcloud': False, 'observer': False, 'endpose': True, 'qpos': True, 'mesh_segmentation': False, 'actor_segmentation': False},
            'pcd_down_sample_num': 1024,
            'pcd_crop': True,
            'save_path': './data',
            'clear_cache_freq': 5,
            'collect_data': True,
            'eval_video_log': True,
        }
    return template


def domain_randomization(name):
    # RoboTwin treats random_head_camera_dis and random_table_height as numeric distances.
    # Never write bool True here: in Python True == 1, which can move the camera/table by ~1m
    # and produces empty/table-edge videos. Keep camera fixed for WorldArena v0.
    if name=='wa_clean_fixed':
        return {'cluttered_table':False,'random_background':False,'clean_background_rate':1.0,'random_light':False,'crazy_random_light_rate':0.0,'random_table_height':0.0,'random_head_camera_dis':0.0}
    if name=='wa_mild_random':
        return {'cluttered_table':True,'random_background':True,'clean_background_rate':0.7,'random_light':True,'crazy_random_light_rate':0.03,'random_table_height':0.02,'random_head_camera_dis':0.0}
    if name=='wa_hard_success':
        return {'cluttered_table':True,'random_background':True,'clean_background_rate':0.35,'random_light':True,'crazy_random_light_rate':0.08,'random_table_height':0.03,'random_head_camera_dis':0.0}
    return {'cluttered_table':False,'random_background':False,'clean_background_rate':1.0,'random_light':False,'crazy_random_light_rate':0.0,'random_table_height':0.0,'random_head_camera_dis':0.0}


def cfg(name, embodiment, template):
    base=copy.deepcopy(template)
    base['episode_num']=5
    base['use_seed']=False
    base['collect_data']=True
    base['save_path']='./data'
    base['render_freq']=0
    base['clear_cache_freq']=5
    base['save_freq']=base.get('save_freq',15) or 15
    base['language_num']=base.get('language_num',100) or 100
    base['embodiment']=[embodiment]
    base['camera']=base.get('camera') or {}
    base['camera'].update({'head_camera_type':'D435','collect_head_camera':True,'wrist_camera_type':'D435','collect_wrist_camera':False})
    base['data_type']=base.get('data_type') or {}
    base['data_type'].update({'rgb':True,'third_view':False,'depth':False,'pointcloud':False,'observer':False,'endpose':True,'qpos':True,'mesh_segmentation':False,'actor_segmentation':False})
    base['domain_randomization']=domain_randomization(name)
    base['worldarena_v0_constraints']={'target_domain':'RoboTwin2 Clean-50 Aloha-AgileX dual-arm gripper','expected_action_dim':14,'action_schema_required':'joint14','is_dual_arm_required':True}
    return base

def safe(emb): return emb.replace('-', '_').replace('+','_')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='/root/autodl-tmp/worldarena_data_factory_v0')
    ap.add_argument('--robotwin-root')
    ap.add_argument('--apply',action='store_true')
    ap.add_argument('--include-secondary-embodiment',action='store_true',help='Optional ablation configs: add <=5% Piper/ARX-X5 formal configs. Default formal configs are aloha-agilex only.')
    ap.add_argument('--secondary-embodiment',default='piper')
    ap.add_argument('--write-probe-configs',action='store_true',help='Write optional diagnostic probe configs for all candidate embodiments.')
    ap.add_argument('--embodiment',help='Deprecated; ignored for formal v0 unless used with --write-probe-configs in custom workflows.')
    ap.add_argument('--main-embodiment',default='aloha-agilex',help='Deprecated for v0; formal main embodiment is fixed to aloha-agilex.')
    args=ap.parse_args()
    out=Path(args.out); ensure_dirs(out); rw=detect_robotwin_root(args.robotwin_root); written=[]; template=load_template(rw)
    embs=[e for e,_ in embodiment_weights(out,args.main_embodiment,args.secondary_embodiment,args.include_secondary_embodiment)]
    for name in ['wa_clean_fixed','wa_mild_random','wa_hard_success']:
        for emb in embs:
            cfg_name=f'{name}__{safe(emb)}'; path=out/'configs_to_apply'/f'{cfg_name}.yml'; write_yaml(path,cfg(name,emb,template)); written.append(str(path))
            if args.apply and rw: shutil.copy2(path,rw/'task_config'/f'{cfg_name}.yml')
    if args.write_probe_configs:
        for emb in [normalize_embodiment(e) for e in DUAL_ARM_CANDIDATE_EMBODIMENTS]:
            cfg_name=f'wa_probe_dual_arm__{safe(emb)}'; path=out/'configs_to_apply'/f'{cfg_name}.yml'; write_yaml(path,cfg('wa_clean_fixed',emb,template)); written.append(str(path))
            if args.apply and rw: shutil.copy2(path,rw/'task_config'/f'{cfg_name}.yml')
    readme=out/'configs_to_apply'/'README_apply_configs.md'
    readme.write_text('Formal v0 configs inherit RoboTwin demo_clean schema, pin Aloha-AgileX, enable rgb/endpose/qpos data_type, and require joint14. Probe configs are optional diagnostics only.\n',encoding='utf-8')
    print('\n'.join(written))
if __name__=='__main__': main()
