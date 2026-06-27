#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, random
sys.path.append(str(Path(__file__).resolve().parent))
from utils import TASK_QUOTAS, CONFIG_RATIOS, embodiment_plan, embodiment_weights, ensure_dirs, write_yaml, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='/root/autodl-tmp/worldarena_data_factory_v0')
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--include-secondary-embodiment',action='store_true',help='Optional diagnostic/ablation mode: allow <=5% Piper or ARX-X5. Default v0 is 100% aloha-agilex.')
    ap.add_argument('--secondary-embodiment',default='piper')
    ap.add_argument('--main-embodiment',default='aloha-agilex',help='Deprecated for v0; formal main embodiment is fixed to aloha-agilex.')
    args=ap.parse_args()
    out=Path(args.out); ensure_dirs(out); random.seed(args.seed)
    total=sum(TASK_QUOTAS.values())
    emb=embodiment_plan(total, out, args.main_embodiment, args.secondary_embodiment, args.include_secondary_embodiment)
    weights=[{'embodiment': e, 'ratio': r} for e,r in embodiment_weights(out,args.main_embodiment,args.secondary_embodiment,args.include_secondary_embodiment)]
    spec={'version':'v0_smoke','target_successful_episodes':1500,'task_family_quotas':TASK_QUOTAS,'config_ratios':CONFIG_RATIOS,
          'target_domain':'RoboTwin2 Clean-50 / WorldArena Track1 Aloha-AgileX dual-arm gripper manipulation',
          'action_schema_required':'joint14','expected_action_dim':14,'is_dual_arm_required':True,
          'embodiment_strategy':{'main_embodiment':'aloha-agilex','default_weights':[{'embodiment':'aloha-agilex','ratio':1.0}],'include_secondary_embodiment':bool(args.include_secondary_embodiment),'weights':weights,'plan_counts':{},'note':'v0 defaults to 100% Aloha-AgileX; optional secondary is capped at 5% and intended for ablation only.'},
          'jobs':[]}
    i=0
    for fam,quota in TASK_QUOTAS.items():
        cfg_counts={'wa_clean_fixed':round(quota*.70),'wa_mild_random':round(quota*.20)}; cfg_counts['wa_hard_success']=quota-sum(cfg_counts.values())
        for cfg,n in cfg_counts.items():
            for _ in range(n):
                e=emb[i]
                spec['jobs'].append({'target_id':f'target_{i:04d}','task_family':fam,'config_name':cfg,'embodiment':e,'target_success':1,'expected_action_dim':14,'action_schema_required':'joint14','is_dual_arm_required':True})
                spec['embodiment_strategy']['plan_counts'][e]=spec['embodiment_strategy']['plan_counts'].get(e,0)+1
                i+=1
    write_yaml(out/'manifests'/'worldarena_target_spec.yaml', spec)
    write_json(out/'manifests'/'worldarena_target_spec.json', spec)
    print(out/'manifests'/'worldarena_target_spec.yaml')
if __name__=='__main__': main()
