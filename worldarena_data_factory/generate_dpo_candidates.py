#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
sys.path.append(str(Path(__file__).resolve().parent))
from utils import ensure_dirs, read_csv, write_jsonl, write_csv, write_table, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='/root/autodl-tmp/worldarena_data_factory_v0')
    ap.add_argument('--analysis',default='/root/autodl-tmp/worldarena_testset/analysis_v2')
    ap.add_argument('--target-pairs',type=int,default=300)
    ap.add_argument('--include-wrong-embodiment-dpo',action='store_true',help='Off by default for v0; wrong-embodiment negatives are v1/v2 robustness ablations.')
    args=ap.parse_args(); out=Path(args.out); ensure_dirs(out)
    cross_path=out/'cross_analysis.csv'
    if not cross_path.exists() and (Path(args.analysis)/'cross_analysis.csv').exists():
        cross_path.write_text((Path(args.analysis)/'cross_analysis.csv').read_text(encoding='utf-8'),encoding='utf-8')
    cross=read_csv(cross_path); pairs=[]; meta=[]; rej=[]
    for r in cross:
        if len(pairs)>=args.target_pairs: break
        if r.get('recommended_dpo_candidate_type')!='hard_negative_prompt_action_mismatch': continue
        rej.append({'split':r.get('split',''),'episode_id':r.get('episode_id',''),'candidate_type':'prompt-action mismatch','winner_domain':'aloha-agilex','loser_domain':'aloha-agilex','reason':'no winner/loser generated in v0 without rendered Aloha-AgileX RoboTwin success videos'})
    policy={'target_pairs':args.target_pairs,
            'winner_loser_domain':'aloha-agilex by default',
            'mix':{'prompt_action_mismatch':150,'physics_contact_failure':75,'wrong_object_target':45,'temporal_unfinished':30},
            'note':'v0 focuses on prompt-action mismatch inside the Aloha-AgileX visual/action domain.'}
    if args.include_wrong_embodiment_dpo:
        policy['embodiment_loser_types']=['wrong_embodiment_or_arm_count','wrong_gripper_morphology','left_right_arm_swap']
        policy['embodiment_loser_cap_ratio']=0.05
    else:
        policy['embodiment_loser_types']=[]
        policy['embodiment_loser_cap_ratio']=0.0
    write_json(out/'dpo_prompt_action'/'dpo_policy.json', policy)
    write_jsonl(out/'dpo_prompt_action'/'dpo_pairs.jsonl',pairs)
    actual,mode=write_table(out/'dpo_prompt_action'/'pair_meta.parquet',meta)
    write_csv(out/'rejected'/'rejected_dpo_candidates.csv',rej)
    print(f'pairs={len(pairs)} rejected={len(rej)} meta={actual}')
if __name__=='__main__': main()
