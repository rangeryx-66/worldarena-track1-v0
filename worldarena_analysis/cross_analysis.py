"""Cross-modal merge and training/inference recommendations."""
from __future__ import annotations
import csv, json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

POLICY_TO_MODE={
 'SAME_ACTION_OK':'action_driven_original_hdf5',
 'ACTION_MAYBE_OK':'action_driven_original_or_retrieved_hdf5',
 'TARGET_CHANGED':'action_retrieval_or_robotwin_replan',
 'VERB_CHANGED':'action_retrieval_or_robotwin_replan',
 'AMBIGUOUS':'manual_review_or_text_driven_fallback',
}
FIELDS=['split','episode_id','task_name','base_task_family','base_main_verbs','base_main_objects','policy_1','policy_2','T','action_complexity_score','dominant_arm','visual_cluster_id','brightness_mean','edge_density','recommended_sft_weight','recommended_a2v_weight','recommended_dpo_candidate_type','recommended_inference_mode_base','recommended_inference_mode_1','recommended_inference_mode_2']

def read_csv(path: Path):
    with path.open('r',encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict[str,Any]], fields: list[str]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def fnum(x, default=0.0):
    try: return float(x)
    except Exception: return default

def run_cross_analysis(out_dir: Path, logger) -> dict[str,Any]:
    sem=read_csv(out_dir/'prompt_semantics.csv'); pol=read_csv(out_dir/'prompt_action_policy.csv'); act=read_csv(out_dir/'action_stats_episode.csv'); vis=read_csv(out_dir/'visual_stats_episode.csv')
    base_sem={(r['split'],r['episode_id']):r for r in sem if r['prompt_set']=='base'}
    act_i={(r['split'],r['episode_id']):r for r in act}; vis_i={(r['split'],r['episode_id']):r for r in vis}
    pol_i=defaultdict(dict)
    for r in pol:
        key=(r['split'],r['episode_id']); idx='1' if r['variant_prompt_set']=='variant_1' else '2'; pol_i[key][idx]=r
    keys=sorted(set(base_sem)|set(act_i)|set(vis_i), key=lambda k:(k[0],int(k[1])))
    rows=[]
    for key in keys:
        s,e=key; bs=base_sem.get(key,{}); ar=act_i.get(key,{}); vr=vis_i.get(key,{})
        p1=pol_i.get(key,{}).get('1',{}); p2=pol_i.get(key,{}).get('2',{})
        complexity=fnum(ar.get('action_complexity_score')); edge=fnum(vr.get('edge_density')); bright=fnum(vr.get('brightness_mean'))
        sft=1.0
        if bs.get('task_family') in {'unknown'}: sft*=0.7
        if edge>0.12: sft*=1.1
        if bright<45 or bright>210: sft*=0.85
        a2v=1.0+min(complexity/10.0,0.8)
        if ar.get('dominant_arm')=='balanced': a2v+=0.15
        risky=sum(1 for p in [p1,p2] if p.get('estimated_action_reuse_policy') in {'VERB_CHANGED','TARGET_CHANGED','AMBIGUOUS'})
        dpo='hard_negative_prompt_action_mismatch' if risky==2 else ('mixed_policy_pair' if risky==1 else 'same_action_preference_pair')
        rows.append({'split':s,'episode_id':e,'task_name':ar.get('task_name',''),'base_task_family':bs.get('task_family',''),'base_main_verbs':bs.get('main_verbs',''),'base_main_objects':bs.get('main_objects',''),'policy_1':p1.get('estimated_action_reuse_policy',''),'policy_2':p2.get('estimated_action_reuse_policy',''),'T':ar.get('T',''),'action_complexity_score':ar.get('action_complexity_score',''),'dominant_arm':ar.get('dominant_arm',''),'visual_cluster_id':vr.get('visual_cluster_id',''),'brightness_mean':vr.get('brightness_mean',''),'edge_density':vr.get('edge_density',''),'recommended_sft_weight':round(sft,4),'recommended_a2v_weight':round(a2v,4),'recommended_dpo_candidate_type':dpo,'recommended_inference_mode_base':'action_driven_original_hdf5','recommended_inference_mode_1':POLICY_TO_MODE.get(p1.get('estimated_action_reuse_policy','AMBIGUOUS')),'recommended_inference_mode_2':POLICY_TO_MODE.get(p2.get('estimated_action_reuse_policy','AMBIGUOUS'))})
    write_csv(out_dir/'cross_analysis.csv', rows, FIELDS)
    fam=Counter(r['base_task_family'] for r in rows); clusters=Counter(r['visual_cluster_id'] for r in rows); dpo=Counter(r['recommended_dpo_candidate_type'] for r in rows)
    plan={'sft_sampling':'Stratify by split, task_family, visual_cluster_id, and action_complexity quantiles. Use recommended_sft_weight as sampling multiplier; downweight unknown semantics and extreme brightness.', 'a2v_sampling':'Use recommended_a2v_weight to oversample complex, bimanual, and long-horizon trajectories. Keep a balanced cluster/task mix.', 'task_family_counts':dict(fam), 'visual_cluster_counts':dict(clusters), 'dpo_candidate_counts':dict(dpo)}
    inf={'base':'base instruction should use action_driven_original_hdf5','variant_policy_mapping':POLICY_TO_MODE,'mode_counts_1':dict(Counter(r['recommended_inference_mode_1'] for r in rows)),'mode_counts_2':dict(Counter(r['recommended_inference_mode_2'] for r in rows))}
    (out_dir/'training_sampling_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (out_dir/'inference_policy_summary.json').write_text(json.dumps(inf,indent=2),encoding='utf-8')
    md=out_dir/'abot_training_recommendations.md'
    md.write_text('\n'.join(['# ABot-PhysWorld Training Recommendations','','## SFT Sampling','Use stratified sampling across split, task_family, visual_cluster_id, and action complexity. Use `recommended_sft_weight` from `cross_analysis.csv`; keep all base examples, downweight unknown semantics and extreme visual outliers, and ensure rare families such as `handover`, `hanging`, `ranking_arrangement`, and `dumping_pouring` are not lost.','','## A2V Representation','Export `joint14` as the primary baseline because it directly matches `/joint_action/vector`. Also export `ee16` from left/right endpose plus grippers for ablations, and export `joint14+ee16` for the high-capacity A2V condition. Treat joint14 as the minimum reliable representation.','','## DPO Pair Mining','Mine pairs from `prompt_action_policy.csv` and `cross_analysis.csv`: SAME_ACTION_OK/ACTION_MAYBE_OK can form positive or near-positive action reuse pairs; TARGET_CHANGED and VERB_CHANGED are strong hard negatives for prompt-action mismatch; AMBIGUOUS should be manual-review or low-confidence preference data.','','## Final Generation Policy','Base/test prompts: use `action_driven_original_hdf5`. For test_1/test_2 variants, use SAME_ACTION_OK -> original HDF5, ACTION_MAYBE_OK -> original or retrieved HDF5, TARGET_CHANGED/VERB_CHANGED -> action retrieval or RoboTwin replan, AMBIGUOUS -> manual review or text-driven fallback.','']),encoding='utf-8')
    return {'available':True,'rows':len(rows),'dpo_candidate_counts':dict(dpo),'training_sampling_plan':'training_sampling_plan.json','inference_policy_summary':'inference_policy_summary.json','abot_training_recommendations':'abot_training_recommendations.md'}
