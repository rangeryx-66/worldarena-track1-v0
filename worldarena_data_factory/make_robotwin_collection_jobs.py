#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, itertools
sys.path.append(str(Path(__file__).resolve().parent))
from utils import TASK_QUOTAS, TASK_CANDIDATES, embodiment_plan, ensure_dirs, detect_robotwin_root, available_robotwin_tasks, write_csv

FIELDS=['job_id','task_family','robotwin_task_name','base_task_config','task_config','embodiment','expected_action_dim','action_schema_required','is_dual_arm_required','target_success','max_attempts','gpu_id','output_dir']

def safe(emb): return emb.replace('-', '_').replace('+','_')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='/root/autodl-tmp/worldarena_data_factory_v0')
    ap.add_argument('--robotwin-root')
    ap.add_argument('--main-embodiment',default='aloha-agilex',help='Deprecated for v0; formal jobs are aloha-agilex unless --include-secondary-embodiment is set.')
    ap.add_argument('--secondary-embodiment',default='piper')
    ap.add_argument('--include-secondary-embodiment',action='store_true',help='Optional ablation: allow <=5% Piper or ARX-X5 jobs. Default is 100% aloha-agilex.')
    ap.add_argument('--gpus',default='0')
    ap.add_argument('--target-success-per-job',type=int,default=5)
    ap.add_argument('--max-attempts-multiplier',type=int,default=6)
    args=ap.parse_args()
    out=Path(args.out); ensure_dirs(out); rw=detect_robotwin_root(args.robotwin_root); tasks=set(available_robotwin_tasks(rw)); all_tasks=sorted(tasks)
    gpus=[x.strip() for x in args.gpus.split(',') if x.strip()]; gpu_cycle=itertools.cycle(gpus or ['0'])
    emb_slots=embodiment_plan(sum(TASK_QUOTAS.values()), out, args.main_embodiment, args.secondary_embodiment, args.include_secondary_embodiment)
    slot_idx=0; rows=[]; missing=[]; jid=0
    for fam,quota in TASK_QUOTAS.items():
        cands=TASK_CANDIDATES.get(fam) or all_tasks
        cands=[t for t in cands if not tasks or t in tasks]
        if not cands:
            missing.append({'task_family':fam,'requested_candidates':';'.join(TASK_CANDIDATES.get(fam,[]))}); slot_idx += quota; continue
        cfg_counts={'wa_clean_fixed':round(quota*.70),'wa_mild_random':round(quota*.20)}; cfg_counts['wa_hard_success']=quota-sum(cfg_counts.values())
        for cfg,n in cfg_counts.items():
            remaining=n
            while remaining>0:
                emb=emb_slots[slot_idx]
                same_run=0
                while slot_idx+same_run < len(emb_slots) and emb_slots[slot_idx+same_run]==emb and same_run < remaining and same_run < args.target_success_per_job:
                    same_run += 1
                ts=max(1,same_run)
                job_id=f'job_{jid:05d}'
                task=cands[jid%len(cands)]
                base_task_cfg=f"{cfg}__{safe(emb)}"
                task_cfg=f"{base_task_cfg}__{job_id}"
                rows.append({'job_id':job_id,'task_family':fam,'robotwin_task_name':task,'base_task_config':base_task_cfg,'task_config':task_cfg,'embodiment':emb,'expected_action_dim':14,'action_schema_required':'joint14','is_dual_arm_required':True,'target_success':ts,'max_attempts':ts*args.max_attempts_multiplier,'gpu_id':next(gpu_cycle),'output_dir':str(out/'robotwin_raw'/task/task_cfg)})
                jid+=1; remaining-=ts; slot_idx+=ts
    write_csv(out/'manifests'/'robotwin_collection_jobs.csv', rows, FIELDS); write_csv(out/'manifests'/'missing_tasks.csv', missing, ['task_family','requested_candidates'])
    print(f'wrote {len(rows)} jobs to {out}/manifests/robotwin_collection_jobs.csv')
if __name__=='__main__': main()
