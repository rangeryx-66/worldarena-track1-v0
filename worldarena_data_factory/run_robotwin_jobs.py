#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, subprocess, concurrent.futures, time, os, shutil
sys.path.append(str(Path(__file__).resolve().parent))
from utils import ensure_dirs, detect_robotwin_root, read_csv, write_csv, available_robotwin_tasks


def success_count(job):
    p=Path(job['output_dir'])/'data'
    return len(list(p.glob('episode*.hdf5'))) if p.exists() else 0

def target_count(job): return int(job.get('target_success') or 0)

def capped_success(job): return min(success_count(job), target_count(job))

def aggregate(jobs):
    total=sum(target_count(j) for j in jobs)
    done=sum(capped_success(j) for j in jobs)
    completed=sum(1 for j in jobs if capped_success(j)>=target_count(j))
    return done,total,completed,len(jobs)

def fmt_time(sec):
    sec=max(0,int(sec)); h=sec//3600; m=(sec%3600)//60; s=sec%60
    return f'{h:02d}:{m:02d}:{s:02d}'

def bar(done,total,width=30):
    ratio=0 if total<=0 else min(1.0,done/total); fill=int(ratio*width)
    return '['+'#'*fill+'.'*(width-fill)+f'] {ratio*100:5.1f}%'

def cmd_for(root, job): return [str(root/'collect_data.sh'), job['robotwin_task_name'], job['task_config'], str(job['gpu_id'])]

def ensure_job_config(root: Path, out: Path, job: dict):
    import yaml
    task_cfg=job['task_config']
    cfg_dst=root/'task_config'/f'{task_cfg}.yml'
    base=job.get('base_task_config') or task_cfg.rsplit('__job_',1)[0]
    candidates=[out/'configs_to_apply'/f'{base}.yml', root/'task_config'/f'{base}.yml']
    src=next((p for p in candidates if p.exists()), None)
    if not src:
        raise FileNotFoundError(f'base task config not found for {task_cfg}: checked '+', '.join(str(p) for p in candidates))
    with src.open('r',encoding='utf-8') as f:
        data=yaml.safe_load(f) or {}
    data['episode_num']=target_count(job)
    data['save_path']=str(out/'robotwin_raw')
    data.setdefault('worldarena_v0_constraints',{})
    data['worldarena_v0_constraints'].update({'job_id':job['job_id'],'base_task_config':base,'expected_action_dim':14,'action_schema_required':'joint14','is_dual_arm_required':True})
    cfg_dst.parent.mkdir(parents=True,exist_ok=True)
    with cfg_dst.open('w',encoding='utf-8') as f:
        yaml.safe_dump(data,f,sort_keys=False,allow_unicode=True)
    return cfg_dst

def run_one(root,out,job):
    log=out/'logs'/f"collect_{job['job_id']}.log"; log.parent.mkdir(exist_ok=True)
    start=time.time()
    try:
        cfg_path=ensure_job_config(root,out,job)
        with log.open('a',encoding='utf-8') as f:
            f.write('\n===== WORLD_ARENA_JOB_START =====\n')
            f.write('JOB: '+str(job)+'\n')
            f.write('CONFIG: '+str(cfg_path)+'\n')
            f.write('CMD: '+' '.join(cmd_for(root,job))+'\n')
            f.flush()
            rc=subprocess.run(cmd_for(root,job),cwd=root,stdout=f,stderr=subprocess.STDOUT).returncode
    except Exception as e:
        with log.open('a',encoding='utf-8') as f:
            f.write('RUNNER_ERROR: '+repr(e)+'\n')
        rc=997
    got=success_count(job)
    target=target_count(job)
    if got < target and rc == 0:
        rc=996
        with log.open('a',encoding='utf-8') as f:
            f.write(f'RUNNER_INCOMPLETE: got {got}/{target} hdf5 files; marking job failed despite shell rc=0\n')
    return {'job_id':job['job_id'],'rc':rc,'seconds':time.time()-start,'success_count':got,'target_success':target,'log':str(log)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='/root/autodl-tmp/worldarena_data_factory_v0'); ap.add_argument('--robotwin-root'); ap.add_argument('--jobs-csv'); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--execute',action='store_true'); ap.add_argument('--max-parallel-gpus',type=int,default=1); ap.add_argument('--resume',action='store_true'); args=ap.parse_args()
    out=Path(args.out); ensure_dirs(out); root=detect_robotwin_root(args.robotwin_root)
    if not root: raise SystemExit('RoboTwin root not found; pass --robotwin-root')
    jobs=read_csv(Path(args.jobs_csv) if args.jobs_csv else out/'manifests'/'robotwin_collection_jobs.csv'); tasks=set(available_robotwin_tasks(root)); missing=[]; todo=[]
    for j in jobs:
        if j['robotwin_task_name'] not in tasks: missing.append(j); continue
        if args.resume and success_count(j)>=target_count(j): continue
        todo.append(j)
    write_csv(out/'manifests'/'missing_tasks.csv', missing)
    for j in todo[:20]: print(' '.join(cmd_for(root,j)))
    done,total,completed,total_jobs=aggregate(jobs)
    print(f'{len(todo)} jobs ready; missing={len(missing)}')
    print(f'progress {bar(done,total)} episodes {done}/{total} jobs_done {completed}/{total_jobs}')
    print('note: job-specific task_config files are generated automatically on --execute')
    if args.dry_run or not args.execute: return
    start=time.time(); finished=0; failed=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel_gpus) as ex:
        futs={ex.submit(run_one,root,out,j):j for j in todo}
        for fut in concurrent.futures.as_completed(futs):
            res=fut.result(); finished+=1
            if res['rc']!=0: failed+=1
            done,total,completed,total_jobs=aggregate(jobs)
            elapsed=time.time()-start
            rate=done/elapsed if elapsed>1 else 0
            eta=(total-done)/rate if rate>0 else 0
            print(f"{bar(done,total)} episodes {done}/{total} jobs_done {completed}/{total_jobs} run_done {finished}/{len(todo)} failed {failed} elapsed {fmt_time(elapsed)} eta {fmt_time(eta)} last {res['job_id']} rc={res['rc']} got={res['success_count']}/{res['target_success']} log={res['log']}", flush=True)
if __name__=='__main__': main()
