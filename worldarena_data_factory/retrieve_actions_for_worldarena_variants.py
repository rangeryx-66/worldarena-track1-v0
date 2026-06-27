#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, csv, json, math
sys.path.append(str(Path(__file__).resolve().parent))
from utils import ensure_dirs, read_csv, write_jsonl, POLICY_TO_MODE

def toks(s): return set(x for x in (s or '').split(';') if x)
def score(q,c):
    sc=0
    sc+=4*(q.get('task_family')==c.get('task_family'))
    sc+=3*bool(toks(q.get('main_verbs')) & toks(c.get('main_verbs')))
    sc+=3*bool(toks(q.get('main_objects')) & toks(c.get('main_objects')))
    sc+=2*bool(toks(q.get('receptacles_or_targets')) & toks(c.get('receptacles_or_targets')))
    sc+=1*bool(toks(q.get('spatial_relations')) & toks(c.get('spatial_relations')))
    try: sc-=abs(float(q.get('T',0))-float(c.get('T',0)))/500
    except Exception: pass
    try: sc-=max(0,1-float(c.get('quality_score',1)))
    except Exception: pass
    return sc

def best(q,lib,k=5): return sorted(lib,key=lambda c:score(q,c),reverse=True)[:k]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--worldarena-root',default='/root/autodl-tmp/worldarena_testset'); ap.add_argument('--analysis',default='/root/autodl-tmp/worldarena_testset/analysis_v2'); ap.add_argument('--out',default='/root/autodl-tmp/worldarena_data_factory_v0'); args=ap.parse_args(); out=Path(args.out); ensure_dirs(out); analysis=Path(args.analysis)
    lib=read_csv(out/'manifests'/'action_library.csv') or read_csv(out/'manifests'/'action_library.parquet'.replace if False else out/'manifests'/'action_library.csv')
    if not lib: lib=read_csv(out/'manifests'/'action_library.csv')
    sem=read_csv(analysis/'prompt_semantics.csv'); sem_i={(r['split'],r['episode_id'],r['prompt_set']):r for r in sem}; pol=read_csv(analysis/'prompt_action_policy.csv')
    pol_i={(r['split'],r['episode_id'],r['variant_prompt_set']):r for r in pol}; ep=read_csv(analysis/'episode_level.csv')
    for split,label,n in [('val_dataset','val',500),('test_dataset','test',1000)]:
        base=[]; v1=[]; v2=[]
        for r in [x for x in ep if x['dataset']==split]:
            eid=r['episode_id']; bs=sem_i.get((split,eid,'base'),{})
            base.append({'worldarena_episode_id':int(eid),'prompt_set':'base','inference_mode':'action_driven_original_hdf5','first_frame':r['first_frame_path'],'action_source':'worldarena_original','action_joint14_path':str(Path(args.out)/'worldarena_actions'/split/f'episode{eid}'/'action_joint14.npy'),'prompt':bs.get('raw_prompt','')})
            for vn,arr in [('variant_1',v1),('variant_2',v2)]:
                p=pol_i.get((split,eid,vn),{}); ss=sem_i.get((split,eid,vn),{}); policy=p.get('estimated_action_reuse_policy','AMBIGUOUS'); cand=best({**ss,'T':''},lib,5)
                arr.append({'worldarena_episode_id':int(eid),'prompt_set':vn,'policy':policy,'inference_mode':POLICY_TO_MODE.get(policy,'second_stage_review_or_text_fallback'),'first_frame':r['first_frame_path'],'original_action_joint14_path':str(Path(args.out)/'worldarena_actions'/split/f'episode{eid}'/'action_joint14.npy'),'retrieved_candidates':[{'library_id':c.get('library_id'),'score':score(ss,c),'action_joint14_path':c.get('action_joint14_path'),'source':c.get('source')} for c in cand],'prompt':ss.get('raw_prompt','')})
        write_jsonl(out/'inference_manifests'/f'worldarena_{label}_base_a2v.jsonl',base); write_jsonl(out/'inference_manifests'/f'worldarena_{label}_1_retrieval.jsonl',v1); write_jsonl(out/'inference_manifests'/f'worldarena_{label}_2_retrieval.jsonl',v2)
    print('wrote inference manifests')
if __name__=='__main__': main()
