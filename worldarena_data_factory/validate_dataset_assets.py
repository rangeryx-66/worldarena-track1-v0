#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, json, os
sys.path.append(str(Path(__file__).resolve().parent))
from utils import ensure_dirs, read_csv, read_table, read_jsonl, write_json

def exists(p): return bool(p) and Path(p).exists()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='/root/autodl-tmp/worldarena_data_factory_v0'); args=ap.parse_args(); out=Path(args.out); ensure_dirs(out)
    report={'checks':{},'errors':[]}
    ep=read_table(out/'manifests'/'episode_manifest.parquet'); report['checks']['episode_manifest_rows']=len(ep); report['checks']['success_count']=sum(str(r.get('success')).lower()=='true' for r in ep)
    for rel in ['sft_worldarena_style/metadata.jsonl','a2v_worldarena_joint14/metadata.jsonl','a2v_worldarena_ee16/metadata.jsonl','a2v_worldarena_joint14_ee16/metadata.jsonl','dpo_prompt_action/dpo_pairs.jsonl']:
        rows=read_jsonl(out/rel); missing=0
        for r in rows:
            for k in ['video','action_path','intrinsic_path','extrinsic_path','winner_video','loser_video']:
                if k in r and r[k] and not exists(r[k]): missing+=1
        report['checks'][rel]={'rows':len(rows),'missing_paths':missing}
    for rel,expected in [('inference_manifests/worldarena_val_base_a2v.jsonl',500),('inference_manifests/worldarena_val_1_retrieval.jsonl',500),('inference_manifests/worldarena_val_2_retrieval.jsonl',500),('inference_manifests/worldarena_test_base_a2v.jsonl',1000),('inference_manifests/worldarena_test_1_retrieval.jsonl',1000),('inference_manifests/worldarena_test_2_retrieval.jsonl',1000)]:
        rows=read_jsonl(out/rel); report['checks'][rel]={'rows':len(rows),'expected':expected,'covered':len(rows)==expected}
    write_json(out/'v0_dataset_report.json',report)
    md=['# WorldArena Data Factory v0 Report','',f"Episode manifest rows: {report['checks']['episode_manifest_rows']}",f"Successful collected episodes: {report['checks']['success_count']} / 1500",'']
    for k,v in report['checks'].items(): md.append(f'- `{k}`: `{v}`')
    (out/'v0_dataset_report.md').write_text('\n'.join(md)+'\n',encoding='utf-8'); print(out/'v0_dataset_report.md')
if __name__=='__main__': main()
