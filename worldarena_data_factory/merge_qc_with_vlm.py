#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from v0_1_vlm_qc_common import FLAG_KEYS, SCORE_KEYS, make_contact_sheet, read_jsonl, read_table, write_table


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def flatten_vlm(row: dict[str, Any]) -> dict[str, Any]:
    scores = row.get('scores') if isinstance(row.get('scores'), dict) else {}
    flags = row.get('flags') if isinstance(row.get('flags'), dict) else {}
    rec = row.get('recommended_use') if isinstance(row.get('recommended_use'), dict) else {}
    out = {
        'episode_id': row.get('episode_id', ''),
        'vlm_decision': str(row.get('overall_decision', 'NEED_HUMAN_REVIEW')).upper(),
        'vlm_confidence': float(row.get('confidence') or 0.0),
        'vlm_evidence': '; '.join(str(x) for x in (row.get('evidence') or [])[:6]),
        'raw_output_path': row.get('raw_output_path', ''),
    }
    for k in SCORE_KEYS:
        out[f'score_{k}'] = int(scores.get(k, 1))
    for k in FLAG_KEYS:
        out[f'flag_{k}'] = bool(flags.get(k, False))
    for k in ['use_for_sft', 'use_for_a2v', 'use_for_dpo_winner', 'use_for_dpo_loser']:
        out[k] = bool(rec.get(k, False))
    return out


def is_hard_fail(row: pd.Series) -> bool:
    if truthy(row.get('deterministic_hard_fail', False)):
        return True
    if str(row.get('video_readable', 'true')).lower() == 'false':
        return True
    reason = str(row.get('hard_fail_reason') or row.get('qc_reason') or '').lower()
    hard_tokens = ['unreadable', 'black_or_bad', 'bad_frame', 'resolution_invalid', 'fps_invalid', 'action_shape_invalid', 'action_nan', 'action_inf']
    return any(t in reason for t in hard_tokens)


def vlm_obvious_visual_hard_reject(row: pd.Series) -> bool:
    if str(row.get('vlm_decision', '')).upper() != 'REJECT':
        return False
    zero_score_keys = [
        'score_robot_visibility', 'score_gripper_contact_visibility', 'score_object_visibility',
        'score_physical_plausibility', 'score_temporal_consistency', 'score_visual_quality',
        'score_task_relevance', 'score_sft_positive_suitability', 'score_a2v_positive_suitability',
    ]
    zero_count = 0
    for key in zero_score_keys:
        try:
            zero_count += int(float(row.get(key, 1)) <= 0)
        except Exception:
            pass
    evidence = str(row.get('vlm_evidence', '')).lower()
    hard_phrases = [
        'no robot', 'robot or gripper is visible', 'no gripper', 'no object',
        'not visible in any frame', 'blank', 'gray', 'no manipulation',
        'no visual content', 'missing robot', 'wrong_robot_or_missing_robot',
    ]
    hard_flags = any(str(row.get(k, '')).lower() == 'true' for k in [
        'flag_wrong_robot_or_missing_robot', 'flag_critical_contact_invisible', 'flag_object_moves_without_contact',
    ])
    return zero_count >= 6 or hard_flags or any(x in evidence for x in hard_phrases)


def vlm_visual_artifact_only(row: pd.Series) -> bool:
    if str(row.get('vlm_decision', '')).upper() != 'REJECT':
        return False
    good_keys = [
        'score_domain_match', 'score_robot_visibility', 'score_gripper_contact_visibility',
        'score_object_visibility', 'score_physical_plausibility', 'score_task_relevance',
    ]
    try:
        semantic_ok = all(float(row.get(k, 0)) >= 2 for k in good_keys)
    except Exception:
        semantic_ok = False
    bad_physical_flags = any(str(row.get(k, '')).lower() == 'true' for k in [
        'flag_critical_contact_invisible', 'flag_object_moves_without_contact',
        'flag_wrong_robot_or_missing_robot', 'flag_prompt_action_mismatch',
    ])
    visual_flag = str(row.get('flag_severe_flicker_or_exposure_jump', '')).lower() == 'true' or str(row.get('flag_severe_noise_or_compression', '')).lower() == 'true'
    evidence = str(row.get('vlm_evidence', '')).lower()
    mentions_only_visual = any(x in evidence for x in ['flicker', 'color shift', 'exposure', 'visual artifact', 'compression'])
    return semantic_ok and visual_flag and mentions_only_visual and not bad_physical_flags


def decide(row: pd.Series) -> tuple[str, str, bool]:
    rule_status = str(row.get('qc_status', '')).lower()
    vlm = str(row.get('vlm_decision', 'NEED_HUMAN_REVIEW')).upper()
    conf = float(row.get('vlm_confidence') or 0.0)
    hard = is_hard_fail(row)
    visual_artifact_only = vlm_visual_artifact_only(row)
    obvious_visual_hard = vlm_obvious_visual_hard_reject(row)
    strong_rule_reject = rule_status == 'reject'
    strong_vlm_reject = vlm == 'REJECT' and conf >= 0.75
    strong_vlm_pass = vlm == 'PASS' and conf >= 0.75
    conflict = (strong_rule_reject and strong_vlm_pass) or (rule_status == 'pass' and strong_vlm_reject and not obvious_visual_hard and not visual_artifact_only)
    if hard:
        return 'REJECT', 'deterministic_hard_fail', conflict
    if visual_artifact_only:
        return 'WARN_KEEP', 'vlm_visual_artifact_only_warn_keep', False
    if obvious_visual_hard:
        return 'REJECT', 'vlm_obvious_visual_hard_reject', False
    if conflict:
        return 'NEED_HUMAN_REVIEW', 'rule_vlm_conflict', True
    if strong_vlm_reject:
        return 'REJECT', 'vlm_reject_confident', False
    if vlm == 'DPO_LOSER_CANDIDATE' and conf >= 0.70:
        return 'DPO_LOSER_CANDIDATE', 'vlm_dpo_loser_confident', False
    if vlm == 'PASS':
        return 'PASS', 'vlm_pass_no_hard_fail', False
    if vlm == 'WARN_KEEP' or conf < 0.75:
        return 'WARN_KEEP', 'vlm_warn_or_low_confidence', False
    return 'NEED_HUMAN_REVIEW', 'unhandled_vlm_decision', False


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or ['episode_id'])
        w.writeheader()
        w.writerows(rows)


def report_text(df: pd.DataFrame, out: Path) -> str:
    counts = df['final_decision'].value_counts().to_dict() if not df.empty else {}
    matrix = pd.crosstab(df.get('qc_status', pd.Series(dtype=str)), df.get('vlm_decision', pd.Series(dtype=str)), dropna=False)
    evidence = Counter()
    dpo_evidence = Counter()
    for _, row in df.iterrows():
        texts = [x.strip() for x in str(row.get('vlm_evidence', '')).split(';') if x.strip()]
        if row.get('final_decision') == 'REJECT':
            evidence.update(texts[:2])
        if row.get('final_decision') == 'DPO_LOSER_CANDIDATE':
            dpo_evidence.update(texts[:2])
    lines = [
        '# v0.1 VLM QC Report',
        '',
        f'Total processed: {len(df)}',
        '',
        '## Final Decisions',
        '',
    ]
    for k in ['PASS', 'WARN_KEEP', 'REJECT', 'DPO_LOSER_CANDIDATE', 'NEED_HUMAN_REVIEW']:
        lines.append(f'- {k}: {counts.get(k, 0)}')
    lines += ['', '## Rule vs VLM Decision Matrix', '', matrix.to_markdown() if not matrix.empty else 'No rows.', '', f'Conflict cases: {int(df.get("is_conflict", pd.Series(dtype=bool)).sum())}', '', '## Top Reject Reasons', '']
    if evidence:
        lines += [f'- {k}: {v}' for k, v in evidence.most_common(12)]
    else:
        lines.append('- none')
    lines += ['', '## Top DPO Loser Reasons', '']
    if dpo_evidence:
        lines += [f'- {k}: {v}' for k, v in dpo_evidence.most_common(12)]
    else:
        lines.append('- none')
    lines += [
        '',
        '## Contact Sheets',
        '',
        '- `contact_sheets/vlm_PASS_samples.jpg`',
        '- `contact_sheets/vlm_WARN_KEEP_samples.jpg`',
        '- `contact_sheets/vlm_REJECT_samples.jpg`',
        '- `contact_sheets/vlm_DPO_LOSER_CANDIDATE_samples.jpg`',
        '- `contact_sheets/vlm_NEED_HUMAN_REVIEW_samples.jpg`',
        '',
        '## Recommended Use',
        '',
        '- SFT: use final PASS only.',
        '- A2V: use final PASS plus WARN_KEEP where `a2v_positive_suitability >= 1`.',
        '- DPO loser bank: use final DPO_LOSER_CANDIDATE.',
        '- REJECT: never use as positive SFT/A2V data.',
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser(description='Merge deterministic/rule QC with VLM QC')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--rule-qc', required=True)
    ap.add_argument('--vlm-results', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_table(Path(args.manifest))
    rule = pd.read_csv(args.rule_qc) if Path(args.rule_qc).exists() else pd.DataFrame(columns=['episode_id'])
    vlm_rows = [flatten_vlm(x) for x in read_jsonl(Path(args.vlm_results))]
    vlm = pd.DataFrame(vlm_rows)
    if vlm.empty:
        vlm = pd.DataFrame(columns=['episode_id', 'vlm_decision', 'vlm_confidence'])
    df = manifest.merge(rule, on='episode_id', how='left', suffixes=('', '_rule'))
    df = df.merge(vlm, on='episode_id', how='left')
    df['vlm_decision'] = df['vlm_decision'].fillna('NEED_HUMAN_REVIEW')
    df['vlm_confidence'] = df['vlm_confidence'].fillna(0.0)
    decisions = df.apply(decide, axis=1)
    df['final_decision'] = [x[0] for x in decisions]
    df['final_reason'] = [x[1] for x in decisions]
    df['is_conflict'] = [x[2] for x in decisions]

    score_cols = [f'score_{k}' for k in SCORE_KEYS]
    for c in score_cols:
        if c not in df.columns:
            df[c] = 1
    for c in [f'flag_{k}' for k in FLAG_KEYS]:
        if c not in df.columns:
            df[c] = False

    write_table(out / 'episode_manifest_qc_pass_vlm.parquet', df[df['final_decision'] == 'PASS'])
    write_table(out / 'episode_manifest_qc_warn_vlm.parquet', df[df['final_decision'].isin(['WARN_KEEP', 'NEED_HUMAN_REVIEW'])])
    write_table(out / 'episode_manifest_qc_reject_vlm.parquet', df[df['final_decision'] == 'REJECT'])
    df.to_csv(out / 'vlm_qc_scores.csv', index=False)
    df[df['final_decision'] == 'REJECT'].to_csv(out / 'rejected_episodes_vlm.csv', index=False)
    df[df['final_decision'] == 'DPO_LOSER_CANDIDATE'].to_csv(out / 'dpo_loser_candidates_vlm.csv', index=False)
    df[df['is_conflict']].to_csv(out / 'conflict_review.csv', index=False)

    for decision in ['PASS', 'WARN_KEEP', 'REJECT', 'DPO_LOSER_CANDIDATE', 'NEED_HUMAN_REVIEW']:
        make_contact_sheet(df[df['final_decision'] == decision], out / 'contact_sheets' / f'vlm_{decision}_samples.jpg', f'VLM {decision} samples')
    (out / 'vlm_qc_report.md').write_text(report_text(df, out), encoding='utf-8')
    print(f'wrote {out / "vlm_qc_scores.csv"} rows={len(df)}')


if __name__ == '__main__':
    main()
