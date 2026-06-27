#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from v0_1_vlm_qc_common import read_jsonl
from merge_qc_with_vlm import flatten_vlm


def suggest_threshold(df: pd.DataFrame, decision: str, default: float) -> float:
    if df.empty:
        return default
    best = default
    for t in [x / 100 for x in range(50, 96, 5)]:
        pred = (df['vlm_decision'] == decision) & (df['vlm_confidence'] >= t)
        if pred.sum() == 0:
            continue
        precision = ((df['human_decision'] == decision) & pred).sum() / pred.sum()
        if precision >= 0.8:
            return t
        best = t
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate VLM QC calibration labels')
    ap.add_argument('--calibration', required=True)
    ap.add_argument('--vlm-results', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = pd.read_csv(args.calibration).fillna('')
    if 'human_decision' not in cal.columns:
        raise ValueError('calibration CSV must contain human_decision')
    vlm = pd.DataFrame([flatten_vlm(x) for x in read_jsonl(Path(args.vlm_results))])
    df = cal.merge(vlm, on='episode_id', how='inner')
    df = df[df['human_decision'].astype(str).str.len() > 0]
    matrix = pd.crosstab(df['human_decision'], df['vlm_decision'], dropna=False)
    matrix.to_csv(out / 'vlm_confusion_matrix.csv')
    reject_t = suggest_threshold(df, 'REJECT', 0.75)
    dpo_t = suggest_threshold(df, 'DPO_LOSER_CANDIDATE', 0.70)
    report = [
        '# VLM Calibration Report', '',
        f'Labeled samples matched to VLM results: {len(df)}', '',
        '## Confusion Matrix', '',
        matrix.to_markdown() if not matrix.empty else 'No labeled rows.', '',
        '## Suggested Thresholds', '',
        f'- reject_confidence_threshold: {reject_t:.2f}',
        f'- dpo_confidence_threshold: {dpo_t:.2f}', '',
        'Allowed human labels: PASS, WARN_KEEP, REJECT, DPO_LOSER_CANDIDATE, NEED_HUMAN_REVIEW.',
    ]
    (out / 'vlm_calibration_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(out / 'vlm_calibration_report.md')


if __name__ == '__main__':
    main()
