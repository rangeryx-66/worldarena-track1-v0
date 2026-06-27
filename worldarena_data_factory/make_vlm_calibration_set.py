#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from v0_1_vlm_qc_common import make_contact_sheet, read_table


def main() -> None:
    ap = argparse.ArgumentParser(description='Build a human calibration sample set for VLM QC')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--rule-qc', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-per-class', type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_table(Path(args.manifest))
    rule = pd.read_csv(args.rule_qc)
    df = manifest.merge(rule, on='episode_id', how='left', suffixes=('', '_rule'))
    parts = []
    for status in ['pass', 'warn', 'reject']:
        sub = df[df['qc_status'].fillna('').astype(str).str.lower() == status]
        if not sub.empty:
            parts.append(sub.sample(min(args.n_per_class, len(sub)), random_state=41))
    sample = pd.concat(parts).drop_duplicates('episode_id') if parts else df.head(0)
    sample = sample.copy()
    sample['human_decision'] = ''
    sample.to_csv(out / 'calibration_samples.csv', index=False)
    make_contact_sheet(sample, out / 'contact_sheets' / 'calibration_samples.jpg', 'VLM calibration samples', n=len(sample))
    print(f'wrote {len(sample)} samples to {out / "calibration_samples.csv"}')


if __name__ == '__main__':
    main()
