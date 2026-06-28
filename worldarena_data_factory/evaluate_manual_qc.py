#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from collections import Counter
import pandas as pd


def norm_current(x):
    s = str(x).upper()
    if s in {"PASS"}:
        return "PASS"
    if s in {"REJECT", "DPO_LOSER", "DPO_LOSER_CANDIDATE"}:
        return "REJECT"
    return "WARN"


def metrics(df):
    y = df["human_label"].astype(str).str.upper()
    p = df["current_pred"].astype(str).str.upper()
    tp = ((p == "PASS") & (y == "PASS")).sum()
    fp = ((p == "PASS") & (y == "REJECT")).sum()
    fn = ((p == "REJECT") & (y == "PASS")).sum()
    tn = ((p == "REJECT") & (y == "REJECT")).sum()
    total = max(len(df), 1)
    pass_prec = tp / max(tp + fp, 1)
    pass_rec = tp / max(tp + fn, 1)
    rej_prec = tn / max(tn + fn, 1)
    rej_rec = tn / max(tn + fp, 1)
    return {
        "n": len(df),
        "overall_accuracy": (tp + tn) / total,
        "pass_precision": pass_prec,
        "pass_recall": pass_rec,
        "reject_precision": rej_prec,
        "reject_recall": rej_rec,
        "balanced_accuracy": (pass_rec + rej_rec) / 2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.labels).fillna("")
    labeled = df[
        df["human_label"].astype(str).str.upper().isin(["PASS", "REJECT"])
    ].copy()
    labeled["current_pred"] = labeled.get("current_qc_status", "unknown").map(
        norm_current
    )
    labeled["human_label"].value_counts().rename_axis("human_label").reset_index(
        name="count"
    ).to_csv(out / "human_label_counts.csv", index=False)
    reasons = Counter()
    for v in labeled.get("human_reason", pd.Series(dtype=str)).astype(str):
        for x in v.replace("|", ",").replace(";", ",").split(","):
            x = x.strip()
            if x:
                reasons[x] += 1
    pd.DataFrame(
        [{"human_reason": k, "count": v} for k, v in reasons.most_common()]
    ).to_csv(out / "human_reason_counts.csv", index=False)
    pd.crosstab(labeled["current_pred"], labeled["human_label"], dropna=False).to_csv(
        out / "current_qc_vs_human_confusion.csv"
    )
    labeled[
        (labeled["current_pred"] == "PASS")
        & (labeled["human_label"].str.upper() == "REJECT")
    ].to_csv(out / "false_pass_cases.csv", index=False)
    labeled[
        (labeled["current_pred"] == "REJECT")
        & (labeled["human_label"].str.upper() == "PASS")
    ].to_csv(out / "false_reject_cases.csv", index=False)
    labeled[
        pd.to_numeric(labeled.get("human_confidence", 0), errors="coerce").fillna(0)
        <= 1
    ].to_csv(out / "low_confidence_cases.csv", index=False)
    overall = metrics(labeled) if len(labeled) else {"n": 0}
    lines = [
        "# Manual QC Evaluation",
        "",
        f"Total rows: `{len(df)}`",
        f"Labeled rows: `{len(labeled)}`",
        "",
        "## Overall Metrics",
        "",
    ]
    for k, v in overall.items():
        lines.append(
            f"- `{k}`: `{v:.4f}`" if isinstance(v, float) else f"- `{k}`: `{v}`"
        )
    lines += ["", "## By Sample Group", ""]
    for group, g in labeled.groupby("sample_group"):
        m = metrics(g)
        lines.append(f"### {group}")
        lines += [
            f"- `{k}`: `{v:.4f}`" if isinstance(v, float) else f"- `{k}`: `{v}`"
            for k, v in m.items()
        ]
        lines.append("")
    (out / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "eval_report.md")


if __name__ == "__main__":
    main()
