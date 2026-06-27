# v0.1 VLM-Assisted Video QC

WorldArena v0.1 QC uses a hybrid rule-based + Qwen3-VL review flow. Deterministic failures remain hard rejects: unreadable video, black/broken frames, invalid fps/resolution, invalid action shape, action NaN/Inf. WorldArena-normal properties such as white background, light simulator render grain, and partially out-of-frame arms are only heuristic context for the VLM.

The default VLM mode is contact-sheet image input because it is stable and cheap to inspect. It does not modify original videos, expand the dataset, or start training.

## Dependencies

Recommended real VLM environment:

```bash
/root/autodl-tmp/conda/envs/fantasyworld/bin/python -m pip install qwen-vl-utils pyarrow
```

If those dependencies or the model are unavailable, use `--backend dummy --dry-run` to test the full file pipeline.

## 1. Conservative Rule QC

```bash
/root/autodl-tmp/conda/envs/RoboTwin/bin/python worldarena_data_factory/v0_1_qc_video_assets.py \
  --manifest /root/autodl-tmp/worldarena_data_factory_v0/episode_manifest.parquet \
  --out /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc \
  --workers 8 \
  --max-sampled-frames 12
```

The manifest path may also point to `/root/autodl-tmp/worldarena_data_factory_v0/manifests/episode_manifest.parquet`.

## 2. Calibration Set

```bash
/root/autodl-tmp/conda/envs/RoboTwin/bin/python worldarena_data_factory/make_vlm_calibration_set.py \
  --manifest /root/autodl-tmp/worldarena_data_factory_v0/episode_manifest.parquet \
  --rule-qc /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc/qc_scores.csv \
  --out /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc_vlm
```

Fill `human_decision` manually if you want threshold calibration.

## 3. Dry-Run Dummy VLM

```bash
/root/autodl-tmp/conda/envs/RoboTwin/bin/python worldarena_data_factory/v0_1_vlm_qc.py \
  --manifest /root/autodl-tmp/worldarena_data_factory_v0/episode_manifest.parquet \
  --rule-qc /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc/qc_scores.csv \
  --out /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc_vlm_dummy \
  --backend dummy \
  --max-samples 5 \
  --dry-run \
  --resume
```

## 4. Real Qwen3-VL Run

```bash
/root/autodl-tmp/conda/envs/fantasyworld/bin/python worldarena_data_factory/v0_1_vlm_qc.py \
  --manifest /root/autodl-tmp/worldarena_data_factory_v0/episode_manifest.parquet \
  --rule-qc /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc/qc_scores.csv \
  --out /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc_vlm \
  --model qwen3-vl \
  --model-path /root/autodl-tmp/qwen3vl8b \
  --mode contact_sheet \
  --num-frames 16 \
  --batch-size 1 \
  --run-all \
  --resume
```

For a small sanity test first, add `--max-samples 3`.

## 5. Merge Rule + VLM

```bash
/root/autodl-tmp/conda/envs/RoboTwin/bin/python worldarena_data_factory/merge_qc_with_vlm.py \
  --manifest /root/autodl-tmp/worldarena_data_factory_v0/episode_manifest.parquet \
  --rule-qc /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc/qc_scores.csv \
  --vlm-results /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc_vlm/vlm_qc_results.jsonl \
  --out /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc_vlm
```

Main outputs are `vlm_qc_scores.csv`, pass/warn/reject parquet files, `dpo_loser_candidates_vlm.csv`, `conflict_review.csv`, and `vlm_qc_report.md`.

## 6. Optional Calibration Evaluation

```bash
/root/autodl-tmp/conda/envs/RoboTwin/bin/python worldarena_data_factory/evaluate_vlm_calibration.py \
  --calibration /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc_vlm/calibration_samples.csv \
  --vlm-results /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc_vlm/vlm_qc_results.jsonl \
  --out /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc_vlm
```

## 7. Re-Export v0.1 Metadata

```bash
/root/autodl-tmp/conda/envs/RoboTwin/bin/python worldarena_data_factory/export_abot_sft.py \
  --out /root/autodl-tmp/worldarena_data_factory_v0 \
  --qc-source vlm \
  --vlm-qc-dir /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc_vlm

/root/autodl-tmp/conda/envs/RoboTwin/bin/python worldarena_data_factory/export_abot_a2v.py \
  --out /root/autodl-tmp/worldarena_data_factory_v0 \
  --qc-source vlm \
  --vlm-qc-dir /root/autodl-tmp/worldarena_data_factory_v0/v0_1_qc_vlm
```

This creates:

- `sft_worldarena_style_v0_1_vlm/metadata.jsonl`: final PASS only.
- `a2v_worldarena_ee16_v0_1_vlm/metadata.jsonl`: final PASS plus WARN_KEEP with `a2v_positive_suitability >= 1`.
- `dpo_loser_bank_v0_1_vlm.csv`: final DPO_LOSER_CANDIDATE.

v0.1 intentionally exports formal A2V validation metadata for `ee16`; joint14 A2V is held back until the ABot condition map path is fixed.
