# Manual QC Streamlit Workflow

This tool creates a 500-sample manual QC set and a simple Streamlit single-episode annotation UI.

Install Streamlit if needed:

```bash
/root/autodl-tmp/conda/envs/fantasyworld/bin/python -m pip install streamlit
```

## 1. Prepare 20 Test Samples

```bash
/root/autodl-tmp/conda/envs/fantasyworld/bin/python /root/autodl-tmp/worldarena_data_factory/prepare_manual_qc_500.py \
  --manifest /root/autodl-tmp/worldarena_data_factory_v0/manifests/episode_manifest.parquet \
  --out /root/autodl-tmp/worldarena_data_factory_v0/manual_qc_500 \
  --n-total 20 \
  --seed 42 \
  --force-regenerate
```

## 2. Start Annotation UI

```bash
/root/autodl-tmp/conda/envs/fantasyworld/bin/streamlit run /root/autodl-tmp/worldarena_data_factory/manual_qc_app.py -- \
  --csv /root/autodl-tmp/worldarena_data_factory_v0/manual_qc_500/manual_qc_500.csv \
  --out /root/autodl-tmp/worldarena_data_factory_v0/manual_qc_500/manual_qc_500_labeled.csv
```

PASS means suitable as SFT/A2V positive training data. REJECT means it should not be used as a positive sample. Do not reject just for white background, mild render grain, partial robot framing, gripper entering from outside the frame, large black/white robot area changes, or mild blur/noise.

Reject for broken/black videos, no task progress, object teleport/floating/disappearing, key object motion without visible or plausible robot contact, action/video mismatch, severe exposure/flicker, severe physics issues, or wrong domain/robot.

## 3. Generate Full 500 Samples

```bash
/root/autodl-tmp/conda/envs/fantasyworld/bin/python /root/autodl-tmp/worldarena_data_factory/prepare_manual_qc_500.py \
  --manifest /root/autodl-tmp/worldarena_data_factory_v0/manifests/episode_manifest.parquet \
  --out /root/autodl-tmp/worldarena_data_factory_v0/manual_qc_500 \
  --n-total 500 \
  --random-n 200 \
  --hard-n 300 \
  --seed 42 \
  --force-regenerate
```

## 4. Evaluate Labels

```bash
/root/autodl-tmp/conda/envs/fantasyworld/bin/python /root/autodl-tmp/worldarena_data_factory/evaluate_manual_qc.py \
  --labels /root/autodl-tmp/worldarena_data_factory_v0/manual_qc_500/manual_qc_500_labeled.csv \
  --out /root/autodl-tmp/worldarena_data_factory_v0/manual_qc_500/eval
```
