# worldarena-track1-v0

WorldArena Track-1 leaderboard toolkit for base-model adaptation, synthetic trajectory generation, data filtering, LoRA fine-tuning, and benchmark evaluation.

This repository contains the code developed for a WorldArena Track1 v0 smoke pipeline:

- WorldArena val/test semantic, action, visual, and cross-modal analysis.
- RoboTwin2/Aloha-AgileX WorldArena-style data factory.
- SFT/A2V/DPO metadata export for ABot-PhysWorld.
- WorldArena val/test action retrieval manifests.
- ABot-PhysWorld compatibility patch used for local smoke tests.

Large datasets, videos, checkpoints, HDF5 files, parquet manifests, and training outputs are intentionally excluded from git.

## Main Local Paths Used

- WorldArena root: `/root/autodl-tmp/worldarena_testset`
- Analysis output: `/root/autodl-tmp/worldarena_testset/analysis_v2`
- Data factory output: `/root/autodl-tmp/worldarena_data_factory_v0`
- RoboTwin root: `/root/autodl-tmp/RoboTwin`
- ABot repo: `/root/autodl-tmp/ABot-PhysWorld`

## Current v0 Data Status

- Collected/converted RoboTwin-style successful episodes: 1500
- SFT metadata rows: 1500
- A2V joint14 metadata rows: 1500
- A2V ee16 metadata rows: 1500
- A2V joint14+ee16 metadata rows: 1500
- Formal DPO pairs: 0
- DPO smoke pair: 1

See `docs/v0_dataset_report.md` for the local validation report.

## ABot-PhysWorld Smoke Status

Smoke tests were run locally from:

`/root/autodl-tmp/ABot-PhysWorld/inference/checkpoints/amap_cvlab/Abot-PhysWorld/abotpw_i2v_480p.safetensors`

Validated locally:

- SFT LoRA smoke: passed
- A2V LoRA smoke: passed with ee16 action path
- DPO LoRA smoke: passed with one synthetic winner/loser pair

Important: current ABot A2V code still needs a proper `joint14 -> condition map` adapter before using joint14 as the primary production A2V representation. The local ee16 smoke proves the training chain, not final joint14 conditioning quality.

## Repository Layout

- `worldarena_analysis/`: dataset analysis scripts.
- `worldarena_data_factory/`: v0 target spec, RoboTwin job generation, conversion, ABot export, retrieval, validation.
- `abot_patches/`: patch for the local ABot-PhysWorld checkout used in smoke testing.
- `docs/`: lightweight reports copied from local outputs.
