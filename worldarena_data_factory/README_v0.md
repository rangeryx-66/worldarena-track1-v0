# WorldArena Track1 v0_smoke Data Factory

This project builds a staged data pipeline for WorldArena-style RoboTwin2 collection and ABot-PhysWorld exports. It is designed to dry-run safely first; long-running RoboTwin collection only starts with `run_robotwin_jobs.py --execute`.

## Paths

Default inputs:

- WorldArena root: `/root/autodl-tmp/worldarena_testset`
- WorldArena analysis: `/root/autodl-tmp/worldarena_testset/analysis_v2`
- Output root: `/root/autodl-tmp/worldarena_data_factory_v0`
- RoboTwin root: auto-detected from `--robotwin-root`, `ROBOTWIN_ROOT`, or `/root/autodl-tmp/RoboTwin`

Use RoboTwin Python for HDF5/video conversions:

```bash
PY=/root/autodl-tmp/conda/envs/RoboTwin/bin/python
```

## v0 Embodiment Policy

WorldArena Track1 is treated as a RoboTwin2 Clean-50 / Aloha-AgileX dual-arm gripper target domain. The evidence is consistent across WorldArena README, RoboTwin2 benchmark setting, RoboTwin2 paper descriptions of the 50 clean expert demonstrations, local WorldArena `joint14` HDF5 schema, and demo visuals showing an AgileX-style dual-arm gripper setup.

v0 is therefore optimized to score WorldArena, not to train a cross-embodiment world model. Formal v0 SFT, A2V, action library training rows, and DPO winners/losers default to `aloha-agilex` only. Other embodiments are reserved for v1/v2 robustness ablations and do not enter the v0 mainline.

All v0 positives must export WorldArena-compatible `joint14`: left_arm 6 + left_gripper 1 + right_arm 6 + right_gripper 1. Single-arm, dexterous hand, human hand, Franka, UR5, Piper, ARX-X5, or any non-matching action schema are rejected by default. A narrow `--include-secondary-embodiment` switch exists only for explicit ablation jobs and caps Piper/ARX-X5 at 5%; it is off by default.

## Optional Diagnostic Probe

`embodiment_probe.py` is optional diagnostics only. It can be useful for visual/action sanity checks, but it does not block or steer the v0 pipeline.

```bash
$PY worldarena_data_factory/create_robotwin_configs.py --out /root/autodl-tmp/worldarena_data_factory_v0 --robotwin-root /root/autodl-tmp/RoboTwin --write-probe-configs
$PY worldarena_data_factory/embodiment_probe.py --out /root/autodl-tmp/worldarena_data_factory_v0 --worldarena-root /root/autodl-tmp/worldarena_testset --robotwin-root /root/autodl-tmp/RoboTwin --dry-run
```

The probe outputs `embodiment_probe/embodiment_probe_manifest.csv`, `embodiment_probe/embodiment_probe_report.md`, and optional `contact_sheets/embodiment_probe_<embodiment>.jpg`.

## 1. Dry-run target/jobs/configs

```bash
$PY worldarena_data_factory/build_worldarena_target_spec.py --out /root/autodl-tmp/worldarena_data_factory_v0
$PY worldarena_data_factory/make_robotwin_collection_jobs.py --out /root/autodl-tmp/worldarena_data_factory_v0 --robotwin-root /root/autodl-tmp/RoboTwin --gpus 0
$PY worldarena_data_factory/create_robotwin_configs.py --out /root/autodl-tmp/worldarena_data_factory_v0 --robotwin-root /root/autodl-tmp/RoboTwin
$PY worldarena_data_factory/run_robotwin_jobs.py --out /root/autodl-tmp/worldarena_data_factory_v0 --robotwin-root /root/autodl-tmp/RoboTwin --dry-run --resume
```

Review `configs_to_apply/wa_*.yml`. To copy them into RoboTwin after review:

```bash
$PY worldarena_data_factory/create_robotwin_configs.py --out /root/autodl-tmp/worldarena_data_factory_v0 --robotwin-root /root/autodl-tmp/RoboTwin --apply
```

## 2. Execute Collection

Only run this when ready:

```bash
$PY worldarena_data_factory/run_robotwin_jobs.py --out /root/autodl-tmp/worldarena_data_factory_v0 --robotwin-root /root/autodl-tmp/RoboTwin --execute --resume --max-parallel-gpus 1
```

Each job logs to `logs/collect_<job_id>.log`. Missing RoboTwin tasks are written to `manifests/missing_tasks.csv`.

## 3. Convert Collected RoboTwin Assets

```bash
$PY worldarena_data_factory/convert_robotwin_to_manifest.py --out /root/autodl-tmp/worldarena_data_factory_v0 --robotwin-root /root/autodl-tmp/RoboTwin
```

The converter accepts only `aloha-agilex` by default and enforces `joint14` shape `(T,14)`, left/right arm and gripper fields, `T >= 60`, and no NaN/Inf. Rejected episodes are written to `rejected/rejected_episodes.csv` and do not enter SFT/A2V/action-library positives.

This creates `episode_manifest.parquet` when parquet dependencies exist, otherwise a CSV fallback plus sidecar note. It exports `joint14`, `joint14_norm`, `ee16`, and `joint14+ee16`; v0 default training uses `joint14`.

## 4. Export ABot Datasets

```bash
$PY worldarena_data_factory/export_abot_sft.py --out /root/autodl-tmp/worldarena_data_factory_v0
$PY worldarena_data_factory/export_abot_a2v.py --out /root/autodl-tmp/worldarena_data_factory_v0
```

SFT defaults to successful `aloha-agilex` videos only. ABot real replay is excluded unless `--include-real-replay` is explicitly passed. A2V defaults to `a2v_worldarena_joint14/metadata.jsonl`; `ee16` and `joint14+ee16` are exported for ablation, not the v0 default train path.

## 5. Build Action Library and Inference Manifests

This works even before RoboTwin collection because it can use existing WorldArena input HDF5 actions for inference retrieval. WorldArena val/test rows are marked `inference_only`; they are not training positives.

```bash
$PY worldarena_data_factory/build_action_library.py --worldarena-root /root/autodl-tmp/worldarena_testset --analysis /root/autodl-tmp/worldarena_testset/analysis_v2 --out /root/autodl-tmp/worldarena_data_factory_v0
$PY worldarena_data_factory/retrieve_actions_for_worldarena_variants.py --worldarena-root /root/autodl-tmp/worldarena_testset --analysis /root/autodl-tmp/worldarena_testset/analysis_v2 --out /root/autodl-tmp/worldarena_data_factory_v0
$PY worldarena_data_factory/package_inference_manifests.py --out /root/autodl-tmp/worldarena_data_factory_v0
```

## 6. Generate DPO Candidates

```bash
$PY worldarena_data_factory/generate_dpo_candidates.py --out /root/autodl-tmp/worldarena_data_factory_v0
```

v0 DPO winners and losers are expected to come from the Aloha-AgileX visual/action domain. Main DPO type is prompt-action mismatch. Wrong-embodiment DPO is disabled by default and only enabled with `--include-wrong-embodiment-dpo` for explicit ablations.

## 7. Validate

```bash
$PY worldarena_data_factory/validate_dataset_assets.py --out /root/autodl-tmp/worldarena_data_factory_v0
```

Reports:

- `v0_dataset_report.md`
- `v0_dataset_report.json`

## ABot Training Env Hints

SFT:

```bash
export DATASET_BASE_PATH=/root/autodl-tmp/worldarena_data_factory_v0
export DATASET_METADATA_PATH=/root/autodl-tmp/worldarena_data_factory_v0/sft_worldarena_style/metadata.jsonl
```

A2V joint14:

```bash
export DATASET_BASE_PATH=/root/autodl-tmp/worldarena_data_factory_v0
export DATASET_METADATA_PATH=/root/autodl-tmp/worldarena_data_factory_v0/a2v_worldarena_joint14/metadata.jsonl
```

DPO:

```bash
export DATASET_BASE_PATH=/root/autodl-tmp/worldarena_data_factory_v0
export DATASET_METADATA_PATH=/root/autodl-tmp/worldarena_data_factory_v0/dpo_prompt_action/dpo_pairs.jsonl
```
