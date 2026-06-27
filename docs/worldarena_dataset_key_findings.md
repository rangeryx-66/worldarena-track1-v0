# WorldArena Dataset Key Findings

Source reports:

- `/root/autodl-tmp/worldarena_testset/worldarena_dataset_report.md`
- `/root/autodl-tmp/worldarena_testset/worldarena_dataset_report.json`

## Dataset Scale

- Both `val_dataset` and `test_dataset` contain only one task: `fixed_scene_task`.
- `val_dataset`: 500 episodes, numbered 1-500.
- `test_dataset`: 1000 episodes, numbered 1-1000.
- Each episode is fully aligned across `data`, `first_frame`, `instructions`, `instructions_1`, and `instructions_2`.
- No missing episodes were found in any modality.

## HDF5 Structure

1500 HDF5 files were analyzed with no read errors.

Fields present in every file:

- `/endpose/left_endpose`: `(T, 7) float64`
- `/endpose/right_endpose`: `(T, 7) float64`
- `/endpose/left_gripper`: `(T,) float64`
- `/endpose/right_gripper`: `(T,) float64`
- `/joint_action/left_arm`: `(T, 6) float64`
- `/joint_action/right_arm`: `(T, 6) float64`
- `/joint_action/left_gripper`: `(T,) float64`
- `/joint_action/right_gripper`: `(T,) float64`
- `/joint_action/vector`: `(T, 14) float64`
- `/pointcloud`: `(T, 0) float64`

No HDF5 attributes were found.

## Action And Trajectory Lengths

- Main action vector: `/joint_action/vector`, shape `(T, 14)`.
- The 14 action dimensions are consistent with two arms:
  - left arm: 6
  - left gripper: 1
  - right arm: 6
  - right gripper: 1
- Trajectory length `T` is variable:
  - min: 74
  - median: 164
  - mean: 214.34
  - max: 1074
  - 90th percentile: 445
  - 95th percentile: 474
  - 99th percentile: 669

## Frames And Videos

- First frames:
  - `val_dataset`: 500 images, all `320x240`, RGB.
  - `test_dataset`: 1000 images, all `320x240`, RGB.
- Example videos:
  - `example_val`, `example_val_1`, `example_val_2`: 500 files each, all valid in the report.
  - Video format: `640x480`, MPEG-4, 24 fps, about 5.0417 seconds.
  - `example_test`, `example_test_1`, `example_test_2`: 1000 files each, but only 634 files per directory were readable by `ffprobe`; unreadable files report `moov atom not found`.
- `convert_vscode_visible` was also detected as a video directory. It appears to be an extra converted/preview directory, not part of the original submission examples.

## Embodiment Appearance

The dominant instruction prefix describes the embodiment as:

> a rigid, physically consistent embodied robotic arm in a fixed robotic workspace, maintaining high stability with no deformation.

Contact sheets were generated for visual inspection:

- `/root/autodl-tmp/worldarena_testset/contact_sheets/test_dataset__fixed_scene_task.jpg`
- `/root/autodl-tmp/worldarena_testset/contact_sheets/val_dataset__fixed_scene_task.jpg`

## Instruction Variants

Each episode has three instruction sets:

- `instructions`
- `instructions_1`
- `instructions_2`

None of the corresponding instruction triples are exactly identical.

`instructions_1` and `instructions_2` are generally longer and describe different actions or object manipulations while keeping the same embodiment/workspace prefix.

### Validation Set

- Counts: 500 per instruction set.
- Exact equality:
  - base vs 1: 0
  - base vs 2: 0
  - 1 vs 2: 0
- Average similarity:
  - base vs 1: 0.7096
  - base vs 2: 0.6639
  - 1 vs 2: 0.6073
- Average word counts:
  - `instructions`: 41.97
  - `instructions_1`: 51.41
  - `instructions_2`: 53.65

### Test Set

- Counts: 1000 per instruction set.
- Exact equality:
  - base vs 1: 0
  - base vs 2: 0
  - 1 vs 2: 0
- Average similarity:
  - base vs 1: 0.7004
  - base vs 2: 0.6629
  - 1 vs 2: 0.6030
- Average word counts:
  - `instructions`: 42.77
  - `instructions_1`: 52.39
  - `instructions_2`: 54.37

## Practical Takeaways

- This split is structurally clean: no missing modalities, no missing episodes.
- For action-conditioned generation, use `/joint_action/vector` as the compact 14-D action sequence.
- Models must handle variable-length action trajectories; padding/truncation/resampling should be explicit.
- For image-conditioned generation, the provided first frame is `320x240`, while example output videos are `640x480`.
- For text-conditioned generation, treat `instructions`, `instructions_1`, and `instructions_2` as three materially different prompts per episode, not as duplicate paraphrases.
- For submission-format checks, use the `example_*` directory names and episode naming pattern, but do not treat all example test videos as valid visual references because some have broken MP4 metadata.
