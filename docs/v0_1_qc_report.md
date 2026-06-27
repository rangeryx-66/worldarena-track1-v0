# WorldArena v0.1 Video QC Report

Manifest: `/root/autodl-tmp/worldarena_data_factory_v0/manifests/episode_manifest.parquet`
Episodes checked: `1500`

## QC Status

- `pass`: `1038`
- `warn`: `409`
- `reject`: `53`

## Top Reasons

- `ok`: `1038`
- `color_shift`: `409`
- `temporal_flicker`: `129`
- `strong_color_or_exposure_jump`: `53`
- `strong_temporal_flicker`: `3`

## Metric Summary

- `brightness_mean`: mean=198.8116, p50=211.7874, p95=227.8316, max=231.8239
- `brightness_temporal_std`: mean=9.0668, p50=9.1961, p95=20.1850, max=28.3420
- `contrast_mean`: mean=52.6596, p50=57.1058, p95=71.1856, max=83.5106
- `color_shift_score`: mean=30.5970, p50=30.1111, p95=70.6676, max=155.7096
- `temporal_flicker_score`: mean=17.6088, p50=17.4463, p95=40.4028, max=75.4527
- `compression_artifact_score`: mean=0.1030, p50=0.0904, p95=0.2370, max=0.4317
- `sharpness_laplacian`: mean=88.0260, p50=89.8907, p95=116.9885, max=511.1406
- `arm_visible_ratio`: mean=0.9977, p50=1.0000, p95=1.0000, max=1.0000
- `end_effector_visible_ratio`: mean=0.9971, p50=1.0000, p95=1.0000, max=1.0000
- `contact_region_visible_ratio`: mean=0.9576, p50=1.0000, p95=1.0000, max=1.0000
- `object_motion_without_visible_contact_score`: mean=0.0003, p50=0.0000, p95=0.0000, max=0.1818
- `bad_frame_ratio`: mean=0.0000, p50=0.0000, p95=0.0000, max=0.0000
- `motion_score`: mean=17.1530, p50=16.1642, p95=35.4343, max=48.5368

## Notes

- White backgrounds, partially out-of-frame robot arms, and light render grain are treated as target-domain style and are not reject reasons by themselves.
- Arm/gripper/contact visibility uses image-processing proxies, not a learned detector. Warn/reject samples should be manually spot-checked with the contact sheets.
- Reject is reserved for unreadable videos, bad/black frames, strong exposure/color jumps, strong flicker, severe compression artifacts, or object motion without visible contact.
