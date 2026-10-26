# Data preparation

Obtain NeRSemble or another compatible multiview avatar dataset under its own terms. Raw images, tracked subject parameters, and checkpoints are not distributed in this repository. Use the GaussianAvatars / [VHAP](https://github.com/ShenhanQian/VHAP) tracking and export workflow to prepare a subject.

The input directory contains:

```text
prepared_subject/
  transforms_train.json
  transforms_val.json
  transforms_test.json
  canonical_flame_param.npz
```

Frame records reference images and per-frame FLAME parameter files, commonly in sibling sequence directories. Preserve their relative paths when moving the dataset. Records contain `file_path`, `flame_param_path`, `timestep_index`, camera calibration, and camera identity.

The entry points expect disjoint training/validation/test image paths. Validation contains held-out camera views at training timesteps; test contains held-out expression timesteps. The streaming exporter uses training views for its contribution scores and does not use ground-truth image errors for ranking.
