# Training

Run commands from the repository root after installing dependencies and FLAME assets.

```bash
python train.py --source /path/to/prepared_subject --output output/avatar_final
```

One command runs the complete recipe: 600k steps from scratch, then automatically reloads that checkpoint and refines to 650k. No second training command is needed. The final checkpoint is `output/avatar_final/chkpnt650000.pth`; the intermediate stage is kept in `output/avatar_final/base/`. Rendering and evaluation are separate commands.

Use a new output directory for each run. The source checkpoint is preserved. Output includes `run_config.json`, JSONL logs, fixed-view preview images, checkpoints, and offline W&B logs. The default execution does not upload data to W&B.

## Release recipe

The fresh stage runs 600k steps, with maximum subdivision depth 5 and a scheduled budget of 130,144 Gaussians. It uses cumulative multi-level supervision, signed screen-space gradient scores corrected for level-loss weighting, optical-depth redistribution, and covariance-based child initialization. Parent Gaussians are retained.

The refinement stage continues to 650k. It adds MSE (weight 1.25) and LPIPS (weight 0.01) to the existing L1/DSSIM and geometric regularizers. Learning rates decay to one tenth of the restored values. Bounded gradient-based triangle subdivision is allowed until 620k, with at most 600 new Gaussians per event and a 150k ceiling. Thereafter, the existing representation is optimized without further splitting or opacity resets.

All supervised levels remain active. The release does not enable periodic rebinding, prefix freezing, a teacher, or additional within-triangle Gaussian splitting. The adaptive `densify_and_prune` entry point currently performs subdivision only; it does not implement hierarchical pruning. Covariance/opacity initialization does not imply exact pixel-invariance at a split.

These are the maintained implementation defaults, including the later refinement stage; do not assume they are identical to every setting described in the paper.

## Short integration run

```bash
python train.py --source /path/to/prepared_subject --output output/smoke --smoke
```

This runs 500 fresh steps followed automatically by 50 refinement steps, checking checkpoint restoration, finite losses, and final model saving. It is an integration check, not a convergence experiment.

## Evaluation

```bash
python scripts/evaluate.py --model output/avatar_final
```

The evaluator checks checkpoint render consistency and processes all validation/test images at each hierarchy level. Its metric protocol is whole-image RGB on white background, with LPIPS-Alex v0.1 inputs mapped to [-1, 1]. This differs from foreground-mask-only evaluation. Numeric benchmark results are intentionally not included in this repository.
