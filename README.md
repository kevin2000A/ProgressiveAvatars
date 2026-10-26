<div align="center">

# ProgressiveAvatars
### Progressive Animatable 3D Gaussian Avatars

**CVPR 2026**

Kaiwen Song · Jinkai Cui · Juyong Zhang  
University of Science and Technology of China

[Project Page](https://ustc3dv.github.io/ProgressiveAvatars/) · [Paper](https://arxiv.org/abs/2603.16447) · [Installation](docs/installation.md) · [Training](docs/training.md) · [Progressive Rendering](docs/streaming.md)

</div>

<table>
  <tr>
    <td width="50%" align="center"><img src="assets/demo/avatar_218.gif" alt="Avatar 218: contribution-ranked incremental rendering" width="100%"></td>
    <td width="50%" align="center"><img src="assets/demo/avatar_253.gif" alt="Avatar 253: contribution-ranked incremental rendering" width="100%"></td>
  </tr>
</table>

Each avatar is shown from three synchronized views. The progress bars show the fraction of Gaussians loaded; loading starts slowly and accelerates as finer details arrive.

ProgressiveAvatars represents an animatable head as a hierarchy of mesh-bound 3D Gaussians. Adaptive triangle subdivision adds detail where it is needed. At inference time, the avatar can be delivered progressively: a coarse representation appears first, and incoming Gaussians refine the same asset.

## What is included

- Adaptive triangle subdivision with face-local Gaussian parameters and cumulative levels of detail.
- Multi-level training, covariance-based child initialization, and opacity redistribution when splitting.
- Budgeted growth followed by a bounded refinement stage.
- Rendering-contribution ranking based on accumulated **alpha × transmittance** across training views.
- Packet export, independent incremental loading, and multi-view video rendering with a Gaussian-count progress bar.

This release packages the maintained training and streaming implementation. See [the training recipe](docs/training.md) for the release defaults and their scope.

## Quick start

Follow [installation and asset setup](docs/installation.md), then train on a prepared dataset:

```bash
python train.py --source /path/to/prepared_subject --output output/avatar_final
```

Training automatically runs both stages and saves the final model in `output/avatar_final`. Export and render a progressive stream separately:

```bash
python scripts/export_stream.py --model output/avatar_final \
  --checkpoint output/avatar_final/chkpnt650000.pth \
  --subject avatar --output stream_assets/avatar

python scripts/render_stream.py --assets stream_assets/avatar \
  --output output/avatar_demo --duration 12 --fps 30 --load-duration 10 --growth 3
```

The export command defaults to the `EXP-3` sequence; select another prepared sequence with `--animation-sequence`. Details: [data preparation](docs/data.md), [training](docs/training.md), [streaming](docs/streaming.md).

## Repository layout

```text
train.py                  Complete training pipeline (fresh + refinement)
training/                 Fresh and refinement optimization loops
scene/                    Mesh hierarchy, Gaussians, and datasets
scripts/                  Asset setup, evaluation, stream export and rendering
submodules/               Vendored CUDA extension source
assets/demo/              Incremental-rendering GIFs
```

## Acknowledgments and licenses

Built upon [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), FLAME, and their dependencies. Original notices are retained.

The inherited [GaussianAvatars license](LICENSE.md) and [Gaussian Splatting license](LICENSE_GS.md) include **non-commercial / research-use restrictions**; this repository does not replace them with a permissive license. FLAME code/assets and datasets must be obtained separately under their own terms. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@inproceedings{song2026progressiveavatars,
  title     = {ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars},
  author    = {Song, Kaiwen and Cui, Jinkai and Zhang, Juyong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
