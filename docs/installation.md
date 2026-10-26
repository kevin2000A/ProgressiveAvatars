# Installation

Use Linux, Python 3.10, an NVIDIA CUDA GPU, a CUDA toolkit compatible with PyTorch, a C++ compiler, and FFmpeg. The training implementation uses CUDA extensions and is not a CPU-only package.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Example CUDA 12.1 wheel pair; use a toolkit compatible with your environment.
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./submodules/simple-knn
```

CUDA sources are vendored; a recursive Git clone is not required. The modified rasterizer installs as `diff_gaussian_rasterization_density`. Do not replace it with the unmodified upstream rasterizer: the screen-space gradient interface differs.

FFmpeg and a TrueType font are required for demo videos. On Ubuntu:

```bash
sudo apt-get install ffmpeg fonts-dejavu-core
```

## FLAME dependency

Obtain FLAME assets through the [official FLAME site](https://flame.is.tue.mpg.de/) and prepare an authorized [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars) checkout with its FLAME implementation and assets. These separately licensed files are not redistributed here.

```bash
python scripts/setup_flame.py --from-checkout /path/to/GaussianAvatars
```

The resulting `flame_model/*.py` and `flame_model/assets/` files are ignored by Git. See [required files](../flame_model/README.md). LPIPS downloads its publicly hosted network weights on first use; model caches and training checkpoints are not included in this repository.

## Check the environment

```bash
python -c 'import torch; import diff_gaussian_rasterization_density; import simple_knn._C; assert torch.cuda.is_available()'
python -m pytest tests -q
```

The integration tests require the dependencies above. A short end-to-end training check is documented in [training](training.md).
