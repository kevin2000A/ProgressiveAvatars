# Separately licensed FLAME dependency

FLAME implementation files and model assets are not redistributed in this repository. Obtain them under their respective licenses and prepare a GaussianAvatars checkout following its data setup instructions. Then run:

```bash
python scripts/setup_flame.py --from-checkout /path/to/GaussianAvatars
```

Required files include `flame.py`, `lbs.py`, and `assets/flame/{flame2023.pkl,FLAME_masks.pkl,head_template_mesh.obj,landmark_embedding_with_eyes.npy}`. The setup script copies the existing asset directory; it does not download assets or grant any additional rights. Installed files are Git-ignored.

Sources: [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars), [FLAME](https://flame.is.tue.mpg.de/).
