# Third-party notices

This is a derivative research implementation, not a replacement license for upstream components. Preserve the original copyright headers and license files when redistributing source.

- **GaussianAvatars**: https://github.com/ShenhanQian/GaussianAvatars. Its CC-BY-NC-SA-4.0 license is retained in `LICENSE.md`. Original Toyota Motor Europe notices remain in the inherited Gaussian/FLAME binding code.
- **3D Gaussian Splatting and Gaussian rasterization**: https://github.com/graphdeco-inria/gaussian-splatting and https://github.com/graphdeco-inria/diff-gaussian-rasterization. The inherited research/non-commercial terms are retained in `LICENSE_GS.md` and the rasterizer directory. Our changes include the screen-space gradient interface and hierarchical density-control integration.
- **simple-knn**: https://gitlab.inria.fr/bkerbl/simple-knn. Vendored source retains its original notices; it is used by the Gaussian Splatting implementation.
- **GLM**: vendored under the rasterizer's `third_party/glm`; retain its original license files and headers.
- **LPIPS**: the inherited `lpipsPyTorch` wrapper loads the published perceptual-network weights from https://github.com/richzhang/PerceptualSimilarity. Pretrained weights are not bundled.
- **FLAME / SMPL-X-related implementation and assets**: obtained separately under their applicable terms. Neither FLAME implementation files nor model assets are distributed here. The local setup helper only copies files the user already obtained.
- **nvdiffrast, PyTorch, torchvision, RoMa, and other Python dependencies**: installed separately and governed by their own licenses.

The project title, authorship, citation and project link follow https://ustc3dv.github.io/ProgressiveAvatars/. The README GIFs are newly generated qualitative demonstrations from the project's trained avatars. Raw NeRSemble data and model checkpoints are not included.
