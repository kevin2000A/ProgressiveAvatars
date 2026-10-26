# Progressive stream export and rendering

```bash
python scripts/export_stream.py --model output/avatar_final \
  --checkpoint output/avatar_final/chkpnt650000.pth \
  --subject avatar --output stream_assets/avatar \
  --score-views 48 --animation-sequence EXP-3 --animation-frames 120

python scripts/render_stream.py --assets stream_assets/avatar \
  --output output/avatar_demo --duration 12 --fps 30 --load-duration 10 --growth 3
```

Use `--source` on export/evaluation if the prepared dataset has moved. The selected animation sequence must have at least the requested number of frames. Choose a font with `--font /path/to/font.ttf` if DejaVu Sans is not installed.

## Contribution order

For each sampled training view, sum the rendered red-channel pixels and differentiate with respect to each Gaussian's red color. That derivative is the sum of alpha times accumulated transmittance over pixels. Average these scores across sampled training views, then order Gaussians by level first and descending contribution within each level. No reconstruction-loss gradient or ground-truth image is used for this score.

## Packet format

`header.pt` contains camera matrices, base mesh animation, hierarchy counts, and rendering metadata. Each `chunk_*.pt` contains only its Gaussian parameter tensors and the newly required subdivision vertex records. `manifest.json` records packet order and sizes. Base-layer packets contain up to 64 Gaussians; later packets up to 512. The exporter also writes reference renders used to verify full reconstruction, outside the streamed payload.

The receiver has no checkpoint dependency. It loads packets when their Gaussian-count budget permits, evaluates the received barycentric hierarchy for the current animation frame, and depth-sorts/rasterizes the received set normally. Existing Gaussian parameter values remain unchanged as packets arrive. The animation header is shared and preloaded; this demo is not a deployed network transport or an arbitrary-expression browser viewer.

The default count schedule is cubic during the first ten seconds, followed by two seconds at full detail. The bar is **received Gaussian count / total Gaussian count**. A separate footer reports bytes read. Animation is interpolated forward through the exported base-mesh sequence without ping-pong repetition.

The generated MP4 contains three synchronized viewpoints. `load_trace.json` records the active count and animation time for every frame. At completion the receiver checks full-loaded renders against the exported checkpoint references.
