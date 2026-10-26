"""Reload saved models and evaluate every held-out view with shared metric definitions."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time

parser = argparse.ArgumentParser(description='Evaluate all views and hierarchy levels')
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--source',type=Path,help='Override saved dataset location')
args=parser.parse_args()
out=args.model.resolve()
config=json.loads((out/'run_config.json').read_text())
done=json.loads((out/'training_complete.json').read_text())
repo=Path(__file__).resolve().parents[1]
os.chdir(repo);sys.path.insert(0,str(repo))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import Scene, FlameGaussianModel
from gaussian_renderer import render
from lpipsPyTorch import LPIPS
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.general_utils import safe_state

safe_state(False)
cfg = argparse.Namespace(**config['args'])
cfg.model_path=str(out)
if args.source is not None:cfg.source_path=str(args.source.resolve())
mp = ModelParams(argparse.ArgumentParser())
op = OptimizationParams(argparse.ArgumentParser())
pp = PipelineParams(argparse.ArgumentParser())
dataset, opt, pipe = mp.extract(cfg), op.extract(cfg), pp.extract(cfg)
progressive = config['method'] == 'progressive'
if progressive:
    from scene import AdaptiveFlameGaussianModel, TriangleSplitMode
    model = AdaptiveFlameGaussianModel(
        dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params,
        split_mode=TriangleSplitMode.ONE_TO_THREE, max_level=dataset.adaptive_max_level)
    model.set_lod_include_descendants(True)
    scene = Scene(dataset, model, shuffle=False)
    state, step = torch.load(out / f"chkpnt{done['iterations']}.pth", map_location='cuda')
    assert step == done['iterations']
    model.restore(state, opt)
    assert torch.equal(model.geo.node_levels, state[1]['adaptive_geo']['node_levels'].cuda())
    for name, value in state[1]['flame_param'].items():
        assert torch.equal(model.flame_param[name], value.cuda()), name
    levels = model.available_levels()
else:
    model = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset,
                               dataset.not_finetune_flame_params)
    scene = Scene(dataset, model, load_iteration=done['iterations'], shuffle=False)
    levels = [None]

background = torch.ones(3, device='cuda')
criterion = LPIPS(net_type='alex', version='0.1').cuda().eval()


def render_at(camera, level):
    if progressive:
        return render(camera, model, pipe, background, level=level)['render'].clamp(0, 1)
    return render(camera, model, pipe, background)['render'].clamp(0, 1)


def png(path, tensor):
    # (3,H,W) -> (H,W,3)
    Image.fromarray((tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
                     * 255 + 0.5).astype(np.uint8)).save(path)


with torch.no_grad():
    reference = torch.load(out / 'reload_reference.pt', map_location='cpu')
    val = scene.getValCameras()
    idx = next(i for i, cam in enumerate(val.cameras) if cam.image_path == reference['image_path'])
    cam = val[idx]
    model.select_mesh_by_timestep(cam.timestep)
    finest = levels[-1]
    reload_image = render_at(cam, finest)
    reload_error = float((reload_image.cpu() - reference['image']).abs().max())
    assert reload_error < 2e-5, f'Checkpoint render mismatch: {reload_error}'
    (out / 'reload_verified_lpips_standard.json').write_text(json.dumps({
        'max_pixel_error': reload_error, 'iteration': done['iterations'],
        'gaussians': len(model._xyz), 'levels': levels,
    }, indent=2))

    result = {'method': config['method'], 'subject': config['subject'],
              'iterations': done['iterations'], 'smoke': config['smoke'],
              'metric_protocol': 'whole RGB image, white background; PSNR/SSIM [0,1]; LPIPS-Alex v0.1 [-1,1]; all LODs, finest for primary comparison',
              'results': {}, 'benchmark': {}}
    eval_dir = out / 'evaluation_lpips_standard'
    eval_dir.mkdir(exist_ok=True)
    for level in levels:
        label = 'baseline' if level is None else f'level_{level}'
        result['results'][label] = {}
        for split, cameras in [('val', scene.getValCameras()), ('test', scene.getTestCameras())]:
            sums = {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0}
            start = time.monotonic()
            sample_ids = {i * len(cameras) // 8 for i in range(8)}
            with (eval_dir / f'{label}_{split}.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=['index','image_path','timestep','psnr','ssim','lpips'])
                writer.writeheader()
                for i, camera in enumerate(DataLoader(cameras, batch_size=None, num_workers=4, shuffle=False)):
                    model.select_mesh_by_timestep(camera.timestep)
                    image = render_at(camera, level)
                    gt = camera.original_image.cuda().clamp(0, 1)
                    row = {'psnr': float(psnr(image, gt).mean()),
                           'ssim': float(ssim(image, gt)),
                           'lpips': float(criterion(image[None] * 2 - 1, gt[None] * 2 - 1).mean())}
                    assert all(np.isfinite(v) for v in row.values()), (label, split, i, row)
                    writer.writerow({'index': i, 'image_path': cameras.cameras[i].image_path,
                                     'timestep': camera.timestep, **row})
                    for key in sums:
                        sums[key] += row[key]
                    if i in sample_ids:
                        png(eval_dir / f'{label}_{split}_{i}_render.png', image)
                        if level == finest:
                            png(eval_dir / f'{split}_{i}_gt.png', gt)
                    if (i + 1) % 200 == 0:
                        print(f'EVAL_PROGRESS {label} {split} {i+1}/{len(cameras)}', flush=True)
            result['results'][label][split] = {
                **{key: value / len(cameras) for key, value in sums.items()},
                'views': len(cameras), 'seconds': time.monotonic() - start,
            }
            print('EVAL_RESULT ' + json.dumps({label: {split: result['results'][label][split]}}), flush=True)
            (out / 'evaluation_lpips_standard_progress.json').write_text(json.dumps(result, indent=2))

        # Disk I/O and metric networks are excluded from both timing protocols.
        camera = scene.getValCameras()[0]
        model.select_mesh_by_timestep(camera.timestep)
        for _ in range(20):
            render_at(camera, level)
        torch.cuda.synchronize()
        timings = {}
        for animated in [False, True]:
            times = []
            for _ in range(100):
                torch.cuda.synchronize()
                start = time.perf_counter()
                if animated:
                    model.select_mesh_by_timestep(camera.timestep)
                render_at(camera, level)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            tag = 'flame_plus_render' if animated else 'render_only'
            timings[tag] = {'mean_ms': statistics.mean(times),
                            'median_ms': statistics.median(times),
                            'fps_from_mean': 1000 / statistics.mean(times)}
        timings['gaussians'] = int(model.get_render_indices(level).numel()) if progressive else len(model._xyz)
        result['benchmark'][label] = timings
    result['status'] = 'evaluation_complete'
    (out / 'evaluation_lpips_standard_complete.json').write_text(json.dumps(result, indent=2))
    print('EVALUATION_COMPLETE ' + json.dumps(result), flush=True)
