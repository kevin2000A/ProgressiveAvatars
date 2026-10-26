"""Training entry point for the ProgressiveAvatars release recipe."""
import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
cli = argparse.ArgumentParser(description="Train ProgressiveAvatars with the release recipe")
cli.add_argument('--source', type=Path, required=True, help='Prepared dataset directory')
cli.add_argument('--output', type=Path, required=True, help='New output directory')
cli.add_argument('--subject', default='avatar')
cli.add_argument('--smoke', action='store_true', help='Run a 500-step fresh-training check')
cli.add_argument('--checkpoint', type=Path, help='Refine a 600k checkpoint; omit to train from scratch')
cli.add_argument('--iterations', type=int, default=650000, help='Final iteration for checkpoint refinement (default: 650000)')
args = cli.parse_args()
refining = args.checkpoint is not None
checkpoint = str(args.checkpoint.resolve()) if refining else None
start_iteration = (500 if args.smoke else 600000) if refining else 0
args.source = args.source.resolve()
args.output = args.output.resolve()
args.method = 'progressive'
args.variant = 'initialization'
args.run_name = args.output.name
args.ramp_start = 600000
args.ramp_end = 625000
repo = ROOT
os.chdir(repo)
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_DIR'] = str(args.output.parent)
os.environ['WANDB_DISABLE_GIT'] = 'true'
args.end_iteration = args.iterations if refining else (500 if args.smoke else 600000)
args.lambda_perceptual = 0.01 if refining else 0.0
args.max_depth = 5
args.lr_final_factor = 0.1

import numpy as np
from PIL import Image
import torch
import wandb
if refining:
    from training import refine as trainer
else:
    from training import fresh as trainer
from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state
from utils.image_utils import psnr

out = args.output
out.mkdir(parents=True, exist_ok=refining)
iterations = args.end_iteration
source = args.source
parser = argparse.ArgumentParser()
mp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
common = [
    '-s', str(source), '-m', str(out), '--eval', '--bind_to_mesh', '--white_background',
    '--resolution', '1', '--iterations', str(iterations),
    '--position_lr_max_steps', '600000',
    '--densification_interval', '50' if args.smoke else '2000',
    '--densify_from_iter', '50' if args.smoke else '10000',
    '--densify_until_iter', '450' if args.smoke else '600000',
    '--densify_grad_threshold', '0' if args.smoke else '0.0002',
    '--lambda_xyz', '0.01', '--lambda_scale', '1.0',
]
if args.method == 'progressive':
    common += [
        '--adaptive_mesh', '--adaptive_split_mode', '1to3',
        '--adaptive_max_level', str(args.max_depth),
        '--progressive_adaptive_max_level_period', '50' if args.smoke else '50000',
        '--densify_grad_threshold_base', '0' if args.smoke else '0.0002',
        '--densify_grad_threshold_decay', '1.0',
        '--visualize_interval', '0', '--render_visualize_interval', '0',
        '--extra_checkpoint_interval', '0',
    ]
common += ['--supervision_mode', 'multi', '--supervision_ramp_start', str(args.ramp_start), '--supervision_ramp_end', str(args.ramp_end)]
common += ['--density_score', 'absolute' if args.variant in ['score','both'] else 'signed',
           '--child_initialization', 'covariance' if args.variant in ['initialization','both'] else 'original',
           '--density_budget', '10264' if args.smoke else '130144',
           '--density_budget_start', '100' if args.smoke else '50000',
           '--density_budget_end', '400' if args.smoke else '598000']
common += ['--lambda_perceptual', str(args.lambda_perceptual)]
if refining:
    common += [
        '--lambda_dssim', '0.2', '--lambda_mse', '1.25',
        '--densify_from_iter', str(start_iteration),
        '--densify_until_iter', str(start_iteration + (20 if args.smoke else 20000)),
        '--densification_interval', '10' if args.smoke else '2000',
        '--densify_grad_threshold_base', '0' if args.smoke else '0.00002',
        '--opacity_reset_interval', '100000000',
    ]
cfg = parser.parse_args(common)
cfg.late_split = refining
if refining:
    cfg.natural_growth_fraction = .02
    cfg.natural_ceiling = 150000
    cfg.natural_max_new_points = 600
    cfg.lr_final_factor = args.lr_final_factor
(out / 'run_config.json').write_text(json.dumps({
    'method': args.method, 'subject': args.subject, 'smoke': args.smoke, 'variant': args.variant,
    'repo': str(repo), 'argv': common, 'args': vars(cfg), 'seed': 0,
    'tracking': 'wandb offline plus JSONL and PNG; no hashes',
    'source_checkpoint': checkpoint, 'start_iteration': start_iteration, 'initialization_seed': 0,
    'supervised_levels': [0,1,2,3,'finest'],
}, indent=2))

splits = {s: json.loads((source / f'transforms_{s}.json').read_text())['frames']
          for s in ['train', 'val', 'test']}
paths = {s: {os.path.normpath(str(source / f['file_path'])) for f in frames}
         for s, frames in splits.items()}
assert not paths['train'] & paths['val']
assert not paths['train'] & paths['test']
assert not paths['val'] & paths['test']
times = {s: {f['timestep_index'] for f in frames} for s, frames in splits.items()}
assert not times['train'] & times['test']
assert times['val'] <= times['train']
(out / 'split_audit.json').write_text(json.dumps({
    s: {'views': len(splits[s]), 'timesteps': len(times[s])} for s in splits
}, indent=2))

safe_state(False)
run = wandb.init(project='ProgressiveAvatars', name=args.run_name,
                 config={'method': args.method, 'subject': args.subject,
                         'smoke': args.smoke, **vars(cfg)},
                 mode='offline', save_code=False, settings=wandb.Settings(disable_git=True))
metrics_file = (out / 'training_metrics.jsonl').open('a', buffering=1)
state = {}
started = time.monotonic()


def save_image(path, tensor):
    # (3,H,W) -> (H,W,3)
    a = (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    Image.fromarray(a).save(path)


def report(writer, iteration, losses, elapsed, testing_iterations, scene, render_func, render_args):
    state['scene'] = scene
    assert torch.isfinite(losses['total']).item(), f'Nonfinite loss at {iteration}'
    if iteration == 1 or iteration % 100 == 0 or iteration == iterations:
        row = {'iteration': iteration, 'wall_seconds': time.monotonic() - started,
               'iteration_ms': elapsed, 'gaussians': len(scene.gaussians._xyz),
               **{f'loss/{k}': float(v.detach()) for k, v in losses.items()}}
        if refining:
            row['lr_factor'] = args.lr_final_factor ** ((iteration - start_iteration - 1) / (iterations - start_iteration - 1))
            row['learning_rates'] = {g['name']: g['lr'] for g in scene.gaussians.optimizer.param_groups}
        if args.method == 'progressive':
            row['max_level'] = int(scene.gaussians.geo.node_levels.max())
            if iteration % 1000 == 0 or iteration == iterations:
                opacity = scene.gaussians.get_opacity[:,0]
                row['inactive_by_level'] = [int(((scene.gaussians.geo.node_levels==level)&(opacity<1/255)).sum()) for level in range(row['max_level']+1)]
                row['nodes_by_level'] = torch.bincount(scene.gaussians.geo.node_levels.long()).tolist()
            from utils.lod_supervision import level_loss_weights
            row['level_weights'] = level_loss_weights(
                min(5,len(scene.gaussians.available_levels())), cfg.level_loss_decay,
                iteration, 'multi', args.ramp_start, args.ramp_end)
        metrics_file.write(json.dumps(row) + '\n')
        run.log(row, step=iteration)
        if iteration == 1 or iteration % 1000 == 0 or iteration == iterations:
            print('PROGRESS ' + json.dumps(row), flush=True)
    if iteration % (100 if args.smoke else 5000) == 0 or iteration == iterations:
        timestep = scene.gaussians.timestep
        cams = scene.getValCameras()
        ids = sorted(range(len(cams)), key=lambda i: cams.cameras[i].image_path)
        chosen = [ids[i * len(ids) // 4] for i in range(4)]
        visual_dir = out / 'visuals' / str(iteration)
        visual_dir.mkdir(parents=True, exist_ok=True)
        vals = []
        for j, idx in enumerate(chosen):
            camera = cams[idx]
            scene.gaussians.select_mesh_by_timestep(camera.timestep)
            image = render_func(camera, scene.gaussians, *render_args)['render'].clamp(0, 1)
            gt = camera.original_image.cuda().clamp(0, 1)
            save_image(visual_dir / f'{j}_render.png', image)
            save_image(visual_dir / f'{j}_gt.png', gt)
            vals.append(float(psnr(image, gt).mean()))
            if iteration == iterations and j == 0:
                torch.save({'image': image.cpu(), 'image_path': camera.image_path},
                           out / 'reload_reference.pt')
        run.log({'preview/psnr_4_fixed_views': sum(vals) / len(vals)}, step=iteration)
        scene.gaussians.select_mesh_by_timestep(timestep)


trainer.training_report = report
trainer.network_gui.init('127.0.0.1', 0)
save_interval = 10000 if refining else 60000
save_steps = sorted(set(range(start_iteration + save_interval, iterations + 1, save_interval)) | {iterations})
kwargs = {'lr_final_factor': args.lr_final_factor} if refining else {}
if args.method == 'progressive':
    kwargs['train_levels'] = [0, 1, 2, 3, None]
safe_state(False)
train_opt = op.extract(cfg)
train_opt.root_position = False
train_opt.root_scale = False
train_opt.late_split = cfg.late_split
if refining:
    train_opt.natural_growth_fraction = cfg.natural_growth_fraction
    train_opt.natural_ceiling = cfg.natural_ceiling
    train_opt.natural_max_new_points = cfg.natural_max_new_points
trainer.training(mp.extract(cfg), train_opt, pp.extract(cfg), [], save_steps,
                 save_steps, checkpoint, -1, **kwargs)
model = state['scene'].gaussians
if refining:
    assert len(model._xyz) <= cfg.natural_ceiling
    if not args.smoke:
        assert int(model.geo.node_levels.max()) == 5
else:
    assert len(model._xyz) == cfg.density_budget
assert (out / f'chkpnt{iterations}.pth').is_file()
assert (out / f'point_cloud/iteration_{iterations}/point_cloud.ply').is_file()
if args.method == 'progressive':
    assert int(model.geo.node_levels.max()) > 0, 'Adaptive training never subdivided'
receipt = {'status': 'training_complete', 'method': args.method, 'subject': args.subject,
           'iterations': iterations, 'gaussians': len(model._xyz),
           'wall_seconds': time.monotonic() - started, 'smoke': args.smoke,
           'source_checkpoint': checkpoint, 'additional_iterations': iterations - start_iteration, 'variant': args.variant}
(out / 'training_complete.json').write_text(json.dumps(receipt, indent=2))
run.finish()
metrics_file.close()
print('TRAINING_COMPLETE ' + json.dumps(receipt), flush=True)
