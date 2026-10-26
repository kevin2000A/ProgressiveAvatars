#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.lod_supervision import level_loss_weights
from utils.supervision_render import supervised_images
from gaussian_renderer import render, network_gui
from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, FlameGaussianModel, AdaptiveFlameGaussianModel, TriangleSplitMode
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, train_levels=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    if dataset.bind_to_mesh:
        if getattr(dataset, "adaptive_mesh", False):
            split_flag = (getattr(dataset, "adaptive_split_mode", "1to3") or "1to3").lower()
            if split_flag in {"1to4", "1-4", "one_to_four"}:
                split_mode = TriangleSplitMode.ONE_TO_FOUR
            else:
                split_mode = TriangleSplitMode.ONE_TO_THREE

            gaussians = AdaptiveFlameGaussianModel(
                dataset.sh_degree,
                dataset.disable_flame_static_offset,
                dataset.not_finetune_flame_params,
                split_mode=split_mode,
                max_level=int(getattr(dataset, "adaptive_max_level", 5)),
            )
            # Default to enabling descendant inclusion when level is unspecified
            gaussians.set_lod_include_descendants(getattr(dataset, "lod_include_descendants", True))
        else:
            gaussians = FlameGaussianModel(
                dataset.sh_degree,
                dataset.disable_flame_static_offset,
                dataset.not_finetune_flame_params,
            )
        mesh_renderer = NVDiffRenderer()
    else:
        gaussians = GaussianModel(dataset.sh_degree)
    gaussians.configure_debug(
        enable=getattr(dataset, "debug_vis", False),
        output_dir=getattr(dataset, "model_path", None),
        export_interval=getattr(dataset, "debug_vis_interval", None),
    )
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    gaussians.density_log_path = os.path.join(dataset.model_path, "densification.jsonl")
    assert checkpoint is None, 'This experiment must start from scratch'
    assert first_iter == 0
    from lpipsPyTorch import LPIPS
    perceptual = LPIPS(net_type='alex', version='0.1').cuda().eval()
    perceptual.requires_grad_(False)
    torch.save({name: getattr(gaussians,name).detach().cpu().clone() for name in
        ['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','binding']},
        os.path.join(dataset.model_path,'initial_gaussians.pt'))


    train_levels_config = None if train_levels is None else list(train_levels)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Visualization setup (canonical-space mesh export + render snapshots)
    vis_interval = int(getattr(opt, "visualize_interval", 0) or 0)
    render_vis_interval = int(getattr(opt, "render_visualize_interval", 0) or 0)
    render_vis_top_k = int(getattr(opt, "render_visualize_top_k", 0) or 0)
    vis_dir = None
    if vis_interval > 0:
        vis_dir = os.path.join(dataset.model_path, "vis")
        os.makedirs(vis_dir, exist_ok=True)
    elif render_vis_interval > 0:
        # Even if mesh export disabled, we still want a vis folder for renders
        vis_dir = os.path.join(dataset.model_path, "vis")
        os.makedirs(vis_dir, exist_ok=True)

    def _save_image(path: str, img_chw: torch.Tensor) -> None:
        """Save a [3,H,W] float tensor in [0,1] as PNG."""
        try:
            img = img_chw.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
            Image.fromarray((img * 255.0 + 0.5).astype(np.uint8)).save(path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[visualize] failed to save image at {path}: {exc}")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    loader_camera_train = DataLoader(scene.getTrainCameras(), batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, generator=torch.Generator().manual_seed(0))
    iter_camera_train = iter(loader_camera_train)
    # viewpoint_stack = None
    ema_loss_for_log = 0.0

    # --------------------------------------------------------------
    # Pre-loop snapshots at iteration 0 (ckpt, renders, canonical mesh)
    # --------------------------------------------------------------
    try:
        vp0 = next(iter_camera_train)
    except StopIteration:
        iter_camera_train = iter(loader_camera_train)
        vp0 = next(iter_camera_train)

    if gaussians.binding is not None:
        gaussians.select_mesh_by_timestep(vp0.timestep)

    # Determine level candidates same as in-loop
    if hasattr(gaussians, "available_levels"):
        available_levels_0 = gaussians.available_levels()
    else:
        available_levels_0 = [None]

    if train_levels_config is None:
        level_candidates_0 = available_levels_0
    else:
        resolved_levels_0 = []
        max_available_0 = available_levels_0[-1] if len(available_levels_0) > 0 else None
        for lvl in train_levels_config:
            if lvl is None:
                resolved_levels_0.append(max_available_0)
            else:
                if max_available_0 is None:
                    resolved_levels_0.append(None)
                else:
                    resolved_levels_0.append(max(min(int(lvl), int(max_available_0)), 0))
        level_candidates_0 = []
        seen_levels_0 = set()
        for lvl in resolved_levels_0:
            if lvl not in seen_levels_0:
                level_candidates_0.append(lvl)
                seen_levels_0.add(lvl)
        if len(level_candidates_0) == 0:
            level_candidates_0 = available_levels_0

    def level_key_0(level_value):
        return -1 if level_value is None else int(level_value)

    level_targets_0 = sorted(level_candidates_0, key=level_key_0)

    render_results_0 = []
    for lvl in level_targets_0:
        render_results_0.append((lvl, render(vp0, gaussians, pipe, background, level=lvl)))

    # Save canonical FLAME base-topology mesh at iter 0
    if vis_interval > 0:  # (0 % vis_interval == 0)
        try:
            if gaussians.binding is not None and hasattr(gaussians, "flame_model") and hasattr(gaussians, "geo"):
                device_geo = gaussians.geo.device
                faces_base = gaussians.flame_model.faces.to(device=device_geo, dtype=torch.long)
                if hasattr(gaussians, "_last_base_verts_cano") and (gaussians._last_base_verts_cano is not None):
                    verts_cano_base = gaussians._last_base_verts_cano.to(device_geo)
                else:
                    verts_cano_base = gaussians.verts_cano.squeeze(0).to(device_geo)
                n_base = int(gaussians.geo.n_base_triangles)
                root_ids = gaussians.geo.root_triangle_ids
                node_lvls = gaussians.geo.node_levels
                face_levels_0 = torch.zeros((n_base,), dtype=torch.int16, device=device_geo)
                uniq_roots = torch.unique(root_ids)
                for rid in uniq_roots.tolist():
                    mask = (root_ids == int(rid))
                    if mask.any():
                        face_levels_0[int(rid)] = node_lvls[mask].max()
                out_path_0 = os.path.join(vis_dir, f"mesh_cano_{0:06d}.ply")
                gaussians._write_mesh_ply_with_face_scalar(out_path_0, verts_cano_base, faces_base, face_levels_0, prop_name="level")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[visualize] mesh export failed at iter 0: {exc}")

    # Save renders for all/current levels at iter 0
    if render_vis_interval > 0:
        try:
            renders_dir_0 = os.path.join(vis_dir if vis_dir else dataset.model_path, "renders")
            os.makedirs(renders_dir_0, exist_ok=True)
            subset_0 = render_results_0 if len(render_results_0) > 0 else []
            for lvl, pkg in subset_0:
                img = pkg["render"]
                lvl_tag = "none" if lvl is None else str(int(lvl))
                out_path = os.path.join(renders_dir_0, f"iter_{0:06d}_lvl{lvl_tag}.png")
                _save_image(out_path, img)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[visualize] render snapshot failed at iter 0: {exc}")

    # Save extra checkpoint at iter 0
    extra_ckpt_0 = int(getattr(opt, "extra_checkpoint_interval", 0) or 0)
    if extra_ckpt_0 > 0:
        torch.save((gaussians.capture(), 0), scene.model_path + "/chkpnt0.pth")

    sampling_trace = open(os.path.join(scene.model_path, "sampling_trace.tsv"), "w", buffering=1)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):   
        # print(iteration)     
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                # receive data
                net_image = None
                # custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, use_original_mesh = network_gui.receive()
                custom_cam, msg = network_gui.receive()

                # render
                if custom_cam != None:
                    # mesh selection by timestep
                    if gaussians.binding != None:
                        gaussians.select_mesh_by_timestep(custom_cam.timestep, msg['use_original_mesh'])
                    
                    # gaussian splatting rendering
                    if msg['show_splatting']:
                        net_image = render(custom_cam, gaussians, pipe, background, msg['scaling_modifier'])["render"]
                    
                    # mesh rendering
                    if gaussians.binding != None and msg['show_mesh']:
                        out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, custom_cam)

                        rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                        rgb_mesh = rgba_mesh[:3, :, :]
                        alpha_mesh = rgba_mesh[3:, :, :]

                        mesh_opacity = msg['mesh_opacity']
                        if net_image is None:
                            net_image = rgb_mesh
                        else:
                            net_image = rgb_mesh * alpha_mesh * mesh_opacity  + net_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

                    # send data
                    active_render_indices = gaussians.get_render_indices(None)
                    net_dict = {
                        'num_timesteps': gaussians.num_timesteps,
                        'num_points': int(active_render_indices.numel()),
                    }
                    network_gui.send(net_image, net_dict)
                if msg['do_training'] and ((iteration < int(opt.iterations)) or not msg['keep_alive']):
                    break
            except Exception as e:
                # print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.set_debug_iteration(iteration)
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Gradually open adaptive MAX_LEVEL (configurable schedule)
        # Applies only to adaptive models (has geo + _max_level)
        try:
            if hasattr(gaussians, "geo") and hasattr(gaussians, "_max_level"):
                base = int(getattr(opt, "progressive_adaptive_max_level_base", 0))
                step = int(getattr(opt, "progressive_adaptive_max_level_step", 1))
                period = int(getattr(opt, "progressive_adaptive_max_level_period", 50_000))
                target_cap = int(getattr(gaussians, "_max_level", 5))

                if period > 0 and step > 0:
                    increments = int(iteration // period)
                else:
                    increments = 0
                progressive_max = max(0, min(base + increments * step, target_cap))
                if int(gaussians.geo.MAX_LEVEL) != progressive_max:
                    gaussians.geo.MAX_LEVEL = int(progressive_max)
        except Exception:
            # Non-adaptive models or early init; ignore silently
            pass

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        sampling_trace.write(f"{iteration}\t{viewpoint_cam.image_path}\n")
        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        if hasattr(gaussians, "available_levels"):
            available_levels = gaussians.available_levels()
        else:
            available_levels = [None]

        if train_levels_config is None:
            level_candidates = available_levels
        else:
            resolved_levels = []
            max_available = available_levels[-1] if len(available_levels) > 0 else None
            for lvl in train_levels_config:
                if lvl is None:
                    resolved_levels.append(max_available)
                else:
                    if max_available is None:
                        resolved_levels.append(None)
                    else:
                        resolved_levels.append(max(min(int(lvl), int(max_available)), 0))
            level_candidates = []
            seen_levels = set()
            for lvl in resolved_levels:
                if lvl not in seen_levels:
                    level_candidates.append(lvl)
                    seen_levels.add(lvl)
            if len(level_candidates) == 0:
                level_candidates = available_levels

        def level_key(level_value):
            return -1 if level_value is None else int(level_value)

        level_targets = sorted(level_candidates, key=level_key)

        render_results = []
        for lvl in level_targets:
            render_results.append((lvl, render(viewpoint_cam, gaussians, pipe, background, level=lvl,
                detach_parent_params=(opt.supervision_mode == 'detach'
                                      and iteration >= opt.supervision_ramp_start))))

        highest_level, render_pkg_highest = render_results[-1]
        viewspace_point_tensor = render_pkg_highest["viewspace_points"]
        visibility_filter = render_pkg_highest["visibility_filter"]
        radii = render_pkg_highest["radii"]
        render_indices = render_pkg_highest["render_indices"]

        gt_image = viewpoint_cam.original_image.cuda()
        # Per-level geometric weights (finest=1.0, coarser *= decay), normalized to sum=1
        decay = float(getattr(opt, "level_loss_decay", 0.8))
        num_lvls = len(render_results)
        level_weights = level_loss_weights(
            num_lvls, decay, iteration, opt.supervision_mode,
            opt.supervision_ramp_start, opt.supervision_ramp_end)

        losses = {}
        for idx, (lvl, pkg) in enumerate(render_results):
            lw = level_weights[idx] if idx < len(level_weights) else 1.0
            image_lvl = pkg["render"]
            target_lvl = gt_image
            if opt.supervision_mode == 'multiscale' and iteration >= opt.supervision_ramp_start:
                image_lvl, target_lvl = supervised_images(image_lvl, gt_image, lvl)
            l1_term = l1_loss(image_lvl, target_lvl) * (1.0 - opt.lambda_dssim) * lw
            ssim_term = (1.0 - ssim(image_lvl, target_lvl)) * opt.lambda_dssim * lw
            losses['l1'] = losses['l1'] + l1_term if 'l1' in losses else l1_term
            losses['ssim'] = losses['ssim'] + ssim_term if 'ssim' in losses else ssim_term

        if gaussians.binding != None:
            # Root IDs follow topology; geometric frames are constants for this penalty.
            if opt.root_position or opt.root_scale:
                root_ids = gaussians.binding.long().clone()
                for _ in range(dataset.adaptive_max_level):
                    children = gaussians.geo.node_levels[root_ids] > 0
                    root_ids[children] = gaussians.geo.parent_ids[root_ids[children]].long()
                face_scale = gaussians.face_scaling.detach() # (N,1)
                root_scale = face_scale[root_ids]
                if opt.root_position:
                    world_offset = torch.bmm(gaussians.face_orien_mat.detach(), gaussians._xyz[...,None]).squeeze(-1)
                    world_position = world_offset * face_scale + gaussians.face_center.detach()
                    root_position = (world_position - gaussians.face_center.detach()[root_ids]) / root_scale
            subset_xyz = gaussians._xyz[render_indices]
            if visibility_filter.any():
                if opt.metric_xyz:
                    face_scaling_subset = gaussians.face_scaling[render_indices]
                    xyz_metric = (subset_xyz * face_scaling_subset)[visibility_filter]
                    losses['xyz'] = F.relu(xyz_metric - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
                else:
                    position_params = root_position[render_indices] if opt.root_position else subset_xyz
                    xyz_norm = position_params[visibility_filter].norm(dim=1)
                    losses['xyz'] = F.relu(xyz_norm - opt.threshold_xyz).mean() * opt.lambda_xyz

            if opt.lambda_scale != 0 and visibility_filter.any():
                scaling_subset = gaussians.get_scaling[render_indices]
                if opt.metric_scale:
                    losses['scale'] = F.relu(scaling_subset[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                else:
                    scale_params = gaussians._scaling.exp()
                    if opt.root_scale:
                        scale_params = scale_params * face_scale / root_scale
                    losses['scale'] = F.relu(scale_params[render_indices][visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale

            if opt.lambda_dynamic_offset != 0:
                losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:
                ti = viewpoint_cam.timestep
                t_indices =[ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std
        
            if opt.lambda_laplacian != 0:
                losses['lap'] = gaussians.compute_laplacian_loss() * opt.lambda_laplacian
        
        # Visualization snapshot: export canonical FLAME mesh (base topology) with per-face max level
        if vis_interval > 0 and (iteration % vis_interval == 0):
            try:
                if gaussians.binding is not None and hasattr(gaussians, "flame_model") and hasattr(gaussians, "geo"):
                    device_geo = gaussians.geo.device
                    # Base FLAME topology
                    faces_base = gaussians.flame_model.faces.to(device=device_geo, dtype=torch.long)
                    # Canonical base vertices from current timestep
                    if hasattr(gaussians, "_last_base_verts_cano") and (gaussians._last_base_verts_cano is not None):
                        verts_cano_base = gaussians._last_base_verts_cano.to(device_geo)
                    else:
                        # Fallback: if not available, try current canonical verts (may be subdivided) and index base
                        verts_cano_base = gaussians.verts_cano.squeeze(0).to(device_geo)

                    # Per-base-face max level across its current forest nodes
                    n_base = int(gaussians.geo.n_base_triangles)
                    root_ids = gaussians.geo.root_triangle_ids
                    node_lvls = gaussians.geo.node_levels
                    face_levels = torch.zeros((n_base,), dtype=torch.int16, device=device_geo)
                    # Compute max per root by unique scan (Python loop at vis interval; acceptable)
                    uniq_roots = torch.unique(root_ids)
                    for rid in uniq_roots.tolist():
                        mask = (root_ids == int(rid))
                        if mask.any():
                            face_levels[int(rid)] = node_lvls[mask].max()

                    out_path = os.path.join(vis_dir, f"mesh_cano_{iteration:06d}.ply")
                    gaussians._write_mesh_ply_with_face_scalar(out_path, verts_cano_base, faces_base, face_levels, prop_name="level")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[visualize] mesh export failed at iter {iteration}: {exc}")

        # Visualization snapshot: save renders for available levels (all or top-K)
        if render_vis_interval > 0 and (iteration % render_vis_interval == 0):
            try:
                renders_dir = os.path.join(vis_dir if vis_dir else dataset.model_path, "renders")
                os.makedirs(renders_dir, exist_ok=True)
                # Choose subset: all if top_k<=0, otherwise last K (highest levels at the end)
                if len(render_results) > 0:
                    if render_vis_top_k <= 0:
                        subset = render_results
                    else:
                        subset = render_results[-min(render_vis_top_k, len(render_results)):]
                else:
                    subset = []
                for lvl, pkg in subset:
                    img = pkg["render"]  # [3,H,W]
                    lvl_tag = "none" if lvl is None else str(int(lvl))
                    out_path = os.path.join(renders_dir, f"iter_{iteration:06d}_lvl{lvl_tag}.png")
                    _save_image(out_path, img)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[visualize] render snapshot failed at iter {iteration}: {exc}")

        if opt.lambda_perceptual > 0:
            # (1,3,H,W): full-resolution finest render, LPIPS standard [-1,1].
            pred = render_results[-1][1]['render'].clamp(0,1).unsqueeze(0)
            target = gt_image.unsqueeze(0)
            losses['perceptual'] = perceptual(pred*2-1, target*2-1).mean() * opt.lambda_perceptual
        losses['total'] = sum(losses.values())
        losses['total'].backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                # Log total number of 3DGS (all levels aggregated)
                try:
                    total_3dgs = int(gaussians._xyz.shape[0]) if hasattr(gaussians, "_xyz") else 0
                    postfix["3dgs_total"] = f"{total_3dgs}"
                except Exception:
                    pass
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                if visibility_filter.any():
                    visible_nodes = render_indices[visibility_filter]
                    if visible_nodes.numel() > 0:
                        gaussians.max_radii2D[visible_nodes] = torch.maximum(
                            gaussians.max_radii2D[visible_nodes],
                            radii[visibility_filter],
                        )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter, render_indices,
                    loss_weight=level_weights[-1])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # Progressive densify grad threshold synchronized with MAX_LEVEL schedule
                    # Piecewise schedule: constant before warmup, geometric decay every step within [warmup, stop], then constant after stop
                    warmup_iters = int(getattr(opt, "densify_grad_threshold_warmup_iters", 30_000))
                    stop_iters = int(getattr(opt, "densify_grad_threshold_stop_iters", 50_000))
                    step_iters = int(getattr(opt, "densify_grad_threshold_step_iters", 5_000))
                    base_thr = float(getattr(opt, "densify_grad_threshold_base", getattr(opt, "densify_grad_threshold", 0.0002)))
                    decay = float(getattr(opt, "densify_grad_threshold_decay", 0.9))
                    if iteration < warmup_iters or step_iters <= 0:
                        increments = 0
                    else:
                        clamped_it = min(iteration, stop_iters)
                        if clamped_it < warmup_iters:
                            increments = 0
                        else:
                            increments = (clamped_it - warmup_iters) // step_iters + 1
                    current_grad_thr = base_thr * (decay ** max(int(increments), 0))
                    gaussians.density_iteration = iteration
                    gaussians.densify_and_prune(
                        current_grad_thr,
                        getattr(opt, "densify_min_opacity", 0.005),
                        scene.cameras_extent,
                        size_threshold,
                    )
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Extra periodic checkpoint
            extra_ckpt = int(getattr(opt, "extra_checkpoint_interval", 0) or 0)
            if extra_ckpt > 0 and (iteration % extra_ckpt == 0) and (iteration not in checkpoint_iterations):
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
        if 'xyz' in losses:
            tb_writer.add_scalar('train_loss_patches/xyz_loss', losses['xyz'].item(), iteration)
        if 'scale' in losses:
            tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale'].item(), iteration)
        if 'dynamic_offset' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset', losses['dynamic_offset'].item(), iteration)
        if 'laplacian' in losses:
            tb_writer.add_scalar('train_loss_patches/laplacian', losses['laplacian'].item(), iteration)
        if 'dynamic_offset_std' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if scene.gaussians.num_timesteps > 1:
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx % (len(config['cameras']) // num_vis_img) == 0):
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        vis_ct += 1
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            active_indices = scene.gaussians.get_render_indices(None)
            tb_writer.add_scalar('total_points', int(active_indices.numel()), iteration)
        torch.cuda.empty_cache()

