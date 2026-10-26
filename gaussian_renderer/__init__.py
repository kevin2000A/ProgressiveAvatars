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

import torch
import sys
from pathlib import Path
from utils.supervision_render import gaussian_gradient_view
import math
from typing import Union
from diff_gaussian_rasterization_density import GaussianRasterizationSettings, GaussianRasterizer
from scene import GaussianModel, FlameGaussianModel
from utils.sh_utils import eval_sh

def render(
    viewpoint_camera,
    pc: Union[GaussianModel, FlameGaussianModel],
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color: torch.Tensor = None,
    level: int = None,
    detach_parent_params: bool = False,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    if detach_parent_params:
        pc = gaussian_gradient_view(pc, level)

    render_indices = pc.get_render_indices(level)
    render_indices = render_indices.to(device=bg_color.device, dtype=torch.long)

    pc_xyz = pc.get_xyz
    means3D = pc_xyz[render_indices]

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((means3D.shape[0], 4), dtype=means3D.dtype, device=means3D.device)
    screenspace_points.requires_grad_(True)
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points

    pc_opacity = pc.get_opacity
    opacity = pc_opacity[render_indices]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        pc_scaling = pc.get_scaling
        cov3D_precomp = pc.covariance_activation(
            pc_scaling[render_indices],
            scaling_modifier,
            pc._rotation[render_indices],
        )
    else:
        pc_scaling = pc.get_scaling
        scales = pc_scaling[render_indices]
        pc_rotation = pc.get_rotation
        rotations = pc_rotation[render_indices]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            features = pc.get_features
            shs_view = features[render_indices].transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = means3D - viewpoint_camera.camera_center.unsqueeze(0)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features[render_indices]
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "render_indices": render_indices}


def render_wmask(
    viewpoint_camera,
    pc: Union[GaussianModel, FlameGaussianModel],
    pipe,
    bg_color: torch.Tensor,
    mask: torch.Tensor | None = None,
    scaling_modifier: float = 1.0,
    override_color: torch.Tensor | None = None,
    level: int | None = None,
):
    """
    Render with an explicit mask over Gaussians.

    - If mask is None: same as render().
    - If mask is Long[Ks]: interpreted as global Gaussian indices to include.
    - If mask is Bool[N_total]: interpreted as a boolean mask over all Gaussians.

    The final set is intersect(render_indices(level), mask).
    """
    # Base indices by level policy
    render_indices = pc.get_render_indices(level)
    render_indices = render_indices.to(device=bg_color.device, dtype=torch.long)

    if mask is not None:
        # Build boolean inclusion over global space
        if mask.dtype == torch.bool:
            # Expect mask over all Gaussians
            # Clip/expand device
            mask_bool = mask.to(device=bg_color.device)
            # Guard: if shapes mismatch, fall back to base indices
            try:
                take = mask_bool[render_indices]
                render_indices = render_indices[take]
            except Exception:
                pass
        else:
            # Assume LongTensor of global indices
            try:
                mask_idx = mask.to(device=bg_color.device, dtype=torch.long)
                # Efficient membership via scatter boolean map
                total = int(pc._xyz.shape[0]) if hasattr(pc, "_xyz") and isinstance(pc._xyz, torch.Tensor) else 0
                if total > 0:
                    inc = torch.zeros((total,), dtype=torch.bool, device=bg_color.device)
                    inc[mask_idx.clamp_min(0).clamp_max(max(total - 1, 0))] = True
                    take = inc[render_indices]
                    render_indices = render_indices[take]
            except Exception:
                pass

    # From here identical to render(), with filtered render_indices
    pc_xyz = pc.get_xyz
    means3D = pc_xyz[render_indices]

    screenspace_points = torch.zeros((means3D.shape[0], 4), dtype=means3D.dtype, device=means3D.device)
    screenspace_points.requires_grad_(True)
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points

    pc_opacity = pc.get_opacity
    opacity = pc_opacity[render_indices]

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        pc_scaling = pc.get_scaling
        cov3D_precomp = pc.covariance_activation(
            pc_scaling[render_indices],
            scaling_modifier,
            pc._rotation[render_indices],
        )
    else:
        pc_scaling = pc.get_scaling
        scales = pc_scaling[render_indices]
        pc_rotation = pc.get_rotation
        rotations = pc_rotation[render_indices]

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            features = pc.get_features
            shs_view = features[render_indices].transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = means3D - viewpoint_camera.camera_center.unsqueeze(0)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features[render_indices]
    else:
        colors_precomp = override_color

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "render_indices": render_indices,
    }
