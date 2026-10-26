import os
import json
from typing import Optional

import torch

from .flame_gaussian_model import FlameGaussianModel
from .adaptive_geometry import AdaptiveMeshForest, TriangleSplitMode
from utils.density_control import budget_target, covariance_children
from utils.natural_split_selection import select_split_nodes
from utils.graphics_utils import compute_face_orientation
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz


class AdaptiveFlameGaussianModel(FlameGaussianModel):
    """FLAME Gaussian model enriched with an adaptive triangle forest."""

    def __init__(
        self,
        sh_degree: int,
        disable_flame_static_offset: bool = False,
        not_finetune_flame_params: bool = False,
        n_shape: int = 300,
        n_expr: int = 100,
        split_mode: int = TriangleSplitMode.ONE_TO_THREE,
        max_level: int = 5,
    ) -> None:
        super().__init__(
            sh_degree,
            disable_flame_static_offset=disable_flame_static_offset,
            not_finetune_flame_params=not_finetune_flame_params,
            n_shape=n_shape,
            n_expr=n_expr,
        )

        self.density_score = "signed"
        self.child_initialization = "original"
        self.geo = AdaptiveMeshForest()
        self._split_mode = split_mode
        self._max_level = int(max_level)
        self._forest_initialized: bool = False
        self.leaf_node_ids: Optional[torch.Tensor] = None
        self._training_args_cache = None
        self._last_base_verts: Optional[torch.Tensor] = None
        self._last_base_verts_cano: Optional[torch.Tensor] = None
        self.debug_vis: bool = False
        self._debug_out_dir: Optional[str] = None
        self._debug_export_interval: int = 1000
        self._debug_current_iteration: Optional[int] = None
        self._debug_last_export_iter: Optional[int] = None
        self._debug_last_logged_iter: Optional[int] = None

        self.faces_all: Optional[torch.Tensor] = None
        self._leaf_render_indices: Optional[torch.Tensor] = None
        self._lod_include_descendants: bool = False

    # --------------------------------------------------------------------- #
    # Lifecycle helpers                                                     #
    # --------------------------------------------------------------------- #

    def training_setup(self, training_args):
        self._training_args_cache = training_args
        self.density_score = training_args.density_score
        self.child_initialization = training_args.child_initialization
        super().training_setup(training_args)

    # ------------------------ Checkpoint I/O ------------------------ #
    def capture(self):
        base_capture = super().capture()
        # Unpack potential (base, flame_extra) from parent
        flame_extra = None
        base = base_capture
        if isinstance(base_capture, tuple) and len(base_capture) == 2 and isinstance(base_capture[1], dict):
            base, flame_extra = base_capture

        geo_state = None
        try:
            if hasattr(self, "geo") and self._forest_initialized:
                geo_state = self.geo.get_state()
        except Exception:
            geo_state = None
        # Persist adaptive + flame options
        combined_extra = {}
        if isinstance(flame_extra, dict):
            combined_extra.update(flame_extra)
        combined_extra.update({
            "adaptive_geo": geo_state,
            "adaptive_opts": {
                "lod_include_descendants": bool(getattr(self, "_lod_include_descendants", False)),
                "split_mode": int(getattr(self, "_split_mode", TriangleSplitMode.ONE_TO_THREE)),
                "max_level": int(getattr(self, "_max_level", 5)),
            },
        })
        return (base, combined_extra)

    def restore(self, model_args, training_args):
        """Restore parameters and adaptive geometry in a safe order.

        We avoid calling base .restore() directly because it triggers
        training_setup() before geometry/bindings are consistent, which can
        cause out-of-bounds indexing on GPU. Instead, we:
          1) parse args (support legacy vs extended formats)
          2) assign tensors directly (no training_setup yet)
          3) restore adaptive geometry if available
          4) initialize mesh properties for a safe timestep (0)
          5) run training_setup and then load optimizer state
        """
        adaptive = None
        adaptive_opts = None
        flame_extra = None
        base_args = model_args
        if isinstance(model_args, tuple) and len(model_args) == 2 and isinstance(model_args[1], dict):
            base_args, extra = model_args
            adaptive = extra.get("adaptive_geo", None)
            adaptive_opts = extra.get("adaptive_opts", None)
            flame_extra = {
                "flame_param": extra.get("flame_param", None),
                "num_timesteps": extra.get("num_timesteps", None),
            }

        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.binding,
            self.binding_counter,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = base_args

        # Restore adaptive forest if present
        if adaptive is not None:
            try:
                device = self._xyz.device
                self.geo = AdaptiveMeshForest()
                self.geo.load_state(adaptive, device)
                self._forest_initialized = True
            except Exception:
                self._forest_initialized = False

        # Restore FLAME parameters if present in extras
        try:
            if flame_extra is not None and flame_extra.get("flame_param", None) is not None:
                fp = {k: v.to(self._xyz.device).float() for k, v in flame_extra["flame_param"].items()}
                self.flame_param = fp
                nt = flame_extra.get("num_timesteps", None)
                if nt is not None:
                    self.num_timesteps = int(nt)
        except Exception:
            pass

        # Initialize mesh-dependent properties at timestep 0 to make get_xyz safe
        try:
            if hasattr(self, "select_mesh_by_timestep"):
                self.select_mesh_by_timestep(0)
        except Exception:
            pass

        # Restore density settings alongside the optimizer.
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        try:
            self.optimizer.load_state_dict(opt_dict)
        except Exception:
            pass

        # Restore adaptive options after setup
        try:
            if adaptive_opts is not None:
                self.set_lod_include_descendants(bool(adaptive_opts.get("lod_include_descendants", False)))
            else:
                # Fallback to training args/dataset flag if present
                if hasattr(self, "_training_args_cache"):
                    from argparse import Namespace
                    ds_flag = getattr(self._training_args_cache, "lod_include_descendants", None)
                    if ds_flag is not None:
                        self.set_lod_include_descendants(bool(ds_flag))
        except Exception:
            pass

    def configure_debug(
        self,
        enable: bool,
        output_dir: Optional[str] = None,
        export_interval: Optional[int] = None,
    ) -> None:
        self.debug_vis = bool(enable)
        if not self.debug_vis:
            self._debug_out_dir = None
            return

        if export_interval is not None and export_interval > 0:
            self._debug_export_interval = int(export_interval)

        if output_dir:
            self._debug_out_dir = os.path.join(os.path.abspath(output_dir), "debug_vis")
        else:
            self._debug_out_dir = None

    def set_debug_iteration(self, iteration: int) -> None:
        self._debug_current_iteration = int(iteration)

    def _maybe_init_forest(self, faces: torch.Tensor) -> None:
        if not self._forest_initialized:
            device = faces.device
            self.geo.initialize_from_faces(
                faces=faces,
                device=device,
                split_mode=self._split_mode,
                max_level=self._max_level,
            )
            self._forest_initialized = True

    # --------------------------------------------------------------------- #
    # Mesh update                                                           #
    # --------------------------------------------------------------------- #

    def select_mesh_by_timestep(self, timestep, original=False):
        super().select_mesh_by_timestep(timestep, original)

        base_faces = self.flame_model.faces.to(self.verts.device)
        self._maybe_init_forest(base_faces)

        base_verts = self.verts.squeeze(0)
        base_verts_cano = self.verts_cano.squeeze(0)
        self._last_base_verts = base_verts.detach()
        self._last_base_verts_cano = base_verts_cano.detach()

        subdiv_verts = self.geo.bary_interpolation_hierarchical_subdivision(base_verts)
        subdiv_verts_cano = self.geo.bary_interpolation_hierarchical_subdivision(base_verts_cano)

        self._assign_face_properties(subdiv_verts, subdiv_verts_cano)

    def _assign_face_properties(
        self,
        subdiv_verts: torch.Tensor,
        subdiv_verts_cano: torch.Tensor,
    ) -> None:
        device = subdiv_verts.device
        leaf_node_ids = self.geo.get_leaf_node_ids().to(device, dtype=torch.int32)
        all_faces = self.geo.all_tris.to(device)

        triangles_all = subdiv_verts[all_faces]
        face_center_all = triangles_all.mean(dim=1)
        face_orien_mat_all, face_scaling_all = compute_face_orientation(
            subdiv_verts, all_faces, return_scale=True
        )
        face_orien_quat_all = quat_xyzw_to_wxyz(rotmat_to_unitquat(face_orien_mat_all))

        self.faces_all = all_faces
        self.face_center = face_center_all
        self.face_orien_mat = face_orien_mat_all
        self.face_orien_quat = face_orien_quat_all
        self.face_scaling = face_scaling_all

        leaf_long = leaf_node_ids.to(device=self._xyz.device, dtype=torch.long)
        self.leaf_node_ids = leaf_node_ids
        self._leaf_render_indices = leaf_long

        derived_faces = all_faces[leaf_long]
        self.faces = derived_faces
        self.verts = subdiv_verts[None, ...]
        self.verts_cano = subdiv_verts_cano[None, ...]

        self.binding = torch.arange(all_faces.shape[0], device=device, dtype=torch.int32)
        self.binding_counter = torch.ones_like(self.binding)
        self._maybe_log_debug(derived_faces)
        self._maybe_export_debug(subdiv_verts, derived_faces)

    def _refresh_face_properties(self) -> None:
        if self._last_base_verts is None or self._last_base_verts_cano is None:
            return
        with torch.no_grad():
            subdiv_verts = self.geo.bary_interpolation_hierarchical_subdivision(self._last_base_verts)
            subdiv_verts_cano = self.geo.bary_interpolation_hierarchical_subdivision(self._last_base_verts_cano)
        self._assign_face_properties(subdiv_verts, subdiv_verts_cano)

    def available_levels(self):
        if not self._forest_initialized:
            return super().available_levels()
        max_level = int(self.geo.current_max_level)
        return list(range(max_level + 1))

    def set_lod_include_descendants(self, enable: bool) -> None:
        self._lod_include_descendants = bool(enable)

    def get_render_indices(self, level=None):
        if not self._forest_initialized:
            return super().get_render_indices(level)
        device = self._xyz.device
        if level is None:
            # When no explicit level is requested:
            # - If including descendants, return ALL current nodes (levels <= current_max_level)
            # - Otherwise, return only leaves (default behavior)
            if getattr(self, "_lod_include_descendants", False):
                mask_all = self.geo.get_level_mask(int(self.geo.current_max_level), include_descendants=True).to(device)
                if mask_all.dtype != torch.bool:
                    mask_all = mask_all.to(torch.bool)
                return torch.nonzero(mask_all, as_tuple=False).squeeze(-1).to(device=device, dtype=torch.long)
            if self._leaf_render_indices is None:
                return super().get_render_indices(level)
            return self._leaf_render_indices.to(device=device, dtype=torch.long)
        level_mask = self.geo.get_level_mask(
            int(level), include_descendants=self._lod_include_descendants
        ).to(device)
        if level_mask.dtype != torch.bool:
            level_mask = level_mask.to(torch.bool)
        return torch.nonzero(level_mask, as_tuple=False).squeeze(-1).to(device=device, dtype=torch.long)

    # --------------------------------------------------------------------- #
    # Adaptive densification                                                #
    # --------------------------------------------------------------------- #

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        if not self._forest_initialized:
            return super().densify_and_prune(max_grad, min_opacity, extent, max_screen_size)

        denom_safe = torch.where(self.denom > 0, self.denom, torch.ones_like(self.denom))
        grads = torch.nan_to_num(self.xyz_gradient_accum / denom_safe, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            self._adaptive_split(grads.squeeze(-1), max_grad, extent)

    def _adaptive_split(self, grads: torch.Tensor, grad_threshold: float, scene_extent: float) -> bool:
        if self._training_args_cache.late_split:
            return self._natural_split(grads, grad_threshold, scene_extent)
        if self._leaf_render_indices is None or self._leaf_render_indices.numel() == 0:
            return False
        device = self._xyz.device
        leaf_indices = self._leaf_render_indices
        leaf_grads = grads[leaf_indices]
        if leaf_grads.numel() == 0:
            return False

        opt = self._training_args_cache
        target = budget_target(self.density_iteration, self.geo.n_base_triangles,
                               opt.density_budget, opt.density_budget_start,
                               opt.density_budget_end, opt.densification_interval)
        count_before = len(self._xyz)
        n_split = (target - count_before) // 3
        if n_split == 0:
            return False
        # Filter capped leaves before ranking, so budget is actually usable.
        eligible = self.geo.node_levels[leaf_indices] < self.geo.MAX_LEVEL
        candidates = leaf_indices[eligible]
        scores = leaf_grads[eligible]
        assert n_split > 0 and len(candidates) >= n_split
        assert torch.isfinite(scores).all()
        # Stable tie order by node id; same count even when score scales differ.
        ranked = torch.argsort(scores, descending=True, stable=True)
        parent_nodes = candidates[ranked[:n_split]]

        split_mask = torch.zeros(self.geo.current_size, dtype=torch.bool, device=device)
        split_mask[parent_nodes.long()] = True

        _, new_child_ids = self.geo.split_triangles(split_mask)
        if new_child_ids.numel() == 0:
            return False

        # Debug logging removed to avoid overhead

        self._rebuild_gaussians_after_split()
        assert len(self._xyz) == target
        record = {"iteration": self.density_iteration, "before": count_before,
                  "target": target, "after": len(self._xyz), "splits": n_split,
                  "score": self.density_score, "initialization": self.child_initialization,
                  "selected_score_mean": float(grads[parent_nodes].mean()),
                  "selected_score_max": float(grads[parent_nodes].max()),
                  "layer_counts": torch.bincount(self.geo.node_levels.long()).tolist()}
        with open(self.density_log_path, "a") as log:
            log.write(json.dumps(record) + "\n")
        return True

    def _natural_split(self, grads, grad_threshold, scene_extent):
        opt = self._training_args_cache
        count_before = len(self._xyz)
        parents = select_split_nodes(
            self._leaf_render_indices, self.geo.node_levels, grads, grad_threshold,
            count_before, opt.natural_growth_fraction, opt.natural_ceiling, self.geo.MAX_LEVEL, opt.natural_max_new_points)
        n_split = len(parents)
        record = dict(iteration=self.density_iteration, before=count_before,
                      threshold=grad_threshold, splits=n_split,
                      growth_fraction=opt.natural_growth_fraction, ceiling=opt.natural_ceiling,
                      max_new_points=opt.natural_max_new_points,
                      selected_score_min=float(grads[parents].min()) if n_split else None)
        if n_split:
            split_mask = torch.zeros(self.geo.current_size, dtype=torch.bool, device=self._xyz.device)
            split_mask[parents.long()] = True
            _, children = self.geo.split_triangles(split_mask)
            assert len(children) == 3 * n_split
            self._rebuild_gaussians_after_split()
        assert len(self._xyz) == count_before + 3 * n_split
        # Every decision uses one statistics window, including decisions with no split.
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
        record.update(after=len(self._xyz), layer_counts=torch.bincount(self.geo.node_levels.long()).tolist())
        with open(self.density_log_path, 'a') as log:
            log.write(json.dumps(record) + chr(10))
        return n_split > 0

    def _rebuild_gaussians_after_split(self) -> None:
        if not self._forest_initialized:
            return

        device = self._xyz.device
        prev_size = self._xyz.shape[0]
        new_size = int(self.geo.current_size)

        if new_size < prev_size:
            raise RuntimeError("Adaptive subdivision reduced the number of nodes, which is not supported.")

        reset_stats = False

        if new_size > prev_size:
            new_child_ids = torch.arange(prev_size, new_size, device=device, dtype=torch.long)
            parent_ids = self.geo.parent_ids[new_child_ids.to(self.geo.device)].to(device=device, dtype=torch.long)

            parent_scaling_activated = self.scaling_activation(self._scaling[parent_ids].detach())
            child_scaling = self.scaling_inverse_activation(parent_scaling_activated * 0.6)

            new_xyz = torch.zeros_like(self._xyz[parent_ids]).detach()
            new_features_dc = self._features_dc[parent_ids].detach().clone()
            new_features_rest = self._features_rest[parent_ids].detach().clone()
            # (P, 1): share optical depth across each newly split family.
            # Equal footprints then preserve transmittance: prod(1-alpha_i)=1-alpha.
            split_parents, child_to_parent, child_counts = torch.unique(
                parent_ids, return_inverse=True, return_counts=True)
            family_size = child_counts + int(self._lod_include_descendants)
            parent_depth = torch.nn.functional.softplus(
                self._opacity[split_parents].detach())
            split_depth = parent_depth / family_size[:, None]
            split_logits = torch.log(-torch.expm1(-split_depth)) + split_depth
            new_opacity = split_logits[child_to_parent]
            if self._lod_include_descendants:
                with torch.no_grad():
                    self._opacity[split_parents] = split_logits
            new_rotation = self._rotation[parent_ids].detach().clone()
            if self.child_initialization == "covariance":
                # Build canonical frames for old parents and new bound child faces.
                vertices = self.geo.bary_interpolation_hierarchical_subdivision(
                    self._last_base_verts_cano)
                faces = self.geo.all_tris.long()
                centers = vertices[faces].mean(dim=1)
                frames, face_scales = compute_face_orientation(vertices, faces, return_scale=True)
                new_xyz, child_scaling, new_rotation = covariance_children(
                    self._xyz.detach(), self._scaling.detach(), self._rotation.detach(),
                    parent_ids, new_child_ids, centers, frames, face_scales)

            extension = {
                "xyz": new_xyz.to(device),
                "f_dc": new_features_dc.to(device),
                "f_rest": new_features_rest.to(device),
                "opacity": new_opacity.to(device),
                "scaling": child_scaling.to(device),
                "rotation": new_rotation.to(device),
            }

            optimizable_tensors = self.cat_tensors_to_optimizer(extension)
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            # Parent opacity changed coordinates; discard its stale Adam moments.
            if self._lod_include_descendants and self._opacity in self.optimizer.state:
                opacity_state = self.optimizer.state[self._opacity]
                opacity_state["exp_avg"][split_parents] = 0
                opacity_state["exp_avg_sq"][split_parents] = 0
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            reset_stats = True

        total_size = self._xyz.shape[0]
        self.binding = torch.arange(total_size, device=device, dtype=torch.int32)
        self.binding_counter = torch.ones_like(self.binding)
        if reset_stats:
            self.max_radii2D = torch.zeros((total_size,), device=device)
            self.xyz_gradient_accum = torch.zeros((total_size, 1), device=device)
            self.denom = torch.zeros((total_size, 1), device=device)

        self._refresh_face_properties()
        leaf_ids = self.geo.get_leaf_node_ids().to(device=self.geo.device, dtype=torch.int32).to(device)
        self.leaf_node_ids = leaf_ids
        self._leaf_render_indices = leaf_ids.to(device=device, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # Debug helpers                                                      #
    # ------------------------------------------------------------------ #

    def _maybe_log_debug(self, derived_faces: torch.Tensor) -> None:
        if not self.debug_vis:
            return
        if self._debug_current_iteration is None:
            return
        if self._debug_export_interval > 0 and (self._debug_current_iteration % self._debug_export_interval != 0):
            return
        if self._debug_last_logged_iter == self._debug_current_iteration:
            return

        leaf_count = int(derived_faces.shape[0])
        total_nodes = int(self.geo.current_size)
        max_level = int(self.geo.current_max_level)
        subdiv_points = int(self.geo.parent_subdivision_ids.numel()) if self.geo.parent_subdivision_ids is not None else 0
        print(
            f"[adaptive-debug] iter={self._debug_current_iteration} leaves={leaf_count} "
            f"triangles_total={total_nodes} max_level={max_level} subdiv_points={subdiv_points}"
        )
        self._debug_last_logged_iter = self._debug_current_iteration

    def _maybe_export_debug(self, subdiv_verts: torch.Tensor, derived_faces: torch.Tensor) -> None:
        if not self.debug_vis or self._debug_out_dir is None:
            return
        if self._debug_current_iteration is None:
            return
        if self._debug_export_interval > 0 and (self._debug_current_iteration % self._debug_export_interval != 0):
            return
        if self._debug_last_export_iter == self._debug_current_iteration:
            return

        os.makedirs(self._debug_out_dir, exist_ok=True)
        iter_tag = f"{self._debug_current_iteration:06d}"
        mesh_path = os.path.join(self._debug_out_dir, f"mesh_{iter_tag}.ply")
        cloud_path = os.path.join(self._debug_out_dir, f"gaussians_{iter_tag}.ply")

        try:
            self._write_mesh_ply(mesh_path, subdiv_verts, derived_faces)
            with torch.no_grad():
                self._write_point_cloud_ply(cloud_path, self.get_xyz)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[adaptive-debug] failed to export debug assets at iter={self._debug_current_iteration}: {exc}")
        else:
            self._debug_last_export_iter = self._debug_current_iteration

    @staticmethod
    def _write_point_cloud_ply(path: str, points: torch.Tensor) -> None:
        pts = points.detach().cpu()
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("Point cloud tensor must be of shape [N, 3].")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {pts.shape[0]}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("end_header\n")
            for xyz in pts.tolist():
                handle.write(f"{xyz[0]} {xyz[1]} {xyz[2]}\n")

    @staticmethod
    def _write_mesh_ply(path: str, verts: torch.Tensor, faces: torch.Tensor) -> None:
        v_cpu = verts.detach().cpu()
        f_cpu = faces.detach().cpu()
        if v_cpu.ndim != 2 or v_cpu.shape[1] != 3:
            raise ValueError("Mesh vertices must be of shape [N, 3].")
        if f_cpu.ndim != 2 or f_cpu.shape[1] != 3:
            raise ValueError("Mesh faces must be of shape [F, 3].")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {v_cpu.shape[0]}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write(f"element face {f_cpu.shape[0]}\n")
            handle.write("property list uchar int vertex_indices\n")
            handle.write("end_header\n")
            for xyz in v_cpu.tolist():
                handle.write(f"{xyz[0]} {xyz[1]} {xyz[2]}\n")
            for tri in f_cpu.tolist():
                handle.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

    @staticmethod
    def _write_mesh_ply_with_face_scalar(
        path: str,
        verts: torch.Tensor,
        faces: torch.Tensor,
        values: torch.Tensor,
        prop_name: str = "level",
    ) -> None:
        """Write a triangle mesh to ASCII PLY with one per-face scalar property.

        Args:
            path: Output file path.
            verts: [V,3] float tensor.
            faces: [F,3] long/int tensor.
            values: [F] int/float tensor; per-face scalar to write after indices.
            prop_name: Name of the face property.
        """
        v_cpu = verts.detach().cpu()
        f_cpu = faces.detach().cpu()
        val_cpu = values.detach().cpu().view(-1)
        if v_cpu.ndim != 2 or v_cpu.shape[1] != 3:
            raise ValueError("Mesh vertices must be of shape [N, 3].")
        if f_cpu.ndim != 2 or f_cpu.shape[1] != 3:
            raise ValueError("Mesh faces must be of shape [F, 3].")
        if val_cpu.ndim != 1 or val_cpu.shape[0] != f_cpu.shape[0]:
            raise ValueError("values must be shape [F] matching number of faces.")
        # Coerce to Python list of ints for stable ASCII output
        try:
            val_list = [int(x) for x in val_cpu.tolist()]
            face_prop_decl = f"property int {prop_name}\n"
        except Exception:
            val_list = [float(x) for x in val_cpu.tolist()]
            face_prop_decl = f"property float {prop_name}\n"

        with open(path, "w", encoding="utf-8") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {v_cpu.shape[0]}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write(f"element face {f_cpu.shape[0]}\n")
            handle.write("property list uchar int vertex_indices\n")
            handle.write(face_prop_decl)
            handle.write("end_header\n")
            for xyz in v_cpu.tolist():
                handle.write(f"{xyz[0]} {xyz[1]} {xyz[2]}\n")
            for tri, val in zip(f_cpu.tolist(), val_list):
                handle.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])} {val}\n")
