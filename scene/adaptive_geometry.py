import torch
import torch.nn as nn
from typing import Optional, Tuple


class TriangleSplitMode:
    """Split mode constants for implicit triangle subdivision."""

    ONE_TO_FOUR = 1
    ONE_TO_THREE = 2


class AdaptiveMeshForest(nn.Module):
    """Adaptive triangle forest that grows a 1:1 face↔Gaussian mapping."""

    def __init__(self) -> None:
        super().__init__()
        self.device: Optional[torch.device] = None
        self.split_mode: int = TriangleSplitMode.ONE_TO_THREE
        self.MAX_LEVEL: int = 5
        # Debug/validation flag: when True, run topology invariants assertions
        self.enable_debug_topology_checks: bool = True

        # Topology (triangles)
        self.all_tris: Optional[torch.Tensor] = None  # [F, 3] long
        self.n_base_triangles: int = 0
        self.current_size: int = 0  # number of triangles (nodes)
        self.current_max_level: int = 0

        # Forest structure per triangle (size == current_size)
        self.node_levels: Optional[torch.Tensor] = None  # int16
        self.parent_ids: Optional[torch.Tensor] = None  # int32
        self.first_child_id: Optional[torch.Tensor] = None  # int32
        self.root_triangle_ids: Optional[torch.Tensor] = None  # int32

        # Subdivision vertices bookkeeping
        self.parent_subdivision_ids: Optional[torch.Tensor] = None  # int32 (len == n_subdiv_pts)
        self.point_level: Optional[torch.Tensor] = None  # int16 (len == n_subdiv_pts)
        self._parent_barycentric_coords: Optional[torch.Tensor] = None  # raw coords (len == n_subdiv_pts, 2 or 3)

        # Derived vertex count; inferred from first interpolation call
        self.n_base_vertices: Optional[int] = None

    def initialize_from_faces(
        self,
        faces: torch.Tensor,
        device: torch.device,
        split_mode: int = TriangleSplitMode.ONE_TO_THREE,
        max_level: int = 5,
    ) -> None:
        """Initialize forest from base mesh faces.

        Args:
            faces: LongTensor of shape [F, 3]
            device: torch device for allocations
            split_mode: ONE_TO_THREE or ONE_TO_FOUR
            max_level: maximum subdivision depth
        """
        assert faces.ndim == 2 and faces.shape[1] == 3, "faces must be [F,3]"
        self.device = device
        self.split_mode = split_mode
        self.MAX_LEVEL = int(max_level)

        faces = faces.to(device=device, dtype=torch.long)
        n_faces = int(faces.shape[0])

        # Base topology
        self.all_tris = faces.clone()
        self.n_base_triangles = n_faces
        self.current_size = n_faces
        self.current_max_level = 0

        # Forest arrays (size == current_size)
        self.node_levels = torch.zeros((n_faces,), dtype=torch.int16, device=device)
        self.parent_ids = torch.full((n_faces,), -1, dtype=torch.int32, device=device)
        self.first_child_id = torch.full((n_faces,), -1, dtype=torch.int32, device=device)
        self.root_triangle_ids = torch.arange(n_faces, dtype=torch.int32, device=device)

        # Subdivision vertices (none yet)
        self.parent_subdivision_ids = torch.empty((0,), dtype=torch.int32, device=device)
        self.point_level = torch.empty((0,), dtype=torch.int16, device=device)
        self._parent_barycentric_coords = torch.empty((0, 3), dtype=torch.float32, device=device)

        # Will be set on first interpolation call
        self.n_base_vertices = None

    @staticmethod
    def bary_coords_activation(raw_coords: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        activated = torch.relu(raw_coords)
        sum_coords = torch.sum(activated, dim=-1, keepdim=True) + eps
        return activated / sum_coords

    @staticmethod
    def bary_coords_inverse_activation(bary_coords: torch.Tensor) -> torch.Tensor:
        return torch.relu(bary_coords)

    def bary_interpolation_hierarchical_subdivision(self, features: torch.Tensor) -> torch.Tensor:
        """Vectorized interpolation of subdivided vertex attributes.

        If no subdivision has occurred, this is a no-op that returns the base
        features. When subdivision points exist, this function computes new
        vertex attributes per level using the parent triangle vertices and the
        stored per-parent barycentric coordinates (to be added in a later step).

        Args:
            features: Tensor of shape [N_base_verts, C]

        Returns:
            Tensor of shape [N_total_verts, C]
        """
        assert features.ndim == 2, "features must be [N, C]"
        device = features.device

        # Lazy-set base vertex count on first call
        if self.n_base_vertices is None:
            self.n_base_vertices = int(features.shape[0])

        n_subdiv = int(self.parent_subdivision_ids.numel()) if self.parent_subdivision_ids is not None else 0
        if n_subdiv == 0:
            return features

        if self._parent_barycentric_coords is None:
            return features

        n_total_points = self.n_base_vertices + n_subdiv
        feature_dim = features.shape[-1]
        all_points = torch.zeros((n_total_points, feature_dim), device=device, dtype=features.dtype)
        all_points[: self.n_base_vertices] = features

        if self.point_level is None or self.point_level.numel() == 0:
            return all_points

        max_level = int(self.point_level.max().item())
        for level in range(1, max_level + 1):
            level_mask = self.point_level == level
            if not torch.any(level_mask):
                continue

            parent_ids = self.parent_subdivision_ids[level_mask]
            parent_tris = self.all_tris[parent_ids.long()]
            parent_vertices = all_points[parent_tris]

            # Topology invariant (debug): parent triangles of level-L subdiv points
            # may only reference base vertices or subdiv vertices strictly below L.
            if self.enable_debug_topology_checks:
                base = int(self.n_base_vertices) if self.n_base_vertices is not None else 0
                tri_flat = parent_tris.reshape(-1).to(torch.long)
                mask_subdiv = tri_flat >= base
                if mask_subdiv.any():
                    sub_ids = tri_flat[mask_subdiv] - base
                    ref_lvls = self.point_level[sub_ids]
                    ok = ref_lvls < level
                    if not torch.all(ok):
                        # Pick one offending reference for readable error
                        bad_idx = torch.nonzero(~ok, as_tuple=False).squeeze(1)[0]
                        offending_vid = int(sub_ids[bad_idx].item())
                        offending_lvl = int(ref_lvls[bad_idx].item())
                        raise AssertionError(
                            f"[AdaptiveMeshForest] Topology invariant violated at level={level}: "
                            f"parent triangle references subdiv vertex (id={offending_vid}) with level={offending_lvl} "
                            f">= current level. Ensure splits are processed per-level or parent indices remain stable."
                        )

            raw_coords = self._parent_barycentric_coords[level_mask]
            bary_coords = self.bary_coords_activation(raw_coords)

            interpolated = torch.bmm(
                bary_coords.unsqueeze(1),
                parent_vertices,
            ).squeeze(1)

            level_indices = torch.nonzero(level_mask, as_tuple=False).squeeze(1)
            global_indices = level_indices + self.n_base_vertices
            all_points[global_indices] = interpolated

        return all_points

    # ------------------------------------------------------------------ #
    # Debug controls                                                      #
    # ------------------------------------------------------------------ #

    def set_topology_debug(self, enable: bool) -> None:
        """Enable or disable topology invariant assertions during subdivision.

        When enabled, each level-L interpolation verifies that parent triangles
        only reference base vertices or subdiv vertices from levels < L. This
        helps catch ordering/consistency issues when splitting across levels.
        """
        self.enable_debug_topology_checks = bool(enable)

    # ----------------------- Persistence helpers ----------------------- #
    def get_state(self) -> dict:
        """Serialize forest topology state to a CPU dictionary."""
        def cpu(x):
            return None if x is None else x.detach().cpu()

        return {
            "device": str(self.device) if self.device is not None else None,
            "split_mode": int(self.split_mode),
            "MAX_LEVEL": int(self.MAX_LEVEL),
            "all_tris": cpu(self.all_tris),
            "n_base_triangles": int(self.n_base_triangles),
            "current_size": int(self.current_size),
            "current_max_level": int(self.current_max_level),
            "node_levels": cpu(self.node_levels),
            "parent_ids": cpu(self.parent_ids),
            "first_child_id": cpu(self.first_child_id),
            "root_triangle_ids": cpu(self.root_triangle_ids),
            "parent_subdivision_ids": cpu(self.parent_subdivision_ids),
            "point_level": cpu(self.point_level),
            "_parent_barycentric_coords": cpu(self._parent_barycentric_coords),
            "n_base_vertices": int(self.n_base_vertices) if self.n_base_vertices is not None else None,
        }

    def load_state(self, state: dict, device: torch.device) -> None:
        """Restore forest topology state from a state dictionary to given device."""
        self.device = device
        self.split_mode = int(state.get("split_mode", self.split_mode))
        self.MAX_LEVEL = int(state.get("MAX_LEVEL", self.MAX_LEVEL))

        def to_dev(x, dtype=None):
            if x is None:
                return None
            t = x.to(device)
            return t.to(dtype) if dtype is not None else t

        self.all_tris = to_dev(state.get("all_tris"), torch.long)
        self.n_base_triangles = int(state.get("n_base_triangles", 0))
        self.current_size = int(state.get("current_size", self.all_tris.shape[0] if self.all_tris is not None else 0))
        self.current_max_level = int(state.get("current_max_level", 0))
        self.node_levels = to_dev(state.get("node_levels"), torch.int16)
        self.parent_ids = to_dev(state.get("parent_ids"), torch.int32)
        self.first_child_id = to_dev(state.get("first_child_id"), torch.int32)
        self.root_triangle_ids = to_dev(state.get("root_triangle_ids"), torch.int32)
        self.parent_subdivision_ids = to_dev(state.get("parent_subdivision_ids"), torch.int32)
        self.point_level = to_dev(state.get("point_level"), torch.int16)
        self._parent_barycentric_coords = to_dev(state.get("_parent_barycentric_coords"), torch.float32)
        nbv = state.get("n_base_vertices", None)
        self.n_base_vertices = int(nbv) if nbv is not None else None

    @property
    def triangles_per_parent(self) -> int:
        return 3 if self.split_mode == TriangleSplitMode.ONE_TO_THREE else 4

    @property
    def subdivision_points_per_parent(self) -> int:
        return 1 if self.split_mode == TriangleSplitMode.ONE_TO_THREE else 3

    def get_leaf_mask(self) -> torch.Tensor:
        assert self.first_child_id is not None, "Forest not initialized."
        return self.first_child_id == -1

    def get_leaf_node_ids(self) -> torch.Tensor:
        leaf_mask = self.get_leaf_mask()
        return torch.nonzero(leaf_mask, as_tuple=False).squeeze(1).to(torch.int32)

    def split_triangles(self, split_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split selected triangles according to `split_mode` and update forest state.

        Returns:
            actual_mask: 布尔掩码，表示哪些节点真正触发了细分。
            new_child_node_ids: 新增子节点（按内部追加顺序）的索引。
        """
        assert self.all_tris is not None, "Forest not initialized."
        assert split_mask.dtype == torch.bool and split_mask.shape[0] == self.current_size
        if not torch.any(split_mask):
            empty = torch.empty((0,), dtype=torch.int32, device=self.device)
            return torch.zeros_like(split_mask), empty

        split_node_ids = torch.nonzero(split_mask, as_tuple=False).squeeze(1)
        if split_node_ids.numel() == 0:
            empty = torch.empty((0,), dtype=torch.int32, device=self.device)
            return torch.zeros_like(split_mask), empty

        if self.node_levels is None:
            empty = torch.empty((0,), dtype=torch.int32, device=self.device)
            return torch.zeros_like(split_mask), empty

        level_ok = self.node_levels[split_node_ids] < self.MAX_LEVEL
        split_node_ids = split_node_ids[level_ok]
        actual_mask = torch.zeros_like(split_mask)
        if split_node_ids.numel() == 0:
            empty = torch.empty((0,), dtype=torch.int32, device=self.device)
            return actual_mask, empty

        actual_mask[split_node_ids] = True

        triangles_per_parent = self.triangles_per_parent
        subdivision_points_per_parent = self.subdivision_points_per_parent

        n_new_tris = split_node_ids.numel() * triangles_per_parent
        n_new_subdiv = split_node_ids.numel() * subdivision_points_per_parent

        new_parent_coords = self._generate_new_subdivision_coordinates(n_new_subdiv)

        prev_size = self.current_size
        self._update_adaptive_forest_structure(
            split_node_ids,
            triangles_per_parent,
            subdivision_points_per_parent,
        )

        if new_parent_coords.numel() > 0 and self._parent_barycentric_coords is not None:
            self._parent_barycentric_coords = torch.cat(
                [self._parent_barycentric_coords, new_parent_coords], dim=0
            )


        new_child_ids = torch.arange(
            prev_size,
            prev_size + n_new_tris,
            device=self.device,
            dtype=torch.int32,
        )

        return actual_mask, new_child_ids

    def get_level_mask(self, level: int, include_descendants: bool = False) -> torch.Tensor:
        """Return the leaf mask when truncating at depth ``level``.

        Args:
            level: Target depth.
            include_descendants: If ``True``, mark every node with depth ``<= level``.
                Otherwise, keep the default behaviour that returns the leaf set after
                truncating the forest at ``level``.

        Note:
            If ``level < 0`` the result is all False. If the forest has no hierarchy
            yet, the result is all True regardless of the flag.
        """
        assert self.all_tris is not None, "Forest not initialized."
        if level < 0:
            return torch.zeros(self.current_size, dtype=torch.bool, device=self.device)

        if self.first_child_id is None:
            return torch.ones(self.current_size, dtype=torch.bool, device=self.device)

        if include_descendants:
            clamped_level = max(0, int(level))
            return self.node_levels <= clamped_level

        at_level = (self.node_levels == level)
        below_level = (self.node_levels < level)
        is_leaf = (self.first_child_id == -1)
        below_level_leaves = below_level & is_leaf
        return at_level | below_level_leaves

    # ------------------------------------------------------------------ #
    # Export helpers                                                      #
    # ------------------------------------------------------------------ #

    def export_mesh(
        self,
        path: str,
        base_verts: torch.Tensor,
        level: Optional[int] = None,
        include_descendants: bool = False,
        file_format: str = "ply",
    ) -> None:
        """Export the current forest mesh to disk as PLY/OBJ.

        Args:
            path: Output file path.
            base_verts: Base mesh vertex positions (shape [V, 3]) to drive
                hierarchical subdivision.
            level: Target LOD. If None, exports current leaf faces.
            include_descendants: If True, include all nodes with depth <= level.
            file_format: "ply" (default) or "obj".

        Notes:
            - This method is frame-agnostic; caller provides vertex positions.
            - Vertices are the full subdivided set. Faces are filtered by level.
        """
        assert self.all_tris is not None, "Forest not initialized."
        assert base_verts.ndim == 2 and base_verts.shape[1] == 3, "base_verts must be [V, 3]."

        # Compute subdivided vertices for the provided base positions
        with torch.no_grad():
            all_verts = self.bary_interpolation_hierarchical_subdivision(base_verts.to(self.device))

        # Select faces (leaves by default or per level)
        if level is None:
            face_mask = self.first_child_id == -1
        else:
            face_mask = self.get_level_mask(int(level), include_descendants=include_descendants)

        faces = self.all_tris[face_mask].detach().cpu()
        verts = all_verts.detach().cpu()

        if file_format.lower() == "ply":
            self._write_mesh_ply(path, verts, faces)
        elif file_format.lower() == "obj":
            self._write_mesh_obj(path, verts, faces)
        else:
            raise ValueError(f"Unsupported mesh format: {file_format}")

    @staticmethod
    def _write_mesh_ply(path: str, verts: torch.Tensor, faces: torch.Tensor) -> None:
        """Write a triangle mesh to ASCII PLY."""
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("Mesh vertices must be of shape [N, 3].")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Mesh faces must be of shape [F, 3].")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {verts.shape[0]}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write(f"element face {faces.shape[0]}\n")
            handle.write("property list uchar int vertex_indices\n")
            handle.write("end_header\n")
            for xyz in verts.tolist():
                handle.write(f"{xyz[0]} {xyz[1]} {xyz[2]}\n")
            for tri in faces.tolist():
                handle.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

    @staticmethod
    def _write_mesh_obj(path: str, verts: torch.Tensor, faces: torch.Tensor) -> None:
        """Write a triangle mesh to OBJ (1-indexed faces)."""
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("Mesh vertices must be of shape [N, 3].")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Mesh faces must be of shape [F, 3].")
        with open(path, "w", encoding="utf-8") as handle:
            for xyz in verts.tolist():
                handle.write(f"v {xyz[0]} {xyz[1]} {xyz[2]}\n")
            for tri in faces.tolist():
                # OBJ uses 1-based indexing
                handle.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")

    def _generate_new_triangle_gs_coords(self, n_new_triangles: int) -> torch.Tensor:
        if n_new_triangles == 0:
            return torch.empty((0, 3), device=self.device, dtype=torch.float32)
        default_coords = torch.full(
            (n_new_triangles, 3), 1.0 / 3.0, device=self.device, dtype=torch.float32
        )
        return self.bary_coords_inverse_activation(default_coords)

    def _generate_new_subdivision_coordinates(self, n_new_subdivision_points: int) -> torch.Tensor:
        if n_new_subdivision_points == 0:
            return torch.empty((0, 3), device=self.device, dtype=torch.float32)

        if self.split_mode == TriangleSplitMode.ONE_TO_THREE:
            center = torch.full(
                (n_new_subdivision_points, 3), 1.0 / 3.0, device=self.device, dtype=torch.float32
            )
            return self.bary_coords_inverse_activation(center)

        patterns = torch.tensor(
            [
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        repeats = n_new_subdivision_points // 3
        raw = patterns.repeat(repeats, 1)
        return self.bary_coords_inverse_activation(raw)

    def _update_adaptive_forest_structure(
        self,
        split_node_ids: torch.Tensor,
        triangles_per_parent: int,
        subdivision_points_per_parent: int,
    ) -> None:
        old_size = self.current_size
        n_split = split_node_ids.numel()
        n_new_tris = n_split * triangles_per_parent
        n_new_subdiv = n_split * subdivision_points_per_parent

        parent_levels = self.node_levels[split_node_ids]
        child_levels = parent_levels + 1
        self.current_max_level = max(
            int(self.current_max_level), int(child_levels.max().item()) if child_levels.numel() else 0
        )

        self.node_levels = torch.cat(
            [self.node_levels, child_levels.repeat_interleave(triangles_per_parent).to(torch.int16)],
            dim=0,
        )
        self.parent_ids = torch.cat(
            [self.parent_ids, split_node_ids.repeat_interleave(triangles_per_parent).to(torch.int32)],
            dim=0,
        )
        self.first_child_id = torch.cat(
            [self.first_child_id, torch.full((n_new_tris,), -1, dtype=torch.int32, device=self.device)], dim=0
        )

        child_roots = self.root_triangle_ids[split_node_ids].repeat_interleave(triangles_per_parent)
        self.root_triangle_ids = torch.cat([self.root_triangle_ids, child_roots], dim=0)

        first_child_ids = torch.arange(
            old_size, old_size + n_new_tris, triangles_per_parent, device=self.device, dtype=torch.int32
        )
        self.first_child_id[split_node_ids] = first_child_ids

        if n_new_subdiv > 0:
            new_parent_subdiv = split_node_ids.repeat_interleave(subdivision_points_per_parent).to(torch.int32)
            new_levels = (parent_levels + 1).repeat_interleave(subdivision_points_per_parent).to(torch.int16)

            self.parent_subdivision_ids = torch.cat([self.parent_subdivision_ids, new_parent_subdiv], dim=0)
            self.point_level = torch.cat([self.point_level, new_levels], dim=0)
        else:
            new_parent_subdiv = torch.empty((0,), dtype=torch.int32, device=self.device)

        base_vertices = int(self.n_base_vertices) if self.n_base_vertices is not None else 0
        existing_subdiv = self._parent_barycentric_coords.shape[0] if self._parent_barycentric_coords is not None else 0
        old_num_vertices = base_vertices + existing_subdiv
        new_triangles = self._generate_subdivision_triangles(
            split_node_ids,
            old_num_vertices,
            triangles_per_parent,
            subdivision_points_per_parent,
        )
        self.all_tris = torch.cat([self.all_tris, new_triangles.to(torch.long)], dim=0)

        self.current_size = self.all_tris.shape[0]

    def _generate_subdivision_triangles(
        self,
        split_node_ids: torch.Tensor,
        old_num_vertices: int,
        triangles_per_parent: int,
        subdivision_points_per_parent: int,
    ) -> torch.Tensor:
        if split_node_ids.numel() == 0:
            return torch.empty((0, 3), dtype=torch.int32, device=self.device)

        parent_tris = self.all_tris[split_node_ids]
        if triangles_per_parent == 3:
            centers = torch.arange(
                old_num_vertices,
                old_num_vertices + split_node_ids.numel(),
                device=self.device,
                dtype=torch.int32,
            )
            v0 = parent_tris[:, 0]
            v1 = parent_tris[:, 1]
            v2 = parent_tris[:, 2]

            tri0 = torch.stack([v0, v1, centers], dim=1)
            tri1 = torch.stack([v1, v2, centers], dim=1)
            tri2 = torch.stack([v2, v0, centers], dim=1)
            # Parent-major ordering: for each parent i, emit tri0[i], tri1[i], tri2[i]
            return torch.stack([tri0, tri1, tri2], dim=1).reshape(-1, 3)

        edge_indices = torch.arange(
            old_num_vertices,
            old_num_vertices + subdivision_points_per_parent * split_node_ids.numel(),
            device=self.device,
            dtype=torch.int32,
        ).view(-1, 3)

        edge01 = edge_indices[:, 0]
        edge02 = edge_indices[:, 1]
        edge12 = edge_indices[:, 2]

        tri0 = torch.stack([parent_tris[:, 0], edge01, edge02], dim=1)
        tri1 = torch.stack([edge01, parent_tris[:, 1], edge12], dim=1)
        tri2 = torch.stack([edge02, edge12, parent_tris[:, 2]], dim=1)
        tri3 = torch.stack([edge01, edge12, edge02], dim=1)

        # Parent-major ordering: tri0[i], tri1[i], tri2[i], tri3[i] per parent i
        return torch.stack([tri0, tri1, tri2, tri3], dim=1).reshape(-1, 3).to(torch.int32)

    def _validate_data_consistency(self) -> None:
        """Basic invariants for sizes and indexing."""
        assert self.all_tris is not None, "Forest not initialized."
        assert self.current_size == int(self.all_tris.shape[0]), "all_tris length must match current_size"
        assert self.node_levels.shape[0] == self.current_size, "node_levels size mismatch"
        assert self.parent_ids.shape[0] == self.current_size, "parent_ids size mismatch"
        assert self.first_child_id.shape[0] == self.current_size, "first_child_id size mismatch"
        assert self.root_triangle_ids.shape[0] == self.current_size, "root_triangle_ids size mismatch"
