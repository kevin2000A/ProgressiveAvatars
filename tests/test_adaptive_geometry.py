import torch

from scene.adaptive_geometry import AdaptiveMeshForest, TriangleSplitMode


def _init_forest(split_mode: int):
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    forest = AdaptiveMeshForest()
    device = torch.device("cpu")
    forest.initialize_from_faces(faces, device=device, split_mode=split_mode, max_level=5)

    base_verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    forest.bary_interpolation_hierarchical_subdivision(base_verts)  # sets n_base_vertices
    return forest, base_verts


def test_split_one_to_three_counts_and_barycentric():
    forest, base_verts = _init_forest(TriangleSplitMode.ONE_TO_THREE)

    split_mask = torch.tensor([True])
    actual_mask, child_ids = forest.split_triangles(split_mask)
    assert actual_mask.tolist() == [True]
    assert child_ids.numel() == 3

    assert forest.current_size == 4
    assert forest.node_levels.shape[0] == 4
    assert forest.parent_subdivision_ids.numel() == 1
    assert forest.point_level.tolist() == [1]

    parent_coords = forest.bary_coords_activation(forest._parent_barycentric_coords)
    assert torch.allclose(parent_coords.sum(dim=-1), torch.ones_like(parent_coords[:, 0]))

    all_points = forest.bary_interpolation_hierarchical_subdivision(base_verts)
    assert all_points.shape[0] == 4
    expected_centroid = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 0.0])
    assert torch.allclose(all_points[-1], expected_centroid, atol=1e-6)

    child_tris = forest.all_tris[1:]
    assert torch.all((child_tris == 3).any(dim=1))


def test_split_one_to_four_counts_and_midpoints():
    forest, base_verts = _init_forest(TriangleSplitMode.ONE_TO_FOUR)

    split_mask = torch.tensor([True])
    actual_mask, child_ids = forest.split_triangles(split_mask)
    assert actual_mask.tolist() == [True]
    assert child_ids.numel() == 4

    assert forest.current_size == 5
    assert forest.parent_subdivision_ids.numel() == 3
    assert forest.point_level.tolist() == [1, 1, 1]

    parent_coords = forest.bary_coords_activation(forest._parent_barycentric_coords)
    assert torch.allclose(parent_coords.sum(dim=-1), torch.ones_like(parent_coords[:, 0]))

    all_points = forest.bary_interpolation_hierarchical_subdivision(base_verts)
    assert all_points.shape[0] == 6
    new_points = all_points[-3:]

    expected_midpoints = torch.stack(
        [
            0.5 * (base_verts[0] + base_verts[1]),
            0.5 * (base_verts[0] + base_verts[2]),
            0.5 * (base_verts[1] + base_verts[2]),
        ],
        dim=0,
    )
    assert torch.allclose(new_points, expected_midpoints, atol=1e-6)

    child_tris = forest.all_tris[1:]
    new_indices = {3, 4, 5}
    assert all(set(tri.tolist()) & new_indices for tri in child_tris)


def test_get_level_mask_after_split():
    forest, _ = _init_forest(TriangleSplitMode.ONE_TO_THREE)
    forest.split_triangles(torch.tensor([True]))

    mask_level0 = forest.get_level_mask(0)
    mask_level1 = forest.get_level_mask(1)
    mask_level2 = forest.get_level_mask(2)

    assert mask_level0.shape == torch.Size([4])
    assert mask_level0[0]
    assert not mask_level0[1:].any()

    assert mask_level1.tolist() == [False, True, True, True]
    # A deeper requested LOD keeps existing leaves instead of dropping geometry.
    assert torch.equal(mask_level2, mask_level1)

    inclusive_level0 = forest.get_level_mask(0, include_descendants=True)
    inclusive_level1 = forest.get_level_mask(1, include_descendants=True)
    inclusive_level2 = forest.get_level_mask(2, include_descendants=True)

    assert inclusive_level0.tolist() == [True, False, False, False]
    assert inclusive_level1.tolist() == [True, True, True, True]
    assert inclusive_level2.tolist() == [True, True, True, True]
