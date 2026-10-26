"""Matched-budget selection and canonical covariance-based child initialization."""
import math

import torch
from roma import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz, unitquat_to_rotmat, rotmat_to_unitquat


def budget_target(iteration, initial, final, start, end, interval):
    """A shared integer schedule; retaining parents adds three points per split."""
    assert (final - initial) % 3 == 0
    assert start % interval == end % interval == 0 and end >= start
    events = (end - start) // interval + 1
    completed = max(0, min(events, (iteration - start) // interval + 1))
    return initial + 3 * (((final - initial) // 3) * completed // events)


def covariance_children(xyz, log_scales, rotations, parents, children,
                        centers, frames, face_scales):
    """Three equal-weight children match the parent's canonical first two moments.

    Retained parents and opacity redistribution are handled by the caller.
    Moment matching is not pixel-exact alpha-compositing preservation.
    """
    # (K,3), (K,3,3): learned parent means and principal axes in canonical space.
    local_rotation = unitquat_to_rotmat(
        quat_wxyz_to_xyzw(torch.nn.functional.normalize(rotations[parents], dim=-1)))
    world_rotation = frames[parents] @ local_rotation
    world_scales = log_scales[parents].exp() * face_scales[parents]
    world_mean = (frames[parents] @ xyz[parents, :, None]).squeeze(-1)
    world_mean = world_mean * face_scales[parents] + centers[parents]
    axis = world_scales.argmax(dim=-1)
    rows = torch.arange(len(parents), device=parents.device)
    direction = world_rotation[rows, :, axis]  # (K,3): longest principal axis.
    shrink = 0.6
    # Each family is appended contiguously in the forest's parent order.
    rank = torch.arange(len(parents), device=parents.device) % 3 - 1
    assert torch.equal(parents.reshape(-1, 3), parents.reshape(-1, 3)[:, :1].expand(-1, 3))
    offset = rank * world_scales[rows, axis] * math.sqrt(1.5 * (1 - shrink ** 2))
    child_mean = world_mean + direction * offset[:, None]
    child_scales = world_scales.clone()
    child_scales[rows, axis] *= shrink
    # (K,3): inverse child-face transform; no second face-scale shrink.
    child_xyz = (frames[children].transpose(-1, -2)
                 @ ((child_mean - centers[children]) / face_scales[children])[:, :, None])
    child_xyz = child_xyz.squeeze(-1)
    child_rotation = frames[children].transpose(-1, -2) @ world_rotation
    child_rotation = quat_xyzw_to_wxyz(rotmat_to_unitquat(child_rotation))
    return child_xyz, (child_scales / face_scales[children]).log(), child_rotation
