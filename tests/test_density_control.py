"""Numerical invariants for the factorial densification experiment."""
import torch
from roma import unitquat_to_rotmat, quat_wxyz_to_xyzw
from utils.density_control import budget_target, covariance_children


def test_same_integer_budget_schedule():
    targets = [budget_target(i, 10144, 130144, 50000, 598000, 2000)
               for i in range(0, 600001, 2000)]
    assert targets[0] == 10144 and targets[-1] == 130144
    assert all((n - 10144) % 3 == 0 for n in targets)
    assert all(a <= b for a, b in zip(targets, targets[1:]))
    assert budget_target(48000, 10144, 130144, 50000, 598000, 2000) == 10144


def test_canonical_moments_and_inverse_binding():
    torch.manual_seed(13)
    dtype = torch.float64
    frames = unitquat_to_rotmat(torch.nn.functional.normalize(torch.randn(8, 4, dtype=dtype), dim=-1))
    centers = torch.randn(8, 3, dtype=dtype)
    face_scales = torch.rand(8, 1, dtype=dtype) + 0.1
    xyz = torch.randn(2, 3, dtype=dtype)
    scales = torch.tensor([[0.3, 0.8, 0.2], [1.2, 0.3, 0.5]], dtype=dtype).log()
    rotations = torch.nn.functional.normalize(torch.randn(2, 4, dtype=dtype), dim=-1)
    parents = torch.tensor([0, 0, 0, 1, 1, 1])
    children = torch.arange(2, 8)
    cx, cs, cr = covariance_children(xyz, scales, rotations, parents, children,
                                    centers, frames, face_scales)
    parent_mean = (frames[:2] @ xyz[..., None]).squeeze(-1) * face_scales[:2] + centers[:2]
    child_mean = (frames[children] @ cx[..., None]).squeeze(-1) * face_scales[children] + centers[children]
    parent_rot = frames[:2] @ unitquat_to_rotmat(quat_wxyz_to_xyzw(rotations))
    child_rot = frames[children] @ unitquat_to_rotmat(quat_wxyz_to_xyzw(cr))
    parent_cov = parent_rot @ torch.diag_embed((scales.exp() * face_scales[:2]) ** 2) @ parent_rot.transpose(-1, -2)
    child_cov = child_rot @ torch.diag_embed((cs.exp() * face_scales[children]) ** 2) @ child_rot.transpose(-1, -2)
    delta = child_mean.reshape(2, 3, 3) - parent_mean[:, None]
    mixture_cov = (child_cov.reshape(2, 3, 3, 3) + delta[..., :, None] * delta[..., None, :]).mean(1)
    torch.testing.assert_close(child_mean.reshape(2, 3, 3).mean(1), parent_mean, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(mixture_cov, parent_cov, atol=1e-12, rtol=1e-12)
    assert torch.isfinite(cx).all() and torch.isfinite(cs).all() and torch.isfinite(cr).all()
