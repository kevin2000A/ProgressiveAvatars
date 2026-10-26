"""Regression tests for split transmittance and loss-independent densification."""
import unittest
import torch
from scene.gaussian_model import GaussianModel
from scene.adaptive_gaussian_model import AdaptiveFlameGaussianModel
from scene.adaptive_geometry import AdaptiveMeshForest, TriangleSplitMode


class SplitBudgetTests(unittest.TestCase):
    def test_family_transmittance_and_optimizer(self):
        for mode in [TriangleSplitMode.ONE_TO_THREE, TriangleSplitMode.ONE_TO_FOUR]:
            for cumulative in [True, False]:
                with self.subTest(mode=mode, cumulative=cumulative):
                    model = AdaptiveFlameGaussianModel.__new__(AdaptiveFlameGaussianModel)
                    GaussianModel.__init__(model, 0)
                    model.geo = AdaptiveMeshForest()
                    model.geo.initialize_from_faces(
                        torch.tensor([[0, 1, 2], [0, 2, 3], [1, 2, 3]]),
                        device=torch.device('cpu'), split_mode=mode, max_level=5)
                    model.child_initialization = "original"
                    model._forest_initialized = True
                    model._lod_include_descendants = cumulative
                    # Geometry refresh is independent of this optimizer/opacity unit test.
                    model._refresh_face_properties = lambda: None
                    tensors = {'xyz': ('_xyz', torch.zeros(3, 3)),
                               'f_dc': ('_features_dc', torch.ones(3, 1, 3)),
                               'f_rest': ('_features_rest', torch.zeros(3, 0, 3)),
                               'scaling': ('_scaling', torch.zeros(3, 3)),
                               'rotation': ('_rotation', torch.ones(3, 4)),
                               'opacity': ('_opacity', torch.tensor([[-12.], [0.], [12.]]))}
                    groups = []
                    for name, (attr, value) in tensors.items():
                        parameter = torch.nn.Parameter(value.clone())
                        setattr(model, attr, parameter)
                        groups.append({'name': name, 'params': [parameter]})
                    model.optimizer = torch.optim.Adam(groups, lr=.001)
                    # Create nonzero Adam state, including an unaffected parent row.
                    sum(p.sum() for g in groups for p in g['params']).backward()
                    model.optimizer.step()
                    model.optimizer.zero_grad(set_to_none=True)
                    old_logits = model._opacity.detach().clone()
                    old_state = model.optimizer.state[model._opacity]['exp_avg'].clone()
                    model.geo.split_triangles(torch.tensor([True, False, True]))
                    model._rebuild_gaussians_after_split()
                    for parent in [0, 2]:
                        children = torch.nonzero(model.geo.parent_ids == parent).flatten()
                        indices = torch.cat([torch.tensor([parent]), children]) if cumulative else children
                        # Log transmittance avoids catastrophic cancellation near alpha=1.
                        new_log_t = -torch.nn.functional.softplus(model._opacity[indices]).sum()
                        old_log_t = -torch.nn.functional.softplus(old_logits[parent]).sum()
                        torch.testing.assert_close(new_log_t, old_log_t)
                        if not cumulative:
                            torch.testing.assert_close(model._opacity[parent], old_logits[parent])
                    torch.testing.assert_close(model._opacity[1], old_logits[1])
                    state = model.optimizer.state[model._opacity]
                    torch.testing.assert_close(state['exp_avg'][1], old_state[1])
                    if cumulative:
                        self.assertEqual(float(state['exp_avg'][[0, 2]].abs().sum()), 0.)
                    self.assertEqual(float(state['exp_avg'][3:].abs().sum()), 0.)
                    model._opacity.sigmoid().sum().backward()
                    model.optimizer.step()
                    self.assertTrue(torch.isfinite(model._opacity).all())

    def test_gradient_statistics_ignore_loss_weight(self):
        for weight in [1., 16/31, .2]:
            model = GaussianModel(0)
            model.xyz_gradient_accum = torch.zeros(4, 1)
            model.denom = torch.zeros(4, 1)
            points = torch.zeros(3, 3, requires_grad=True)
            coeff = torch.tensor([[3., 4., 9.], [5., 12., 7.], [8., 15., 1.]])
            (weight * (points * coeff).sum()).backward()
            original_grad = points.grad.clone()
            model.add_densification_stats(points, torch.tensor([True, False, True]),
                                          torch.tensor([2, 1, 0]), loss_weight=weight)
            torch.testing.assert_close(model.xyz_gradient_accum[:, 0],
                                       torch.tensor([17., 0., 5., 0.]))
            torch.testing.assert_close(model.denom[:, 0], torch.tensor([1., 0., 1., 0.]))
            torch.testing.assert_close(points.grad, original_grad)


if __name__ == '__main__':
    unittest.main()
