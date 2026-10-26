import unittest
from types import SimpleNamespace
import torch
from utils.supervision_render import gaussian_gradient_view, supervised_images, GAUSSIAN_PARAMETERS

class SupervisionRenderTests(unittest.TestCase):
    def test_only_own_gaussian_parameters_receive_gradients(self):
        model=SimpleNamespace(geo=SimpleNamespace(node_levels=torch.tensor([0,1,2,1])),binding=torch.tensor([2,0,3,1]))
        for name in GAUSSIAN_PARAMETERS:setattr(model,name,torch.ones(4,3,requires_grad=True))
        shared=torch.ones(4,3,requires_grad=True)
        view=gaussian_gradient_view(model,1)
        loss=0
        for name in GAUSSIAN_PARAMETERS:
            original=getattr(model,name);filtered=getattr(view,name)
            torch.testing.assert_close(filtered,original)
            loss+=(filtered*shared).sum()
        loss.backward()
        for name in GAUSSIAN_PARAMETERS:
            grad=getattr(model,name).grad
            torch.testing.assert_close(grad,torch.tensor([[0.,0.,0.],[0.,0.,0.],[1.,1.,1.],[1.,1.,1.]]))
        self.assertTrue((shared.grad>0).all())
        self.assertTrue(all(getattr(model,name).is_leaf for name in GAUSSIAN_PARAMETERS))

    def test_matched_antialiased_image_sizes_and_gradient(self):
        prediction=torch.rand(3,96,128,requires_grad=True)
        target=prediction.detach().clone()
        for level,shape in [(0,(3,12,16)),(1,(3,24,32)),(2,(3,48,64)),(3,(3,96,128)),(4,(3,96,128))]:
            a,b=supervised_images(prediction,target,level)
            self.assertEqual(tuple(a.shape),shape)
            torch.testing.assert_close(a,b)
            a.sum().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue((prediction.grad>0).all())

if __name__=='__main__':unittest.main()
