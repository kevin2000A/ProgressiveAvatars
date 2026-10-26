"""The color derivative equals visible alpha/transmittance mass."""
import math
import torch
from diff_gaussian_rasterization_density import GaussianRasterizationSettings,GaussianRasterizer
from utils.graphics_utils import getProjectionMatrix

def test_contribution_mass():
    device='cuda'
    color=torch.ones((2,3),device=device,requires_grad=True)
    settings=GaussianRasterizationSettings(
        image_height=64,image_width=64,tanfovx=1.,tanfovy=1.,
        bg=torch.zeros(3,device=device),scale_modifier=1.,
        viewmatrix=torch.eye(4,device=device),
        projmatrix=getProjectionMatrix(.01,100.,math.pi/2,math.pi/2).T.cuda(),
        sh_degree=0,campos=torch.zeros(3,device=device),prefiltered=False,debug=False)
    image,_=GaussianRasterizer(settings)(
        means3D=torch.tensor([[0.,0.,2.],[0.,0.,2.5]],device=device),
        means2D=torch.zeros((2,4),device=device),colors_precomp=color,
        opacities=torch.full((2,1),.8,device=device),
        scales=torch.full((2,3),.3,device=device),
        rotations=torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]],device=device))
    weights=torch.autograd.grad(image[0].sum(),color)[0][:,0]
    assert (weights>0).all()
    torch.testing.assert_close(weights.sum(),image[0].sum(),rtol=1e-5,atol=1e-5)
    assert weights[0]>weights[1]
