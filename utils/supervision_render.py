"""Per-LOD gradient routing and antialiased image-pyramid supervision."""
import copy
import torch
import torch.nn.functional as F

GAUSSIAN_PARAMETERS = ('_xyz', '_features_dc', '_features_rest',
                       '_opacity', '_scaling', '_rotation')

def gaussian_gradient_view(model, level):
    """Preserve rendering values and shared geometry; gate only Gaussian parameters."""
    view = copy.copy(model)
    own_level = model.geo.node_levels[model.binding.long()] == level  # (N,)
    for name in GAUSSIAN_PARAMETERS:
        value = getattr(model, name)
        mask = own_level.view(-1, *([1] * (value.ndim - 1)))
        setattr(view, name, torch.where(mask, value, value.detach()))
    return view

def supervised_images(prediction, target, level):
    """Downsample both (3,H,W) images; L0/1/2 use 1/8,1/4,1/2, L3+ full size."""
    divisor = {0: 8, 1: 4, 2: 2}.get(level, 1)
    if divisor == 1:
        return prediction, target
    size = (prediction.shape[-2] // divisor, prediction.shape[-1] // divisor)
    prediction = F.interpolate(prediction[None], size=size, mode='bilinear',
                               align_corners=False, antialias=True)[0]
    target = F.interpolate(target[None], size=size, mode='bilinear',
                          align_corners=False, antialias=True)[0]
    return prediction, target
