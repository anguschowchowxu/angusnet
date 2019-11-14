import pdb
import logging
import torch
from torchvision.utils import make_grid

def select_rand(output, target, length=4):
    n, d, h, w = output.shape
    output = output.reshape(-1, h, w).float()
    target = target.reshape(-1, h, w)#.to(output.dtype)
    index = torch.randperm(len(output))[:length]

    # pdb.set_trace()
    output = torch.unsqueeze(output[index], dim=1)
    target = torch.unsqueeze(target[index], dim=1)
    # logging.debug('{}'.format(target.shape))
    img_grid = torch.cat((output, target))
    img_grid = make_grid(img_grid, nrow=length)
    # logging.debug('{}'.format(img_grid.shape))

    return img_grid
