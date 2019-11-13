import pdb
import torch
from torchvision.utils import make_grid

def select_rand(output, target, length=4):
    index = torch.randperm(len(output))[:length]

    # pdb.set_trace()
    output = torch.unsqueeze(output[index], dim=1)
    target = torch.unsqueeze(target[index], dim=1)
    img_grid = torch.cat((output, target))
    img_grid = make_grid(img_grid, nrow=length)

    return img_grid
