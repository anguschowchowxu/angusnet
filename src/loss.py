import torch
from torch import nn
from torch.nn import functional as F

class weighted_bce_dice_loss_2d(nn.Module):
    def __init__(self, size_average=True, is_weight=True):
        super().__init__()
        self.size_average = size_average
        self.is_weight = is_weight

    def dice_loss(self, output, target):
        intersection = (output*target).sum(dim=1) # tensor([batch_Size])
        smooth = 1.
        dice_loss = 1.-(2.*intersection+smooth)/(output.sum(dim=1)+target.sum(dim=1)+smooth) # tensor([batch_size])
        dice_loss = dice_loss.mean()
        return dice_loss       

    # output: tensor([batch_size, z, y, x]), target: tensor([batch_size, z, y, x]), z = 3 or num_slices
    def forward(self, output, target):
        batch_size = output.size(0)
        output = output.reshape(batch_size, -1) # tensor([batch_size, 3*y*x])
        target = target.reshape(batch_size, -1) # tensor([batch_size, 3*y*x])
        dice_loss = self.dice_loss(output, target)

        if self.is_weight:
            class_num = torch.sum(target, dim=1, keepdim=True) # (batch_size, 1)
            total_num = target.size(1)
            weight = 1 - class_num / total_num 
            weight = torch.pow(weight, 2) # (batch_size, 1)
            weight = weight.expand(-1, target.size(1)) # (batch_size, 3*y*x)
        else:
            weight = None
        bce_loss = F.binary_cross_entropy(output, target, weight, self.size_average)

        loss = dice_loss#  + bce_loss

        return loss