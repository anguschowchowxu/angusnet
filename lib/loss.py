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
        output = torch.clamp(output,0,1)
        target = torch.clamp(target,0,1)
        bce_loss = F.binary_cross_entropy(output, target, weight, self.size_average)

        loss = dice_loss#  + bce_loss

        return loss


# BUG: negative value
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def forward(self, input, target):
        N = target.size(0)
        smooth = 1
 
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
 
        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
 
        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """
    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()
 
    def forward(self, input, target, weights=None):
 
        C = target.shape[1]
 
        # if weights is None:
        #   weights = torch.ones(C) #uniform weights for all classes
 
        dice = DiceLoss()
        totalLoss = 0
 
        for i in range(C):
            diceLoss = dice(input[:,i], target[:,i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss
 
        return totalLoss

    def to_one_hot(self, tensor, nb_digits):
        b, c, d, h, w = tensor.shape
        tensor = tensor.to(torch.int64).reshape(-1, 1)

        tensor_onehot = torch.FloatTensor(torch.numel(tensor), nb_digits).to(tensor.get_device())
        tensor_onehot.zero_()
        tensor_onehot.scatter_(1, tensor, 1)
        return tensor_onehot
