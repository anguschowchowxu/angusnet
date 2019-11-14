import torch

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# output/target: tensor([batch_size, z, y, x]), z = 3 or num_slcies
def binary_dice(output, target, threshold=0.5):

    batch_size = output.size(0)
    output = output.reshape(batch_size, -1) # tensor([batch_size, z*y*x])
    output = torch.ge(output, threshold).float()
    target = target.reshape(batch_size, -1)

    intersection = torch.sum(output*target, dim=1) # tensor([batch_Size])
    smooth = 1.

    intersection_multiplier = torch.ge(intersection, 0).float()
    dice_loss = (2.*intersection+smooth)/(torch.sum(output, dim=1)+torch.sum(target, dim=1)+smooth) # tensor([batch_size])
    dice_loss = dice_loss*intersection_multiplier
    dice_coeff = torch.mean(dice_loss)
    
    return dice_coeff

def multi_dice(input, target, weights=None):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
"""
    C = target.shape[1]

    # if weights is None:
    #   weights = torch.ones(C) #uniform weights for all classes

    total_coef = 0

    for i in range(C):
        dice_coef = binary_dice(input[:,i], target[:,i])
        if weights is not None:
            dice_coef *= weights[i]
        total_coef += dice_coef

    return total_coef/C

def to_one_hot(tensor, nb_digits):
    b, c, d, h, w = tensor.shape
    tensor = tensor.to(torch.int64).reshape(-1, 1)

    tensor_onehot = torch.FloatTensor(torch.numel(tensor), nb_digits).to(tensor.get_device())
    tensor_onehot.zero_()
    tensor_onehot.scatter_(1, tensor, 1)
    return tensor_onehot