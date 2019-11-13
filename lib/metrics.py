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