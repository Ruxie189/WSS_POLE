import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''
    Implementation of Focal Loss.
    Reference:
    [1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
        arXiv preprint arXiv:1708.02002, 2017.
    '''
    def __init__(self, gamma=0, ignore_index=-100, size_average=False, one_hot_target=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.one_hot_target = one_hot_target

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        if self.ignore_index >= 0:
            index = torch.nonzero(target.squeeze() != self.ignore_index).squeeze()
            input = input[index, :]
            target = target[index, :]

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()



def cross_entropy_loss(input, target, size_average=True):
    input = F.log_softmax(input, dim=1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        return cross_entropy_loss(input, target, self.size_average)



#https://github.com/hysts/pytorch_mixup.git
def onehot(label, n_classes):
    return torch.zeros(label.detach().cpu().size(0), n_classes).scatter_(
        1, label.detach().cpu().view(-1, 1), 1)


def mixup(data, targets, device, alpha, n_classes):
    indices = torch.randperm(data.size(0)).to(device)
    data2 = data[indices].to(device)
    targets2 = targets[indices].to(device)

    targets = onehot(targets, n_classes).to(device)
    targets2 = onehot(targets2, n_classes).to(device)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)]).to(device)
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)
    return data, targets

class MDCA(torch.nn.Module):
    '''
    Implementation of the MDCA and DCA loss function.
    [1] Hebbalaguppe R, Prakash J, Madan N, Arora C. A Stitch in Time Saves Nine: A Train-Time Regularizing Loss for Improved Neural Network Calibration. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2022
    '''
    def __init__(self):
        super(MDCA,self).__init__()

    def forward(self , output, target):
        output = torch.softmax(output, dim=1)
        # [batch, classes]
        loss = torch.tensor(0.0).cuda()
        batch, classes = output.shape
        for c in range(classes):
            avg_count = (target == c).float().mean()
            avg_conf = torch.mean(output[:,c])
            loss += torch.abs(avg_conf - avg_count)
        denom = classes
        loss /= denom
        return loss

