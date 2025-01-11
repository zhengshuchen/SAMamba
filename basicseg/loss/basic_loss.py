import torch
import torch.nn as nn
import torch.nn.functional as F
from basicseg.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class Iou_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        iou = intersection_sum / (pred_sum + target_sum - intersection_sum + self.eps)
        if self.reduction == 'mean':
            return 1 - iou.mean()
        elif self.reduction == 'sum':
            return 1 - iou.mean()
        else: 
            raise NotImplementedError('reduction type {} not implemented'.format(self.reduction))

@LOSS_REGISTRY.register()
class Dice_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * target, dim=(1,2,3))
        total_sum = torch.sum((pred + target), dim=(1,2,3))
        dice = 2 * intersection / (total_sum + self.eps)
        if self.reduction == 'mean':
            return 1 - dice.mean()
        elif self.reduction == 'sum':
            return 1 - dice.sum()
        else: 
            raise NotImplementedError('reduction type {} not implemented'.format(self.reduction))

@LOSS_REGISTRY.register()
class Bce_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
    def forward(self, pred, target):
        loss_fn = nn.BCEWithLogitsLoss(reduction=self.reduction)
        return loss_fn(pred, target)

@LOSS_REGISTRY.register()
class L1_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
    def forward(self, pred, target):
        loss_fn = nn.L1Loss(reduction=self.reduction)
        return loss_fn(pred, target)

@LOSS_REGISTRY.register()
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
@LOSS_REGISTRY.register()
class SoftIoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SoftIoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Convert logits to probabilities
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        union = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection

        soft_iou = (intersection + 1e-6) / (union + 1e-6)
        loss = 1 - soft_iou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
@LOSS_REGISTRY.register()
class WeightedDiceLoss(nn.Module):
    def __init__(self, weight_map=None, reduction='mean'):
        super(WeightedDiceLoss, self).__init__()
        self.weight_map = weight_map  # Optional weight map for per-pixel weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Convert logits to probabilities

        if self.weight_map is None:
            weight_map = torch.ones_like(targets)
        else:
            weight_map = self.weight_map

        intersection = 2 * (inputs * targets * weight_map).sum(dim=(1, 2, 3))
        union = (inputs * weight_map).sum(dim=(1, 2, 3)) + (targets * weight_map).sum(dim=(1, 2, 3))

        dice = (intersection + 1e-6) / (union + 1e-6)
        loss = 1 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

@LOSS_REGISTRY.register()
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, reduction='mean'):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Convert logits to probabilities
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        false_positive = ((1 - targets) * inputs).sum(dim=(1, 2, 3))
        false_negative = ((1 - inputs) * targets).sum(dim=(1, 2, 3))

        tversky = intersection / (intersection + self.alpha * false_positive + self.beta * false_negative + 1e-6)
        loss = 1 - tversky

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
@LOSS_REGISTRY.register()
class BoundaryLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BoundaryLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Convert logits to probabilities

        # Calculate gradients (Sobel operator can also be used)
        grad_inputs = torch.abs(torch.gradient(inputs, dim=(2, 3))[0] + torch.gradient(inputs, dim=(2, 3))[1])
        grad_targets = torch.abs(torch.gradient(targets, dim=(2, 3))[0] + torch.gradient(targets, dim=(2, 3))[1])

        # Boundary loss as L1 distance between gradients
        boundary_loss = torch.abs(grad_inputs - grad_targets).sum(dim=(1, 2, 3))

        if self.reduction == 'mean':
            return boundary_loss.mean()
        elif self.reduction == 'sum':
            return boundary_loss.sum()
        else:
            return boundary_loss
