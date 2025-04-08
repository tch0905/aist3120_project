import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='sum'):
        super().__init__()
        self.alpha = alpha  # Weighting factor per class (can be None)
        self.gamma = gamma  # Focusing parameter (higher γ = more focus on hard examples)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=self.alpha, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # Focal Loss formula

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
