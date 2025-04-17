import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=5.0, ignore_index=-100, reduction='sum'):
        super().__init__()
        if alpha is None:
            alpha = [1.0, 1, 1, 1, 1, 1, 1, 1, 1]
        self.alpha = torch.tensor(alpha)  # Weighting factor per class (can be None)
        self.gamma = gamma  # Focusing parameter (higher γ = more focus on hard examples)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        alpha = self.alpha.to(inputs.device) if self.alpha is not None else None

        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=alpha,
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
