import torch
import numpy as np
from torch import nn
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from seqeval.metrics import classification_report
from torch import Tensor
from typing import Optional


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=-100, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets, mask=None):
        """
        Args:
            logits: (N, C) where N is number of tokens, C is num_classes
            targets: (N,) containing class indices
            mask: (N,) indicating which tokens to include (1) or exclude (0)
        """
        num_classes = logits.size(1)
        
        # Convert targets to one-hot encoding (N, C)
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Get probabilities (N, C)
        probs = F.softmax(logits, dim=1)
        
        # Apply mask if provided
        if mask is not None:
            # Reshape mask to (N, 1) to broadcast correctly
            mask = mask.unsqueeze(1)
            probs = probs * mask
            targets_onehot = targets_onehot * mask
        
        # Calculate intersection and union
        intersection = (probs * targets_onehot).sum(dim=0)  # (C,)
        cardinality = (probs + targets_onehot).sum(dim=0)    # (C,)
        
        # Calculate Dice coefficient per class
        dice_coeff = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1. - dice_coeff  # (C,)
        
        # Handle reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        return dice_loss