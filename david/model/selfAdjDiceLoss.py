import torch

class SelfAdjDiceLoss(torch.nn.Module):
    r"""
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)

    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.

    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.gamma) / (probs_with_factor + 1 + self.gamma)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return loss
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")
            

class DiceLoss(torch.nn.Module):
    def __init__(self, mode: str = "standard", alpha: float = 1.0, gamma: float = 0.1, reduction: str = "sum"):
        super().__init__()
        self.mode = mode  # "self_adj" or "standard"
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        
        if self.mode == "self_adj":
            # Self-adjusting Dice Loss
            probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))
            probs_with_factor = ((1 - probs) ** self.alpha) * probs
            loss = 1 - (2 * probs_with_factor + self.gamma) / (probs_with_factor + 1 + self.gamma)
        elif self.mode == "standard":
            # Standard Dice Loss (requires one-hot targets)
            targets_one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
            numerator = 2 * probs * targets_one_hot + self.gamma
            denominator = probs.pow(2) + targets_one_hot.pow(2) + self.gamma
            loss = 1 - (numerator / denominator).mean(dim=1)  # Mean over classes
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # "none"