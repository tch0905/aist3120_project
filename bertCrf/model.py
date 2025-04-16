import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torchcrf import CRF

tokenizer = AutoTokenizer.from_pretrained(
    "/root/aist3120_project/roberta-base-local",
    add_prefix_space=True
)



class BertWithMLPForNER(nn.Module):
    def __init__(self, model_name, num_labels, hidden_dim=256, loss_type='focal', loss_kwargs=None):
        super().__init__()
        self.lstm_hidden_dim = 384
        self.bert = AutoModelForTokenClassification.from_pretrained(
            "/root/aist3120_project/roberta-base-local",
            num_labels=num_labels,
            output_hidden_states=True,
        )
        # Freeze BERT (optional)
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # BiLSTM Layer
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )

        self.lstm_norm = nn.LayerNorm(self.bert.config.hidden_size)

        # MLP Head (now takes BiLSTM output which is 2*lstm_hidden_dim)
        # self.mlp = nn.Sequential(
        #     nn.Linear(2 * self.lstm_hidden_dim, hidden_dim),  # 2* because bidirectional
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(hidden_dim, num_labels),
        # )

        # Custom MLP Head
        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_labels),
        )

        # CRF Layer
        self.crf = CRF(num_labels, batch_first=True)
        self.use_crf = True

        # self.loss_fn = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=-100)
        self.num_labels = num_labels
        self.loss_type = loss_type
        loss_kwargs = loss_kwargs or {}

        if loss_type == 'dice':
            self.loss_fn = DiceLoss(**loss_kwargs)
        elif loss_type == 'focal':
            self.loss_fn = FocalLoss(**loss_kwargs)
        elif loss_type == 'self_adj_dice':
            self.loss_fn = SelfAdjDiceLoss(**loss_kwargs)
        else:
            class_weights = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            # self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.hidden_states[-1]  # Last hidden state

        # Pass through MLP
        logits = self.mlp(sequence_output)

        outputs = self.bert(input_ids, attention_mask=attention_mask)

        loss = None
        if labels is not None:
            if self.use_crf:
                # CRF requires mask for padding tokens
                mask = (input_ids != tokenizer.pad_token_id).bool()

                # Convert -100 to 0 for CRF (it will be masked anyway)
                labels_crf = labels.clone()
                labels_crf[labels == -100] = 0

                # CRF loss calculation
                loss = -self.crf(logits, labels_crf, mask=mask, reduction='mean')
            elif self.loss_type == 'self_adj_dice':
                active_logits = logits.view(-1, self.num_labels)
                active_labels = labels.view(-1)
                valid_indices = active_labels != -100
                active_logits = active_logits[valid_indices]
                active_labels = active_labels[valid_indices]
                loss = self.loss_fn(active_logits, active_labels)
            else:
                loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}



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
        cardinality = (probs + targets_onehot).sum(dim=0)  # (C,)

        # Calculate Dice coefficient per class
        dice_coeff = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1. - dice_coeff  # (C,)

        # Handle reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        return dice_loss



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
