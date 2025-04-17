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
from torchcrf import CRF

from focal_loss import FocalLoss
from dice_loss import DiceLoss
from selfAdjDiceLoss import SelfAdjDiceLoss


tokenizer = AutoTokenizer.from_pretrained(
    "../../roberta-base-local",
    add_prefix_space=True
)

class BertWithMLPForNER(nn.Module):
    def __init__(self, num_labels, hidden_dim=256, loss_type='focal', loss_kwargs=None):
        super().__init__()
        self.lstm_hidden_dim = 384
        self.bert = AutoModelForTokenClassification.from_pretrained(
            "../../roberta-base-local",
            num_labels=num_labels,
            output_hidden_states=True,
        )

        self.lstm_norm = nn.LayerNorm(self.bert.config.hidden_size)

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
        sequence_output = outputs.hidden_states[-1]  # Last hidden state from BERT

        # MLP Head for prediction
        logits = self.mlp(sequence_output)

        loss = None
        if labels is not None:
            if self.loss_type in ['crf'] and self.use_crf:
                # CRF expects a mask and labels with no -100
                mask = (input_ids != tokenizer.pad_token_id).bool()

                labels_crf = labels.clone()
                labels_crf[labels == -100] = 0  # Replace -100 with dummy label (ignored by mask)

                # Compute CRF loss (negative log-likelihood)
                loss = -self.crf(logits, labels_crf, mask=mask, reduction='mean')

            elif self.loss_type in ['self_adj_dice', 'focal', 'dice']:
                # Flatten logits and labels, remove ignored indices
                active_logits = logits.view(-1, self.num_labels)
                active_labels = labels.view(-1)
                valid_indices = active_labels != -100
                active_logits = active_logits[valid_indices]
                active_labels = active_labels[valid_indices]

                # Compute custom loss
                loss = self.loss_fn(active_logits, active_labels)

            else:
                # Default: CrossEntropyLoss (unweighted or weighted)
                loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}
