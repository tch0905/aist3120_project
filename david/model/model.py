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
from datasets import load_dataset, load_from_disk, concatenate_datasets
from seqeval.metrics import classification_report
from torch import Tensor
from typing import Optional
from torchcrf import CRF

from focal_loss import FocalLoss
from dice_loss import DiceLoss
from selfAdjDiceLoss import SelfAdjDiceLoss

from utils.save_best_model import save_model_params_and_f1, save_model_and_hparams, save_test_results_and_hparams


tokenizer = AutoTokenizer.from_pretrained(
    "../../roberta-base-local",
    add_prefix_space=True
)

class BertWithMLPForNER(nn.Module):
    def __init__(self, model_name, num_labels, hidden_dim=256, loss_type='focal', loss_kwargs=None):
        super().__init__()
        self.lstm_hidden_dim = 384
        self.bert = AutoModelForTokenClassification.from_pretrained(
            "../../roberta-base-local",
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
        sequence_output = outputs.hidden_states[-1]

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
