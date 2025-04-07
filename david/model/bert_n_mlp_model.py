### https://huggingface.co/google-bert/bert-base-cased
### BERT + MLP

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

from focal_loss import FocalLoss
from dice_loss import DiceLoss
from selfAdjDiceLoss import SelfAdjDiceLoss

# Step 1: Load Dataset and Labels
dataset = load_from_disk("../../conll2003_local")
wikiann_dataset = load_from_disk("../../wikiann_local")

# CoNLL-2003 Label Names
label_names = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_names)

# Verify WikiANN tags (should only contain 0-6)
def validate_wikiann_tags(example):
    for tag in example["ner_tags"]:
        assert tag in {0, 1, 2, 3, 4, 5, 6}, f"Invalid WikiANN tag: {tag}"
    return example

wikiann_dataset = wikiann_dataset.map(validate_wikiann_tags)

# Step 2: Load Tokenizer and Model with Custom MLP Head
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained("../../bert-base-cased-local")

# Custom Model Architecture
class BertWithMLPForNER(nn.Module):
    def __init__(self, model_name, num_labels, hidden_dim=256, loss_type='focal', loss_kwargs=None):
        super().__init__()
        self.lstm_hidden_dim = 128
        self.bert = AutoModelForTokenClassification.from_pretrained(
            "../../bert-base-cased-local",
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
        )
        
        # MLP Head (now takes BiLSTM output which is 2*lstm_hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.lstm_hidden_dim, hidden_dim),  # 2* because bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_labels),
        )

        # Custom MLP Head
        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_labels),
        )
        
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
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        # outputs = self.bert(input_ids, attention_mask=attention_mask)
        # sequence_output = outputs.hidden_states[-1]  # Last hidden state

        # # Pass through MLP
        # logits = self.mlp(sequence_output)
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.hidden_states[-1]  # Last hidden state
        
        # Pass through BiLSTM
        lstm_output, _ = self.bilstm(sequence_output)
        
        # Pass through MLP
        logits = self.mlp(lstm_output)

        # loss = None
        # if labels is not None:
        #     # loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        #     # loss = loss_fn(logits.view(-1, num_labels), labels.view(-1))
        #     loss = self.loss_fn(logits.view(-1, num_labels), labels.view(-1))
        loss = None
        if labels is not None:
            if self.loss_type == 'self_adj_dice':
                # Flatten logits and labels
                active_logits = logits.view(-1, self.num_labels)  # (batch_size * seq_len, num_labels)
                active_labels = labels.view(-1)  # (batch_size * seq_len)
                
                # Filter out padding tokens (ignore_index=-100)
                valid_indices = active_labels != -100
                active_logits = active_logits[valid_indices]
                active_labels = active_labels[valid_indices]
                
                # Compute loss (SelfAdjDiceLoss handles softmax internally)
                loss = self.loss_fn(active_logits, active_labels)
            else:
                # Original loss handling (Dice/Focal/CrossEntropy)
                loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))


        return {"loss": loss, "logits": logits}

# Original
model = BertWithMLPForNER(
    model_name, 
    num_labels, 
    loss_type='focal',
    loss_kwargs={'alpha': None, 'gamma': 2.0, 'ignore_index': -100} 
)

# For Dice Loss
# model = BertWithMLPForNER(
#     model_name,
#     num_labels,
#     loss_type='dice',
#     loss_kwargs={'smooth': 1e-5, 'reduction': 'mean'}
# )

# For Focal Loss
# focal_loss_kwargs = {
#     'alpha': None,
#     'gamma': 2.0,
#     'ignore_index': -100
# }
# model = BertWithMLPForNER(
#     model_name, 
#     num_labels, 
#     loss_type='focal',
#     loss_kwargs=focal_loss_kwargs
# )

# For Self Adj Dice Loss
# model = BertWithMLPForNER(
#     model_name, 
#     num_labels, 
#     loss_type='self_adj_dice',
#     loss_kwargs={'alpha': 1.0, 'gamma': 1.0, 'reduction': 'mean'}
# )

# Step 3: Tokenize and Align Labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # New word
            else:
                label_ids.append(-100)  # Subword (optional: use label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets_conll = dataset.map(tokenize_and_align_labels, batched=True)
tokenized_datasets_wikiann = wikiann_dataset.map(tokenize_and_align_labels, batched=True)

# Step 4: Data Collator
data_collator = DataCollatorForTokenClassification(tokenizer)


def compute_metrics(p):
    predictions, labels = p.predictions, p.label_ids
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (-100)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Get full classification report with float precision
    report = classification_report(
        true_labels, true_predictions, output_dict=True, digits=6
    )  # Increased decimal places

    # Calculate exact match accuracy with high precision
    correct = sum(
        1 for true, pred in zip(true_labels, true_predictions) if true == pred
    )
    total = len(true_labels)
    accuracy = correct / total if total > 0 else 0.0

    # Format all metrics to 6 decimal places
    metrics = {
        # Overall scores
        "accuracy": round(accuracy, 6),
        "overall_precision": round(report["micro avg"]["precision"], 6),
        "overall_recall": round(report["micro avg"]["recall"], 6),
        "overall_f1": round(report["micro avg"]["f1-score"], 6),
        # Macro averages
        "macro_precision": round(report["macro avg"]["precision"], 6),
        "macro_recall": round(report["macro avg"]["recall"], 6),
        "macro_f1": round(report["macro avg"]["f1-score"], 6),
    }

    # Add per-class F1 scores with high precision
    for label_name in label_names:
        if (
            label_name in report
            and label_name != "macro avg"
            and label_name != "micro avg"
        ):
            metrics.update(
                {
                    f"{label_name}_precision": round(
                        report[label_name]["precision"], 6
                    ),
                    f"{label_name}_recall": round(report[label_name]["recall"], 6),
                    f"{label_name}_f1": round(report[label_name]["f1-score"], 6),
                }
            )

    # Print beautifully formatted report
    print("\nDetailed Classification Report (6 decimal places):")
    print(classification_report(true_labels, true_predictions, digits=6))

    return metrics


# Step 6: Training
training_args = TrainingArguments(
    output_dir="../../../autodl-fs/ner_results",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,  # NEW: Limit to 3 checkpoint (best 3 model only)
    load_best_model_at_end=True,  # NEW: Load the best model at the end
    metric_for_best_model="eval_overall_f1",  # NEW: Define "best" based on F1
    greater_is_better=True,  # NEW: Higher F1 is better
    logging_dir="./logs",
    report_to="none",
    logging_steps=10,
    warmup_ratio=0.1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets_wikiann["train"],
    eval_dataset=tokenized_datasets_conll["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# Then train on mixed dataset
# mixed_train = concatenate_datasets([
#     tokenized_datasets_wikiann["train"].select(range(10000)),  # subset of WikiANN
#     tokenized_datasets_conll["train"]
# ]).shuffle(seed=42)

# trainer.train_dataset = mixed_train
# trainer.learning_rate = 2e-5  # reduced learning rate
# trainer.train()

trainer.train_dataset = tokenized_datasets_conll["train"]
trainer.learning_rate = 1e-5
trainer.train()

# Step 7: Evaluate
results = trainer.evaluate(tokenized_datasets_conll["test"])
print(results)

print("\n=== Best Model Information ===")
print(f"Best Model Checkpoint: {trainer.state.best_model_checkpoint}")
print(f"Best Validation F1 Score: {trainer.state.best_metric:.6f}")

# Evaluate the best model on the test set
print("\n=== Evaluation of Best Model on Test Set ===")
best_model_results = trainer.evaluate(tokenized_datasets_conll["test"])
print("Test Set Results for Best Model:")
for metric, value in best_model_results.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.6f}")
    else:
        print(f"{metric}: {value}")
