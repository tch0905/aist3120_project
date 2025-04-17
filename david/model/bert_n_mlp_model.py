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
from model import BertWithMLPForNER

from utils.save_best_model import save_model_params_and_f1, save_model_and_hparams, save_test_results_and_hparams

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
# tokenizer = AutoTokenizer.from_pretrained("../../bert-base-cased-local")
tokenizer = AutoTokenizer.from_pretrained(
    "../../roberta-base-local",
    add_prefix_space=True
)


def augment_batch_with_random_concat(batch, tokenizer, max_length=512, pad_label_id=-100):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    batch_size = input_ids.size(0)

    # Generate a random permutation of indices for pairing
    permuted_indices = torch.randperm(batch_size)

    # New containers for augmented samples
    new_input_ids = []
    new_attention_masks = []
    new_labels = []

    for i in range(batch_size):
        # Get original and random sample
        original_input = input_ids[i]
        original_mask = attention_mask[i]
        original_label = labels[i]

        random_input = input_ids[permuted_indices[i]][1:]
        random_mask = attention_mask[permuted_indices[i]][1:]
        random_label = labels[permuted_indices[i]][1:]

        # Remove padding from input and label using attention_mask
        original_input_trimmed = original_input[original_mask.bool()]
        random_input_trimmed = random_input[random_mask.bool()]

        original_label_trimmed = original_label[original_mask.bool()]
        random_label_trimmed = random_label[random_mask.bool()]

        # Concatenate input and label
        concat_input = torch.cat([original_input_trimmed, random_input_trimmed[1:]], dim=0)
        concat_label = torch.cat([original_label_trimmed, random_label_trimmed[1:]], dim=0)

        # Truncate if needed
        concat_input = concat_input[:max_length]
        concat_label = concat_label[:max_length]

        # Create attention mask
        concat_attention_mask = torch.ones_like(concat_input)

        # Pad input, label, and attention mask to max_length
        pad_len = max_length - concat_input.size(0)
        if pad_len > 0:
            concat_input = torch.cat([concat_input, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
            concat_attention_mask = torch.cat([concat_attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            concat_label = torch.cat([concat_label, torch.full((pad_len,), pad_label_id, dtype=torch.long)])

        # Append to new batch
        new_input_ids.append(concat_input)
        new_attention_masks.append(concat_attention_mask)
        new_labels.append(concat_label)

    # Stack into tensors
    augmented_batch = {
        "input_ids": torch.stack(new_input_ids).to(input_ids.device),
        "attention_mask": torch.stack(new_attention_masks).to(attention_mask.device),
        "labels": torch.stack(new_labels).to(labels.device)
    }

    return augmented_batch

class AugmentingDataCollator:
    def __init__(self, tokenizer, max_length=512, pad_label_id=-100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_label_id = pad_label_id

    def __call__(self, features):
        # Convert list of dicts to batch dict of tensors
        batch = {key: torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(f[key]) for f in features],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id if key != "labels" else self.pad_label_id
                )
                for key in features[0].keys()}

        # Apply your custom augmentation
        return augment_batch_with_random_concat(batch, self.tokenizer, self.max_length, self.pad_label_id)


# Original
model = BertWithMLPForNER(
    num_labels, 
    loss_type='focal',
)
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

    # ## Original
    # predictions = np.argmax(predictions, axis=2)

    # # Remove ignored index (-100)
    # true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    # true_predictions = [
    #     [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
    #     for prediction, label in zip(predictions, labels)
    # ]

    # Handle CRF case where predictions might come as lists of lists
    if isinstance(predictions, list):
        # CRF returns tags directly
        true_predictions = predictions
    else:
        # Standard argmax for non-CRF
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
    
    true_labels = [
        [label_names[l] for l in label if l != -100] 
        for label in labels
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

# sample = tokenized_datasets_conll["train"][0]
# print("First sample labels:", sample["labels"])
# print("Label names:", label_names)
# print("Tokenized text:", tokenizer.convert_ids_to_tokens(sample["input_ids"]))

# Step 6: Training
training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_overall_f1",
    greater_is_better=True,
    logging_dir="./logs",
    report_to="none",
    logging_steps=10,
    lr_scheduler_type="cosine_with_restarts",
    warmup_steps=500,
    # gradient_accumulation_steps=2,
    # fp16=True,
    # label_smoothing_factor=0.1,
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets_wikiann["train"],
    eval_dataset=tokenized_datasets_conll["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
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

results = trainer.evaluate(tokenized_datasets_conll["test"])
print("Test Result:")
print(results)

print("=== Now training on conll ===")
trainer.train_dataset = tokenized_datasets_conll["train"]
trainer.eval_dataset = tokenized_datasets_conll["test"]
trainer.learning_rate = 2e-5
trainer.train()

# Step 7: Evaluate
results = trainer.evaluate(tokenized_datasets_conll["test"])
print("Test Result:")
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

# After training, save the best model parameters AND hyperparameters to a text file
best_model = trainer.model

# Save both model and hyperparameters
save_model_and_hparams(best_model, trainer, "best_model_info.txt")

# save_model_params_and_f1(trainer, output_file="model_params_and_f1.txt")

# print("Best model information saved to best_model_info.txt")

save_test_results_and_hparams(
    trainer,
    best_model,
    results,  # This is the output from trainer.evaluate()
    "test_results_and_hparams.txt"
)

print("Test results and hyperparameters saved to test_results_and_hparams.txt")