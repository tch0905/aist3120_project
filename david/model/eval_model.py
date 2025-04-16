import argparse
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk
from model import BertWithMLPForNER  # Replace with your actual model file
from utils.save_best_model import save_test_results_and_hparams
from seqeval.metrics import classification_report
import numpy as np

# Argument parser
parser = argparse.ArgumentParser(description="Evaluate BERT+MLP model from a checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved model checkpoint.")
args = parser.parse_args()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Load dataset
dataset = load_from_disk("../../conll2003_local")
label_names = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_names)

# Tokenization function
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
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Load model from checkpoint
model = BertWithMLPForNER(
    model_name="bert-base-cased",
    num_labels=num_labels,
    loss_type='ce'  # Adjust if using focal, dice, etc.
)
model.load_state_dict(torch.load(f"{args.checkpoint}/training_args.bin"))
model.eval()

# Define compute_metrics function
def compute_metrics(p):
    predictions, labels = p.predictions, p.label_ids

    if isinstance(predictions, list):
        true_predictions = predictions
    else:
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_names[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]

    true_labels = [
        [label_names[l] for l in label if l != -100]
        for label in labels
    ]

    report = classification_report(true_labels, true_predictions, output_dict=True, digits=6)

    accuracy = sum(1 for t, p in zip(true_labels, true_predictions) if t == p) / len(true_labels)

    metrics = {
        "accuracy": round(accuracy, 6),
        "overall_precision": round(report["micro avg"]["precision"], 6),
        "overall_recall": round(report["micro avg"]["recall"], 6),
        "overall_f1": round(report["micro avg"]["f1-score"], 6),
        "macro_precision": round(report["macro avg"]["precision"], 6),
        "macro_recall": round(report["macro avg"]["recall"], 6),
        "macro_f1": round(report["macro avg"]["f1-score"], 6),
    }

    for label_name in label_names:
        if label_name in report:
            metrics[f"{label_name}_precision"] = round(report[label_name]["precision"], 6)
            metrics[f"{label_name}_recall"] = round(report[label_name]["recall"], 6)
            metrics[f"{label_name}_f1"] = round(report[label_name]["f1-score"], 6)

    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, true_predictions, digits=6))

    return metrics

# Create training args and trainer (for evaluation only)
training_args = TrainingArguments(
    output_dir="./",
    per_device_eval_batch_size=128,
    report_to="none",
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate
print("\n=== Evaluating Model ===\n")
results = trainer.evaluate(tokenized_datasets["test"])

# Print results
print("\n=== Test Set Evaluation Results ===")
for metric, value in results.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.6f}")
    else:
        print(f"{metric}: {value}")

# Optionally save
save_test_results_and_hparams(
    trainer=trainer,
    model=model,
    test_results=results,
    file_path="test_results_and_hparams_from_checkpoint.txt"
)

print("\n✅ Test results and hyperparameters saved.")