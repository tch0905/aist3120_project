import argparse
import torch
from datasets import load_from_disk

from model import BertWithMLPForNER
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer, TrainingArguments,
)
from seqeval.metrics import classification_report

# Argument parser
parser = argparse.ArgumentParser(description="Evaluate BERT-based NER model")
parser.add_argument("--model_path", type=str, required=True, help="Path to the .pth model file")
args = parser.parse_args()

# Load dataset and tokenizer
dataset = load_from_disk("../conll2003_local")
tokenizer = AutoTokenizer.from_pretrained("../bert-base-cased-local")


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



# Tokenization and alignment
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=50,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(label[word_id])
            else:
                label_ids.append(label[word_id])  # Same label for subwords
            previous_word_id = word_id
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Preprocess data
tokenized_datasets_conll = dataset.map(tokenize_and_align_labels, batched=True)

# Load model and weights
model = BertWithMLPForNER(num_labels=9)
model.load_state_dict(torch.load(args.model_path))

# training_args = TrainingArguments(
#     output_dir="../../../autodl-fs/ner_results",
#     per_device_train_batch_size=128,
#     per_device_eval_batch_size=128,
#     num_train_epochs=10,
#     learning_rate=5e-5,
#     weight_decay=0.01,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     metric_for_best_model="eval_overall_f1",  # NEW: Define "best" based on F1
#     greater_is_better=True,  # NEW: Higher F1 is better
#     logging_dir="./logs",
#     report_to="none",
#     logging_steps=10,
#     lr_scheduler_type="cosine_with_restarts",  # Better than linear
#     warmup_steps=500,  # More precise than ratio
#     # gradient_accumulation_steps=2,
#     # fp16=True,
#     # label_smoothing_factor=0.1,
# )

# Initialize Trainer
trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)
# print(f"Best Validation F1 Score: {trainer.state.best_metric:.6f}")
# Evaluate
results = trainer.evaluate(tokenized_datasets_conll["test"])
print(results)
f1_score = results.get("f1", None)  # Assuming "f1" is the key for the F1 score
if f1_score is not None:
    print(f"F1 Score: {f1_score:.6f}")
else:
    print("F1 Score not found in the results.")

for metric, value in results.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.6f}")
    else:
        print(f"{metric}: {value}")