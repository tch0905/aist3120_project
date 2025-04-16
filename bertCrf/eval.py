import argparse
import torch
from datasets import load_from_disk

from model import BertWithMLPForNER
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer, TrainingArguments,
)

# Argument parser
parser = argparse.ArgumentParser(description="Evaluate BERT-based NER model")
parser.add_argument("--model_path", type=str, required=True, help="Path to the .pth model file")
args = parser.parse_args()

# Load dataset and tokenizer
dataset = load_from_disk("../conll2003_local")
tokenizer = AutoTokenizer.from_pretrained("../bert-base-cased-local")

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
trainer = Trainer(model=model)
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