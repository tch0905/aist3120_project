import argparse
import torch
from datasets import load_from_disk

from model import BertWithMLPForNER
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
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

# Initialize Trainer
trainer = Trainer(model=model)

# Evaluate
results = trainer.evaluate(tokenized_datasets_conll["test"])
print(results)

for metric, value in results.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.6f}")
    else:
        print(f"{metric}: {value}")