import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from position import PositionalEncoding
from tranfomer import TransformerBlock


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)


class CustomBERTNER(nn.Module):
    def __init__(self, vocab_size, embed_size=768, num_layers=12, heads=12, forward_expansion=8, dropout=0.1, max_len=512,
                 num_classes=9):
        print(
            f"vocab_size={vocab_size}, embed_size={embed_size}, num_layers={num_layers}, heads={heads}, forward_expansion={forward_expansion}, dropout={dropout}, max_len={max_len}, num_classes={num_classes}")
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_encoding = PositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_size, num_classes)  # NER Tagging

    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x)  # Output shape: (batch_size, seq_length, num_classes)


from datasets import load_from_disk
from transformers import AutoTokenizer
from collections import Counter
# Load CoNLL-2003 dataset
dataset = load_from_disk("../conll2003_local")

# Check available splits
print(dataset)

# Use a BERT tokenizer (you can choose any, but we won't use its weights)
tokenizer = AutoTokenizer.from_pretrained("../bert-base-cased-local")

# Tokenize a sample sentence
example = dataset["train"][0]["tokens"]
tokenized_example = tokenizer(example, is_split_into_words=True, padding="max_length", truncation=True, max_length=50)

print(tokenized_example)

# Get all labels from the dataset
all_labels = [label for data in dataset["train"]["ner_tags"] for label in data]

# Count occurrences of each class
label_counts = Counter(all_labels)
total_samples = sum(label_counts.values())

# Compute class weights (inverse of frequency)
num_classes = len(dataset["train"].features["ner_tags"].feature.names)
class_weights = torch.tensor([total_samples / (label_counts[i] + 1e-6) for i in range(num_classes)], dtype=torch.float)

# Normalize weights (optional, but recommended)
class_weights = class_weights / class_weights.sum()
print("Class Weights:", class_weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_size = 5
batch_size = 4
learning_rate = 3e-5
print(f"Training in epoch_size: {epoch_size}, batch_size: {batch_size}, learning_rate: {learning_rate}, device: {device}")

model = CustomBERTNER(vocab_size=tokenizer.vocab_size,
                      num_classes=len(dataset['train'].features['ner_tags'].feature.names)).to(device)



optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Move class_weights to the same device as the model
class_weights = class_weights.to(device)



def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding="max_length", max_length=50,
                                 is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get word IDs
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Ignore special tokens
            elif word_id != previous_word_id:
                label_ids.append(label[word_id])  # Use original label
            else:
                label_ids.append(label[word_id])  # Use same label for subwords
            previous_word_id = word_id
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Preprocess dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

class CoNLLDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        self.labels = tokenized_data["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "labels": torch.tensor(self.labels[idx])
        }



train_dataset = CoNLLDataset(tokenized_datasets["train"])
val_dataset = CoNLLDataset(tokenized_datasets["validation"])
test_dataset = CoNLLDataset(tokenized_datasets["test"])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # Define the weighted loss function
# criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
#
# # Training loop
# for epoch in range(epoch_size//2):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)
#
#         optimizer.zero_grad()
#         outputs = model(input_ids, attention_mask)
#
#         # Compute loss with class weights
#         loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
#

criterion = nn.CrossEntropyLoss(ignore_index=-100)
for epoch in range(epoch_size):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        # Compute loss with class weights
        loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

from sklearn.metrics import classification_report


def evaluate(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=-1)

            for i in range(len(labels)):
                label_seq = labels[i][labels[i] != -100].cpu().numpy()
                pred_seq = predictions[i][labels[i] != -100].cpu().numpy()

                all_labels.extend(label_seq)
                all_preds.extend(pred_seq)

    return classification_report(all_labels, all_preds, digits=4)


print("Evaluation Report on Training Data:")
print(evaluate(model, train_loader))

print("Evaluation Report on Validation Data:")
print(evaluate(model, val_loader))

print("Evaluation Report on Test Data:")
print(evaluate(model, test_loader))