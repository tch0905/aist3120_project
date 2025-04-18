import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import load_from_disk
from transformers import AutoTokenizer
from collections import Counter

from position import PositionalEncoding
from tranfomer import TransformerBlock

# Model Components
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)

class CustomBERTNER(nn.Module):
    def __init__(self, vocab_size, embed_size=768, num_layers=6, heads=8, forward_expansion=8, dropout=0.2, max_len=512,
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
        return self.classifier(x)  # Output shape: (batch_size, seq_length, num_classes)

# Dataset Class
class CoNLLDataset(Dataset):
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

# Data Processing Functions
def tokenize_and_align_labels(examples, tokenizer):
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

def compute_class_weights(dataset):
    all_labels = [label for data in dataset["train"]["ner_tags"] for label in data]
    label_counts = Counter(all_labels)
    total_samples = sum(label_counts.values())
    num_classes = len(dataset["train"].features["ner_tags"].feature.names)
    
    class_weights = torch.tensor([total_samples / (label_counts[i] + 1e-6) for i in range(num_classes)], dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    return class_weights

# Evaluation Function
def evaluate(model, data_loader, device):
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

# Training Setup
def setup_training():
    # Load dataset and tokenizer
    dataset = load_from_disk("../conll2003_local")
    tokenizer = AutoTokenizer.from_pretrained("../bert-base-cased-local")
    
    # Compute class weights
    class_weights = compute_class_weights(dataset)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training parameters
    epoch_size = 30
    batch_size = 12
    learning_rate = 5.0e-6
    
    # Initialize model
    model = CustomBERTNER(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(dataset['train'].features['ner_tags'].feature.names)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    class_weights = class_weights.to(device)
    
    # Process datasets
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )
    
    # Create datasets and dataloaders
    train_dataset = CoNLLDataset(tokenized_datasets["train"])
    val_dataset = CoNLLDataset(tokenized_datasets["validation"])
    test_dataset = CoNLLDataset(tokenized_datasets["test"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return model, optimizer, train_loader, val_loader, test_loader, device, class_weights, epoch_size

# Training Loop
def train_model(model, optimizer, train_loader, val_loader, test_loader, device, class_weights, epoch_size):
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    
    for epoch in range(epoch_size):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
        
        # Evaluate after each epoch
        print("\nEvaluation Reports:")
        print("Training Data:")
        print(evaluate(model, train_loader, device))
        print("\nValidation Data:")
        print(evaluate(model, val_loader, device))
        print("\nTest Data:")
        print(evaluate(model, test_loader, device))

if __name__ == "__main__":
    model, optimizer, train_loader, val_loader, test_loader, device, class_weights, epoch_size = setup_training()
    train_model(model, optimizer, train_loader, val_loader, test_loader, device, class_weights, epoch_size)
