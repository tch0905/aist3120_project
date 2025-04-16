import torch
from torch import nn, optim
from transformers import (
    AutoTokenizer,
)
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, concatenate_datasets
from conllDataset import CoNLLDataset
from model import BertWithMLPForNER
from utils import augment_batch_with_random_concat


BATCH_SIZE = 128
EPOCH_SIZE = 5

num_classes = 9
learning_rate = 2e-5
epoch_size = EPOCH_SIZE

model = BertWithMLPForNER(model_name="", num_labels=9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = BATCH_SIZE

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

dataset = load_from_disk("../conll2003_local")
wikiann_dataset = load_from_disk("../wikiann_local")

# tokenizer = AutoTokenizer.from_pretrained("../../bert-base-cased-local")
tokenizer = AutoTokenizer.from_pretrained(
    "../roberta-base-local",
    add_prefix_space=True
)

def tokenize_and_align_labels(examples):
    tokenizer = AutoTokenizer.from_pretrained(
        "../roberta-base-local",
        add_prefix_space=True
    )
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

def validate_wikiann_tags(example):
    for tag in example["ner_tags"]:
        assert tag in {0, 1, 2, 3, 4, 5, 6}, f"Invalid WikiANN tag: {tag}"
    return example


wikiann_dataset = wikiann_dataset.map(validate_wikiann_tags)
tokenized_datasets_conll = dataset.map(tokenize_and_align_labels, batched=True)
tokenized_datasets_wikiann = wikiann_dataset.map(tokenize_and_align_labels, batched=True)


label_names = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_names)

train_dataset_conll = CoNLLDataset(tokenized_datasets_conll["train"])
val_dataset_conll = CoNLLDataset(tokenized_datasets_conll["validation"])
test_dataset_conll = CoNLLDataset(tokenized_datasets_conll["test"])

train_loader = torch.utils.data.DataLoader(train_dataset_conll, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset_conll, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset_conll, batch_size=batch_size, shuffle=False)

class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

for epoch in range(epoch_size):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]

        # Backward pass
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss

        progress_bar.set_postfix(loss=batch_loss)


    torch.save(model.state_dict(), f"bert_with_mlp_for_ner_{epoch}.pth")


for epoch in range(epoch_size):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Augment the batch
        augmented_batch = augment_batch_with_random_concat(batch, tokenizer)
        aug_input_ids = batch["input_ids"].to(device)
        aug_attention_mask = batch["attention_mask"].to(device)
        aug_labels = batch["labels"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(aug_input_ids, aug_attention_mask, aug_labels)
        loss = outputs["loss"]

        # Backward pass
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss

        progress_bar.set_postfix(loss=batch_loss)

    # Save model after each epoch
    torch.save(model.state_dict(), f"bert_with_mlp_for_ner_arg_{epoch}.pth")

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")