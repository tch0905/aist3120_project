import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset

# Load dataset (CoNLL-2003)
dataset = load_dataset("conll2003")

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("../bert-base-cased-local")
model = BertForTokenClassification.from_pretrained("../bert-base-cased-local", num_labels=9)


# Tokenization function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs


# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results2",
    evaluation_strategy="epoch",
    learning_rate=5.0e-6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    weight_decay=0.,
)

# Use DataCollatorForTokenClassification to handle padding
data_collator = DataCollatorForTokenClassification(tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator,  # Add collator here
)

# Train and evaluate
trainer.train()
trainer.evaluate()