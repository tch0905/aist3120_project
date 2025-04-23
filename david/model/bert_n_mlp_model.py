import datasets
import torch
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
from safetensors.torch import load_file

from dataset import dataset, label_names, num_labels
from compute_metrics import compute_metrics
from model import BertWithMLPForNER
from tokenize_and_align_labels import tokenize_and_align_labels, tokenizer
from utils.save_best_model import save_model_params_and_f1, save_model_and_hparams, save_test_results_and_hparams

BATCH_SIZE = 4


# Verify WikiANN tags (should only contain 0-6)
def validate_wikiann_tags(example):
    for tag in example["ner_tags"]:
        assert tag in {0, 1, 2, 3, 4, 5, 6}, f"Invalid WikiANN tag: {tag}"
    return example

wikiann_dataset = load_from_disk("../../wikiann_local")
wikiann_dataset = wikiann_dataset.map(validate_wikiann_tags)


# Original
model = BertWithMLPForNER(
    num_labels, 
    loss_type='focal',
)


def preprocess_dataset(dataset):
    new_data = {"tokens": [], "ner_tags": [], "id": [], "pos_tags": [], "chunk_tags": []}
    id = 0
    for example in dataset:
        tokens, ner_tags = example["tokens"], example["ner_tags"]
        pos_tags, chunk_tags = example["pos_tags"], example["chunk_tags"]


        # Original example
        new_data["id"].append(id)
        id += 1
        new_data["tokens"].append(tokens)
        new_data["pos_tags"].append(pos_tags)
        new_data["chunk_tags"].append(chunk_tags)
        new_data["ner_tags"].append(ner_tags)

        # Duplicate if first tag is not 0
        if ner_tags[0] != 0:
            new_data["id"].append(id)
            id += 1
            modified_tokens = tokens.copy()
            modified_tokens[0] = " " + modified_tokens[0]
            new_data["tokens"].append(modified_tokens)
            new_data["pos_tags"].append(pos_tags)
            new_data["chunk_tags"].append(chunk_tags)
            new_data["ner_tags"].append(ner_tags)

    return new_data


processed_data = preprocess_dataset(dataset["train"])
processed_dataset = datasets.Dataset.from_dict(processed_data)
tokenized_datasets_conll = processed_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_datasets_conll_test = dataset.map(tokenize_and_align_labels, batched=True)
# if wikinn need another ffucnito
# processed_data_wikiann = preprocess_dataset(wikiann_dataset["train"])
# processed_dataset_wikiann = datasets.Dataset.from_dict(processed_data_wikiann)
tokenized_datasets_wikiann = wikiann_dataset.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer)

# Step 6: Training
training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
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
    eval_dataset=tokenized_datasets_conll_test["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.train()


results = trainer.evaluate(tokenized_datasets_conll_test["test"])
print("Test Result:")
print(results)

trainer.save_model("./best_model")
print("=== Now training on conll ===")
training_args.num_train_epochs = 25  # Update to 25 epochs for CoNLL

# Create a new trainer for CoNLL
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets_conll["train"],  # Use CoNLL training dataset
    eval_dataset=tokenized_datasets_conll_test["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


state_dict = load_file(f"./best_model/model.safetensors")
model.load_state_dict(state_dict)

# Update the trainer with the new model for the next dataset
trainer.train_dataset = tokenized_datasets_conll["train"]
trainer.eval_dataset = tokenized_datasets_conll_test["test"]
trainer.learning_rate = 2e-5
trainer.train()



# Step 7: Evaluate
results = trainer.evaluate(tokenized_datasets_conll_test["test"])
print("Test Result:")
print(results)

print("\n=== Best Model Information ===")
print(f"Best Model Checkpoint: {trainer.state.best_model_checkpoint}")
print(f"Best Validation F1 Score: {trainer.state.best_metric:.6f}")

# Evaluate the best model on the test set
print("\n=== Evaluation of Best Model on Test Set ===")
best_model_results = trainer.evaluate(tokenized_datasets_conll_test["test"])
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