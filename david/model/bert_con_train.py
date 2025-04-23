import argparse
import torch
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
from safetensors.torch import load_file
from model import BertWithMLPForNER  # Ensure this is your custom model class
from concat.save_best_model import save_model_and_hparams, save_test_results_and_hparams

# Load your tokenizer and datasets (assumed preloaded)
# Step 1: Load Dataset and Labels
dataset = load_from_disk("../../conll2003_local")
wikiann_dataset = load_from_disk("../../wikiann_local")

# CoNLL-2003 Label Names
label_names = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_names)

tokenized_datasets_conll = dataset.map(tokenize_and_align_labels, batched=True)
tokenized_datasets_wikiann = wikiann_dataset.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer)


# Parse checkpoint path from command line
parser = argparse.ArgumentParser(description="Continue training from checkpoint")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint (.safetensors)")
args = parser.parse_args()

# Load label names and number
label_names = tokenized_datasets_conll["train"].features["ner_tags"].feature.names
num_labels = len(label_names)

# Load model
model = BertWithMLPForNER(num_labels=num_labels, loss_type='focal')
state_dict = load_file(args.checkpoint_path)
model.load_state_dict(state_dict)

# Define training arguments for 10 more epochs
training_args = TrainingArguments(
    output_dir="./continued_training",
    per_device_train_batch_size=64 + 24,
    per_device_eval_batch_size=64 + 24,
    num_train_epochs=10,
    learning_rate=2e-5,
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
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets_conll["train"],
    eval_dataset=tokenized_datasets_conll["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train for 10 more epochs
trainer.train()

# Final evaluation
results = trainer.evaluate(tokenized_datasets_conll["test"])
print("Final Evaluation on Test Set:")
for metric, value in results.items():
    print(f"{metric}: {value:.6f}" if isinstance(value, float) else f"{metric}: {value}")

# Save model and results
save_model_and_hparams(trainer.model, trainer, "continued_best_model_info.txt")
save_test_results_and_hparams(
    trainer,
    trainer.model,
    results,
    "continued_test_results_and_hparams.txt"
)

print("✅ Continued training complete.")
print("📁 Model and results saved.")