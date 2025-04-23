import datasets
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

from model import BertWithMLPForNER
from dataset import dataset, num_labels
from compute_metrics import compute_metrics
from tokenize_and_align_labels import tokenize_and_align_labels, tokenizer
from safetensors.torch import load_file

def preprocess_dataset_test(dataset):
    new_data = {"tokens": [], "ner_tags": [], "id": [], "pos_tags": [], "chunk_tags": []}
    id = 0
    for example in dataset:
        tokens, ner_tags = example["tokens"], example["ner_tags"]
        pos_tags, chunk_tags = example["pos_tags"], example["chunk_tags"]

        new_data["id"].append(id)
        id += 1
        modified_tokens = tokens.copy()
        modified_tokens[0] = modified_tokens[0]
        new_data["tokens"].append(modified_tokens)
        new_data["pos_tags"].append(pos_tags)
        new_data["chunk_tags"].append(chunk_tags)
        new_data["ner_tags"].append(ner_tags)

    return new_data

processed_data_test = preprocess_dataset_test(dataset["test"])
processed_dataset_test = datasets.Dataset.from_dict(processed_data_test)
tokenized_datasets_conll_test = processed_dataset_test.map(tokenize_and_align_labels, batched=True)

BATCH_SIZE = 32
model = BertWithMLPForNER(
    num_labels,
    loss_type='focal',
)
state_dict = load_file(f"./checkpoint-1768/model.safetensors")
model.load_state_dict(state_dict)
model.eval()

data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=0,
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
    eval_dataset=tokenized_datasets_conll_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

results = trainer.evaluate(tokenized_datasets_conll_test)