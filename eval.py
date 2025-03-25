import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import Trainer, DataCollatorForTokenClassification, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np

# Load dataset (e.g., CoNLL-2003)
dataset = load_dataset("conll2003")

# Load tokenizer and model from the checkpoint
checkpoint_path = "C:\\Users\\tch0905\\PycharmProjects\\aist3120\\proj\\aist3120_project\\results\\checkpoint-2634"
tokenizer = BertTokenizerFast.from_pretrained(checkpoint_path)
model = BertForTokenClassification.from_pretrained(checkpoint_path)

# Load the seqeval metric for NER evaluation
seqeval = evaluate.load("seqeval")


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


# Tokenize the validation dataset
tokenized_eval_dataset = dataset['test'].map(tokenize_and_align_labels, batched=True)

# Use DataCollatorForTokenClassification to handle padding
data_collator = DataCollatorForTokenClassification(tokenizer)

# Label mappings
label_list = dataset["train"].features["ner_tags"].feature.names  # Get label names


# Function to compute metrics (F1 score)
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Convert predictions and labels to entity names
    true_labels = [[label_list[label] for label in labels[i] if label != -100] for i in range(len(labels))]
    true_predictions = [[label_list[pred] for (pred, label) in zip(predictions[i], labels[i]) if label != -100] for i in
                        range(len(labels))]

    # Compute the seqeval metric
    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }


# Create Trainer instance for evaluation
trainer = Trainer(
    model=model,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics  # Pass the metric function
)

# Evaluate the model
evaluation_results = trainer.evaluate()

# Print the F1 score
print(f"F1 Score: {evaluation_results['eval_f1']:.4f}")