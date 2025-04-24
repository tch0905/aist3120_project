import torch
import numpy as np
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
from seqeval.metrics import classification_report
from torch import Tensor
from typing import Optional

def save_model_params_and_f1(trainer, output_file="model_params_and_f1.txt"):
    """
    Saves the shapes of all model parameters and the best F1 score to a text file.
    
    Args:
        trainer: The trained Trainer object containing the model and its state.
        output_file: The path to the output text file (default: 'model_params_and_f1.txt').
    """
    # Get the best model (loaded at the end due to load_best_model_at_end=True)
    model = trainer.model
    best_f1 = trainer.state.best_metric  # Best validation F1 score
    
    # Open the file in write mode
    with open(output_file, 'w') as f:
        f.write("=== Model Parameter Shapes and Best F1 Score ===\n\n")
        
        # Write the best F1 score
        f.write(f"Best Validation F1 Score: {best_f1:.6f}\n\n")
        
        # Write header for parameter shapes
        f.write("Model Parameter Shapes:\n")
        f.write("-" * 50 + "\n")
        
        # Iterate over all named parameters in the model
        for name, param in model.named_parameters():
            shape = list(param.shape)  # Convert torch.Size to a list for readability
            f.write(f"Parameter: {name}\n")
            f.write(f"Shape: {shape}\n")
            f.write(f"Number of elements: {param.numel()}\n")
            f.write("\n")
        
        f.write("-" * 50 + "\n")
        f.write("Total number of parameters: {}\n".format(sum(p.numel() for p in model.parameters())))
    
    print(f"Model parameter shapes and best F1 score saved to '{output_file}'")

def save_model_and_hparams(model, trainer, file_path):
    with open(file_path, 'w') as f:
        # 1. Save hyperparameters first
        f.write("=== TRAINING HYPERPARAMETERS ===\n")
        f.write(f"Model type: {model.__class__.__name__}\n")
        f.write(f"Loss function: {model.loss_type}\n")
        if hasattr(model, 'loss_fn'):
            f.write(f"Loss function details: {str(model.loss_fn)}\n")
        
        f.write("\n=== TRAINING ARGUMENTS ===\n")
        for attr, value in trainer.args.__dict__.items():
            f.write(f"{attr}: {value}\n")
        
        # 2. Save model architecture
        f.write("\n=== MODEL ARCHITECTURE ===\n")
        f.write(str(model) + "\n")
        
        # 3. Save parameter summary
        f.write("\n=== PARAMETER SUMMARY ===\n")
        total_params = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            f.write(f"{name}: {param.shape} | {num_params:,} parameters | "
                   f"{'Trainable' if param.requires_grad else 'Frozen'}\n")
        
        f.write(f"\nTotal parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
        
        # 4. Optionally save full parameter values (commented out as it's very large)
        # f.write("\n=== DETAILED PARAMETER VALUES ===\n")
        # for name, param in model.named_parameters():
        #     f.write(f"\nParameter name: {name}\n")
        #     f.write(f"Shape: {param.shape}\n")
        #     f.write(f"Values:\n{param.data}\n")
        #     f.write("-" * 80 + "\n")

def save_test_results_and_hparams(trainer, model, test_results, file_path):
    with open(file_path, 'w') as f:
        # 1. Save key test metrics (F1, Precision, Recall)
        f.write("=== TEST SET RESULTS ===\n")
        f.write(f"Overall F1: {test_results.get('eval_overall_f1', 0):.6f}\n")
        f.write(f"Overall Precision: {test_results.get('eval_overall_precision', 0):.6f}\n")
        f.write(f"Overall Recall: {test_results.get('eval_overall_recall', 0):.6f}\n")
        f.write(f"Exact Match Accuracy: {test_results.get('eval_accuracy', 0):.6f}\n\n")

        # 2. Save per-class F1 scores (if available)
        if any(k.startswith('eval_') and k.endswith('_f1') for k in test_results.keys()):
            f.write("=== PER-CLASS F1 SCORES ===\n")
            for key in test_results:
                if key.startswith('eval_') and key.endswith('_f1'):
                    label_name = key[5:-3]  # Remove 'eval_' and '_f1'
                    f.write(f"{label_name}: {test_results[key]:.6f}\n")
            f.write("\n")

        # 3. Save hyperparameters
        f.write("=== TRAINING HYPERPARAMETERS ===\n")
        f.write(f"Model type: {model.__class__.__name__}\n")
        f.write(f"Loss function: {model.loss_type}\n")
        if hasattr(model, 'loss_fn'):
            f.write(f"Loss function details: {str(model.loss_fn)}\n")

        # 4. Save training arguments
        f.write("\n=== TRAINING ARGUMENTS ===\n")
        for attr, value in trainer.args.__dict__.items():
            f.write(f"{attr}: {value}\n")

        # 5. Save best model checkpoint info
        f.write("\n=== BEST MODEL INFO ===\n")
        f.write(f"Best checkpoint: {trainer.state.best_model_checkpoint}\n")
        f.write(f"Best validation F1: {trainer.state.best_metric:.6f}\n")