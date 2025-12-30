import os
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding, EarlyStoppingCallback
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, average_precision_score
from sklearn.utils import resample
from collections import Counter
import json

# ============================================================================
# CONFIGURATION - Update these paths for your environment
# ============================================================================
PRETRAINED_MODEL_PATH = "models/pretrained/4096"  # Path to pretrained model
DATA_PATH_TRAIN = "data/finetuning/retinopathy_train.csv"
DATA_PATH_TEST = "data/finetuning/retinopathy_test.csv"
DATA_PATH_VAL = "data/finetuning/retinopathy_val.csv"
OUTPUT_DIR = "outputs/linear_probing/4096/retinopathy"
MAX_LENGTH = 4096

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Model Definition - Frozen LLaMA with Trainable Classification Head
# ============================================================================
class FrozenLlamaForClassification(nn.Module):
    """
    Frozen LLaMA model with trainable classification head (linear probing).
    All LLaMA parameters are frozen, only the classifier is trained.
    """
    def __init__(self, pretrained_model_path, num_labels=1):
        super().__init__()
        self.llama = LlamaForCausalLM.from_pretrained(pretrained_model_path)
        hidden_size = self.llama.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Freeze all LLaMA parameters
        for param in self.llama.parameters():
            param.requires_grad = False
            
        # Only classifier parameters are trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        # LLaMA forward pass with frozen weights
        with torch.no_grad():
            outputs = self.llama(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
    
        # Use last token's hidden state for classification
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden = last_hidden_state[:, -1, :]
        logits = self.classifier(last_token_hidden)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            logits_squeezed = logits.squeeze(-1)
            loss = loss_fct(logits_squeezed, labels.float())
        
        return {'loss': loss, 'logits': logits}

# ============================================================================
# Data Loading
# ============================================================================
def load_and_prepare_data():
    """
    Load and prepare data for linear probing.
    Removes the data subset used for hyperparameter optimization to ensure
    fair comparison with fine-tuning experiments.
    """
    print("Loading and preparing data...")
    
    # Load ALL training data
    finetune_all = pd.read_csv(DATA_PATH_TRAIN) 
    print(f"Loaded {len(finetune_all)} total patients from training set")
    
    # Remove the 15% used for optimization to match fine-tuning exactly
    # IMPORTANT: Use the same random_state as finetuning optimization
    optim_df, finetune_train_df = train_test_split(
        finetune_all,
        test_size=0.85,
        random_state=42,  # MUST match the seed used in fine-tuning optimization
        stratify=finetune_all['retino_10y']
    )
    
    print(f"Excluded {len(optim_df)} patients used for optimization")
    print(f"Using {len(finetune_train_df)} patients for linear probing (same as fine-tuning)")
    
    # Load validation and test sets
    finetune_val_df = pd.read_csv(DATA_PATH_VAL)
    finetune_test_df = pd.read_csv(DATA_PATH_TEST)
    
    # Reset index
    finetune_train_df = finetune_train_df.reset_index(drop=True)
    
    # Rename columns to standard format
    finetune_train_df = finetune_train_df.rename(columns={
        'aggregated_terms': 'text',
        'retino_10y': 'label'
    })[['text', 'label']]

    finetune_val_df = finetune_val_df.rename(columns={
        'aggregated_terms': 'text',
        'retino_10y': 'label'
    })[['text', 'label']]

    finetune_test_df = finetune_test_df.rename(columns={
        'aggregated_terms': 'text',
        'retino_10y': 'label'
    })[['text', 'label']]
    
    print(f"Training dataset label distribution:", Counter(finetune_train_df['label']))
    print(f"Validation dataset label distribution:", Counter(finetune_val_df['label']))
    print(f"Test dataset label distribution:", Counter(finetune_test_df['label']))

    # Create DatasetDict
    all_df = DatasetDict({
        'train': Dataset.from_pandas(finetune_train_df),
        'validation': Dataset.from_pandas(finetune_val_df),
        'test': Dataset.from_pandas(finetune_test_df)
    })
                             
    return all_df

# ============================================================================
# Tokenization
# ============================================================================
def tokenize_dataset(dataset_dict, tokenizer):
    """Tokenize the dataset"""
    def tokenize(batch):
        return tokenizer(batch["text"], 
                        truncation=True,
                        padding="max_length",
                        max_length=MAX_LENGTH,
                        return_tensors="pt")

    tokenized_dataset = dataset_dict.map(tokenize, 
                                        batched=True,
                                        remove_columns=["text"])
    
    # Set format for PyTorch
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    return tokenized_dataset

# ============================================================================
# Metrics and Evaluation
# ============================================================================
def compute_metrics(eval_pred):
    """Compute classification metrics"""
    logits, labels = eval_pred
    
    # Handle shape difference - logits are [batch_size, 1], squeeze to [batch_size]
    if logits.ndim > 1:
        logits = logits.squeeze(-1)
    
    predictions = (logits > 0).astype(int).flatten()
    probs = torch.sigmoid(torch.tensor(logits)).numpy().flatten()

    accuracy_value = accuracy_score(labels, predictions)
    precision_value = precision_score(labels, predictions, average='binary', zero_division=0)
    recall_value = recall_score(labels, predictions, average='binary', zero_division=0)
    f1_value = f1_score(labels, predictions, average='binary', zero_division=0)
    auprc_value = average_precision_score(labels, probs)
    auroc_value = roc_auc_score(labels, probs)

    return {
        "accuracy": accuracy_value,
        "precision": precision_value,
        "recall": recall_value,
        "f1": f1_value,
        "auprc": auprc_value,
        "auroc": auroc_value
    }

def bootstrap_confidence_intervals(y_true, y_pred, y_scores, n_bootstraps=1000, ci=95):
    """Calculate bootstrap confidence intervals for classification metrics"""
    bootstrapped_precision = []
    bootstrapped_recall = []
    bootstrapped_f1 = []
    bootstrapped_auroc = []
    bootstrapped_auprc = []
    
    np.random.seed(42)
    
    for i in range(n_bootstraps):
        indices = resample(range(len(y_true)), replace=True, n_samples=len(y_true))
        
        y_true_bootstrap = np.array(y_true)[indices]
        y_pred_bootstrap = np.array(y_pred)[indices]
        y_scores_bootstrap = np.array(y_scores)[indices]
        
        if len(np.unique(y_true_bootstrap)) > 1:
            bootstrapped_precision.append(precision_score(y_true_bootstrap, y_pred_bootstrap, zero_division=0))
            bootstrapped_recall.append(recall_score(y_true_bootstrap, y_pred_bootstrap, zero_division=0))
            bootstrapped_f1.append(f1_score(y_true_bootstrap, y_pred_bootstrap, zero_division=0))
            bootstrapped_auroc.append(roc_auc_score(y_true_bootstrap, y_scores_bootstrap))
            bootstrapped_auprc.append(average_precision_score(y_true_bootstrap, y_scores_bootstrap))
    
    alpha = (100 - ci) / 2 / 100
    ci_lower = alpha * 100
    ci_upper = (1 - alpha) * 100
    
    results = {}
    for name, values in [
        ('precision', bootstrapped_precision), 
        ('recall', bootstrapped_recall),
        ('f1', bootstrapped_f1),
        ('auroc', bootstrapped_auroc),
        ('auprc', bootstrapped_auprc)
    ]:
        mean_val = np.mean(values)
        lower_val = max(0, np.percentile(values, ci_lower))
        upper_val = min(1, np.percentile(values, ci_upper))
        results[name] = {
            'mean': mean_val,
            'lower': lower_val,
            'upper': upper_val
        }
    
    return results

# ============================================================================
# Main Training Pipeline
# ============================================================================
def main():
    # Start total timer
    total_start = time.time()
    
    print("Starting Linear Probing: Frozen LLaMA + Trainable Classification Head")
    print("=" * 60)
    
    # Load data
    all_df = load_and_prepare_data()
    
    # Load tokenizer and setup
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    tokenizer.model_max_length = MAX_LENGTH
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
   
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(all_df, tokenizer)
    
    # Save tokenized dataset
    tokenized_dataset.save_to_disk(f"{OUTPUT_DIR}/tokenized_data")
    
    # Initialize model
    print("Initializing model...")
    model = FrozenLlamaForClassification(PRETRAINED_MODEL_PATH, num_labels=1)
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Print parameter statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Statistics:")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Percentage trainable: {100 * trainable_params / total_params:.4f}%")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        num_train_epochs=3,
        learning_rate=5e-4,  # Higher learning rate since only training head
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="auroc",
        greater_is_better=True,
        report_to="none",
        fp16=True,
        seed=42
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train with timer
    print("\nTraining frozen LLaMA with trainable classification head...")
    training_start = time.time()
    trainer.train()
    training_time = time.time() - training_start
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print("Test Results:", test_results)
    
    # Get predictions
    predictions = trainer.predict(tokenized_dataset["test"])
    y_pred_logits = predictions.predictions.flatten()
    y_true = predictions.label_ids.flatten()
    
    # Apply sigmoid and threshold
    y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))
    y_pred_binary = (y_pred_probs > 0.5).astype(int)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=["no_retino", "retino"], zero_division=0))
    
    # Bootstrap confidence intervals
    bootstrap_results = bootstrap_confidence_intervals(y_true, y_pred_binary, y_pred_probs)
    
    print("\nBootstrap Results (95% Confidence Intervals):")
    for metric, values in bootstrap_results.items():
        print(f"{metric.upper()}: {values['mean']:.3f} ({values['lower']:.3f} - {values['upper']:.3f})")
    
    # Calculate total time
    total_time = time.time() - total_start

    # Print timing results
    print("\n" + "="*60)
    print("TIMING RESULTS:")
    print(f"Training time: {training_time/3600:.2f} hours ({training_time:.1f} seconds)")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time:.1f} seconds)")
    print("="*60)
    
    # Save everything
    print(f"\nSaving results to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    with open(f"{OUTPUT_DIR}/training_history.json", "w") as f:
        json.dump(trainer.state.log_history, f)
    
    with open(f"{OUTPUT_DIR}/eval_results.json", "w") as f:
        json.dump(test_results, f)
    
    np.savez(f"{OUTPUT_DIR}/test_predictions.npz", 
             y_true=y_true, 
             y_pred_probs=y_pred_probs, 
             y_pred_binary=y_pred_binary)
    
    # Save bootstrap results
    with open(f"{OUTPUT_DIR}/bootstrap_results.json", "w") as f:
        serializable_results = {}
        for metric, values in bootstrap_results.items():
            serializable_results[metric] = {
                'mean': float(values['mean']),
                'lower': float(values['lower']),
                'upper': float(values['upper'])
            }
        json.dump(serializable_results, f)
    
    # Save timing results
    timing_results = {
        'training_time_seconds': training_time,
        'training_time_hours': training_time/3600,
        'total_time_seconds': total_time,
        'total_time_hours': total_time/3600
    }
    
    with open(f"{OUTPUT_DIR}/timing_results.json", "w") as f:
        json.dump(timing_results, f)
    
    print("\nLinear probing complete!")
    
    return test_results

if __name__ == "__main__":
    results = main()
