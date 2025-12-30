import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset, Dataset, DatasetDict
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, EvalPrediction 
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import EarlyStoppingCallback
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, multilabel_confusion_matrix, average_precision_score
import json
import optuna
import joblib

# ============================================================================
# CONFIGURATION - Update these paths for your environment
# ============================================================================
BASE_MODEL_PATH = "models/pretrained/512"  # Path to pretrained model
OPTIMIZATION_OUTPUT_DIR = "outputs/finetuning/optimization/retinopathy"
FINAL_OUTPUT_DIR = "outputs/finetuning/retinopathy"

# Data paths
DATA_PATH_TRAIN = "data/finetuning/retinopathy_train.csv"
DATA_PATH_TEST = "data/finetuning/retinopathy_test.csv"
DATA_PATH_VAL = "data/finetuning/retinopathy_val.csv"

# Optimization settings
NUM_TRIALS = 15
OPTIMIZATION_STEPS = 2000
GRADIENT_ACCUMULATION_STEPS = 4
WEIGHT_DECAY = 0.01
MAX_LENGTH = 512 # this can be adjusted to any model length you require (512, 1024, 2048, 4096)

# Create directories
os.makedirs(OPTIMIZATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Data Loading Functions
# ============================================================================
def load_optimization_data():
    """Load subset data for hyperparameter optimization"""
    print("Loading data for optimization...")
    
    # Load data (expects columns: 'aggregated_terms' for text, 'retino_10y' for label)
    finetune_all = pd.read_csv(DATA_PATH_TRAIN) 
    print(f"Loaded {len(finetune_all)} total patients")
    
    # 15% for optimization, 85% saved for final training
    optim_df, finetune_train_df = train_test_split(
        finetune_all,
        test_size=0.85,
        random_state=42,
        stratify=finetune_all['retino_10y']
    )
    
    print(f"Using {len(optim_df)} patients for optimization")
    
    # Rename columns to standard format
    optim_df = optim_df.rename(columns={
        'aggregated_terms': 'text',
        'retino_10y': 'label'
    })[['text', 'label']]
    
    print("Label distribution:", Counter(optim_df['label']))
    
    # Split optimization data into train/val (85%/15%)
    train_df, val_df = train_test_split(
        optim_df, 
        test_size=0.15,
        random_state=42,
        shuffle=True,
        stratify=optim_df['label'] 
    )
    
    print(f"Optimization train: {len(train_df)}, val: {len(val_df)}")
    
    df_subset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df)
    })
    
    return df_subset, finetune_train_df  

def load_full_data(finetune_train_df):
    """Load full dataset for final model training"""
    print("Loading full dataset for final training...")
    
    finetune_val_df = pd.read_csv(DATA_PATH_VAL)
    finetune_test_df = pd.read_csv(DATA_PATH_TEST)

    finetune_train_df = finetune_train_df.reset_index(drop=True)

    # Rename columns to standard format
    finetune_train_df = finetune_train_df.rename(columns={
        'aggregated_terms': 'text',
        'retino_10y': 'label'
    })[['text', 'label']]

    finetune_test_df = finetune_test_df.rename(columns={
        'aggregated_terms': 'text',
        'retino_10y': 'label'
    })[['text', 'label']]

    finetune_val_df = finetune_val_df.rename(columns={
        'aggregated_terms': 'text',
        'retino_10y': 'label'
    })[['text', 'label']]
    
    print(f"Training dataset: {len(finetune_train_df)}, labels:{Counter(finetune_train_df['label'])}")
    print(f"Validation dataset: {len(finetune_val_df)}, labels: {Counter(finetune_val_df['label'])}")
    print(f"Test dataset: {len(finetune_test_df)}, labels: {Counter(finetune_test_df['label'])}")
   
    df = DatasetDict({
        'train': Dataset.from_pandas(finetune_train_df),
        'validation': Dataset.from_pandas(finetune_val_df),
        'test': Dataset.from_pandas(finetune_test_df)
    })
                             
    return df

# ============================================================================
# Tokenization
# ============================================================================
def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset with left-side padding and truncation"""
    tokenizer.padding_side = "left" 
    tokenizer.truncation_side = "left"
    
    print(f"Using left truncation and left padding")
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        labels = torch.tensor(examples["label"], dtype=torch.float).unsqueeze(1)
        tokenized["labels"] = labels
        return tokenized
    
    encoded_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset['train'].column_names
    )

    return encoded_dataset

# ============================================================================
# Model Definition
# ============================================================================
class LlamaForBinaryClassification(nn.Module):
    """LLaMA model with binary classification head"""
    def __init__(self, base_model_path, num_labels=1):
        super().__init__()
        self.llama = AutoModelForCausalLM.from_pretrained(base_model_path)
        hidden_size = self.llama.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
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
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}

# ============================================================================
# Metrics and Evaluation
# ============================================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

def compute_metrics(p: EvalPrediction):
    """Compute classification metrics"""
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds_flat = preds.flatten()
    labels_flat = p.label_ids.flatten()
    
    y_pred = (sigmoid(preds_flat) >= 0.5).astype(int)
    y_true = labels_flat
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, sigmoid(preds_flat))
        auprc = average_precision_score(y_true, sigmoid(preds_flat))
    except:
        roc_auc = 0.0
        auprc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'auprc': auprc
    }

# ============================================================================
# Hyperparameter Optimization
# ============================================================================
def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    trial_dir = f"{OPTIMIZATION_OUTPUT_DIR}/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.model_max_length = MAX_LENGTH
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    # Use cached optimization data
    if not hasattr(objective, "encoded_dataset"):
        df_subset, objective.saved_train_df = load_optimization_data()
        objective.encoded_dataset = tokenize_dataset(df_subset, tokenizer)
    
    # Suggest hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)
    
    print(f"Trial {trial.number}: lr={lr}, batch_size={batch_size}, warmup_ratio={warmup_ratio}")
    
    # Initialize model
    model = LlamaForBinaryClassification(base_model_path=BASE_MODEL_PATH, num_labels=1)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=trial_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=lr,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=warmup_ratio,
        max_steps=OPTIMIZATION_STEPS,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        fp16=True,
        seed=42,
        report_to="none",
        optim="adamw_torch",
        save_total_limit=1,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=objective.encoded_dataset["train"],
        eval_dataset=objective.encoded_dataset["validation"],
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    try:
        trainer.train()
        eval_result = trainer.evaluate()
        roc_auc_score = eval_result["eval_roc_auc"]
        
        print(f"Trial {trial.number} - ROC AUC Score: {roc_auc_score}")
        
        del model, trainer
        torch.cuda.empty_cache()
        
        return roc_auc_score
    
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        torch.cuda.empty_cache()
        return 0.0

def optimize_hyperparameters():
    """Run hyperparameter optimization using Optuna"""
    print(f"\n{'='*50}")
    print("PHASE 1: HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*50}")
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    
    study.optimize(objective, n_trials=NUM_TRIALS)
    
    print(f"\nOptimization complete!")
    print(f"Best ROC AUC Score: {study.best_trial.value}")
    print(f"Best parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    joblib.dump(study, f"{OPTIMIZATION_OUTPUT_DIR}/study.pkl")
    
    return study.best_trial.params

# ============================================================================
# Final Model Training
# ============================================================================
def train_final_model(best_params):
    """Train final model with optimized parameters on full dataset"""
    print(f"\n{'='*50}")
    print("PHASE 2: FINAL MODEL TRAINING")
    print(f"{'='*50}")
    
    # Load full dataset
    df_full = load_full_data(objective.saved_train_df)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.model_max_length = MAX_LENGTH
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    # Tokenize full dataset
    print("Tokenizing full dataset...")
    encoded_dataset = tokenize_dataset(df_full, tokenizer)
    
    # Initialize model
    model = LlamaForBinaryClassification(base_model_path=BASE_MODEL_PATH, num_labels=1)
    
    # Training arguments with optimized parameters
    training_args = TrainingArguments(
        output_dir=FINAL_OUTPUT_DIR,
        per_device_train_batch_size=best_params["batch_size"],
        per_device_eval_batch_size=best_params["batch_size"],
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=best_params["learning_rate"],
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=best_params["warmup_ratio"],
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        fp16=True,
        seed=42,
        report_to="none",
        optim="adamw_torch",
        save_total_limit=2,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    # Train final model
    print("Training final model with optimized hyperparameters...")
    trainer.train()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=encoded_dataset["test"])
    print("Test Results:", test_results)
    
    # Get detailed predictions
    predictions = trainer.predict(encoded_dataset["test"])
    y_pred_logits = predictions.predictions.flatten()
    y_true = predictions.label_ids.flatten()
    y_pred_probs = sigmoid(y_pred_logits)
    y_pred_binary = (y_pred_probs > 0.5).astype(int)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_binary, target_names=["no_retino", "retino"], zero_division=0))
    
    # Bootstrap confidence intervals
    bootstrap_results = bootstrap_confidence_intervals(y_true, y_pred_binary, y_pred_probs, n_bootstraps=1000)
    
    print("\nBootstrap Results (95% Confidence Intervals):")
    for metric, values in bootstrap_results.items():
        print(f"{metric.upper()}: {values['mean']:.3f} ({values['lower']:.3f} - {values['upper']:.3f})")
    
    # Save everything
    trainer.save_model(FINAL_OUTPUT_DIR)
    tokenizer.save_pretrained(FINAL_OUTPUT_DIR)
    
    # Save results
    with open(f"{FINAL_OUTPUT_DIR}/test_results.json", "w") as f:
        json.dump(test_results, f)
    
    with open(f"{FINAL_OUTPUT_DIR}/best_params.json", "w") as f:
        json.dump(best_params, f)
    
    np.savez(f"{FINAL_OUTPUT_DIR}/test_predictions.npz", 
             y_true=y_true, 
             y_pred_probs=y_pred_probs, 
             y_pred_binary=y_pred_binary)
    
    # Save bootstrap results
    with open(f"{FINAL_OUTPUT_DIR}/bootstrap_results.json", "w") as f:
        serializable_results = {}
        for metric, values in bootstrap_results.items():
            serializable_results[metric] = {
                'mean': float(values['mean']),
                'lower': float(values['lower']),
                'upper': float(values['upper'])
            }
        json.dump(serializable_results, f)
    
    # Save training history
    with open(f"{FINAL_OUTPUT_DIR}/training_history.json", "w") as f:
        json.dump(trainer.state.log_history, f)
    
    print(f"\nFinal model saved to: {FINAL_OUTPUT_DIR}")
    
    return test_results

# ============================================================================
# Main Pipeline
# ============================================================================
if __name__ == "__main__":
    try:
        print("COMBINED OPTIMIZATION AND TRAINING PIPELINE")
        print("=" * 60)
        
        # Phase 1: Optimize hyperparameters
        best_params = optimize_hyperparameters()
        
        # Phase 2: Train final model
        final_results = train_final_model(best_params)
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"Optimized parameters: {best_params}")
        print(f"Final test ROC AUC: {final_results.get('eval_roc_auc', 'N/A')}")
        print(f"Final test F1 score: {final_results.get('eval_f1', 'N/A')}")
        print(f"Model saved to: {FINAL_OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
