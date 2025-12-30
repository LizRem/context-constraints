import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
import torch
import optuna
import joblib
from datetime import datetime

# ============================================================================
# CONFIGURATION - Update these paths for your environment
# ============================================================================
PRETRAIN_DATA_PATH = "data/pretrain_data.csv"  # Path to pretraining data
TOKENIZER_PATH = "tokenizers/tokenizer_llama"  # Path to trained tokenizer
OUTPUT_DIR = "outputs/hyperparameter_optimization/512"  # Output directory
FINAL_PRETRAIN_OUTPUT = "data/pretrain.csv"  # Saved data for final training

# Optimization settings
NUM_TRIALS = 15
EPOCHS_FOR_OPTIM = 3
GRADIENT_ACCUMULATION_STEPS = 4
MAX_LENGTH = 512  # Context length for this optimization

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prepare_data():
    """Load, clean and split the data using a subset for optimization"""
    print("Loading and preparing data subset...")
    
    # Load data (expects CSV with 'aggregated_terms' column containing patient trajectories)
    pretrain_data = pd.read_csv(PRETRAIN_DATA_PATH)  
    print(f"Loaded {len(pretrain_data)} total patients") 
    
    # Hold out 2.5% for optimization, 97.5% for final training
    optim_df, final_pretrain_df = train_test_split(
        pretrain_data,
        test_size=0.975,
        random_state=102
    )
    
    print(f"Using {len(optim_df)} patients for optimization")
    print(f"Saving {len(final_pretrain_df)} patients for final pretraining")
    
    # Save the held-out data for final pretraining
    final_pretrain_df = final_pretrain_df.rename(columns={'aggregated_terms': 'text'}) 
    final_pretrain_df.to_csv(FINAL_PRETRAIN_OUTPUT, index=False)
    print(f"Saved final pretraining data to: {FINAL_PRETRAIN_OUTPUT}")
    
    # Rename columns for optimization data
    optim_df = optim_df.rename(columns={'aggregated_terms': 'text'}) 
    
    # Split optimization data into train/val (95%/5%)
    train_df, val_df = train_test_split(
        optim_df, 
        test_size=0.05, 
        random_state=42,
        shuffle=True
    )
    
    print(f"Optimization train: {len(train_df)} patients, Optimization val: {len(val_df)} patients")
    
    # Create HuggingFace datasets
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df)
    })
    
    return dataset

def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize the dataset with left-side padding and truncation"""
    print(f"Tokenizing dataset with max_length={max_length}...")

    # Set padding and truncation to left
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            padding="max_length", 
            truncation=True,
            max_length=max_length
        ) 
    
    # Use batched processing
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=1
    )
    
    return tokenized_datasets

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    # Generate unique trial directory
    trial_dir = f"{OUTPUT_DIR}/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    # Suggest hyperparameters to optimize
    lr = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.15)
    
    # Load and prepare data (only do this once and reuse)
    if not hasattr(objective, "tokenized_datasets"):
        dataset = load_and_prepare_data()
        objective.tokenized_datasets = tokenize_dataset(dataset, tokenizer, max_length=MAX_LENGTH)
    
    # Calculate effective batch size
    effective_batch_size = batch_size * GRADIENT_ACCUMULATION_STEPS

    # Calculate steps based on data size
    steps_per_epoch = len(objective.tokenized_datasets["train"]) // batch_size
    total_steps = steps_per_epoch * EPOCHS_FOR_OPTIM
    warmup_steps = int(total_steps * warmup_ratio)
    
    print(f"Trial {trial.number}: lr={lr}, batch_size={batch_size} (effective={effective_batch_size}), warmup_ratio={warmup_ratio}")
    
    # Check tokenizer configuration
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Special token IDs: BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, PAD={tokenizer.pad_token_id}")
    
    # LLaMA config - small architecture for clinical domain
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        intermediate_size=2688,
        num_hidden_layers=8,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=MAX_LENGTH,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True
    )
    
    # Initialize the model with random weights
    model = LlamaForCausalLM(config)
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    # Configure data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # Causal LM, not masked LM
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=trial_dir,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS_FOR_OPTIM,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim="adamw_torch",
        learning_rate=lr,
        weight_decay=0.1,
        warmup_ratio=warmup_ratio,
        
        eval_strategy="epoch",
        save_strategy="epoch", 
        save_total_limit=1,
        logging_steps=steps_per_epoch // 5,
        logging_first_step=True,
        prediction_loss_only=True,
        fp16=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=objective.tokenized_datasets["train"],
        eval_dataset=objective.tokenized_datasets["validation"]
    )
    
    # Train the model
    try:
        train_result = trainer.train()
        eval_result = trainer.evaluate()
        eval_loss = eval_result["eval_loss"]
        
        print(f"Trial {trial.number} - Training completed")
        print(f"Trial {trial.number} - eval_loss: {eval_loss}")
        
        # Clean up to save memory
        del model, trainer
        torch.cuda.empty_cache()
        
        return eval_loss
    
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        # Clean up on error
        torch.cuda.empty_cache()
        return float('inf')

def optimize_hyperparameters():
    """Run the Optuna study for hyperparameter optimization"""
    print(f"Starting LLaMA hyperparameter optimization with {NUM_TRIALS} trials...")

    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
    )

    # Run optimization
    study.optimize(objective, n_trials=NUM_TRIALS)

    # Log results
    print("Hyperparameter optimization finished.")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (eval_loss): {best_trial.value}")
    
    # Add effective batch size calculation
    best_params = best_trial.params.copy()
    best_params["effective_batch_size"] = best_params["batch_size"] * GRADIENT_ACCUMULATION_STEPS
    
    print("  Parameters:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    # Save study
    study_path = f"{OUTPUT_DIR}/optuna_study.pkl"
    joblib.dump(study, study_path)
    
    # Save best parameters to file
    best_params_path = f"{OUTPUT_DIR}/best_parameters.txt"
    with open(best_params_path, 'w') as f:
        f.write("Best LLaMA Hyperparameters:\n")
        f.write("=" * 40 + "\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest eval_loss: {best_trial.value}\n")

    return best_params

if __name__ == "__main__":
    try:
        print(f"=== Starting LLaMA hyperparameter optimization ===")
        
        # Run hyperparameter optimization
        best_params = optimize_hyperparameters()
        
        print("\n=== Optimization complete ===")
        print(f"Best parameters found:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        print(f"\nResults saved to: {OUTPUT_DIR}/best_parameters.txt")
        
    except Exception as e:
        print(f"Error in execution: {e}")
        import traceback
        traceback.print_exc()
