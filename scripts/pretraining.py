import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd 
import numpy as np
import torch
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict, load_from_disk 
from sklearn.model_selection import train_test_split

# Optional: Login to Hugging Face Hub if you need to push models
# from huggingface_hub import login
# login(token="your_token_here")

# ============================================================================
# CONFIGURATION - Update these paths for your environment
# ============================================================================
TOKENIZED_DATA_PATH = "data/tokenized/512"  # Path to save tokenized data
PRETRAIN_DATASET_PATH = "data/pretrain_split"  # Path to load pretrain data, this should exclude any data used during optimisation
TOKENIZER_PATH = "tokenizers/tokenizer_llama"  # Path to your trained tokenizer
OUTPUT_DIR = "models/pretrained/512"  # Output directory for trained model

# Training configuration
VAL_SIZE = 0.05
MAX_LENGTH = 512
NUM_EPOCHS = 7

# Optimized hyperparameters (from hyperparameter optimization)
LEARNING_RATE = 0.00019809338952032915
BATCH_SIZE = 4
WARMUP_RATIO = 0.08253303307632644
GRADIENT_ACCUMULATION_STEPS = 4

# ============================================================================
# Load and prepare data
# ============================================================================
print("Loading datasets...")
dataset_dict = load_from_disk(PRETRAIN_DATASET_PATH)
train_dataset = dataset_dict['train']
val_dataset = dataset_dict['validation']

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

# Check tokenizer configuration
print(f"Tokenizer vocabulary size: {len(tokenizer)}")
print(f"Special tokens: {tokenizer.special_tokens_map}")
print(f"Special token IDs: BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, PAD={tokenizer.pad_token_id}")

# ============================================================================
# Tokenization
# ============================================================================
def tokenize_function(batch):
    """Tokenize text with left-side padding and truncation"""
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

# Apply tokenization
print("Tokenizing datasets...")
train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

val_tokenized = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

tokenized_datasets = DatasetDict({
    'train': train_tokenized,
    'validation': val_tokenized
})

# Save tokenized datasets
os.makedirs(os.path.dirname(TOKENIZED_DATA_PATH), exist_ok=True)
tokenized_datasets.save_to_disk(TOKENIZED_DATA_PATH)
print(f"Tokenized datasets saved to: {TOKENIZED_DATA_PATH}")

# ============================================================================
# Model configuration
# ============================================================================
# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM: predict next token
)

# LLaMA model configuration - small efficient architecture for clinical domain
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

# Initialize model with random weights
model = LlamaForCausalLM(config)
print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
print(f"Model context length: {config.max_position_embeddings}")

# ============================================================================
# Training setup
# ============================================================================
# Calculate training metrics
effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS 
steps_per_epoch = len(tokenized_datasets["train"]) // BATCH_SIZE
total_steps = steps_per_epoch * NUM_EPOCHS 
eval_steps = max(1000, steps_per_epoch // 5)

print(f"Effective batch size: {effective_batch_size}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {total_steps}")
print(f"Evaluation every {eval_steps} steps")

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim="adamw_torch", 
    learning_rate=LEARNING_RATE, 
    weight_decay=0.1, 
    warmup_ratio=WARMUP_RATIO,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_steps=eval_steps,
    save_total_limit=2,
    logging_steps=eval_steps,
    prediction_loss_only=True,
    fp16=True,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # GPU optimization
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# ============================================================================
# Training
# ============================================================================
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

print("Starting training...")
train_result = trainer.train()
print(f"Training completed!")
print(f"Training stats: {train_result}")

# ============================================================================
# Save model and extract metrics
# ============================================================================
print(f"Saving model to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Extract training and evaluation loss from log history
loss_values = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
eval_loss_values = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]

print("\nTraining Loss Values:")
print(loss_values)

print("\nEvaluation Loss Values:")
print(eval_loss_values)

# Save loss values to file
loss_file = os.path.join(OUTPUT_DIR, "training_metrics.txt")
with open(loss_file, 'w') as f:
    f.write("Training Metrics\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Final training loss: {loss_values[-1] if loss_values else 'N/A'}\n")
    f.write(f"Final evaluation loss: {eval_loss_values[-1] if eval_loss_values else 'N/A'}\n\n")
    f.write("Training loss history:\n")
    f.write(str(loss_values) + "\n\n")
    f.write("Evaluation loss history:\n")
    f.write(str(eval_loss_values) + "\n")

print(f"\nTraining metrics saved to: {loss_file}")
print("Pretraining complete!")
