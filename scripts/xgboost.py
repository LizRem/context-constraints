import pandas as pd
import numpy as np
import xgboost as xgb
from collections import Counter
import json
import time
from scipy.sparse import csr_matrix
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, 
                            f1_score, auc, accuracy_score, confusion_matrix, 
                            mean_squared_error, recall_score, ConfusionMatrixDisplay,
                            classification_report)
from xgboost import XGBClassifier
from sklearn.utils import resample
import optuna
import os

# ============================================================================
# CONFIGURATION - Update these paths for your environment
# ============================================================================
TRAIN_DATA_PATH = "data/finetuning/retinopathy_train.csv"
TEST_DATA_PATH = "data/finetuning/retinopathy_test.csv"
VAL_DATA_PATH = "data/finetuning/retinopathy_val.csv"
OUTPUT_DIR = "outputs/xgboost/retinopathy"
MAX_CODES = 512  # Use last 512 codes (most recent)
NUM_TRIALS = 50  # Optuna hyperparameter optimization trials

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Load Data
# ============================================================================
print("Loading data...")
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)
val_df = pd.read_csv(VAL_DATA_PATH)

print(f"Train: {len(train_df)} rows, {train_df['patid'].nunique()} patients")
print(f"Val: {len(val_df)} rows, {val_df['patid'].nunique()} patients")
print(f"Test: {len(test_df)} rows, {test_df['patid'].nunique()} patients")

# ============================================================================
# Build Count Matrix from Last N Codes
# ============================================================================
def build_count_matrix_last_n(df, code_column='aggregated_terms', term_vocab=None, max_codes=512):
    """
    Build count matrix using only the last N codes from each patient's sequence.
    This approach focuses on the most recent clinical events.
    
    Args:
        df: DataFrame with columns [patid, aggregated_terms, retino_10y]
        code_column: Column name containing space-separated codes
        term_vocab: Optional dict mapping term -> column index (from training)
        max_codes: Maximum number of codes to use (from the end of sequence)
    
    Returns:
        count_matrix: Sparse count matrix
        term_vocab: Dictionary mapping terms to indices
        patient_ids: List of patient IDs
        labels: Array of labels
    """
    # First pass: build vocabulary if needed (training only)
    if term_vocab is None:
        all_terms = set()
        for codes in df[code_column].dropna():
            # Get last N codes for vocabulary building
            code_list = codes.strip().split()
            last_n_codes = code_list[-max_codes:] if len(code_list) > max_codes else code_list
            all_terms.update(last_n_codes)
        term_vocab = {term: idx for idx, term in enumerate(sorted(all_terms))}
        print(f"Created vocabulary with {len(term_vocab)} unique terms")
    
    # Initialize lists for sparse matrix construction
    row_indices = []
    col_indices = []
    data = []
    
    # Process each patient
    patient_ids = []
    labels = []
    
    for idx, (_, row) in enumerate(df.iterrows()):
        patient_ids.append(row['patid'])
        labels.append(row['retino_10y'])
        
        # Get last N codes for this patient
        if pd.notna(row[code_column]) and row[code_column].strip():
            code_list = row[code_column].strip().split()
            
            # Take only the last max_codes codes
            last_n_codes = code_list[-max_codes:] if len(code_list) > max_codes else code_list
            
            # Count terms in the truncated sequence
            term_counts = Counter(last_n_codes)
            
            for term, count in term_counts.items():
                if term in term_vocab:  # Only include terms in vocabulary
                    row_indices.append(idx)
                    col_indices.append(term_vocab[term])
                    data.append(count)
    
    # Build sparse matrix
    n_patients = len(patient_ids)
    n_terms = len(term_vocab)
    
    count_matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_patients, n_terms)
    )
    
    return count_matrix, term_vocab, patient_ids, np.array(labels)

# Build count matrices using only last N codes
print(f"\nBuilding count matrices (last {MAX_CODES} codes only)...")
X_train, term_vocab, train_patients, y_train = build_count_matrix_last_n(train_df, max_codes=MAX_CODES)
X_val, _, val_patients, y_val = build_count_matrix_last_n(val_df, term_vocab=term_vocab, max_codes=MAX_CODES)
X_test, _, test_patients, y_test = build_count_matrix_last_n(test_df, term_vocab=term_vocab, max_codes=MAX_CODES)

print(f"\nCount matrix shapes:")
print(f"Train: {X_train.shape} (sparsity: {100 * (1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])):.2f}%)")
print(f"Val: {X_val.shape}")
print(f"Test: {X_test.shape}")

print(f"\nLabel distributions:")
print(f"Train: {np.bincount(y_train.astype(int))} (positive rate: {y_train.mean():.3f})")
print(f"Val: {np.bincount(y_val.astype(int))} (positive rate: {y_val.mean():.3f})")
print(f"Test: {np.bincount(y_test.astype(int))} (positive rate: {y_test.mean():.3f})")

# ============================================================================
# Hyperparameter Optimization with Optuna
# ============================================================================
def objective(trial):
    """Optuna objective function for XGBoost hyperparameter optimization"""
    params = {
        'objective': 'binary:logistic',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'auc',
        'early_stopping_rounds': 5,
        'random_state': 42,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    preds = model.predict_proba(X_val)[:, 1]  
    return average_precision_score(y_val, preds)

print(f"\nRunning hyperparameter optimization ({NUM_TRIALS} trials)...")
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=NUM_TRIALS)

print(f"\nBest parameters: {study.best_params}")
print(f"Best validation AUPRC: {study.best_value:.4f}")

# ============================================================================
# Train Final Model with Optimized Parameters
# ============================================================================
optimized_params = {
    'objective': 'binary:logistic',
    'max_depth': study.best_params['max_depth'],
    'learning_rate': study.best_params['learning_rate'],
    'subsample': study.best_params['subsample'],
    'colsample_bytree': study.best_params['colsample_bytree'],
    'eval_metric': 'auc',
    'early_stopping_rounds': 10,
    'random_state': 42
}

print("\nTraining final model with optimized parameters...")
start_time = time.time()

model = xgb.XGBClassifier(**optimized_params)
model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)], 
          verbose=True)

end_time = time.time()
training_time = end_time - start_time
print(f"\nXGBoost Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# ============================================================================
# Evaluation on Test Set
# ============================================================================
print("\nEvaluating on test set...")
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class
y_pred_binary = model.predict(X_test)  # Binary predictions (0/1)

# Calculate all metrics
auprc = average_precision_score(y_test, y_pred_proba)
auroc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

# Print results
print("\nXGBoost Model Performance:")
print("=" * 40)
print(f"AUPRC:     {auprc:.4f}")
print(f"AUROC:     {auroc:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_binary))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_binary)
print(cm)
print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================
def bootstrap_confidence_intervals_xgb(y_true, y_pred_proba, n_bootstraps=1000, ci=95):
    """Calculate bootstrap confidence intervals for XGBoost classification metrics"""
    
    # Get binary predictions using 0.5 threshold
    y_pred = (y_pred_proba > 0.5).astype(int)
    
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
        y_scores_bootstrap = np.array(y_pred_proba)[indices]
        
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

print("\nCalculating bootstrap confidence intervals...")
ci_results = bootstrap_confidence_intervals_xgb(y_test, y_pred_proba)

print("\nXGBoost Bootstrap Confidence Intervals (95%):")
print("=" * 50)
for metric, values in ci_results.items():
    print(f"{metric.upper():10}: {values['mean']:.4f} [{values['lower']:.4f}, {values['upper']:.4f}]")

# ============================================================================
# Save Results
# ============================================================================
print(f"\nSaving results to: {OUTPUT_DIR}")

# Save bootstrap results
results_path = os.path.join(OUTPUT_DIR, "results.json")
with open(results_path, 'w') as f:
    json.dump(ci_results, f, indent=4)
print(f"Bootstrap results saved to: {results_path}")

# Save predictions
predictions_df = pd.DataFrame({
    'patid': test_patients,
    'true_label': y_test,
    'predicted_proba': y_pred_proba,
    'predicted_binary': y_pred_binary
})
predictions_path = os.path.join(OUTPUT_DIR, "predictions.csv")
predictions_df.to_csv(predictions_path, index=False)
print(f"Predictions saved to: {predictions_path}")

# Save optimized hyperparameters
hyperparams_path = os.path.join(OUTPUT_DIR, "optimized_hyperparameters.json")
with open(hyperparams_path, 'w') as f:
    json.dump(study.best_params, f, indent=4)
print(f"Optimized hyperparameters saved to: {hyperparams_path}")

# Save timing information
timing_path = os.path.join(OUTPUT_DIR, "timing.json")
with open(timing_path, 'w') as f:
    json.dump({
        'training_time_seconds': training_time,
        'training_time_minutes': training_time/60
    }, f, indent=4)
print(f"Timing information saved to: {timing_path}")

print("\nXGBoost training and evaluation complete!")
