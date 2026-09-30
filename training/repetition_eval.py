import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add project root to python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.temporal_builder import build_temporal_sequences
from data.dataset import RCATemporalDataset, rca_collate_fn
from models.gru_rca import RCAGRUModel
from training.train_gru import train_temporal_model
from training.run_pipeline import impute_and_scale
from evaluation.compare_models import evaluate_gru_model
from evaluation.metrics import evaluate_predictions, print_evaluation_report

def extract_repetition(case_id):
    """
    Extracts the repetition number from the RCAEval case ID.
    Example: 're2ob_currencyservice_socket_2' -> 2
    """
    try:
        return int(case_id.split('_')[-1])
    except ValueError:
        return -1

def run_repetition_kfold():
    """
    Executes Phase H: Robustness & Repetition-Aware Evaluation.
    Performs 3-Fold Cross Validation based on incident repetitions.
    """
    print("=== Phase H: Repetition-Aware Cross Validation ===")
    
    # 1. Load Metadata
    cases_df = pd.read_parquet("hf://datasets/phamquiluan/RCAEval/cases.parquet")
    re2_cases = cases_df[cases_df["suite"].str.startswith("RE2")].reset_index(drop=True)
    
    # Extract repetition for splitting
    re2_cases['repetition'] = re2_cases['case'].apply(extract_repetition)
    
    # For speed during this demonstration, we'll test on a subset, but in reality 
    # we would process all 90 cases. 
    subset_cases = re2_cases.sample(frac=1, random_state=42).head(20)
    
    # 2. Build Temporal Sequences for all cases
    sequences, labels, lengths, case_ids, metric_cols = build_temporal_sequences(
        subset_cases, 
        window_size=30, 
        time_range=(-120, 120)
    )
    
    if not sequences:
        print("No sequences could be built. Aborting.")
        return
        
    # Convert lists to numpy arrays for easier boolean indexing
    seqs_arr = np.array(sequences, dtype=object)
    labels_arr = np.array(labels)
    lengths_arr = np.array(lengths)
    case_ids_arr = np.array(case_ids)
    
    # Identify the repetition corresponding to each extracted sequence
    # (Since build_temporal_sequences might skip cases with missing data)
    seq_repetitions = np.array([extract_repetition(cid) for cid in case_ids_arr])
    
    services = ["checkoutservice", "currencyservice", "emailservice", "productcatalogservice", "recommendationservice"]
    
    fold_metrics = []
    
    # 3. K-Fold Evaluation (3 Repetitions)
    for test_rep in [1, 2, 3]:
        print(f"\n{'-'*50}")
        print(f"FOLD {test_rep} | Test Repetition: {test_rep} | Train Repetitions: {[r for r in [1,2,3] if r != test_rep]}")
        print(f"{'-'*50}")
        
        # Split Data
        test_mask = (seq_repetitions == test_rep)
        train_mask = ~test_mask
        
        X_train, y_train = seqs_arr[train_mask].tolist(), labels_arr[train_mask].tolist()
        len_train, id_train = lengths_arr[train_mask].tolist(), case_ids_arr[train_mask].tolist()
        
        X_test, y_test = seqs_arr[test_mask].tolist(), labels_arr[test_mask].tolist()
        len_test, id_test = lengths_arr[test_mask].tolist(), case_ids_arr[test_mask].tolist()
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Skipping Fold {test_rep} due to lack of samples in the subset.")
            continue
            
        print(f"Train Cases: {len(X_train)} | Test Cases: {len(X_test)}")
        
        # Impute and Scale
        X_train_proc, X_test_proc = impute_and_scale(X_train, X_test)
        
        # PyTorch DataLoaders
        train_dataset = RCATemporalDataset(X_train_proc, y_train, len_train, id_train)
        test_dataset = RCATemporalDataset(X_test_proc, y_test, len_test, id_test)
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=rca_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=rca_collate_fn)
        
        # Initialize GRU
        final_num_features = X_train_proc[0].shape[1]
        model = RCAGRUModel(input_size=final_num_features, hidden_size=64, num_layers=1, num_classes=5)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        
        # Train Model
        best_model, _ = train_temporal_model(
            model=model, train_loader=train_loader, val_loader=test_loader,
            criterion=criterion, optimizer=optimizer, epochs=5, device="cpu"
        )
        
        # Evaluate Model
        y_true, y_pred, y_prob = evaluate_gru_model(best_model, test_loader)
        metrics = evaluate_predictions(y_true, y_pred, y_prob, class_names=services)
        
        print_evaluation_report(f"Fold {test_rep}", metrics)
        fold_metrics.append(metrics)
        
    print("\n=== K-Fold Cross Validation Complete ===")
    if fold_metrics:
        avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
        avg_f1 = np.mean([m['macro_f1'] for m in fold_metrics])
        print(f"Average Accuracy: {avg_acc:.4f}")
        print(f"Average Macro F1: {avg_f1:.4f}")

if __name__ == "__main__":
    run_repetition_kfold()
