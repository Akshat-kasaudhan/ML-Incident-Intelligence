import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Add project root to python path to allow importing our new modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.temporal_builder import build_temporal_sequences
from data.dataset import RCATemporalDataset, rca_collate_fn
from models.gru_rca import RCAGRUModel
from training.train_gru import train_temporal_model

def impute_and_scale(train_seqs, test_seqs):
    """
    Applies imputation and scaling STRICTLY fitted on the training set
    to prevent data leakage.
    
    Args:
        train_seqs: List of [time, features] numpy arrays
        test_seqs: List of [time, features] numpy arrays
    """
    # 1. Flatten the temporal dimension to fit Sklearn transformers
    # train_seqs is a list of arrays. Let's stack them to [total_train_windows, features]
    train_flat = np.vstack(train_seqs)
    
    # 2. Fit imputer and scaler only on train data!
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    print("Fitting Imputer and Scaler on Training data only (Leakage Prevention)...")
    train_flat_imputed = imputer.fit_transform(train_flat)
    scaler.fit(train_flat_imputed)
    
    # 3. Transform both train and test sequences
    processed_train = []
    for seq in train_seqs:
        # seq is [time, features]
        seq_imp = imputer.transform(seq)
        seq_scaled = scaler.transform(seq_imp)
        processed_train.append(torch.tensor(seq_scaled, dtype=torch.float32))
        
    processed_test = []
    for seq in test_seqs:
        seq_imp = imputer.transform(seq)
        seq_scaled = scaler.transform(seq_imp)
        processed_test.append(torch.tensor(seq_scaled, dtype=torch.float32))
        
    return processed_train, processed_test

def main():
    print("=== 1. Loading Metadata ===")
    cases_df = pd.read_parquet("hf://datasets/phamquiluan/RCAEval/cases.parquet")
    re2_cases = cases_df[cases_df["suite"].str.startswith("RE2")].sample(frac=1, random_state=42).reset_index(drop=True)
    
    # For a quick pipeline test, we'll just process the first 15 cases.
    # In production, this would process all 90 RE2-OB cases.
    subset_cases = re2_cases.head(15)
    print(f"Processing {len(subset_cases)} cases...")
    
    print("\n=== 2. Building Temporal Sequences ===")
    sequences, labels, lengths, case_ids, metric_cols = build_temporal_sequences(
        subset_cases, 
        window_size=30, 
        time_range=(-120, 120)
    )
    
    if not sequences:
        print("No sequences could be built. Check dataset availability.")
        return
        
    num_features = sequences[0].shape[1]
    print(f"Extracted {len(sequences)} valid sequences with {num_features} features each.")
    
    print("\n=== 3. Train / Test Split ===")
    # Split cases (Leakage Prevention: whole cases are split, not just windows!)
    X_train, X_test, y_train, y_test, len_train, len_test, id_train, id_test = train_test_split(
        sequences, labels, lengths, case_ids, test_size=0.2, random_state=42
    )
    print(f"Train cases: {len(X_train)} | Test cases: {len(X_test)}")
    
    print("\n=== 4. Imputation and Scaling ===")
    X_train_proc, X_test_proc = impute_and_scale(X_train, X_test)
    
    print("\n=== 5. Creating PyTorch DataLoaders ===")
    train_dataset = RCATemporalDataset(X_train_proc, y_train, len_train, id_train)
    test_dataset = RCATemporalDataset(X_test_proc, y_test, len_test, id_test)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=rca_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=rca_collate_fn)
    
    print("\n=== 6. Initializing GRU Model ===")
    # Extract the final number of features AFTER imputation/scaling!
    final_num_features = X_train_proc[0].shape[1]
    print(f"Final feature dimension for GRU: {final_num_features}")
    
    model = RCAGRUModel(input_size=final_num_features, hidden_size=64, num_layers=1, num_classes=5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    print("\n=== 7. Training Model ===")
    # Train for 5 epochs for the sake of a quick test
    best_model, best_val_loss = train_temporal_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=5,
        device="cpu"
    )
    
    print(f"\nTraining finished! Best Val Loss: {best_val_loss:.4f}")
    
    print("\n=== 8. Fair Model Comparison (Phase G) ===")
    from evaluation.compare_models import compare_models
    
    # We pass the trained best_model and the test_loader to generate standard RCAEval metrics
    compare_models(best_model, test_loader, rf_model_path="model_service.pkl")
    
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()
