import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.multimodal_builder import build_multimodal_sequences
from data.dataset import RCATemporalDataset, rca_collate_fn
from models.gru_rca import RCAGRUModel
from training.train_gru import train_temporal_model
from training.run_pipeline import impute_and_scale
from evaluation.compare_models import compare_models

def main():
    print("=== 1. Loading Metadata for MULTIMODAL Analysis ===")
    cases_df = pd.read_parquet("hf://datasets/phamquiluan/RCAEval/cases.parquet")
    re2_cases = cases_df[cases_df["suite"].str.startswith("RE2")].sample(frac=1, random_state=42).reset_index(drop=True)
    
    subset_cases = re2_cases.head(15)
    
    print("\n=== 2. Building Multimodal Sequences (Metrics + Logs) ===")
    sequences, labels, lengths, case_ids, all_cols = build_multimodal_sequences(
        subset_cases, 
        window_size=30, 
        time_range=(-120, 120)
    )
    
    if not sequences:
        print("No multimodal sequences could be built.")
        return
        
    print("\n=== 3. Train / Test Split ===")
    X_train, X_test, y_train, y_test, len_train, len_test, id_train, id_test = train_test_split(
        sequences, labels, lengths, case_ids, test_size=0.2, random_state=42
    )
    
    print("\n=== 4. Imputation and Scaling (Leakage-Safe) ===")
    X_train_proc, X_test_proc = impute_and_scale(X_train, X_test)
    
    print("\n=== 5. PyTorch DataLoaders ===")
    train_dataset = RCATemporalDataset(X_train_proc, y_train, len_train, id_train)
    test_dataset = RCATemporalDataset(X_test_proc, y_test, len_test, id_test)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=rca_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=rca_collate_fn)
    
    print("\n=== 6. Initializing GRU Model ===")
    final_num_features = X_train_proc[0].shape[1]
    print(f"Final Multimodal feature dimension: {final_num_features}")
    
    model = RCAGRUModel(input_size=final_num_features, hidden_size=64, num_layers=1, num_classes=5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    print("\n=== 7. Training Multimodal Model ===")
    best_model, _ = train_temporal_model(
        model=model, train_loader=train_loader, val_loader=test_loader,
        criterion=criterion, optimizer=optimizer, epochs=5, device="cpu"
    )
    
    print("\n=== 8. Multimodal Evaluation ===")
    compare_models(best_model, test_loader)
    
if __name__ == "__main__":
    main()
