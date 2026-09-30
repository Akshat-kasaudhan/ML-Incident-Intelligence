import os
import sys
import torch
import numpy as np
import pickle

# Add project root to python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import evaluate_predictions, print_evaluation_report

def evaluate_gru_model(model, test_loader, device="cpu"):
    """
    Evaluates the PyTorch GRU Model.
    Provides case-level predictions based on the sequential processing of windows.
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_seqs, batch_labels, batch_lengths, _ in test_loader:
            batch_seqs = batch_seqs.to(device)
            
            # Get logits
            logits = model(batch_seqs, batch_lengths)
            
            # Convert to probabilities using Softmax
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_labels.numpy())
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def compare_models(gru_model, test_loader, rf_model_path=None):
    """
    Executes the Fair Model Comparison (Phase G).
    Compares the Temporal Sequence Model against the Classical ML Baseline.
    """
    services = ["checkoutservice", "currencyservice", "emailservice", "productcatalogservice", "recommendationservice"]
    
    print("\nEvaluating PyTorch Temporal GRU Model (Case-Level)...")
    y_true_gru, y_pred_gru, y_prob_gru = evaluate_gru_model(gru_model, test_loader)
    
    gru_metrics = evaluate_predictions(y_true_gru, y_pred_gru, y_prob_gru, class_names=services)
    print_evaluation_report("Temporal GRU (Sequential)", gru_metrics)
    
    if rf_model_path and os.path.exists(rf_model_path):
        print("\nEvaluating Baseline Random Forest Model...")
        print("Note: In a full pipeline, the test set sequences must be flattened/aggregated")
        print("into the exact feature space the Random Forest expects (e.g. mean/max delta).")
        # Placeholder for actual RF evaluation logic:
        # 1. Load RF: rf = pickle.load(open(rf_model_path, 'rb'))
        # 2. Extract baseline features from test_loader sequences
        # 3. Predict: rf_preds = rf.predict(X_test_baseline)
        # 4. Compare using evaluate_predictions
    else:
        print("\n[!] Baseline Random Forest model path not provided or not found.")
        print("    Ensure 'model_service.pkl' is correctly integrated for the final fair comparison.")

if __name__ == "__main__":
    print("Model comparison module ready.")
