import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def calculate_top_k_accuracy(y_true, y_prob, k=3):
    """
    Calculates the Top-K accuracy.
    y_true: Array of true class indices.
    y_prob: Array of predicted probabilities (shape: [num_samples, num_classes]).
    k: The number of top predictions to consider.
    """
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] in top_k_preds[i]:
            correct += 1
            
    return correct / len(y_true) if len(y_true) > 0 else 0.0

def evaluate_predictions(y_true, y_pred, y_prob, class_names=None):
    """
    Computes standard RCA classification metrics.
    
    Returns a dictionary of metrics including Accuracy, Macro F1, 
    Top-3 Accuracy, and the Confusion Matrix.
    """
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['top_3_accuracy'] = calculate_top_k_accuracy(y_true, y_prob, k=3)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    if class_names:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, labels=np.arange(len(class_names)), zero_division=0
        )
    else:
        metrics['classification_report'] = classification_report(y_true, y_pred, zero_division=0)
        
    return metrics

def print_evaluation_report(model_name, metrics):
    """
    Pretty-prints the evaluation metrics for a model.
    """
    print(f"\n{'='*40}")
    print(f" Model Evaluation: {model_name}")
    print(f"{'='*40}")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:    {metrics['weighted_f1']:.4f}")
    print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"{'='*40}\n")
