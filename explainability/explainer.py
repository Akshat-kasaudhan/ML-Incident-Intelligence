import pandas as pd
import numpy as np

class RCAExplainer:
    """
    Phase K: Root-Cause Explanation
    Generates structured, telemetry-backed evidence for the model's predictions.
    """
    def __init__(self, metric_names):
        self.metric_names = metric_names
        
    def generate_explanation(self, case_id, sequence, prediction_idx, confidence, services):
        """
        Extracts measurable evidence from the temporal sequence to justify the prediction.
        
        Args:
            sequence: Numpy array of shape [time, features] (e.g. 8x24)
            prediction_idx: The integer index of the predicted service
            confidence: Probability score (0.0 to 1.0)
            services: List of service names mapping to prediction_idx
        """
        predicted_service = services[prediction_idx]
        
        # 1. Identify the most severe window in the sequence (where metrics peaked)
        # We look at the magnitude of standard deviations across all features
        window_magnitudes = np.nansum(np.abs(sequence), axis=1)
        peak_window_idx = np.argmax(window_magnitudes)
        peak_window_time = (peak_window_idx * 30) - 120 # Mapping window index back to relative time
        
        # 2. Extract the Top-3 highest deviating features in that specific peak window
        peak_features = sequence[peak_window_idx]
        top_feature_indices = np.argsort(np.abs(peak_features))[-3:][::-1]
        
        evidence = []
        for feat_idx in top_feature_indices:
            feat_name = self.metric_names[feat_idx]
            val = peak_features[feat_idx]
            
            # Formulate human-readable evidence
            if "error_count" in feat_name:
                evidence.append(f"Elevated error frequency detected in {feat_name.split('_')[0]} (Z-score: {val:.2f})")
            elif "lat" in feat_name or "latency" in feat_name:
                evidence.append(f"Severe p90 latency degradation in {feat_name.split('_')[0]} (Z-score: {val:.2f})")
            elif "cpu" in feat_name:
                evidence.append(f"Anomalous CPU utilization in {feat_name.split('_')[0]} (Z-score: {val:.2f})")
            else:
                evidence.append(f"Anomalous behavior in {feat_name} (Z-score: {val:.2f})")
                
        explanation = {
            "case_id": case_id,
            "predicted_service": predicted_service,
            "confidence": f"{confidence * 100:.2f}%",
            "relevant_time_window": f"T{peak_window_time}s to T{peak_window_time + 30}s",
            "supporting_evidence": evidence
        }
        
        return explanation

if __name__ == "__main__":
    print("Explainer module ready.")
