import json

def run_error_analysis(y_true, y_pred, y_prob, case_ids, services):
    """
    Phase I: Error Analysis
    Identifies and inspects system failures without deleting difficult cases.
    """
    print("\n=== Phase I: Error Analysis Report ===")
    
    errors = []
    
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            true_svc = services[y_true[i]]
            pred_svc = services[y_pred[i]]
            conf = y_prob[i][y_pred[i]]
            
            error_record = {
                "case_id": case_ids[i],
                "true_root_cause": true_svc,
                "predicted_root_cause": pred_svc,
                "prediction_confidence": f"{conf:.2%}",
                "analysis": "Awaiting manual review of temporal window masking and evidence."
            }
            errors.append(error_record)
            
    if not errors:
        print("No errors detected in this subset!")
    else:
        print(f"Detected {len(errors)} misclassifications.")
        for err in errors:
            print(f"\nCase: {err['case_id']}")
            print(f"  Expected: {err['true_root_cause']}")
            print(f"  Predicted: {err['predicted_root_cause']} (Confidence: {err['prediction_confidence']})")
            
    # Save error report
    with open("error_analysis_report.json", "w") as f:
        json.dump(errors, f, indent=4)
        
    print("\nDetailed error report saved to 'error_analysis_report.json'")

if __name__ == "__main__":
    print("Error analysis module ready.")
