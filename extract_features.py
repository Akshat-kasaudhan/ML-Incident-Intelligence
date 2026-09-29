import pandas as pd
from huggingface_hub import snapshot_download
import os

def extract_features(cases_subset, max_cases=10):
    feature_rows = []
    
    count = 0
    for _, row in cases_subset.iterrows():
        if count >= max_cases:
            break
            
        case_id = row['case']
        print(f"Processing {case_id}...")
        
        try:
            # Download case data using huggingface_hub
            case_dir = snapshot_download(
                repo_id="phamquiluan/RCAEval",
                repo_type="dataset",
                allow_patterns=f"{case_id}/*",
                local_dir="./rcaeval_case"
            )
            
            base_dir = os.path.join(case_dir, case_id)
            metrics_path = os.path.join(base_dir, "metrics.parquet")
            
            if not os.path.exists(metrics_path):
                print(f"  Missing metrics for {case_id}")
                continue
                
            metrics = pd.read_parquet(metrics_path)
            
            # The inject_time is often recorded in a file, or we can use an approximation if missing.
            inject_time_path = os.path.join(base_dir, "inject_time.txt")
            if os.path.exists(inject_time_path):
                with open(inject_time_path, "r") as f:
                    inject_time = float(f.read().strip())
            else:
                print(f"  Missing inject_time for {case_id}, skipping.")
                continue
            
            # Calculate time relative to fault injection
            metrics["relative_time"] = metrics["time"] - inject_time
            
            # Split into 60s before and 60s after fault
            before_fault = metrics[(metrics["relative_time"] >= -60) & (metrics["relative_time"] < 0)]
            after_fault = metrics[(metrics["relative_time"] >= 0) & (metrics["relative_time"] <= 60)]
            
            # Initialize our feature row with the labels we want to predict!
            features = {
                "case_id": case_id,
                "root_cause_service": row["root_cause_service"],
                "fault_type": row["fault"]
            }
            
            # Let's track some key microservices from the Online Boutique system
            services = ["checkoutservice", "cartservice", "recommendationservice", "frontend", "paymentservice", "productcatalogservice"]
            
            for svc in services:
                # 1. CPU Difference feature (Mean After - Mean Before)
                cpu_col = f"{svc}_cpu"
                if cpu_col in metrics.columns:
                    diff = after_fault[cpu_col].mean() - before_fault[cpu_col].mean()
                    features[f"{svc}_cpu_diff"] = diff
                else:
                    features[f"{svc}_cpu_diff"] = 0.0
                    
                # 2. Latency Difference feature (Mean P90 After - Mean P90 Before)
                lat_col = f"{svc}_latency-90"
                if lat_col in metrics.columns:
                    diff_lat = after_fault[lat_col].mean() - before_fault[lat_col].mean()
                    features[f"{svc}_lat_diff"] = diff_lat
                else:
                    features[f"{svc}_lat_diff"] = 0.0
                    
            feature_rows.append(features)
            count += 1
            print(f"  Successfully extracted features for {case_id}")
            
        except Exception as e:
            print(f"  Error processing {case_id}: {e}")
            
    return pd.DataFrame(feature_rows)

if __name__ == "__main__":
    print("Loading cases metadata from Hugging Face...")
    cases = pd.read_parquet("hf://datasets/phamquiluan/RCAEval/cases.parquet")
    
    # Filter to cases from the RE2 system (Online Boutique) and shuffle them a bit
    re2_cases = cases[cases["suite"].str.startswith("RE2")].sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total RE2 cases available: {len(re2_cases)}")
    print("Processing a sample of 15 cases to build our initial dataset...")
    
    df_features = extract_features(re2_cases, max_cases=15)
    
    df_features.to_csv("features.csv", index=False)
    print("\nDataset successfully saved to features.csv")
    print(df_features.head())
