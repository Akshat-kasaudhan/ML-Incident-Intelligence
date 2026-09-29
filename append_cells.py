import json

code1 = """import pandas as pd
from huggingface_hub import snapshot_download
import os

def extract_features(cases_subset, max_cases=15):
    feature_rows = []
    
    count = 0
    for _, row in cases_subset.iterrows():
        if count >= max_cases:
            break
            
        case_id = row['case']
        print(f"Processing {case_id}...")
        
        try:
            # Download case data to Colab's /content directory
            case_dir = snapshot_download(
                repo_id="phamquiluan/RCAEval",
                repo_type="dataset",
                allow_patterns=f"{case_id}/*",
                local_dir="/content/rcaeval_cases"
            )
            
            base_dir = os.path.join(case_dir, case_id)
            metrics_path = os.path.join(base_dir, "metrics.parquet")
            inject_time_path = os.path.join(base_dir, "inject_time.txt")
            
            if not os.path.exists(metrics_path) or not os.path.exists(inject_time_path):
                print(f"  Missing files for {case_id}")
                continue
                
            metrics = pd.read_parquet(metrics_path)
            
            with open(inject_time_path, "r") as f:
                inject_time = float(f.read().strip())
            
            # Calculate time relative to fault injection
            metrics["relative_time"] = metrics["time"] - inject_time
            
            before_fault = metrics[(metrics["relative_time"] >= -60) & (metrics["relative_time"] < 0)]
            after_fault = metrics[(metrics["relative_time"] >= 0) & (metrics["relative_time"] <= 60)]
            
            # Labels
            features = {
                "case_id": case_id,
                "root_cause_service": row["root_cause_service"],
                "fault_type": row["fault"]
            }
            
            # Key microservices to track
            services = ["checkoutservice", "cartservice", "recommendationservice", "frontend", "paymentservice", "productcatalogservice"]
            
            for svc in services:
                # CPU Diff
                cpu_col = f"{svc}_cpu"
                if cpu_col in metrics.columns:
                    features[f"{svc}_cpu_diff"] = after_fault[cpu_col].mean() - before_fault[cpu_col].mean()
                else:
                    features[f"{svc}_cpu_diff"] = 0.0
                    
                # Latency P90 Diff
                lat_col = f"{svc}_latency-90"
                if lat_col in metrics.columns:
                    features[f"{svc}_lat_diff"] = after_fault[lat_col].mean() - before_fault[lat_col].mean()
                else:
                    features[f"{svc}_lat_diff"] = 0.0
                    
            feature_rows.append(features)
            count += 1
            
        except Exception as e:
            print(f"  Error processing {case_id}: {e}")
            
    return pd.DataFrame(feature_rows)

print("Loading cases metadata...")
cases = pd.read_parquet("hf://datasets/phamquiluan/RCAEval/cases.parquet")
re2_cases = cases[cases["suite"].str.startswith("RE2")].sample(frac=1, random_state=42).reset_index(drop=True)

print("Extracting features for a sample of 30 cases...")
df_features = extract_features(re2_cases, max_cases=30)
display(df_features.head())
"""

code2 = """from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Prepare data
# Drop identifying columns to get our pure feature matrix X
X = df_features.drop(columns=["case_id", "root_cause_service", "fault_type"])

# Fill any missing/NaN values with 0
X = X.fillna(0)

# We have two targets: what service broke? and what was the fault?
y_service = df_features["root_cause_service"]
y_fault = df_features["fault_type"]

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_service_train, y_service_test, y_fault_train, y_fault_test = train_test_split(
    X, y_service, y_fault, test_size=0.2, random_state=42
)

# 2. Train Model for Root Cause Service
print("Training Root Cause Service Predictor...")
model_service = RandomForestClassifier(n_estimators=100, random_state=42)
model_service.fit(X_train, y_service_train)

# Evaluate Service Predictor
service_preds = model_service.predict(X_test)
print(f"Service Prediction Accuracy: {accuracy_score(y_service_test, service_preds) * 100:.2f}%\\n")
print(classification_report(y_service_test, service_preds, zero_division=0))

# 3. Train Model for Fault Type
print("-" * 50)
print("Training Fault Type Predictor...")
model_fault = RandomForestClassifier(n_estimators=100, random_state=42)
model_fault.fit(X_train, y_fault_train)

# Evaluate Fault Predictor
fault_preds = model_fault.predict(X_test)
print(f"Fault Type Prediction Accuracy: {accuracy_score(y_fault_test, fault_preds) * 100:.2f}%\\n")
print(classification_report(y_fault_test, fault_preds, zero_division=0))

# 4. Save models for deployment
joblib.dump(model_service, "/content/model_service.pkl")
joblib.dump(model_fault, "/content/model_fault.pkl")
joblib.dump(list(X.columns), "/content/feature_names.pkl")

print("\\nModels successfully saved for deployment!")
"""

def split_to_lines(code):
    lines = code.split("\\n")
    return [line + "\\n" for line in lines[:-1]] + [lines[-1]]

notebook_path = "notebooks_01_rcaeval_dataset_inspection (1).ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell1 = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "outputs": [],
  "source": split_to_lines(code1)
}

cell2 = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "outputs": [],
  "source": split_to_lines(code2)
}

nb["cells"].extend([cell1, cell2])

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
