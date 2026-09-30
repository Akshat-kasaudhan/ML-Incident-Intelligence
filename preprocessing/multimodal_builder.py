import pandas as pd
import numpy as np
import os
from huggingface_hub import snapshot_download

def load_case_multimodal(case_id, download_dir="./rcaeval_case"):
    """
    Downloads and loads BOTH metrics and logs for a given case_id.
    """
    case_dir = snapshot_download(
        repo_id="phamquiluan/RCAEval",
        repo_type="dataset",
        allow_patterns=f"{case_id}/*",
        local_dir=download_dir
    )
    
    base_dir = os.path.join(case_dir, case_id)
    metrics_path = os.path.join(base_dir, "metrics.parquet")
    logs_path = os.path.join(base_dir, "logs.parquet")
    inject_time_path = os.path.join(base_dir, "inject_time.txt")
    
    if not os.path.exists(metrics_path) or not os.path.exists(inject_time_path):
        return None, None, None
        
    metrics = pd.read_parquet(metrics_path)
    
    # Safely load logs if they exist
    logs = pd.DataFrame()
    if os.path.exists(logs_path):
        try:
            logs = pd.read_parquet(logs_path)
        except Exception:
            pass
            
    with open(inject_time_path, "r") as f:
        inject_time = float(f.read().strip())
        
    return metrics, logs, inject_time

def process_logs(logs_df, inject_time, core_services, bins, time_range):
    """
    Phase J: Converts the Logs modality into a reproducible numeric representation.
    We extract the frequency of ERROR logs per service per 30-second window.
    """
    # Create empty DataFrame for log features if logs are missing or empty
    log_features = [f"{svc}_error_count" for svc in core_services]
    num_windows = len(bins) - 1
    
    if logs_df.empty or "time" not in logs_df.columns or "level" not in logs_df.columns:
        return pd.DataFrame(0, index=range(num_windows), columns=log_features)
        
    logs_df["relative_time"] = logs_df["time"] - inject_time
    
    # Filter to time range
    mask = (logs_df["relative_time"] >= time_range[0]) & (logs_df["relative_time"] <= time_range[1])
    logs_df = logs_df[mask]
    
    # Filter to only ERROR or FATAL levels
    error_logs = logs_df[logs_df["level"].isin(["ERROR", "FATAL", "WARN"])]
    
    # Assign windows
    error_logs['window_idx'] = pd.cut(error_logs['relative_time'], bins=bins, labels=False, include_lowest=True)
    
    # Count errors per service per window
    # In RCAEval, logs usually have a 'service' or 'container_name' column
    svc_col = "service" if "service" in error_logs.columns else "container_name"
    
    windowed_logs = pd.DataFrame(0, index=range(num_windows), columns=log_features)
    
    if svc_col in error_logs.columns and not error_logs.empty:
        # Group by window and service
        counts = error_logs.groupby(['window_idx', svc_col]).size().unstack(fill_value=0)
        
        for svc in core_services:
            feat_name = f"{svc}_error_count"
            if svc in counts.columns:
                # Update the windowed_logs dataframe
                for idx in counts.index:
                    if not pd.isna(idx):
                        windowed_logs.loc[int(idx), feat_name] = counts.loc[idx, svc]
                        
    return windowed_logs

def build_multimodal_sequences(cases_df, window_size=30, time_range=(-120, 120)):
    """
    Builds temporal sequences using BOTH Metrics and Logs (Phase J).
    """
    services = ["checkoutservice", "currencyservice", "emailservice", "productcatalogservice", "recommendationservice"]
    service_to_idx = {s: i for i, s in enumerate(services)}
    
    core_services = ["checkoutservice", "cartservice", "recommendationservice", "frontend", "paymentservice", "productcatalogservice", "currencyservice", "emailservice"]
    
    # Base metric columns
    metric_cols = []
    for svc in core_services:
        metric_cols.append(f"{svc}_cpu")
        metric_cols.append(f"{svc}_latency-90")
        
    # New Log columns
    log_cols = [f"{svc}_error_count" for svc in core_services]
    
    # Total combined features
    all_cols = metric_cols + log_cols
    
    sequences, labels, lengths, case_ids = [], [], [], []
    
    bins = np.arange(time_range[0], time_range[1] + window_size, window_size)
    num_windows = len(bins) - 1
    
    print(f"Building Multimodal sequences (Metrics + Logs) using {window_size}s windows...")
    print(f"Total features per step: {len(all_cols)} (16 metrics + 8 log frequencies)")
    
    for _, row in cases_df.iterrows():
        case_id = row['case']
        service = row['root_cause_service']
        
        if service not in service_to_idx:
            continue
            
        metrics, logs, inject_time = load_case_multimodal(case_id)
        if metrics is None:
            continue
            
        # --- Process Metrics ---
        metrics["relative_time"] = metrics["time"] - inject_time
        mask = (metrics["relative_time"] >= time_range[0]) & (metrics["relative_time"] <= time_range[1])
        metrics = metrics[mask]
        
        metrics = metrics.reindex(columns=['relative_time'] + metric_cols)
        metrics['window_idx'] = pd.cut(metrics['relative_time'], bins=bins, labels=False, include_lowest=True)
        windowed_metrics = metrics.groupby('window_idx')[metric_cols].mean()
        
        # --- Process Logs ---
        windowed_logs = process_logs(logs, inject_time, core_services, bins, time_range)
        
        # --- Combine Modalities ---
        sequence_data = []
        for i in range(num_windows):
            # 1. Metrics part
            if i in windowed_metrics.index:
                m_vals = windowed_metrics.loc[i].values
            else:
                m_vals = np.full(len(metric_cols), np.nan)
                
            # 2. Logs part (0 if no errors)
            l_vals = windowed_logs.loc[i].values
            
            # Combine [Metrics, Logs]
            combined = np.concatenate([m_vals, l_vals])
            sequence_data.append(combined)
            
        seq_array = np.array(sequence_data, dtype=np.float32)
        
        sequences.append(seq_array)
        labels.append(service_to_idx[service])
        lengths.append(num_windows)
        case_ids.append(case_id)
        
        print(f"  Extracted Multimodal {case_id} | Shape: {seq_array.shape}")
        
    return sequences, labels, lengths, case_ids, all_cols

if __name__ == "__main__":
    print("Multimodal builder (Metrics + Logs) ready.")
