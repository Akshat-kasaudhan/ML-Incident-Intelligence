import pandas as pd
import numpy as np
import os
from huggingface_hub import snapshot_download

def load_case_metrics(case_id, download_dir="./rcaeval_case"):
    """
    Downloads (if necessary) and loads the metrics for a given case_id.
    """
    case_dir = snapshot_download(
        repo_id="phamquiluan/RCAEval",
        repo_type="dataset",
        allow_patterns=f"{case_id}/*",
        local_dir=download_dir
    )
    
    base_dir = os.path.join(case_dir, case_id)
    metrics_path = os.path.join(base_dir, "metrics.parquet")
    inject_time_path = os.path.join(base_dir, "inject_time.txt")
    
    if not os.path.exists(metrics_path) or not os.path.exists(inject_time_path):
        return None, None
        
    metrics = pd.read_parquet(metrics_path)
    
    with open(inject_time_path, "r") as f:
        inject_time = float(f.read().strip())
        
    return metrics, inject_time

def build_temporal_sequences(cases_df, window_size=30, time_range=(-120, 120)):
    """
    Converts raw telemetry into a sequence of 30-second windows.
    """
    # Service to integer label mapping
    services = ["checkoutservice", "currencyservice", "emailservice", "productcatalogservice", "recommendationservice"]
    service_to_idx = {s: i for i, s in enumerate(services)}
    
    # Explicitly define the fixed set of telemetry columns we care about across all cases.
    # This prevents column mismatches where one case has different metrics than another.
    core_services = ["checkoutservice", "cartservice", "recommendationservice", "frontend", "paymentservice", "productcatalogservice", "currencyservice", "emailservice"]
    metric_cols = []
    for svc in core_services:
        metric_cols.append(f"{svc}_cpu")
        metric_cols.append(f"{svc}_latency-90")
    
    sequences = []
    labels = []
    lengths = []
    case_ids = []
    
    bins = np.arange(time_range[0], time_range[1] + window_size, window_size)
    num_windows = len(bins) - 1
    
    print(f"Building temporal sequences using {window_size}s windows...")
    print(f"Total time steps per full sequence: {num_windows}")
    
    for _, row in cases_df.iterrows():
        case_id = row['case']
        service = row['root_cause_service']
        
        if service not in service_to_idx:
            continue
            
        metrics, inject_time = load_case_metrics(case_id)
        
        if metrics is None:
            continue
            
        metrics["relative_time"] = metrics["time"] - inject_time
        mask = (metrics["relative_time"] >= time_range[0]) & (metrics["relative_time"] <= time_range[1])
        metrics = metrics[mask]
        
        feature_df = metrics.drop(columns=["time", "relative_time"], errors="ignore")
        
        if metric_cols is None:
            metric_cols = sorted(feature_df.columns.tolist())
            
        # Ensure the metrics dataframe has exactly the metric_cols we require.
        # Missing columns will be added with NaN, extra columns dropped.
        metrics = metrics.reindex(columns=['relative_time'] + metric_cols)
        
        metrics['window_idx'] = pd.cut(metrics['relative_time'], bins=bins, labels=False, include_lowest=True)
        windowed = metrics.groupby('window_idx')[metric_cols].mean()
        
        sequence_data = []
        for i in range(num_windows):
            if i in windowed.index:
                sequence_data.append(windowed.loc[i].values)
            else:
                sequence_data.append(np.full(len(metric_cols), np.nan))
                
        seq_array = np.array(sequence_data, dtype=np.float32)
        
        sequences.append(seq_array)
        labels.append(service_to_idx[service])
        lengths.append(num_windows)
        case_ids.append(case_id)
        
        print(f"  Extracted {case_id} | Shape: {seq_array.shape}")
        
    return sequences, labels, lengths, case_ids, metric_cols

if __name__ == "__main__":
    print("Temporal builder ready to be imported and used.")
