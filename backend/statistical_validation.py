import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, entropy


def calculate_reuse_entropy(trace_df: pd.DataFrame) -> float:
    """
    Calculate Shannon entropy of reuse distance distribution.
    Higher entropy = more diverse reuse patterns (good for varied workloads)
    Lower entropy = concentrated reuse patterns (good for cache-friendly workloads)
    
    Returns:
        Shannon entropy value (typically 0-4 for realistic workloads)
        - 0-1.5: Very predictable (cache-friendly)
        - 1.5-2.5: Moderate diversity (typical)
        - 2.5+: High diversity (challenging for cache)
    """
    reuse_counts = trace_df['reuse_distance'].value_counts()
    probabilities = reuse_counts / reuse_counts.sum()
    
    # Shannon entropy
    reuse_entropy = entropy(probabilities, base=2)
    
    return float(reuse_entropy)


def calculate_temporal_locality(trace_df: pd.DataFrame) -> float:
    """
    Calculate temporal locality coefficient based on reuse distance distribution.
    Measures how frequently tensors are reused within a short time window.
    
    Returns:
        Temporal locality score (0-1)
        - 0.7-1.0: Excellent temporal locality (Transformer-like)
        - 0.4-0.7: Moderate temporal locality (CNN-like)
        - 0.0-0.4: Poor temporal locality (MLP-like)
    """
    reuse_distances = trace_df['reuse_distance'].values
    
    # Calculate percentage of accesses with low reuse distance (< 5)
    short_reuse = np.sum(reuse_distances < 5)
    total_accesses = len(reuse_distances)
    
    temporal_locality = short_reuse / total_accesses if total_accesses > 0 else 0.0
    
    return float(temporal_locality)


def calculate_reuse_statistics(trace_df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive reuse statistics for the trace.
    
    Returns:
        Dictionary with reuse metrics
    """
    reuse_distances = trace_df['reuse_distance'].values
    
    return {
        "mean_reuse_distance": float(np.mean(reuse_distances)),
        "median_reuse_distance": float(np.median(reuse_distances)),
        "std_reuse_distance": float(np.std(reuse_distances)),
        "max_reuse_distance": int(np.max(reuse_distances)),
        "unique_tensors": int(trace_df['tensor_id'].nunique()),
        "total_accesses": int(len(trace_df))
    }


def validate_tensor_size_distribution(trace_df: pd.DataFrame, model_type: str) -> float:
    """
    Validate that tensor size distribution is reasonable.
    Uses Kolmogorov-Smirnov test.
    
    Returns:
        p-value from KS test (>0.05 indicates good match)
    """
    observed_sizes = trace_df['size_mb'].values
    
    # Expected sizes based on model type
    expected_sizes_map = {
        "Transformer": [4, 8, 16, 32],
        "CNN": [4, 8, 16, 32],
        "MLP": [4, 8, 16],
        "MoE": [4, 8, 16, 32]
    }
    
    expected_sizes = expected_sizes_map.get(model_type, [4, 8, 16, 32])
    expected_sample = np.random.choice(expected_sizes, size=len(observed_sizes))
    
    # KS test
    stat, p_value = ks_2samp(observed_sizes, expected_sample)
    
    return float(p_value)
