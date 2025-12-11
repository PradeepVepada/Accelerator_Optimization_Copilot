import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.ml.models.scheduler_opt import predict_eviction_scores
from backend.workload.generate_synthetic_workload import generate_synthetic_trace
import pandas as pd


def test_predict_eviction_scores():
    """Test that eviction score prediction works"""
    df = generate_synthetic_trace(model_type="Transformer", num_layers=4)
    
    # Predict eviction scores
    df_with_scores = predict_eviction_scores(df)
    
    # Check that eviction_score column was added
    assert "eviction_score" in df_with_scores.columns, "eviction_score column should be added"
    
    # Check that scores are numeric
    assert pd.api.types.is_numeric_dtype(df_with_scores["eviction_score"]), "Scores should be numeric"
    
    # Check that all scores are present (no NaN)
    assert not df_with_scores["eviction_score"].isna().any(), "No scores should be NaN"
    
    print("✅ test_predict_eviction_scores passed")
    print(f"Generated {len(df_with_scores)} eviction scores")
    print(df_with_scores[["tensor_id", "size_mb", "reuse_distance", "eviction_score"]].head())


def test_eviction_scores_range():
    """Test that eviction scores are in reasonable range"""
    df = generate_synthetic_trace(model_type="CNN", num_layers=2)
    df_with_scores = predict_eviction_scores(df)
    
    # Scores should be between 0 and 1 (or at least non-negative)
    assert (df_with_scores["eviction_score"] >= 0).all(), "Scores should be non-negative"
    
    print("✅ test_eviction_scores_range passed")
    print(f"Score range: [{df_with_scores['eviction_score'].min():.4f}, {df_with_scores['eviction_score'].max():.4f}]")


if __name__ == "__main__":
    test_predict_eviction_scores()
    test_eviction_scores_range()
    print("\n✅ All scheduler tests passed!")
