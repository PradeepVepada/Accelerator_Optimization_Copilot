import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.workload.generate_synthetic_workload import generate_synthetic_trace
import pandas as pd


def test_generate_synthetic_trace():
    """Test that synthetic trace generation works correctly"""
    df = generate_synthetic_trace(model_type="Transformer", num_layers=4)
    
    # Check that dataframe is not empty
    assert len(df) > 0, "Generated trace should not be empty"
    
    # Check required columns exist
    required_columns = ["tensor_id", "size_mb", "reuse_distance", "stride", "op", "model_type"]
    for col in required_columns:
        assert col in df.columns, f"Column {col} missing from trace"
    
    # Check data types
    assert pd.api.types.is_numeric_dtype(df["size_mb"]), "size_mb should be numeric"
    assert pd.api.types.is_numeric_dtype(df["reuse_distance"]), "reuse_distance should be numeric"
    
    print("✅ test_generate_synthetic_trace passed")
    print(f"Generated {len(df)} trace entries")
    print(df.head())


def test_different_model_types():
    """Test trace generation for different model types"""
    model_types = ["Transformer", "CNN", "MLP", "MoE"]
    
    for model_type in model_types:
        df = generate_synthetic_trace(model_type=model_type, num_layers=2)
        assert len(df) > 0, f"Trace for {model_type} should not be empty"
        assert all(df["model_type"] == model_type), f"All entries should be {model_type}"
        print(f"✅ {model_type} trace generation passed ({len(df)} entries)")


if __name__ == "__main__":
    test_generate_synthetic_trace()
    test_different_model_types()
    print("\n✅ All workload tests passed!")
