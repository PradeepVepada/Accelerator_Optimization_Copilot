import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.simulator.memory_simulator import MemorySimulator
from backend.workload.generate_synthetic_workload import generate_synthetic_trace
import pandas as pd


def test_memory_simulator_basic():
    """Test basic memory simulator functionality"""
    # Create a simple trace
    trace_data = [
        {"tensor_id": "A", "size_mb": 8, "reuse_distance": 0, "stride": 1, "op": "matmul", "model_type": "Transformer"},
        {"tensor_id": "B", "size_mb": 16, "reuse_distance": 0, "stride": 1, "op": "matmul", "model_type": "Transformer"},
        {"tensor_id": "A", "size_mb": 8, "reuse_distance": 1, "stride": 1, "op": "matmul", "model_type": "Transformer"},
    ]
    df = pd.DataFrame(trace_data)
    
    sim = MemorySimulator(cache_size_mb=64)
    metrics = sim.simulate_trace(df, policy="LRU")
    
    # Check metrics exist
    assert "cache_hits" in metrics
    assert "dram_accesses" in metrics
    assert "total_latency" in metrics
    assert "evictions" in metrics
    
    # Check that we got at least one cache hit (tensor A reused)
    assert metrics["cache_hits"] >= 1, "Should have at least one cache hit"
    
    print("✅ test_memory_simulator_basic passed")
    print(f"Metrics: {metrics}")


def test_cache_eviction():
    """Test that cache eviction works when cache is full"""
    # Create trace that exceeds cache size
    trace_data = []
    for i in range(10):
        trace_data.append({
            "tensor_id": f"tensor_{i}",
            "size_mb": 16,
            "reuse_distance": 0,
            "stride": 1,
            "op": "matmul",
            "model_type": "Transformer"
        })
    
    df = pd.DataFrame(trace_data)
    
    sim = MemorySimulator(cache_size_mb=64)  # Can only fit 4 tensors of 16MB
    metrics = sim.simulate_trace(df, policy="LRU")
    
    # Should have evictions since 10 * 16MB > 64MB
    assert metrics["evictions"] > 0, "Should have evictions when cache is full"
    
    print("✅ test_cache_eviction passed")
    print(f"Evictions: {metrics['evictions']}")


def test_with_synthetic_trace():
    """Test simulator with synthetic trace"""
    df = generate_synthetic_trace(model_type="Transformer", num_layers=4)
    
    sim = MemorySimulator(cache_size_mb=64)
    metrics = sim.simulate_trace(df, policy="LRU")
    
    assert metrics["total_latency"] > 0, "Should have some latency"
    assert metrics["dram_accesses"] > 0, "Should have DRAM accesses"
    
    print("✅ test_with_synthetic_trace passed")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    test_memory_simulator_basic()
    test_cache_eviction()
    test_with_synthetic_trace()
    print("\n✅ All memory simulator tests passed!")
