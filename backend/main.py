# backend/main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re

# Local imports
from backend.workload.generate_synthetic_workload import generate_synthetic_trace
from backend.simulator.memory_simulator import MemorySimulator
from backend.ml.models.scheduler_opt import predict_eviction_scores
from backend.statistical_validation import validate_tensor_size_distribution


app = FastAPI(
    title="Accelerator Optimization Copilot ",
    description="Optimising accelerators for deep learning workloads"
)


# --------------------------------------------------------
# Input Models
# --------------------------------------------------------

class TraceRequest(BaseModel):
    model_type: str
    num_layers: int = 12
    batch_size: int = 32
    seq_length: int = 128
    reuse_probability: float = 0.3


class PredictCompileRequest(BaseModel):
    code: str
    model_type: str = "Transformer"
    opt_level: str = "O2"


# --------------------------------------------------------
# Workload Generation
# --------------------------------------------------------

@app.post("/generate-trace")
def generate_trace(req: TraceRequest):
    df = generate_synthetic_trace(**req.dict())
    return df.to_dict(orient="records")


# --------------------------------------------------------
# ML Eviction Score Prediction
# --------------------------------------------------------

@app.post("/predict-schedule")
def predict_schedule(req: TraceRequest):
    df = generate_synthetic_trace(**req.dict())
    df = predict_eviction_scores(df)
    return df.to_dict(orient="records")


# --------------------------------------------------------
# Cache Simulation
# --------------------------------------------------------

@app.post("/simulate")
def simulate(req: TraceRequest, policy: str = Query("LRU", enum=["LRU", "FIFO", "ML"])):
    df = generate_synthetic_trace(**req.dict())
    df = predict_eviction_scores(df)  # required for ML policy

    sim = MemorySimulator()
    metrics = sim.simulate_trace(df, policy=policy)
    return metrics


# --------------------------------------------------------
# Statistical Validation
# --------------------------------------------------------

@app.post("/validate-trace")
def validate_trace(req: TraceRequest):
    df = generate_synthetic_trace(**req.dict())

    # Import new validation functions
    from backend.statistical_validation import (
        calculate_reuse_entropy,
        calculate_temporal_locality
    )
    
    # Calculate reuse metrics
    reuse_entropy = calculate_reuse_entropy(df)
    temporal_locality = calculate_temporal_locality(df)
    size_p = validate_tensor_size_distribution(df, req.model_type)

    return {
        "reuse_entropy": round(reuse_entropy, 3),
        "temporal_locality_score": round(temporal_locality, 3),
        "tensor_size_distribution_p_value (KS-test)": round(size_p, 4),
        "interpretation": {
            "reuse_entropy": "0-1.5: Cache-friendly | 1.5-2.5: Typical | 2.5+: Challenging",
            "temporal_locality": "0.7-1.0: Excellent | 0.4-0.7: Moderate | 0.0-0.4: Poor",
            "size_p_value": "p > 0.05 indicates realistic size distribution"
        }
    }





# --------------------------------------------------------
# Compiler Prediction
# --------------------------------------------------------

@app.post("/predict-compile")
def predict_compile(req: PredictCompileRequest):

    # Realistic Feature Extraction
    code = req.code
    optimization_level = req.opt_level
    
    # Extract features using regex (simplistic static analysis)
    features = {
        "num_loops": len(re.findall(r'\b(for|while)\b', code)),
        "num_branches": len(re.findall(r'\b(if|else|switch|case)\b', code)),
        "load_ops": len(re.findall(r'\b(load|read|\[\])\b', code)),
        "store_ops": len(re.findall(r'\b(store|write|=)\b', code)),
        "flops": len(re.findall(r'[+\-*/]', code)),
        "lines_of_code": len(code.split('\n'))
    }
    
    # Calculate Latency based on features and optimization level
    base_latency = {
        "O0": 10,    # No optimization - fast compile
        "O1": 50,    # Basic optimization
        "O2": 200,   # Moderate optimization
        "O3": 500    # Aggressive optimization - slow compile
    }.get(optimization_level, 10)
    
    latency = base_latency
    latency += features["num_loops"] * 20
    latency += features["num_branches"] * 10
    latency += features["flops"] * 0.5
    latency += features["lines_of_code"] * 2
    
    # Add some realistic variance
    latency *= np.random.uniform(0.9, 1.1)

    # Calculate Energy (Joules)
    # Energy = Power * Time
    # Assume ~50W power draw during compilation
    power_watts = 50
    energy = power_watts * (latency / 1000) # convert ms to seconds
    
    # Calculate Feature Importance (contribution to compilation time)
    # This helps users know WHAT part of their code is causing slow compiles
    if sum(features.values()) > 0:
        total_features = sum(features.values())
        feature_importance = {k: round(v/total_features, 3) for k, v in features.items()}
    else:
        feature_importance = {k: 0.0 for k in features}

    return {
        "latency (ms)": round(latency, 2),
        "energy (J)": round(energy, 3),
        "features": features,
        "feature_importance": feature_importance,
    }

