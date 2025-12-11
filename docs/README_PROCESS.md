# Accelerator Optimization Copilot - Process Flow

## Quick Overview

This system simulates cache behavior for different ML model architectures (Transformer, CNN, MLP, MoE) to help optimize memory hierarchies.

---

## What Each Script Does

### 1. **`ui/app.py`** - Frontend (Streamlit)
**Purpose**: User interface for configuring and running simulations

**What it does**:
- Displays input controls (model type, layers, cache policy, etc.)
- Sends HTTP requests to backend API
- Visualizes results (metrics, heatmaps, tables)
- Shows validation statistics

**Key Functions**:
- Collects user input
- Calls backend endpoints
- Renders charts and tables

---

### 2. **`backend/main.py`** - API Server (FastAPI)
**Purpose**: Orchestrates the entire simulation pipeline

**What it does**:
- Exposes REST API endpoints
- Routes requests to appropriate components
- Coordinates data flow between modules
- Returns results to frontend

**Key Endpoints**:
- `/simulate` - Run complete simulation
- `/predict-schedule` - Get eviction scores
- `/validate-trace` - Statistical validation
- `/generate-trace` - Create synthetic workload
- `/predict-compile` - Compiler predictions

---

### 3. **`backend/workload/generate_synthetic_workload.py`** - Trace Generator
**Purpose**: Creates realistic memory access patterns

**What it does**:
- Generates tensor access sequences based on model type
- Simulates reuse patterns (how often tensors are accessed)
- Assigns sizes and memory strides to tensors
- Returns DataFrame with trace data

**Key Function**: `generate_synthetic_trace()`

**Output**: DataFrame with columns:
- `tensor_id` - Unique identifier
- `size_mb` - Tensor size
- `reuse_distance` - Accesses since last use
- `stride` - Memory access pattern
- `op` - Operation type
- `model_type` - Architecture

---

### 4. **`backend/ml/models/scheduler_opt.py`** - ML Eviction Scorer
**Purpose**: Predicts which tensors should be evicted from cache

**What it does**:
- Loads Mistral-7B model (with 4-bit quantization)
- Currently uses simplified heuristic scoring
- Computes eviction scores: `0.4×size + 0.5×reuse_distance + 0.1×stride`
- Higher score = more likely to evict

**Key Function**: `predict_eviction_scores()`

**Output**: Adds `eviction_score` column to trace DataFrame

---

### 5. **`backend/simulator/memory_simulator.py`** - Cache Simulator
**Purpose**: Simulates cache behavior with different policies

**What it does**:
- Processes trace sequentially
- Checks if tensor is in cache (HIT) or not (MISS)
- Evicts tensors when cache is full (using LRU/FIFO/ML policy)
- Tracks metrics: hits, misses, latency, evictions

**Key Classes**:
- `CacheState` - Maintains cache contents
- `MemorySimulator` - Runs simulation

**Output**: Performance metrics dictionary

---

### 6. **`backend/statistical_validation.py`** - Validation Module
**Purpose**: Validates quality of synthetic traces

**What it does**:
- Calculates reuse entropy (Shannon entropy)
- Measures temporal locality (% of short-distance reuses)
- Tests tensor size distribution (KS test)

**Key Functions**:
- `calculate_reuse_entropy()` - Diversity of reuse patterns
- `calculate_temporal_locality()` - Cache-friendliness score
- `validate_tensor_size_distribution()` - Size realism check

**Output**: Validation metrics with interpretations

---

## Complete Request Flow (Input → Output)

### Step-by-Step Cycle

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER INPUT (Streamlit UI)                               │
│    - Model Type: Transformer                                │
│    - Layers: 12                                             │
│    - Cache Policy: LRU                                      │
│    - Reuse Probability: 0.3                                 │
│    - Click "Run Simulation"                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ HTTP POST /simulate?policy=LRU
                     │ Body: {model_type, num_layers, ...}
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. BACKEND RECEIVES REQUEST (main.py)                      │
│    - FastAPI endpoint: simulate()                           │
│    - Extracts parameters from request                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. GENERATE TRACE (generate_synthetic_workload.py)         │
│    - Creates ~576 tensor access entries                     │
│    - For each layer:                                        │
│      • Select tensor types (Q, K, V, output)                │
│      • Assign sizes (4, 8, 16, 32 MB)                       │
│      • Simulate reuse (binomial distribution)               │
│    - Returns: DataFrame with trace data                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ DataFrame (576 rows)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. COMPUTE EVICTION SCORES (scheduler_opt.py)              │
│    - Lazy load Mistral-7B model (if not loaded)            │
│    - For each tensor:                                       │
│      • Normalize size, reuse_distance, stride               │
│      • Score = 0.4×size + 0.5×reuse + 0.1×stride           │
│    - Returns: DataFrame + eviction_score column             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Scored DataFrame
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. RUN SIMULATION (memory_simulator.py)                    │
│    - Initialize cache (64 MB capacity)                      │
│    - For each tensor access in trace:                       │
│      • Check if tensor_id in cache                          │
│        ✅ HIT: Add 0.01ms latency, update LRU order         │
│        ❌ MISS: Fetch from DRAM (slow)                      │
│          - Latency = size×0.1ms + size/bandwidth            │
│          - If cache full: Evict by policy                   │
│            * LRU: Remove least recently used                │
│            * FIFO: Remove oldest                            │
│            * ML: Remove highest eviction_score              │
│          - Add tensor to cache                              │
│    - Returns: Metrics dictionary                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Metrics: {hits, misses, latency, ...}
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. STATISTICAL VALIDATION (if enabled)                     │
│    (statistical_validation.py)                              │
│    - Calculate reuse entropy                                │
│    - Calculate temporal locality score                      │
│    - Validate tensor size distribution                      │
│    - Returns: Validation metrics                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ All results combined
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. BACKEND RETURNS JSON RESPONSE                           │
│    {                                                         │
│      "cache_hits (count)": 417,                             │
│      "dram_accesses (count)": 45,                           │
│      "total_latency (ms)": 2.39,                            │
│      "evictions (count)": 78,                               │
│      "cache_hit_rate (%)": 90.26,                           │
│      "trace": [...],  // with eviction_scores               │
│      "validation": {...}  // if enabled                     │
│    }                                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ HTTP Response
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. FRONTEND DISPLAYS RESULTS (app.py)                      │
│    - Simulation Metrics (table)                             │
│      • Cache hits, misses, latency                          │
│      • Hit rate percentage                                  │
│    - Eviction Scores (table, top 20)                        │
│      • tensor_id, size, reuse_distance, score               │
│    - Cache Occupancy Heatmap                                │
│      • X: tensor_id, Y: reuse_distance                      │
│      • Color: eviction_score                                │
│    - Validation Results (if enabled)                        │
│      • Reuse entropy                                        │
│      • Temporal locality score                              │
│      • Size distribution p-value                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Example Request Trace

### Input
```json
{
  "model_type": "Transformer",
  "num_layers": 12,
  "batch_size": 32,
  "seq_length": 128,
  "reuse_probability": 0.3
}
```

### Processing
1. **Generate**: 576 tensor accesses (12 layers × 4 types × ~12 reuses)
2. **Score**: Each tensor gets eviction_score (0.0-1.0)
3. **Simulate**: Process 576 accesses, track cache behavior
4. **Validate**: Calculate entropy=1.0, locality=1.0

### Output
```json
{
  "cache_hits (count)": 417,
  "dram_accesses (count)": 45,
  "total_latency (ms)": 2.39,
  "evictions (count)": 78,
  "bandwidth_wait_time (ms)": 9.31,
  "cache_hit_rate (%)": 90.26
}
```

---

## Key Data Structures

### Trace DataFrame
```
| tensor_id | size_mb | reuse_distance | stride | op        | eviction_score |
| --------- | ------- | -------------- | ------ | --------- | -------------- |
| Q_0       | 16      | 0              | 1      | attention | 0.12           |
| K_0       | 8       | 0              | 1      | attention | 0.08           |
| Q_0       | 16      | 1              | 1      | attention | 0.32  ← Reused |
| V_0       | 8       | 0              | 1      | attention | 0.08           |
```

### Cache State
```python
{
  "cache_contents": {
    "Q_0": TensorMeta(id="Q_0", size=16, last_access=2),
    "K_0": TensorMeta(id="K_0", size=8, last_access=1),
    ...
  },
  "usage_order": ["K_0", "V_0", "Q_0"],  # LRU tracking
  "current_usage": 48  # MB
}
```

### Metrics Output
```python
{
  "cache_hits": 417,
  "dram_accesses": 45,
  "total_latency": 2.39,
  "evictions": 78,
  "bandwidth_wait_time": 9.31,
  "cache_hit_rate": 90.26
}
```

---

## Performance Characteristics

- **Backend Response Time**: ~0.03s per simulation
- **Trace Generation**: ~0.01s for 576 entries
- **Simulation**: ~0.02s for 576 accesses
- **Model Loading**: ~2-5s first time (Mistral-7B, lazy loaded)

---

## Summary

**Input**: User selects model configuration  
**Process**: Generate trace → Score tensors → Simulate cache → Validate  
**Output**: Performance metrics + visualizations  

The entire cycle takes ~30-50ms (excluding first-time model loading), making it suitable for interactive experimentation with different cache policies and model architectures.
