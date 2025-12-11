# 🎓 Accelerator Optimization Copilot - Complete Engineering Guide

## 📋 Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Backend Deep Dive](#backend-deep-dive)
4. [Frontend UI Explained](#frontend-ui-explained)
5. [Output Interpretation](#output-interpretation)
6. [Mix-and-Match Scenarios](#mix-and-match-scenarios)
7. [Performance Analysis](#performance-analysis)
8. [Best Practices](#best-practices)
9. [Common Issues & Solutions](#common-issues--solutions)
10. [Final Takeaways](#final-takeaways)

---

# 1. High-Level Overview

## What Is This System?

The **Accelerator Optimization Copilot** is a memory hierarchy simulator and optimizer for deep learning workloads. It answers the critical question:

> **"How well does my cache/memory system handle tensor operations from different model architectures?"**

### The Big Picture

```
┌─────────────────┐
│  User Selects   │
│  Model Type     │──┐
│  (Transformer,  │  │
│   CNN, MLP,     │  │
│   MoE)          │  │
└─────────────────┘  │
                     ▼
          ┌──────────────────────┐
          │ Generate Synthetic   │
          │ Memory Trace         │
          │ (Tensor Access       │
          │  Patterns)           │
          └──────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Apply ML Eviction    │
          │ Scoring              │
          │ (Predict which       │
          │  tensors to evict)   │
          └──────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Run Cache Simulator  │
          │ with Policy          │
          │ (LRU/FIFO/ML)        │
          └──────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Collect Metrics:     │
          │ - Latency            │
          │ - Cache Hits         │
          │ - Evictions          │
          │ - Bandwidth Usage    │
          └──────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Statistical          │
          │ Validation           │
          │ (Chi-square, KS)     │
          └──────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Visualize in UI:     │
          │ - Heatmaps           │
          │ - Tables             │
          │ - Metrics            │
          └──────────────────────┘
```

### Why This Matters

Different ML models have **drastically different memory access patterns**:

- **Transformers**: High reuse of Q/K/V tensors, attention creates temporal locality
- **CNNs**: Spatial locality, predictable strided access, activation reuse
- **MLPs**: Simple sequential access, less reuse
- **MoE**: Sparse, unpredictable access to expert weights

**The cache policy that works for one model may be terrible for another!**

---

# 2. Architecture & Data Flow

## System Components

```
┌─────────────────────────────────────────────────────┐
│                  FRONTEND (Streamlit)               │
│  ┌──────────────────────────────────────────────┐  │
│  │  Controls: Model Type, Layers, Policy, etc.  │  │
│  └──────────────────────────────────────────────┘  │
│                        │                             │
│                        │ HTTP POST                   │
│                        ▼                             │
└────────────────────────┼──────────────────────────────┘
                         │
┌────────────────────────┼──────────────────────────────┐
│                  BACKEND (FastAPI)                    │
│                        │                              │
│  ┌─────────────────────▼────────────────────────┐   │
│  │         Endpoint: /simulate                   │   │
│  └─────────────────────┬────────────────────────┘   │
│                        │                              │
│  ┌─────────────────────▼────────────────────────┐   │
│  │   1. Generate Synthetic Trace                 │   │
│  │      (workload/generate_synthetic_workload)   │   │
│  └─────────────────────┬────────────────────────┘   │
│                        │                              │
│  ┌─────────────────────▼────────────────────────┐   │
│  │   2. Predict Eviction Scores                  │   │
│  │      (ml/models/scheduler_opt)                │   │
│  └─────────────────────┬────────────────────────┘   │
│                        │                              │
│  ┌─────────────────────▼────────────────────────┐   │
│  │   3. Run Cache Simulation                     │   │
│  │      (simulator/memory_simulator)             │   │
│  └─────────────────────┬────────────────────────┘   │
│                        │                              │
│  ┌─────────────────────▼────────────────────────┐   │
│  │   4. Statistical Validation (optional)        │   │
│  │      (statistical_validation)                 │   │
│  └─────────────────────┬────────────────────────┘   │
│                        │                              │
│  ┌─────────────────────▼────────────────────────┐   │
│  │   Return: Metrics + Trace Data                │   │
│  └───────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────┘
```

## Request Flow Example

Let's trace a complete request where user clicks "Run Simulation":

1. **User Action**: Selects "Transformer", 12 layers, LRU policy, clicks "Run Simulation"

2. **Frontend**: Sends POST request to `http://127.0.0.1:8000/simulate?policy=LRU`
   ```json
   {
     "model_type": "Transformer",
     "num_layers": 12,
     "batch_size": 32,
     "seq_length": 128,
     "reuse_probability": 0.3
   }
   ```

3. **Backend receives request** → `backend/main.py::simulate()`

4. **Step 1: Generate trace**
   - Creates ~576 tensor access entries (12 layers × 4 tensor types × reuse)
   - Each entry has: `tensor_id`, `size_mb`, `reuse_distance`, `stride`, `op`, `model_type`

5. **Step 2: ML scoring**
   - Computes eviction_score for each tensor
   - Score = 0.4×size + 0.5×reuse_distance + 0.1×stride
   - Higher score = more likely to evict

6. **Step 3: Simulation**
   - Processes trace sequentially
   - For each tensor access:
     - Check if in cache → HIT (fast, 0.01ms)
     - If not in cache → MISS (slow, fetch from DRAM)
     - If cache full → Evict by policy (LRU/FIFO/ML)

7. **Step 4: Return metrics**
   ```json
   {
     "cache_hits (count)": 417,
     "dram_accesses (count)": 100,
     "total_latency (ms)": 3.175,
     "evictions (count)": 78,
     "bandwidth_wait_time (ms)": 46.875,
     "cache_hit_rate (%)": 80.61
   }
   ```

8. **Frontend visualizes** → Heatmap, table, charts

---

# 3. Backend Deep Dive

## 3.1 Synthetic Trace Generation

**File**: `backend/workload/generate_synthetic_workload.py`

### What Happens

```python
def generate_synthetic_trace(
    model_type: str = "Transformer",
    num_layers: int = 96,
    batch_size: int = 32,
    seq_length: int = 128,
    reuse_probability: float = 0.3,
    max_tensors_per_layer: int = 16
) -> pd.DataFrame
```

### Process

**For each layer**:
1. Select tensor types based on model:
   - Transformer → Q, K, V, output
   - CNN → weights, activations, output
   - MLP → weights, activations
   - MoE → expert_weights, activations

2. **For each tensor type**:
   - Generate `tensor_id` (e.g., "Q_3" for Query tensor in layer 3)
   - Choose `size_mb` randomly from [4, 8, 16, 32] MB
   - Set `stride`:
     - Non-CNN: stride = 1 (sequential)
     - CNN: stride ∈ {1, 2, 4} (spatial convolutions)
   - Choose `op` (operation type): matmul, attention, conv, relu, gating
   - Compute `reuse_count` ~ Binomial(n=5, p=reuse_probability)

3. **For each reuse** (0 to reuse_count):
   - Create trace entry with `reuse_distance` = access_idx
   - Add to trace list

### Field Meanings

| Field | Meaning | Impact on Simulation |
|-------|---------|---------------------|
| **tensor_id** | Unique identifier (e.g., "Q_5") | Used to track cache presence |
| **size_mb** | Tensor size in MB | Affects cache capacity, eviction decisions |
| **reuse_distance** | # accesses since last use | Higher = colder → more likely to evict |
| **stride** | Memory access pattern | Higher stride = worse spatial locality |
| **op** | Operation type | Contextual info (not used in simulation yet) |
| **model_type** | Architecture | Statistical validation reference |

### Example Trace

```
tensor_id | size_mb | reuse_distance | stride | op        | model_type
----------|---------|----------------|--------|-----------|------------
Q_0       | 16      | 0              | 1      | attention | Transformer
K_0       | 8       | 0              | 1      | attention | Transformer
V_0       | 8       | 0              | 1      | attention | Transformer
Q_0       | 16      | 1              | 1      | attention | Transformer  ← Reuse!
output_0  | 32      | 0              | 1      | matmul    | Transformer
...
```

### Why Reuse Probability Matters

- **High reuse (0.7-0.9)**: Transformer attention patterns
  - Same Q/K/V tensors accessed multiple times
  - Good for LRU caching
  
- **Low reuse (0.1-0.3)**: MLP, some CNN layers
  - Mostly one-time access
  - LRU doesn't help much

- **Medium reuse (0.4-0.6)**: CNN activations
  - Some layers reused for skip connections

---

## 3.2 ML Eviction Scoring

**File**: `backend/ml/models/scheduler_opt.py`

### Purpose

Predict which tensors should be evicted from cache when space is needed.

### Algorithm

```python
def predict_eviction_scores(trace_df: pd.DataFrame):
    inputs = trace_df[["size_mb", "reuse_distance", "stride"]].values
    
    # Normalize to [0, 1]
    inputs_normalized = inputs / (inputs.max(axis=0) + 1e-8)
    
    # Weighted score
    eviction_scores = (
        0.4 * inputs_normalized[:, 0] +  # size_mb
        0.5 * inputs_normalized[:, 1] +  # reuse_distance
        0.1 * inputs_normalized[:, 2]    # stride
    )
    
    return trace_df.with_column("eviction_score", eviction_scores)
```

### Scoring Logic

**Higher eviction score = More likely to evict**

- **40% weight on size**: Large tensors consume cache space
- **50% weight on reuse_distance**: Cold tensors (not accessed recently)
- **10% weight on stride**: Poor spatial locality

### Example Scores

```
tensor_id | size_mb | reuse_distance | stride | eviction_score
----------|---------|----------------|--------|----------------
Q_0       | 4       | 0              | 1      | 0.12  ← Keep (hot, small)
V_5       | 32      | 5              | 1      | 0.85  ← Evict (cold, large)
weights_2 | 16      | 2              | 4      | 0.56  ← Medium priority
```

### When ML Policy is Used

If user selects "ML" policy:
- Simulator uses `eviction_score` to decide what to evict
- **Lowest score** = keep in cache
- **Highest score** = evict first

---

## 3.3 Cache Simulation

**File**: `backend/simulator/memory_simulator.py`

### Components

#### CacheState
```python
class CacheState:
    max_cache_mb: int = 64         # Cache capacity
    cache_contents: Dict[str, TensorMeta] = {}  # What's in cache
    usage_order: List[str] = []    # For LRU tracking
```

#### MemorySimulator
```python
class MemorySimulator:
    cache: CacheState
    dram_latency_per_mb: float = 0.1  # 0.1ms per MB
    dram_bandwidth: int = 32          # 32 MB/s
```

### Simulation Loop

**For each trace entry** (in order):

```python
def access_tensor(tensor_id, size_mb, access_idx, policy):
    if tensor_id in cache:
        ✅ CACHE HIT
        - Add 0.01ms latency (fast cache access)
        - Increment cache_hits counter
        - Update LRU order (move to end)
    else:
        ❌ CACHE MISS
        - Increment dram_accesses counter
        - Calculate fetch_time:
          fetch_time = size_mb * 0.1 + size_mb / 32
          (latency + bandwidth cost)
        - Add to total_latency
        
        while cache is full:
            🗑️ EVICTION NEEDED
            if policy == "LRU":
                evict oldest (usage_order[0])
            elif policy == "FIFO":
                evict first inserted
            elif policy == "ML":
                evict highest eviction_score
            
            Remove from cache
            Increment evictions counter
        
        Add tensor to cache
```

### Latency Calculation

**Cache Hit**: 0.01 ms (fixed, very fast)

**Cache Miss**: 
```
total_time = transfer_latency + bandwidth_time
           = size_mb × 0.1 ms + size_mb / 32 MB/s
```

Example for 16 MB tensor:
```
= 16 × 0.1 + 16 / 32
= 1.6 + 0.5
= 2.1 ms
```

### Policy Comparison

| Policy | Eviction Strategy | Best For |
|--------|------------------|----------|
| **LRU** | Evict least recently used | Temporal locality (Transformers) |
| **FIFO** | Evict oldest insertion | Simple streaming workloads |
| **ML** | Evict highest score | Mixed workloads, learned patterns |

---

## 3.4 Statistical Validation

**File**: `backend/statistical_validation.py`

### Purpose

Validate that synthetic traces are **statistically realistic** for the chosen model type.

### Test 1: Chi-Square (Reuse Distribution)

```python
def validate_reuse_rates(trace_df, model_type):
    observed = trace_df['reuse_distance'].value_counts()
    expected = geometric_distribution()  # Real-world pattern
    
    χ² = Σ((observed - expected)² / expected)
    p_value = chi_square_test(χ²)
    
    return p_value
```

**Interpretation**:
- **p > 0.05**: ✅ Reuse pattern looks realistic
- **p < 0.05**: ⚠️ Unusual pattern, may be over/under-sampled

### Test 2: Kolmogorov-Smirnov (Size Distribution)

```python
def validate_tensor_size_distribution(trace_df, model_type):
    observed_sizes = trace_df['size_mb'].values
    expected_sizes = [4, 8, 16, 32]  # Typical for model_type
    
    KS_statistic, p_value = ks_2samp(observed, expected)
    
    return p_value
```

**Interpretation**:
- **p > 0.05**: ✅ Size distribution is reasonable
- **p < 0.05**: ⚠️ Sizes don't match typical patterns

### Why This Matters

If p-values are too low, your synthetic trace might not represent real workloads accurately, leading to misleading simulation results.

---

# 4. Frontend UI Explained

## 4.1 UI Controls (Top Section)

![UI Controls](file:///C:/Users/vprad/.gemini/antigravity/brain/c35dc305-6f04-47c3-82f2-c36984fda12a/uploaded_image_0_1764222459025.png)

### Input Parameters

| Control | Purpose | Impact |
|---------|---------|--------|
| **Model Type** | Transformer/CNN/MLP/MoE | Changes tensor types and access patterns |
| **Cache Policy** | LRU/FIFO/ML | Eviction algorithm used |
| **Layers** | 4-48 | More layers → more tensors → more evictions |
| **Batch Size** | 1-256 | Affects tensor sizes (not directly in current impl) |
| **Sequence Length** | 32-1024 | Longer = more memory pressure |
| **Reuse Probability** | 0.0-1.0 | Higher = more temporal locality |
| **Run Validation** | Checkbox | Whether to run chi-square/KS tests |

### How They Connect

```
Model Type ──┐
Layers ──────┤
Batch Size ──┤─→ generate_synthetic_trace() ─→ DataFrame
Seq Length ──┤
Reuse Prob ──┘

DataFrame + Policy ─→ simulate() ─→ Metrics

Metrics ─→ Display in UI
```

---

## 4.2 Simulation Metrics

![Simulation Metrics](file:///C:/Users/vprad/.gemini/antigravity/brain/c35dc305-6f04-47c3-82f2-c36984fda12a/uploaded_image_0_1764222459025.png)

### Metrics Explained

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

#### cache_hits (count)
- **What**: # of times tensor found in cache
- **Good**: High (> 80% hit rate)
- **Bad**: Low (< 50% hit rate)
- **Why it matters**: Cache hits are 100× faster than DRAM

#### dram_accesses (count)
- **What**: # of times had to fetch from DRAM
- **Good**: Low
- **Bad**: High (approaching total accesses)
- **Why it matters**: DRAM is the bottleneck

#### total_latency (ms)
- **What**: Total simulated execution time
- **Formula**: Σ(cache_access_time) + Σ(dram_fetch_time)
- **Good**: < 10 ms for small workloads
- **Bad**: > 100 ms indicates poor locality

#### evictions (count)
- **What**: # of times tensors kicked out of cache
- **Good**: Low relative to accesses
- **Bad**: Excessive (> 50% of accesses) = thrashing
- **Why it matters**: Evictions waste bandwidth

#### bandwidth_wait_time (ms)
- **What**: Time waiting for memory bandwidth
- **Formula**: Σ(size_mb / bandwidth)
- **Good**: < 20% of total_latency
- **Bad**: > 50% = bandwidth-bound

#### cache_hit_rate (%)
- **What**: cache_hits / (cache_hits + dram_accesses) × 100
- **Excellent**: > 90%
- **Good**: 70-90%
- **Poor**: 50-70%
- **Bad**: < 50%

---

## 4.3 Predicted Eviction Scores Table

![Eviction Scores](file:///C:/Users/vprad/.gemini/antigravity/brain/c35dc305-6f04-47c3-82f2-c36984fda12a/uploaded_image_1_1764222459025.png)

### What You See

```
| tensor_id     | size_mb | reuse_distance | stride | op        | eviction_score |
|---------------|---------|----------------|--------|-----------|----------------|
| weights_0     | 8       | 1              | 1      | gating    | 0.32           |
| weights_0     | 8       | 2              | 1      | gating    | 0.42           |
| activations_0 | 4       | 6              | 4      | attention | 0.50           |
| activations_0 | 4       | 2              | 4      | attention | 0.28           |
```

### How to Read It

- **Same tensor_id, different reuse_distance**: Same tensor accessed at different times
- **Higher eviction_score**: More likely to be evicted under ML policy
- **Look for patterns**:
  - Small, frequently reused → low score (keep in cache)
  - Large, rarely reused → high score (evict first)

---

## 4.4 Cache Occupancy Heatmap

![Heatmap](file:///C:/Users/vprad/.gemini/antigravity/brain/c35dc305-6f04-47c3-82f2-c36984fda12a/uploaded_image_1_1764222459025.png)

### Axes

- **X-axis**: tensor_id
- **Y-axis**: reuse_distance
- **Color**: eviction_score (darker = higher = more likely to evict)

### What to Look For

✅ **Good Pattern**:
- Dark colors (high scores) at high reuse_distance
- Light colors (low scores) at low reuse_distance
- Clear gradient from light to dark

❌ **Bad Pattern**:
- Random color distribution
- High scores at low reuse_distance
- Uniform colors (no differentiation)

### Example Interpretation

```
Lighter top-left (low reuse, low score) → Frequently accessed, keep in cache
Darker bottom-right (high reuse, high score) → Cold data, evict first
```

---

## 4.5 Statistical Validation

![Validation](file:///C:/Users/vprad/.gemini/antigravity/brain/c35dc305-6f04-47c3-82f2-c36984fda12a/uploaded_image_2_1764222459025.png)

### Output

```json
{
  "reuse_distribution_p_value (chi-square)": 1.0,
  "tensor_size_distribution_p_value (KS-test)": 1.0,
  "interpretation": "p-value > 0.05 indicates good statistical match"
}
```

### What p-values Mean

| p-value | Interpretation | Action |
|---------|---------------|--------|
| **> 0.05** | ✅ Statistically valid | Trace is realistic |
| **0.01-0.05** | ⚠️ Borderline | Check parameters |
| **< 0.01** | ❌ Unrealistic | Adjust reuse_probability |

### Common Issues

**p = 1.0**: Perfect match (may indicate overfitting or placeholder data)
**p ≈ 0**: Distribution completely wrong for model type

---

# 5. Output Interpretation

## 5.1 Reading Simulation Results

### Scenario 1: Excellent Performance

```json
{
  "cache_hits (count)": 450,
  "dram_accesses (count)": 50,
  "total_latency (ms)": 1.2,
  "evictions (count)": 45,
  "bandwidth_wait_time (ms)": 3.5,
  "cache_hit_rate (%)": 90.0
}
```

**Why it's good**:
- ✅ 90% cache hit rate
- ✅ Low latency (1.2ms)
- ✅ Evictions < DRAM accesses (not thrashing)
- ✅ Bandwidth time < 30% of total latency

**What this means**: Policy matches workload well, good temporal locality

---

### Scenario 2: Poor Performance

```json
{
  "cache_hits (count)": 100,
  "dram_accesses (count)": 400,
  "total_latency (ms)": 85.3,
  "evictions (count)": 380,
  "bandwidth_wait_time (ms)": 72.1,
  "cache_hit_rate (%)": 20.0
}
```

**Why it's bad**:
- ❌ Only 20% hit rate
- ❌ High latency (85ms)
- ❌ Evictions ≈ DRAM accesses (thrashing!)
- ❌ Bandwidth-bound (85% of time waiting)

**What this means**: 
- Cache too small OR
- Policy mismatched to workload OR
- No temporal/spatial locality

**Solutions**:
1. Increase cache size
2. Try different policy (LRU → ML)
3. Reduce working set (fewer layers)

---

## 5.2 Bottleneck Identification

### Latency Breakdown

```
total_latency = cache_access_time + dram_fetch_time

cache_access_time = cache_hits × 0.01ms
dram_fetch_time = Σ(size × 0.1 + size/bandwidth)
```

**Example**:
```
cache: 450 hits × 0.01 = 4.5ms
DRAM: 50 misses × 2.1ms avg = 105ms
total = 109.5ms
```

→ **DRAM-bound** (96% time in DRAM)

### Bandwidth vs Latency Bound

**Latency-bound**: `bandwidth_wait_time < 30% of total_latency`
- Dominated by DRAM access latency
- Solution: Improve cache hit rate

**Bandwidth-bound**: `bandwidth_wait_time > 70% of total_latency`
- Saturating memory bandwidth
- Solution: Reduce data movement, compress tensors

---

## 5.3 Reuse Pattern Analysis

### High Reuse Distance Distribution

If most reuse_distance values are high (> 10):
- ❌ Poor temporal locality
- LRU won't help much
- Consider ML policy with predictive prefetch

### Low Reuse Distance (< 5)

- ✅ Good temporal locality
- LRU should work well
- Keep cache small and fast

### Bimodal Distribution

Some tensors hot, some cold:
- Use ML policy to differentiate
- Consider multi-level cache

---

# 6. Mix-and-Match Scenarios

## Scenario 1: Transformer + CNN Policy

**Setup**: Transformer model with stride-optimized policy

**What Happens**:
- Transformer has Q/K/V with stride=1 (sequential)
- CNN policy favors spatial locality (stride-aware)
- **Result**: Policy ignores temporal locality → ❌ poor hit rate

**Metrics**:
```
cache_hit_rate: 45% (bad for Transformer)
evictions: High
latency: 3× higher than LRU
```

**Why it fails**: Transformers need temporal locality (LRU), not spatial (stride)

---

## Scenario 2: MoE + LRU

**Setup**: Mixture-of-Experts with Least Recently Used policy

**What Happens**:
- MoE accesses expert_weights sparsely
- Different experts activated each step
- LRU evicts recently used experts that won't be needed
- **Result**: Moderate performance, could be better with ML

**Metrics**:
```
cache_hit_rate: 55% (mediocre)
evictions: Moderate
latency: Acceptable but not optimal
```

**Why suboptimal**: MoE needs expert-aware eviction, not recency-based

**Better choice**: ML policy that learns expert access patterns

---

## Scenario 3: CNN + Attention-Optimized Policy

**Setup**: CNN with Transformer-style temporal locality policy

**What Happens**:
- CNN has spatial locality (adjacent pixels)
- Attention policy favors temporal reuse
- **Result**: Misses spatial patterns → ❌ poor performance

**Metrics**:
```
cache_hit_rate: 40%
bandwidth_wait_time: 80% of latency (bandwidth-bound)
```

**Why it fails**: Ignores spatial access patterns crucial for CNNs

**Fix**: Use stride-aware policy or tiling strategy

---

## Scenario 4: High Reuse Prob + FIFO

**Setup**: reuse_probability=0.9, policy=FIFO

**What Happens**:
- High reuse → many repeat accesses to same tensors
- FIFO evicts in insertion order, not reuse order
- **Result**: Evicts frequently reused tensors → ❌ thrashing

**Metrics**:
```
cache_hit_rate: 30% (terrible despite high reuse)
evictions: Very high
total_latency: 10× worse than LRU
```

**Why it fails**: FIFO ignores reuse information

**Fix**: Switch to LRU or ML policy

---

## Scenario 5: Large Tensors + Small Cache

**Setup**: size_mb ∈ {16, 32}, cache_size=64MB, many layers

**What Happens**:
- 32MB tensors fill half the cache
- Constant eviction/reload cycle
- **Result**: Cache thrashing

**Metrics**:
```
evictions: ~ dram_accesses (1:1 ratio)
cache_hit_rate: 15%
```

**Solutions**:
1. Increase cache to 256MB
2. Reduce tensor sizes (use smaller batches)
3. Enable tensor compression

---

## Scenario 6: Extreme Sequence Length

**Setup**: seq_length=1024 (very long), Transformer

**What Happens**:
- Attention matrix size ~ seq_length²
- Massive memory footprint
- **Result**: Everything evicted immediately

**Metrics**:
```
total_latency: > 200ms
cache_hit_rate: < 10%
```

**Fix**: Sparse attention, chunking, or Flash Attention

---

## Scenario 7: Stride Conflict

**Setup**: stride=4, policy expects stride=1

**What Happens**:
- Prefetcher loads wrong data
- Cache lines partially used
- **Result**: Wasted bandwidth

**Metrics**:
```
bandwidth_wait_time: 90% of latency
cache_hit_rate: Moderate but inefficient
```

**Fix**: Stride-aware prefetching or padding

---

# 7. Performance Analysis

## 7.1 Best Policy for Each Model

### Transformer
**Best**: LRU or ML  
**Why**: High temporal locality (Q/K/V reuse in attention)  
**Expected hit rate**: 80-95%

**Bad**: FIFO (ignores reuse)

---

### CNN
**Best**: Spatial locality-aware (tiled LRU)  
**Why**: Adjacent pixels accessed together  
**Expected hit rate**: 70-85%

**Bad**: Simple LRU (doesn't exploit spatial patterns)

---

### MLP
**Best**: FIFO or Simple LRU  
**Why**: Mostly sequential, low reuse  
**Expected hit rate**: 50-70%

**Bad**: Complex ML (overhead not worth it)

---

### MoE
**Best**: ML policy with expert tracking  
**Why**: Sparse, unpredictable expert access  
**Expected hit rate**: 60-80% (with ML), 40-60% (without)

**Bad**: LRU (doesn't understand expert usage patterns)

---

## 7.2 Cache Sizing Rules

```
Minimum cache size = sum(hot_working_set)
Optimal cache size = 1.5 × working_set (headroom)
```

**Working Set Estimation**:
- Transformer: batch_size × seq_len × hidden_dim × 4 (Q/K/V/O)
- CNN: batch_size × channels × height × width
- MLP: batch_size × hidden_dim

**Example**:
```
Transformer: 32 × 128 × 768 × 4 = 12.5 MB
Recommended cache: 12.5 × 1.5 = ~20 MB minimum
```

---

## 7.3 Detecting Issues

### Cache Thrashing
**Symptoms**:
- evictions ≈ dram_accesses
- Hit rate < 30%
- Latency 10× higher than expected

**Causes**:
- Cache too small
- Working set > cache capacity
- Poor replacement policy

**Fix**: Increase cache or reduce working set

---

### Bandwidth Saturation
**Symptoms**:
- bandwidth_wait_time > 70% of latency
- High DRAM accesses despite moderate misses

**Causes**:
- Large tensor transfers
- Poor data reuse
- Streaming workload

**Fix**: Increase bandwidth, compress data, or prefetch

---

### False Locality
**Symptoms**:
- Good hit rate but high latency
- Evictions low but DRAM accesses high

**Causes**:
- Cache hits on wrong data
- Incorrect prefetching
- Aliasing in cache

**Fix**: Adjust cache associativity, fix prefetcher

---

# 8. Best Practices

## 8.1 Choosing Parameters

### Model Type Selection
- Match to your actual workload
- Don't use "Transformer" for CNN workloads

### Reuse Probability
- Transformer: 0.6-0.9
- CNN: 0.3-0.6
- MLP: 0.1-0.3
- MoE: 0.4-0.7

### Layer Count
- Start small (4-12) for quick iteration
- Increase to 48+ for realistic models

---

## 8.2 Policy Selection Guide

```
┌─────────────────────────────────────────────────┐
│  Is temporal locality high? (reuse > 0.6)       │
│                                                  │
│  YES ──→ Use LRU                                │
│                                                  │
│  NO ───→ Is access pattern predictable?         │
│           │                                      │
│           YES ──→ Use ML policy                 │
│           NO ───→ Use FIFO                      │
└─────────────────────────────────────────────────┘
```

---

## 8.3 Interpreting Results

### Good Simulation
✅ Hit rate > 70%  
✅ Evictions < 30% of accesses  
✅ Latency proportional to working set  
✅ p-values > 0.05 in validation

### Suspicious Results
⚠️ Hit rate = 100% or 0% (edge case)  
⚠️ Validation p-value = 1.0 (too perfect)  
⚠️ Latency doesn't scale with layers

### Action Items
1. Check if cache_size is realistic
2. Verify reuse_probability matches model
3. Compare multiple policies
4. Run validation tests

---

# 9. Common Issues & Solutions

## Issue 1: Always 100% Hit Rate

**Cause**: Cache too large for working set

**Fix**: Reduce cache_size or increase num_layers

---

## Issue 2: Always 0% Hit Rate

**Cause**: 
- Cache too small
- reuse_probability = 0
- Policy broken

**Fix**: Increase cache or reuse_probability

---

## Issue 3: p-value = 0

**Cause**: Synthetic trace doesn't match real distributions

**Fix**: Adjust reuse_probability, check model_type

---

## Issue 4: Heatmap All Same Color

**Cause**: Eviction scores not normalized properly

**Fix**: Check that tensors have varied size/reuse patterns

---

## Issue 5: UI Shows Error "dataframe() got unexpected keyword"

**Cause**: Old Streamlit version

**Fix**: Already fixed! (removed use_container_width)

---

# 10. Final Takeaways

## 🎯 Top 10 Key Points

1. **Different models need different caching policies**
   - Transformers → LRU (temporal locality)
   - CNNs → Spatial-aware
   - MoE → ML-based

2. **Cache hit rate is the most important metric**
   - > 80% = Excellent
   - 50-80% = Good
   - < 50% = Poor (investigate)

3. **Evictions ≈ DRAM accesses = Thrashing**
   - Cache too small
   - Wrong policy
   - Working set too large

4. **Reuse probability must match model type**
   - Too high for MLP → unrealistic
   - Too low for Transformer → misses temporal locality

5. **Latency breakdown reveals bottlenecks**
   - DRAM-bound: Improve hit rate
   - Bandwidth-bound: Reduce data movement

6. **Statistical validation prevents garbage-in-garbage-out**
   - p-value > 0.05 = realistic trace
   - p-value < 0.01 = synthetic artifacts

7. **Size matters: Large tensors dominate evictions**
   - 32MB tensor can fill half of 64MB cache
   - ML policy helps prioritize small frequent tensors

8. **Policy mismatch = 2-10× performance loss**
   - FIFO for high-reuse workload = disaster
   - LRU for MoE = suboptimal

9. **Heatmap shows eviction behavior visually**
   - Good: Gradient from light (keep) to dark (evict)
   - Bad: Random colors

10. **Start simple, iterate**
    - Begin with 4 layers, tune parameters
    - Scale up to realistic sizes
    - Compare policies systematically

---

## 🚀 Next Steps

1. **Run baseline tests** with all model types
2. **Compare LRU vs ML** for your primary workload
3. **Tune cache_size** to match hardware
4. **Validate** with real traces (when available)
5. **Use heatmaps** to debug unexpected results

---

## 📚 Quick Reference Card

```
┌─────────────────────────────────────────────────────┐
│  METRIC CHEAT SHEET                                  │
├─────────────────────────────────────────────────────┤
│  cache_hit_rate > 80%        ✅ Excellent            │
│  cache_hit_rate 50-80%       ✅ Good                 │
│  cache_hit_rate < 50%        ❌ Poor                 │
│                                                      │
│  evictions < 0.3 × accesses  ✅ Healthy              │
│  evictions > 0.8 × accesses  ❌ Thrashing            │
│                                                      │
│  bandwidth_time < 30%        ✅ Compute-bound        │
│  bandwidth_time > 70%        ❌ Memory-bound         │
│                                                      │
│  p-value > 0.05              ✅ Valid trace          │
│  p-value < 0.01              ❌ Unrealistic          │
└─────────────────────────────────────────────────────┘
```

---

**End of Guide**

You now understand every component from synthetic trace generation to UI visualization. Use this as your reference when debugging, tuning, or explaining results to others!
