# Tensor Size Comparison - All Model Types

## Complete Simulation Results

### 1. TRANSFORMER (hidden_dim=768)

**Tensor Types**: Q, K, V, output

#### Configuration 1: batch=32, seq=128
- **Q**: 12.0 MB
- **K**: 12.0 MB
- **V**: 12.0 MB
- **output**: 48.0 MB
- **TOTAL**: 84.0 MB per layer

#### Configuration 2: batch=64, seq=256
- **Q**: 48.0 MB
- **K**: 48.0 MB
- **V**: 48.0 MB
- **output**: 192.0 MB
- **TOTAL**: 336.0 MB per layer

#### Configuration 3: batch=128, seq=512
- **Q**: 192.0 MB
- **K**: 192.0 MB
- **V**: 192.0 MB
- **output**: 768.0 MB
- **TOTAL**: 1,344.0 MB per layer

---

### 2. CNN (hidden_dim=512)

**Tensor Types**: weights, activations, output

#### Configuration 1: batch=32, seq=128
- **weights**: 1.0 MB (constant - doesn't scale with batch/seq)
- **activations**: 8.0 MB
- **output**: 32.0 MB
- **TOTAL**: 41.0 MB per layer

#### Configuration 2: batch=64, seq=256
- **weights**: 1.0 MB (constant)
- **activations**: 32.0 MB
- **output**: 128.0 MB
- **TOTAL**: 161.0 MB per layer

#### Configuration 3: batch=128, seq=512
- **weights**: 1.0 MB (constant)
- **activations**: 128.0 MB
- **output**: 512.0 MB
- **TOTAL**: 641.0 MB per layer

---

### 3. MLP (hidden_dim=1024)

**Tensor Types**: weights, activations

#### Configuration 1: batch=32, seq=128
- **weights**: 4.0 MB (constant)
- **activations**: 16.0 MB
- **TOTAL**: 20.0 MB per layer

#### Configuration 2: batch=64, seq=256
- **weights**: 4.0 MB (constant)
- **activations**: 64.0 MB
- **TOTAL**: 68.0 MB per layer

#### Configuration 3: batch=128, seq=512
- **weights**: 4.0 MB (constant)
- **activations**: 256.0 MB
- **TOTAL**: 260.0 MB per layer

---

### 4. MoE - Mixture of Experts (hidden_dim=768)

**Tensor Types**: expert_weights, activations

#### Configuration 1: batch=32, seq=128
- **expert_weights**: 18.0 MB (8 experts, constant)
- **activations**: 12.0 MB
- **TOTAL**: 30.0 MB per layer

#### Configuration 2: batch=64, seq=256
- **expert_weights**: 18.0 MB (constant)
- **activations**: 48.0 MB
- **TOTAL**: 66.0 MB per layer

#### Configuration 3: batch=128, seq=512
- **expert_weights**: 18.0 MB (constant)
- **activations**: 192.0 MB
- **TOTAL**: 210.0 MB per layer

---

## Summary Table

| Model Type      | Config       | Q/K/V or weights | activations | output | TOTAL/layer |
| --------------- | ------------ | ---------------- | ----------- | ------ | ----------- |
| **Transformer** | b=32, s=128  | 12 MB each       | -           | 48 MB  | 84 MB       |
|                 | b=64, s=256  | 48 MB each       | -           | 192 MB | 336 MB      |
|                 | b=128, s=512 | 192 MB each      | -           | 768 MB | 1,344 MB    |
| **CNN**         | b=32, s=128  | 1 MB (weights)   | 8 MB        | 32 MB  | 41 MB       |
|                 | b=64, s=256  | 1 MB (weights)   | 32 MB       | 128 MB | 161 MB      |
|                 | b=128, s=512 | 1 MB (weights)   | 128 MB      | 512 MB | 641 MB      |
| **MLP**         | b=32, s=128  | 4 MB (weights)   | 16 MB       | -      | 20 MB       |
|                 | b=64, s=256  | 4 MB (weights)   | 64 MB       | -      | 68 MB       |
|                 | b=128, s=512 | 4 MB (weights)   | 256 MB      | -      | 260 MB      |
| **MoE**         | b=32, s=128  | 18 MB (experts)  | 12 MB       | -      | 30 MB       |
|                 | b=64, s=256  | 18 MB (experts)  | 48 MB       | -      | 66 MB       |
|                 | b=128, s=512 | 18 MB (experts)  | 192 MB      | -      | 210 MB      |

---

## Multi-Layer Impact (12 layers)

### With 64 MB Cache

| Model           | Config       | Total Working Set | Cache Pressure          |
| --------------- | ------------ | ----------------- | ----------------------- |
| **Transformer** | b=32, s=128  | 1,008 MB          | 🔴 Very High (16× cache) |
| **Transformer** | b=64, s=256  | 4,032 MB          | 🔴 Extreme (63× cache)   |
| **Transformer** | b=128, s=512 | 16,128 MB         | 🔴 Critical (252× cache) |
| **CNN**         | b=32, s=128  | 492 MB            | 🟡 High (8× cache)       |
| **CNN**         | b=64, s=256  | 1,932 MB          | 🔴 Very High (30× cache) |
| **CNN**         | b=128, s=512 | 7,692 MB          | 🔴 Extreme (120× cache)  |
| **MLP**         | b=32, s=128  | 240 MB            | 🟡 Moderate (4× cache)   |
| **MLP**         | b=64, s=256  | 816 MB            | 🟡 High (13× cache)      |
| **MLP**         | b=128, s=512 | 3,120 MB          | 🔴 Very High (49× cache) |
| **MoE**         | b=32, s=128  | 360 MB            | 🟡 Moderate (6× cache)   |
| **MoE**         | b=64, s=256  | 792 MB            | 🟡 High (12× cache)      |
| **MoE**         | b=128, s=512 | 2,520 MB          | 🔴 Very High (39× cache) |

---

## Key Insights

### Scaling Behavior

1. **Transformer**: 
   - Most memory-intensive due to Q/K/V + large output
   - Output is 4× larger than Q/K/V (FFN expansion)
   - Scales linearly with batch_size and seq_length

2. **CNN**:
   - Weights are constant (1 MB regardless of batch/seq)
   - Activations and output scale with batch/seq
   - More efficient than Transformer for same config

3. **MLP**:
   - Simplest structure (only 2 tensor types)
   - Weights are constant (4 MB)
   - Most memory-efficient for small configs

4. **MoE**:
   - Expert weights are large (18 MB) but constant
   - Activations scale with batch/seq
   - Middle ground between CNN and Transformer

### Doubling Effects

- **2× batch_size** → 2× tensor size (for non-weight tensors)
- **2× seq_length** → 2× tensor size (for non-weight tensors)
- **2× both** → 4× tensor size (quadratic growth!)

### Cache Implications

With a **64 MB cache**:
- **Small configs** (b=32, s=128): Moderate pressure, decent hit rates
- **Medium configs** (b=64, s=256): High pressure, lower hit rates
- **Large configs** (b=128, s=512): Extreme pressure, cache thrashing

**Recommendation**: For realistic cache simulation with 64 MB cache, use:
- Transformer: b≤32, s≤128
- CNN: b≤64, s≤256
- MLP: b≤128, s≤256
- MoE: b≤64, s≤256

---

## Formula Reference

### General Formula
```
size_mb = (batch_size × seq_length × hidden_dim × 4 bytes) / (1024²)
```

### Model-Specific Hidden Dimensions
- Transformer: 768 (BERT-base, GPT-2 small)
- CNN: 512 (ResNet-like)
- MLP: 1024 (Large MLP)
- MoE: 768 (Mixture of Experts)

### Tensor-Specific Multipliers
- **Q/K/V**: `batch × seq × hidden_dim`
- **output**: `batch × seq × (hidden_dim × 4)` ← 4× FFN expansion
- **weights**: `hidden_dim × hidden_dim` ← constant
- **activations**: `batch × seq × hidden_dim`
- **expert_weights**: `hidden_dim × hidden_dim × 8` ← 8 experts
