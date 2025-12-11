"""
COMPILER PREDICTION - What It Does and How to Use It

PURPOSE:
Predicts compilation latency and energy consumption for GPU/accelerator code
before actually compiling it. This helps developers:
1. Estimate compilation time for large codebases
2. Optimize code before expensive compilation
3. Choose between different optimization levels

REAL-WORLD EXAMPLE:
You have CUDA kernel code and want to know:
- Will it take 1 second or 10 minutes to compile?
- Will it use 0.5J or 5J of energy?
"""

import re
import numpy as np


def extract_code_features(code: str) -> dict:
    """
    Extract features from code that affect compilation time/energy.
    
    Features that matter:
    - Number of loops (more loops = longer compilation)
    - Number of branches (if/else statements)
    - Memory operations (load/store)
    - Arithmetic operations (FLOPs)
    - Code complexity
    """
    features = {}
    
    # Count loops (for, while)
    features["num_loops"] = len(re.findall(r'\b(for|while)\b', code))
    
    # Count branches (if, else, switch)
    features["num_branches"] = len(re.findall(r'\b(if|else|switch|case)\b', code))
    
    # Count memory operations
    features["load_ops"] = len(re.findall(r'\b(load|read|\[\])\b', code))
    features["store_ops"] = len(re.findall(r'\b(store|write|=)\b', code))
    
    # Estimate FLOPs (floating point operations)
    features["flops"] = len(re.findall(r'[+\-*/]', code))
    
    # Estimate memory usage (rough)
    features["mem_bytes"] = len(code) * 10  # Rough estimate
    
    # Code length
    features["lines_of_code"] = len(code.split('\n'))
    
    return features


def predict_compilation_metrics(features: dict, optimization_level: str = "O0") -> dict:
    """
    Predict compilation latency and energy based on code features.
    
    Args:
        features: Dictionary of code features
        optimization_level: O0 (none), O1 (basic), O2 (moderate), O3 (aggressive)
    
    Returns:
        Dictionary with latency (ms) and energy (J) predictions
    """
    
    # Base compilation time (ms) - depends on optimization level
    base_latency = {
        "O0": 10,    # No optimization - fast compile
        "O1": 50,    # Basic optimization
        "O2": 200,   # Moderate optimization
        "O3": 500    # Aggressive optimization - slow compile
    }.get(optimization_level, 10)
    
    # Calculate latency based on features
    # More complex code = longer compilation
    latency = base_latency
    latency += features["num_loops"] * 20        # Each loop adds 20ms
    latency += features["num_branches"] * 10     # Each branch adds 10ms
    latency += features["flops"] * 0.5           # Each FLOP adds 0.5ms
    latency += features["lines_of_code"] * 2     # Each line adds 2ms
    
    # Energy consumption (Joules)
    # Energy ≈ Power × Time
    # Assume compiler uses ~50W during compilation
    power_watts = 50
    time_seconds = latency / 1000  # Convert ms to seconds
    energy = power_watts * time_seconds
    
    # Add some variance (±10%)
    latency *= np.random.uniform(0.9, 1.1)
    energy *= np.random.uniform(0.9, 1.1)
    
    return {
        "latency_ms": round(latency, 2),
        "energy_joules": round(energy, 3),
        "optimization_level": optimization_level
    }


def calculate_feature_importance(features: dict) -> dict:
    """
    Calculate which features contribute most to compilation time.
    """
    total = sum(features.values())
    if total == 0:
        return {k: 0.0 for k in features}
    
    importance = {}
    for key, value in features.items():
        importance[key] = round(value / total, 3)
    
    return importance


# ============================================================================
# EXAMPLE 1: Simple CUDA Kernel
# ============================================================================

simple_cuda_code = """
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"""

print("="*70)
print("EXAMPLE 1: Simple CUDA Vector Addition")
print("="*70)
print(f"Code:\n{simple_cuda_code}")

features_simple = extract_code_features(simple_cuda_code)
print(f"\nExtracted Features:")
for key, value in features_simple.items():
    print(f"  {key:20s}: {value}")

print(f"\nPredictions for different optimization levels:")
for opt_level in ["O0", "O1", "O2", "O3"]:
    result = predict_compilation_metrics(features_simple, opt_level)
    print(f"  {opt_level}: Latency={result['latency_ms']:.1f}ms, Energy={result['energy_joules']:.3f}J")

# ============================================================================
# EXAMPLE 2: Complex Matrix Multiplication
# ============================================================================

complex_cuda_code = """
__global__ void matrixMul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
"""

print("\n" + "="*70)
print("EXAMPLE 2: Complex Matrix Multiplication")
print("="*70)
print(f"Code:\n{complex_cuda_code}")

features_complex = extract_code_features(complex_cuda_code)
print(f"\nExtracted Features:")
for key, value in features_complex.items():
    print(f"  {key:20s}: {value}")

result_complex = predict_compilation_metrics(features_complex, "O2")
print(f"\nPrediction (O2 optimization):")
print(f"  Latency: {result_complex['latency_ms']:.1f} ms")
print(f"  Energy:  {result_complex['energy_joules']:.3f} J")

importance = calculate_feature_importance(features_complex)
print(f"\nFeature Importance (what affects compilation most):")
for key, value in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {key:20s}: {value*100:.1f}%")

# ============================================================================
# EXAMPLE 3: Comparison - Simple vs Complex
# ============================================================================

print("\n" + "="*70)
print("COMPARISON: Simple vs Complex Code")
print("="*70)

result_simple = predict_compilation_metrics(features_simple, "O2")
result_complex = predict_compilation_metrics(features_complex, "O2")

print(f"\nSimple Vector Add:")
print(f"  Latency: {result_simple['latency_ms']:.1f} ms")
print(f"  Energy:  {result_simple['energy_joules']:.3f} J")

print(f"\nComplex Matrix Mul:")
print(f"  Latency: {result_complex['latency_ms']:.1f} ms")
print(f"  Energy:  {result_complex['energy_joules']:.3f} J")

print(f"\nDifference:")
print(f"  Latency: {result_complex['latency_ms'] - result_simple['latency_ms']:.1f} ms slower")
print(f"  Energy:  {result_complex['energy_joules'] - result_simple['energy_joules']:.3f} J more")

# ============================================================================
# EXAMPLE 4: Optimization Level Impact
# ============================================================================

print("\n" + "="*70)
print("OPTIMIZATION LEVEL IMPACT")
print("="*70)
print("\nSame code, different optimization levels:\n")

print(f"{'Level':<10} {'Latency (ms)':<15} {'Energy (J)':<12} {'Speedup'}")
print("-" * 50)

baseline = predict_compilation_metrics(features_complex, "O0")
print(f"{'O0 (none)':<10} {baseline['latency_ms']:<15.1f} {baseline['energy_joules']:<12.3f} 1.0x")

for level in ["O1", "O2", "O3"]:
    result = predict_compilation_metrics(features_complex, level)
    speedup = baseline['latency_ms'] / result['latency_ms']
    print(f"{level:<10} {result['latency_ms']:<15.1f} {result['energy_joules']:<12.3f} {speedup:.1f}x")

# ============================================================================
# KEY INSIGHTS
# ============================================================================

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("""
1. COMPLEXITY MATTERS
   - More loops/branches = longer compilation
   - Complex code can take 10x longer to compile

2. OPTIMIZATION TRADE-OFF
   - O0: Fast compile (10ms), but slow runtime
   - O3: Slow compile (500ms), but fast runtime
   - Choose based on: development (O0) vs production (O3)

3. ENERGY CONSIDERATION
   - Compilation energy = Power × Time
   - Complex code at O3 can use 25J vs 0.5J at O0
   - Important for: battery-powered devices, cloud costs

4. PRACTICAL USE
   - Predict before compiling large codebases
   - Choose optimization level wisely
   - Optimize code structure to reduce compilation time
""")

print("="*70)
print("HOW TO USE IN YOUR PROJECT")
print("="*70)
print("""
1. Paste your GPU/CUDA code in the text box
2. Select optimization level (O0, O1, O2, O3)
3. Click "Predict"
4. See estimated latency and energy
5. Decide: compile now or optimize code first?

Example decision:
- Latency < 1s, Energy < 1J → Compile immediately
- Latency > 10s, Energy > 10J → Optimize code first
""")
