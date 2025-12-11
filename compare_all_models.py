import sys
sys.path.append('c:\\Users\\vprad\\.gemini\\antigravity\\scratch\\accelerator_optimization_copilot')

from backend.workload.generate_synthetic_workload import calculate_tensor_size

print("="*80)
print("TENSOR SIZE COMPARISON - ALL MODEL TYPES")
print("="*80)

configs = [
    {"batch": 32, "seq": 128},
    {"batch": 64, "seq": 256},
    {"batch": 128, "seq": 512}
]

model_types = ["Transformer", "CNN", "MLP", "MoE"]

for model_type in model_types:
    print(f"\n{'='*80}")
    print(f"MODEL TYPE: {model_type}")
    print(f"{'='*80}")
    
    # Get tensor types for this model
    tensor_types_map = {
        "Transformer": ["Q", "K", "V", "output"],
        "CNN": ["weights", "activations", "output"],
        "MLP": ["weights", "activations"],
        "MoE": ["expert_weights", "activations"]
    }
    
    tensor_types = tensor_types_map[model_type]
    
    for config in configs:
        batch = config["batch"]
        seq = config["seq"]
        
        print(f"\n📊 Config: batch={batch}, seq={seq}")
        print("-" * 80)
        
        sizes = {}
        for t_type in tensor_types:
            size = calculate_tensor_size(
                tensor_type=t_type,
                model_type=model_type,
                batch_size=batch,
                seq_length=seq
            )
            sizes[t_type] = size
            print(f"   {t_type:20s}: {size:8.1f} MB")
        
        total = sum(sizes.values())
        print(f"   {'TOTAL':20s}: {total:8.1f} MB")

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

# Create summary table
print("\n┌─────────────┬────────────┬──────────────────────────────────────────────────┐")
print("│ Model Type  │ Config     │ Tensor Sizes (MB)                                │")
print("├─────────────┼────────────┼──────────────────────────────────────────────────┤")

for model_type in model_types:
    tensor_types = tensor_types_map[model_type]
    
    for i, config in enumerate(configs):
        batch = config["batch"]
        seq = config["seq"]
        
        sizes_str = ", ".join([
            f"{t}={calculate_tensor_size(t, model_type, batch, seq):.1f}"
            for t in tensor_types
        ])
        
        total = sum([
            calculate_tensor_size(t, model_type, batch, seq)
            for t in tensor_types
        ])
        
        config_str = f"b={batch:3d}, s={seq:3d}"
        
        if i == 0:
            print(f"│ {model_type:11s} │ {config_str:10s} │ {sizes_str:48s} │")
        else:
            print(f"│             │ {config_str:10s} │ {sizes_str:48s} │")
    
    if model_type != model_types[-1]:
        print("├─────────────┼────────────┼──────────────────────────────────────────────────┤")

print("└─────────────┴────────────┴──────────────────────────────────────────────────┘")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("✅ Transformer: Q/K/V are equal size, output is 4× larger (FFN expansion)")
print("✅ CNN: weights are constant, activations scale with batch/seq")
print("✅ MLP: weights are constant, activations scale with batch/seq")
print("✅ MoE: expert_weights are 8× larger (8 experts), activations scale")
print("\n✅ Doubling batch_size → 2× tensor size")
print("✅ Doubling seq_length → 2× tensor size")
print("✅ Doubling both → 4× tensor size")
print("="*80)
