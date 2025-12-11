import numpy as np
import pandas as pd


def calculate_tensor_size(
    tensor_type: str,
    model_type: str,
    batch_size: int,
    seq_length: int
) -> float:
    """
    Calculate realistic tensor size in MB based on model architecture.
    
    Formula: size_mb = (batch_size × seq_length × hidden_dim × bytes_per_element) / (1024²)
    
    Args:
        tensor_type: Type of tensor (Q, K, V, output, weights, etc.)
        model_type: Model architecture (Transformer, CNN, MLP, MoE)
        batch_size: Number of examples in batch
        seq_length: Sequence length (tokens for NLP, pixels for vision)
    
    Returns:
        Tensor size in MB
    """
    # Hidden dimensions by model type (typical values)
    hidden_dims = {
        "Transformer": 768,   # BERT-base, GPT-2 small
        "CNN": 512,           # ResNet-like
        "MLP": 1024,          # Large MLP
        "MoE": 768            # Mixture of Experts
    }
    
    hidden_dim = hidden_dims.get(model_type, 768)
    bytes_per_element = 4  # float32
    
    # Different tensor types have different size multipliers
    if tensor_type in ["Q", "K", "V"]:
        # Attention matrices: [batch, seq_length, hidden_dim]
        elements = batch_size * seq_length * hidden_dim
    elif tensor_type == "output":
        # Output is typically larger (includes FFN expansion)
        elements = batch_size * seq_length * (hidden_dim * 4)  # 4x expansion in FFN
    elif tensor_type == "weights":
        # Weight matrices: [hidden_dim, hidden_dim]
        elements = hidden_dim * hidden_dim
    elif tensor_type == "activations":
        # Activations: [batch, seq_length, hidden_dim]
        elements = batch_size * seq_length * hidden_dim
    elif tensor_type == "expert_weights":
        # Expert weights (MoE): larger due to multiple experts
        elements = hidden_dim * hidden_dim * 8  # 8 experts typical
    else:
        # Default
        elements = batch_size * seq_length * hidden_dim
    
    size_bytes = elements * bytes_per_element
    size_mb = size_bytes / (1024 * 1024)
    
    # Round to 1 decimal place for readability
    return round(size_mb, 1)


def generate_synthetic_trace(
    model_type: str = "Transformer",
    num_layers: int = 96,                     
    batch_size: int = 32,
    seq_length: int = 128,
    reuse_probability: float = 0.3,
    max_tensors_per_layer: int = 16          
) -> pd.DataFrame:

    trace = []

    for layer in range(num_layers):

        tensor_types = {
            "Transformer": ["Q", "K", "V", "output"],
            "CNN": ["weights", "activations", "output"],
            "MLP": ["weights", "activations"],
            "MoE": ["expert_weights", "activations"]
        }.get(model_type, ["tensor"])

        # Select up to max_tensors_per_layer types (can overflow actual types → cycles)
        selected_tensor_types = (
            tensor_types * (max_tensors_per_layer // len(tensor_types) + 1)
        )[:max_tensors_per_layer]

        for t_type in selected_tensor_types:

            tensor_id = f"{t_type}_{layer}"
            
            # Calculate realistic tensor size based on batch_size and seq_length
            tensor_size = calculate_tensor_size(
                tensor_type=t_type,
                model_type=model_type,
                batch_size=batch_size,
                seq_length=seq_length
            )
            
            stride = 1 if model_type != "CNN" else np.random.choice([1, 2, 4])
            op = np.random.choice(["matmul", "attention", "conv", "relu", "gating"])
            reuse_count = np.random.binomial(n=5, p=reuse_probability)

            # Add multiple accesses per tensor
            for access_idx in range(reuse_count + 1):
                trace.append({
                    "tensor_id": tensor_id,
                    "size_mb": tensor_size,
                    "reuse_distance": access_idx,
                    "stride": stride,
                    "op": op,
                    "model_type": model_type
                })

    return pd.DataFrame(trace)


if __name__ == "__main__":
    df = generate_synthetic_trace(model_type="Transformer")
    print(df.head())

