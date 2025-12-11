import torch
import pandas as pd
import numpy as np


# Lazy loading of model - only load when needed
_model = None
_tokenizer = None


def _get_model():
    """Lazy load the Mistral model with 4-bit quantization"""
    global _model, _tokenizer
    if _model is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model
            import torch
            
            # 4-bit quantization configuration for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Load Mistral-7B-Instruct-v0.2 with 4-bit quantization
            _model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            _tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
            
            # LoRA configuration optimized for Mistral architecture
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            _model = get_peft_model(_model, lora_config)
            print("Successfully loaded Mistral-7B-Instruct-v0.2 with 4-bit quantization")
        except Exception as e:
            print(f"Warning: Could not load Mistral model: {e}")
            print("Using simplified scoring instead")
    return _model, _tokenizer


def predict_eviction_scores(trace_df: pd.DataFrame):
    """
    Predict eviction scores for tensors in the trace.
    Uses a simplified scoring based on size, reuse distance, and stride.
    """
    inputs = trace_df[["size_mb", "reuse_distance", "stride"]].values.astype(float)
    
    # Normalize inputs
    inputs_normalized = inputs / (inputs.max(axis=0) + 1e-8)
    
    # Simple scoring: higher score = more likely to evict
    # Evict large tensors with high reuse distance (not accessed recently)
    eviction_scores = (
        0.4 * inputs_normalized[:, 0] +  # size_mb
        0.5 * inputs_normalized[:, 1] +  # reuse_distance
        0.1 * inputs_normalized[:, 2]    # stride
    )
    
    trace_df["eviction_score"] = eviction_scores
    return trace_df

