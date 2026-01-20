#!/usr/bin/env python3
"""
Verify ModernBERT outputs against HuggingFace transformers.
Run this after running 13_modernbert.nos to compare outputs.

Install: pip install transformers torch
Usage: python verify_modernbert.py
"""

import sys
import math
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    print("Please install: pip install transformers torch")
    sys.exit(1)

def compute_rope_frequencies(dim, seq_len, theta=10000.0):
    """Compute RoPE sin/cos frequencies"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    # Return cos and sin of shape [seq_len, dim]
    cos = torch.cos(freqs).repeat(1, 2)
    sin = torch.sin(freqs).repeat(1, 2)
    return cos, sin

def main():
    print("=== HuggingFace ModernBERT Reference Output ===")
    print("Model: answerdotai/ModernBERT-base\n")

    # Load model and tokenizer
    print("Loading model...")
    model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model.eval()
    print("Model loaded!\n")

    # Config
    print("Config:")
    print(f"  hidden_size: {model.config.hidden_size}")
    print(f"  num_heads: {model.config.num_attention_heads}")
    print(f"  num_layers: {model.config.num_hidden_layers}")
    print(f"  intermediate_size: {model.config.intermediate_size}")
    print(f"  head_dim: {model.config.hidden_size // model.config.num_attention_heads}")
    print()

    # Test input
    test_text = "hello world"
    print(f'Input text: "{test_text}"')

    # Tokenize
    inputs = tokenizer(test_text, return_tensors="pt")
    print(f"Token IDs: {inputs['input_ids'].tolist()}")
    print(f"Token tensor shape: {list(inputs['input_ids'].shape)}")

    # Forward pass with hooks to capture intermediates
    print("\n--- Forward Pass ---\n")

    intermediates = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                intermediates[name] = output[0].detach()
            else:
                intermediates[name] = output.detach()
        return hook

    # Register hooks
    model.embeddings.register_forward_hook(make_hook('embeddings'))
    model.embeddings.norm.register_forward_hook(make_hook('emb_norm'))
    for i in range(min(3, len(model.layers))):  # First 3 layers
        model.layers[i].register_forward_hook(make_hook(f'layer_{i}'))
    model.final_norm.register_forward_hook(make_hook('final_norm'))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Print intermediates
    print("1. Token Embeddings (before norm)")
    # Get raw embeddings
    with torch.no_grad():
        raw_emb = model.embeddings.tok_embeddings(inputs['input_ids'])
    print(f"   Shape: {list(raw_emb.shape)}")
    print(f"   First 5 values of CLS: {raw_emb[0, 0, :5].tolist()}")

    print("\n2. Embeddings (after norm)")
    emb_normed = intermediates.get('emb_norm', intermediates.get('embeddings'))
    print(f"   Shape: {list(emb_normed.shape)}")
    print(f"   First 5 values of CLS: {emb_normed[0, 0, :5].tolist()}")

    print("\n3. After Layer 0")
    layer0_out = intermediates.get('layer_0')
    if layer0_out is not None:
        print(f"   Shape: {list(layer0_out.shape)}")
        print(f"   First 5 values of CLS: {layer0_out[0, 0, :5].tolist()}")

    print("\n4. After Layer 1")
    layer1_out = intermediates.get('layer_1')
    if layer1_out is not None:
        print(f"   Shape: {list(layer1_out.shape)}")
        print(f"   First 5 values of CLS: {layer1_out[0, 0, :5].tolist()}")

    print("\n5. After Layer 2")
    layer2_out = intermediates.get('layer_2')
    if layer2_out is not None:
        print(f"   Shape: {list(layer2_out.shape)}")
        print(f"   First 5 values of CLS: {layer2_out[0, 0, :5].tolist()}")

    print("\n6. Final Output (after final_norm)")
    print(f"   Shape: {list(outputs.last_hidden_state.shape)}")
    print(f"   First 10 values of CLS:")
    print(f"   {outputs.last_hidden_state[0, 0, :10].tolist()}")
    print(f"   Last 5 values of CLS:")
    print(f"   {outputs.last_hidden_state[0, 0, -5:].tolist()}")

    # Print RoPE frequencies for verification
    print("\n7. RoPE Frequencies (for verification)")
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    seq_len = inputs['input_ids'].shape[1]

    # Global attention uses theta=160000
    cos_g, sin_g = compute_rope_frequencies(head_dim, seq_len, theta=160000.0)
    print(f"   Global (theta=160000), head_dim={head_dim}, seq_len={seq_len}")
    print(f"   cos[0, :5]: {cos_g[0, :5].tolist()}")
    print(f"   sin[0, :5]: {sin_g[0, :5].tolist()}")

    # Local attention uses theta=10000
    cos_l, sin_l = compute_rope_frequencies(head_dim, seq_len, theta=10000.0)
    print(f"   Local (theta=10000)")
    print(f"   cos[0, :5]: {cos_l[0, :5].tolist()}")
    print(f"   sin[0, :5]: {sin_l[0, :5].tolist()}")

    print("\n=== Reference Complete ===")
    print("\nCompare the values above with the Nostos output.")
    print("Small numerical differences (< 1e-3) are expected due to floating point precision.")

if __name__ == "__main__":
    main()
