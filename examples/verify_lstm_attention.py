#!/usr/bin/env python3
# Generates shared weights and runs LSTM + attention in PyTorch.
# Compare output with test_lstm_attention.nos (Nostos/candle version).
#
# Usage:
#   python3 verify_lstm_attention.py
#   nostos --use candle test_lstm_attention.nos

import torch
import torch.nn as nn
from safetensors.torch import save_file
import os

torch.manual_seed(42)

input_dim = 4
hidden_dim = 4
seq_len = 3
batch_size = 1
num_heads = 2

# Create fixed input
input_data = torch.randn(batch_size, seq_len, input_dim)

# Create LSTM with PyTorch
lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

# Create attention weight matrices (small values for stability)
torch.manual_seed(123)
wQ = torch.randn(hidden_dim, hidden_dim) * 0.1
wK = torch.randn(hidden_dim, hidden_dim) * 0.1
wV = torch.randn(hidden_dim, hidden_dim) * 0.1
wO = torch.randn(hidden_dim, hidden_dim) * 0.1

# Save everything to safetensors
tensors = {
    "lstm.weight_ih_l0": lstm.weight_ih_l0.data,
    "lstm.weight_hh_l0": lstm.weight_hh_l0.data,
    "lstm.bias_ih_l0": lstm.bias_ih_l0.data,
    "lstm.bias_hh_l0": lstm.bias_hh_l0.data,
    "attn_wQ": wQ,
    "attn_wK": wK,
    "attn_wV": wV,
    "attn_wO": wO,
    "input_data": input_data,  # [batch, seq, input_dim] = [1, 3, 4]
}

weights_path = os.path.join(os.path.dirname(__file__), "lstm_attention_weights.safetensors")
save_file(tensors, weights_path)
print(f"Saved weights to {weights_path}")

# ---- Forward pass ----

# 1. LSTM
lstm_output, _ = lstm(input_data)  # [1, 3, 4]
print(f"\nLSTM output shape: {lstm_output.shape}")
print(f"LSTM output:\n{lstm_output.detach()}")

# 2. Multi-head attention (matching our Rust implementation exactly)
def multi_head_attention(x, num_heads, wq, wk, wv, wo):
    batch, seq, dim = x.shape
    head_dim = dim // num_heads

    # Project Q, K, V: x @ W^T
    q = x @ wq.T  # [batch, seq, dim]
    k = x @ wk.T
    v = x @ wv.T

    # Reshape to [batch, seq, num_heads, head_dim] then transpose to [batch, num_heads, seq, head_dim]
    q = q.reshape(batch, seq, num_heads, head_dim).transpose(1, 2).contiguous()
    k = k.reshape(batch, seq, num_heads, head_dim).transpose(1, 2).contiguous()
    v = v.reshape(batch, seq, num_heads, head_dim).transpose(1, 2).contiguous()

    # Scaled dot-product attention
    scale = head_dim ** 0.5
    scores = q @ k.transpose(-2, -1) / scale  # [batch, heads, seq, seq]
    weights = torch.softmax(scores, dim=-1)
    attended = weights @ v  # [batch, heads, seq, head_dim]

    # Concatenate heads
    concat = attended.transpose(1, 2).contiguous().reshape(batch, seq, dim)

    # Output projection
    output = concat @ wo.T
    return output

result = multi_head_attention(lstm_output.detach(), num_heads, wQ, wK, wV, wO)
print(f"\nFinal output shape: {result.shape}")
print(f"Final output (flat): {result.detach().flatten().tolist()}")

# Print rounded for easier comparison
vals = result.detach().flatten().tolist()
print(f"\nRounded to 4 decimals:")
print([round(v, 4) for v in vals])
