# ModernBERT Implementation Verification

This document describes the verification process for the Nostos Candle ModernBERT implementation against the HuggingFace reference implementation.

## Test Configuration

### Model
- **Model**: `answerdotai/ModernBERT-base`
- **Hidden size**: 768
- **Num attention heads**: 12
- **Num encoder layers**: 22
- **Intermediate size**: 1152
- **Head dim**: 64
- **Vocab size**: 50,368
- **Max position embeddings**: 8,192

### Architecture Features
- **Rotary Position Embeddings (RoPE)**: Replaces absolute position embeddings
  - Global attention: theta=160,000
  - Local attention: theta=10,000
- **GeGLU Activation**: GELU(gate) * up in MLP
- **Pre-Norm**: LayerNorm before attention and MLP
- **No Bias**: All linear layers have no bias
- **Fused QKV**: Single matrix for Q, K, V projections
- **Alternating Attention**: Global every 3rd layer (0, 3, 6, ...)

### Test Input
- **Text**: "hello world"
- **Token IDs**: [50281, 25521, 1533, 50282]
- **Sequence Length**: 4

## Verification Results (First 3 Layers)

### 1. Token Embeddings (before norm)

| Value Index | Nostos Output | HuggingFace Output | Difference |
|-------------|---------------|-------------------|------------|
| CLS[0] | -0.2283676713705063 | -0.2283676713705063 | **0.0 (exact)** |
| CLS[1] | -0.004056947771459818 | -0.004056947771459818 | **0.0 (exact)** |
| CLS[2] | -0.00564997224137187 | -0.00564997224137187 | **0.0 (exact)** |
| CLS[3] | -0.035328127443790436 | -0.035328127443790436 | **0.0 (exact)** |
| CLS[4] | -0.12835195660591125 | -0.12835195660591125 | **0.0 (exact)** |

**Shape**: [1, 4, 768] ✓

### 2. After Embedding Norm

| Value Index | Nostos Output | HuggingFace Output | Difference |
|-------------|---------------|-------------------|------------|
| CLS[0] | -0.8092252016067505 | -0.8092257976531982 | ~6e-7 |
| CLS[1] | 0.06222573667764664 | 0.06222587451338768 | ~1e-7 |
| CLS[2] | 0.040681496262550354 | 0.040681593120098114 | ~1e-7 |
| CLS[3] | -0.10112308710813522 | -0.10112307220697403 | ~1e-8 |
| CLS[4] | -0.5578980445861816 | -0.5578984022140503 | ~4e-7 |

### 3. After Layer 0 (Global Attention)

| Value Index | Nostos Output | HuggingFace Output | Difference |
|-------------|---------------|-------------------|------------|
| CLS[0] | -0.8291528224945068 | -0.8291516900062561 | ~1e-6 |
| CLS[1] | 0.30172163248062134 | 0.30172160267829895 | ~3e-8 |
| CLS[2] | 0.40570491552352905 | 0.4057053029537201 | ~4e-7 |
| CLS[3] | -0.7426408529281616 | -0.7426411509513855 | ~3e-7 |
| CLS[4] | -1.3551666736602783 | -1.3551666736602783 | **0.0 (exact)** |

### 4. After Layer 1 (Local Attention)

| Value Index | Nostos Output | HuggingFace Output | Difference |
|-------------|---------------|-------------------|------------|
| CLS[0] | -0.46129775047302246 | -0.4612967371940613 | ~1e-6 |
| CLS[1] | 0.20409676432609558 | 0.20409640669822693 | ~4e-7 |
| CLS[2] | 0.8494459390640259 | 0.84944748878479 | ~2e-6 |
| CLS[3] | -0.827497124671936 | -0.8274971842765808 | ~6e-8 |
| CLS[4] | -1.1842602491378784 | -1.1842595338821411 | ~7e-7 |

### 5. After Layer 2 (Local Attention)

| Value Index | Nostos Output | HuggingFace Output | Difference |
|-------------|---------------|-------------------|------------|
| CLS[0] | -0.2264343500137329 | -0.22643232345581055 | ~2e-6 |
| CLS[1] | 0.7318850755691528 | 0.7318844795227051 | ~6e-7 |
| CLS[2] | 1.720723271369934 | 1.7207247018814087 | ~1e-6 |
| CLS[3] | -1.2689045667648315 | -1.268904209136963 | ~4e-7 |
| CLS[4] | -1.3040611743927002 | -1.304059386253357 | ~2e-6 |

## Analysis

### Numerical Precision
- **Maximum difference observed**: ~2e-6
- **Average difference**: < 1e-6
- **Exact matches**: Multiple values match exactly

### Conclusion
**PASS** - The Nostos ModernBERT implementation produces outputs that match the HuggingFace reference within floating-point tolerance (< 1e-5).

## How to Reproduce

### Run Nostos ModernBERT
```bash
cd /home/user/nostos
cargo run --release --bin nostos -- --use candle /home/user/nostos-candle/examples/13_modernbert.nos
```

### Run HuggingFace Reference
```bash
cd /home/user/nostos-candle/examples
python3 verify_modernbert.py
```

## Files Used

| File | Description |
|------|-------------|
| `examples/13_modernbert.nos` | Nostos ModernBERT implementation |
| `examples/verify_modernbert.py` | Python/HuggingFace reference implementation |
| `models/modernbert-base.safetensors` | Pretrained model weights (571 MB) |
| `models/modernbert-tokenizer.json` | ModernBERT tokenizer configuration |
| `src/lib.rs` | Candle extension with RoPE, GeGLU, etc. |

## Implementation Notes

### Current Limitations
1. **Partial implementation**: Only first 3 layers verified (register limit workaround)
2. **No local attention window**: Local attention treated as global (not needed for short sequences)

### Key Features Implemented
- RoPE (Rotary Position Embeddings) with configurable theta
- GeGLU activation (GELU(gate) * up)
- Pre-norm architecture
- Fused QKV projection
- No-bias linear layers
- Per-layer weight loading

## Test Date
2026-01-20
