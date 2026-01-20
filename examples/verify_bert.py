#!/usr/bin/env python3
"""
Verify BERT outputs against HuggingFace transformers.
Run this after running 12_bert_pretrained.nos to compare outputs.

Install: pip install transformers torch
Usage: python verify_bert.py
"""

import sys
try:
    import torch
    from transformers import BertModel, BertTokenizer
except ImportError:
    print("Please install: pip install transformers torch")
    sys.exit(1)

def main():
    print("=== HuggingFace BERT Reference Output ===")
    print("Model: google/bert_uncased_L-2_H-128_A-2\n")

    # Load model and tokenizer
    print("Loading model...")
    model = BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    model.eval()
    print("Model loaded!\n")

    # Test input
    test_text = "hello world"
    print(f'Input text: "{test_text}"')

    # Tokenize
    inputs = tokenizer(test_text, return_tensors="pt")
    print(f"Token IDs: {inputs['input_ids'].tolist()}")
    print(f"Token tensor shape: {list(inputs['input_ids'].shape)}")

    # Forward pass
    print("\n--- Forward Pass ---\n")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Embeddings (before encoder)
    embeddings = model.embeddings(inputs['input_ids'], inputs.get('token_type_ids'))
    print("1. Embeddings")
    print(f"   Shape: {list(embeddings.shape)}")
    print(f"   First 5 values of CLS embedding: {embeddings[0, 0, :5].tolist()}")

    # Hidden states after each layer
    hidden_states = outputs.hidden_states
    print(f"\n2. Encoder Layer 0")
    print(f"   Shape: {list(hidden_states[1].shape)}")

    print(f"\n3. Encoder Layer 1")
    print(f"   Shape: {list(hidden_states[2].shape)}")

    # Pooler output
    print("\n4. Pooler (CLS token)")
    print(f"   Pooled shape: {list(outputs.pooler_output.shape)}")
    print(f"\n   First 10 pooled values:")
    print(f"   {outputs.pooler_output[0, :10].tolist()}")

    print("\n=== Reference Complete ===")
    print("\nCompare the values above with the Nostos output.")
    print("Small numerical differences (< 1e-4) are expected due to floating point precision.")

if __name__ == "__main__":
    main()
