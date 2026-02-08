# MNIST Handwritten Digit Recognition with LSTM

Train an LSTM to classify handwritten digits (0-9) from the MNIST dataset,
in both **Nostos** (using the Candle extension) and **Python** (using PyTorch).
Both implementations use the same architecture and hyperparameters so you can
compare the results side by side.

## Architecture

Each 28x28 grayscale image is treated as a **sequence of 28 rows**, where each
row is a 28-dimensional input vector. The LSTM reads the image row by row (top
to bottom) and the final hidden state is projected to 10 classes:

```
Image [28, 28]
  -> LSTM(input=28, hidden=128)   # 28 time steps
  -> last hidden state [128]
  -> linear(128, 10)              # 10 digit classes
  -> cross-entropy loss
```

Optimizer: Adam (lr=0.001), batch size 128, 3 epochs.

Expected result: **~97% test accuracy** in both languages.

## Tip: Implicit Type Conversions

When writing candle code, you can pass `List[Float]` directly where a `Tensor` is expected. The compiler auto-inserts the conversion:

```nostos
# Instead of:
loss = mseLoss(w, tensorFromList([5.0]))

# Just write:
loss = mseLoss(w, [5.0])
```

This MNIST tutorial loads data from safetensors files (already tensors), so it doesn't use this feature much, but it's very handy for quick experiments and simple training scripts.

## Prerequisites

- **Nostos** with the `candle` extension installed at `~/.nostos/extensions/candle/`
- **Python 3** with a virtual environment

## Setup

All commands are run from the **nostos-candle project root**.

### 1. Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision safetensors
```

### 2. Download and prepare the MNIST data

```bash
python tutorial/prepare_mnist.py
```

This downloads the MNIST dataset and saves it as a single safetensors file
at `tutorial/data/mnist.safetensors` (~210 MB). Both the Python and Nostos
training scripts load from this file.

Output:
```
Downloading MNIST...
Train: torch.Size([60000, 28, 28]) images, torch.Size([60000]) labels
Test:  torch.Size([10000, 28, 28]) images, torch.Size([10000]) labels
Saved to tutorial/data/mnist.safetensors
```

## Training

### Python (PyTorch)

```bash
python tutorial/train_mnist.py
```

Example output:
```
Loading MNIST from safetensors...
  Train: torch.Size([60000, 28, 28]), Test: torch.Size([10000, 28, 28])

Training LSTM on MNIST:
  Model: LSTM(28, 128) -> linear(128, 10)
  Optimizer: Adam(lr=0.001)
  Batch size: 128, Epochs: 3

  Epoch 1/3 - avg loss: 0.6281
  Epoch 2/3 - avg loss: 0.1854
  Epoch 3/3 - avg loss: 0.1268

Evaluating on test set...
  Test accuracy: 9704/10000 = 97.0%
```

### Nostos (Candle)

```bash
nostos --use candle tutorial/train_mnist.nos
```

Example output:
```
Loading MNIST from safetensors...
  Train: [60000, 28, 28]
  Test:  [10000, 28, 28]

Training LSTM on MNIST:
  Model: LSTM(28, 128) -> linear(128, 10)
  Optimizer: Adam(lr=0.001)
  Batch size: 128, Epochs: 3

  Epoch 1/3 - avg loss: 0.5161874407787186
  Epoch 2/3 - avg loss: 0.14488186488002538
  Epoch 3/3 - avg loss: 0.08973148639723659

Evaluating on test set...
  Test accuracy: 9757/10000 = 97.57%
```

## Verifying the results

Both implementations should:

1. **Converge similarly** — loss drops from ~0.5-0.6 in epoch 1 to ~0.09-0.13 in epoch 3
2. **Achieve ~97% test accuracy** — slight variation is expected due to different random seeds and the fact that Python shuffles training data while Nostos processes it sequentially

| Metric | Nostos (Candle) | Python (PyTorch) |
|--------|-----------------|------------------|
| Epoch 1 avg loss | ~0.52 | ~0.63 |
| Epoch 2 avg loss | ~0.14 | ~0.19 |
| Epoch 3 avg loss | ~0.09 | ~0.13 |
| Test accuracy | ~97.6% | ~97.0% |

The small differences come from:
- Different random weight initialization
- Python shuffles each epoch; Nostos uses sequential order
- Candle and PyTorch may have minor numerical differences in LSTM implementation

## How the code works

### Data preparation (`prepare_mnist.py`)

Downloads MNIST via torchvision, normalizes pixel values to [0, 1], and saves
all four tensors (train/test images and labels) into a single safetensors file.

### Forward pass

Both implementations do the same thing:

1. Feed the image `[batch, 28, 28]` into an LSTM — each of the 28 rows is one
   time step with 28 input features
2. Take the hidden state at the **last time step** — `hidden.lastOf(1)` in
   Nostos, `hidden[:, -1, :]` in Python — shape `[batch, 128]`
3. Apply a linear projection `[128] -> [10]` to get logits for each digit class

### Training loop

1. Slice a batch of images and labels from the dataset using `narrow`
2. Run the forward pass to get logits
3. Compute cross-entropy loss between logits and target labels
4. Call `trainStep(optimizer, loss)` which runs backpropagation and updates weights
5. Repeat for all batches, for 3 epochs

### How `trainStep` knows which weights to update

A natural question: `trainStep(opt, loss)` only receives the optimizer and a loss
tensor — how does it know about the LSTM and linear layer used in the forward pass?

The answer is that **the loss tensor carries the entire computation graph**.

There are three pieces that connect everything:

**1. The parameter map collects trainable weights.**
`paramMapCreate()` creates a `VarMap` — a container for trainable tensors. When
you call `lstmTrainable(params, ...)` or `paramRandn(params, "wOut", ...)`, their
weights are registered as gradient-tracked `Var` objects inside this map.

**2. The optimizer captures those Vars.**
When you call `adam(params, 0.001)`, the optimizer grabs references to every `Var`
in the parameter map. It knows which weights exist, but not yet how they're used.

**3. The forward pass builds a computation graph automatically.**
Every tensor operation — matmul, tanh, sigmoid, add — records itself in a graph
attached to the resulting tensor. When you compute:

```nostos
hidden = lstmSeq(lstm, images)           # matmul, sigmoid, tanh at each step
logits = linear(last, wOut)              # matmul with wOut
loss = crossEntropyLoss(logits, labels)  # softmax + log
```

the `loss` tensor ends up holding a chain of references all the way back through
every operation to the original `Var` weights. This is called **automatic
differentiation** (autograd).

**4. `trainStep` walks this graph backward.**
`trainStep(opt, loss)` calls `backward_step(&loss)` which:
  1. Traces the computation graph stored in `loss` backward to every `Var`
  2. Computes gradients (dLoss/dWeight) for each `Var` via the chain rule
  3. Updates each `Var` using Adam's update rule (or SGD, depending on optimizer)

This is the same mechanism as PyTorch's `loss.backward()` + `optimizer.step()`.

It also explains why **LSTM backpropagation through time (BPTT) works for free** —
the LSTM's internal operations at each of the 28 time steps (matrix multiplies,
sigmoid gates, tanh activations) are all recorded in the graph. The backward pass
automatically unrolls through all 28 steps to compute gradients, without any
special BPTT code.

### Evaluation

1. Run forward pass on test images in batches of 100
2. Take `argmax` of logits to get predicted digit
3. `countEqual(preds, labels)` counts matching elements directly on tensors (no list conversion needed)

## Files

| File | Description |
|------|-------------|
| `tutorial/mnist.md` | This tutorial |
| `tutorial/prepare_mnist.py` | Downloads MNIST, saves as safetensors |
| `tutorial/train_mnist.py` | PyTorch training script |
| `tutorial/train_mnist.nos` | Nostos training script |
| `tutorial/data/` | Created by prepare_mnist.py (not checked in) |
