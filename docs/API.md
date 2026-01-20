# Candle ML Extension API Reference

This document describes all functions exposed by the Candle ML extension for Nostos.

## Installation

```bash
# Build the extension
cargo build --release

# Install to Nostos extensions directory
mkdir -p ~/.nostos/extensions/candle/target/release
cp target/release/libnostos_candle.so ~/.nostos/extensions/candle/target/release/
cp candle.nos ~/.nostos/extensions/candle/
```

## Usage

```nostos
use candle.*

main() = {
    t = zeros([2, 3])
    println(toList(t))
}
```

Run with:
```bash
nostos --use candle your_script.nos
```

---

## Tensor Creation

### `zeros(shape)`
Create a tensor filled with zeros.

**Parameters:**
- `shape`: List of integers specifying dimensions

**Returns:** Tensor

**Example:**
```nostos
t = zeros([2, 3])  # 2x3 tensor of zeros
```

---

### `ones(shape)`
Create a tensor filled with ones.

**Parameters:**
- `shape`: List of integers specifying dimensions

**Returns:** Tensor

**Example:**
```nostos
t = ones([3, 4])  # 3x4 tensor of ones
```

---

### `randn(shape)`
Create a tensor with random values from standard normal distribution (mean=0, std=1).

**Parameters:**
- `shape`: List of integers specifying dimensions

**Returns:** Tensor

**Example:**
```nostos
t = randn([2, 2])  # 2x2 tensor with random values
```

---

### `fromList(data)`
Create a tensor from a nested list of numbers.

**Parameters:**
- `data`: Nested list of floats/integers

**Returns:** Tensor

**Example:**
```nostos
t = fromList([[1.0, 2.0], [3.0, 4.0]])  # 2x2 tensor
v = fromList([1.0, 2.0, 3.0])           # 1D tensor
```

---

## Binary Operations

### `tensorAdd(a, b)`
Element-wise addition of two tensors.

**Parameters:**
- `a`: Tensor
- `b`: Tensor (must be broadcastable to `a`)

**Returns:** Tensor

**Example:**
```nostos
c = tensorAdd(a, b)  # c = a + b
```

---

### `tensorSub(a, b)`
Element-wise subtraction of two tensors.

**Parameters:**
- `a`: Tensor
- `b`: Tensor

**Returns:** Tensor

**Example:**
```nostos
c = tensorSub(a, b)  # c = a - b
```

---

### `tensorMul(a, b)`
Element-wise multiplication of two tensors.

**Parameters:**
- `a`: Tensor
- `b`: Tensor

**Returns:** Tensor

**Example:**
```nostos
c = tensorMul(a, b)  # c = a * b (element-wise)
```

---

### `tensorDiv(a, b)`
Element-wise division of two tensors.

**Parameters:**
- `a`: Tensor
- `b`: Tensor

**Returns:** Tensor

**Example:**
```nostos
c = tensorDiv(a, b)  # c = a / b
```

---

### `matmul(a, b)`
Matrix multiplication of two tensors.

**Parameters:**
- `a`: Tensor of shape [..., M, K]
- `b`: Tensor of shape [..., K, N]

**Returns:** Tensor of shape [..., M, N]

**Example:**
```nostos
# 2x3 @ 3x2 = 2x2
a = fromList([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = fromList([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = matmul(a, b)
```

---

## Unary Operations

### `tensorExp(t)`
Element-wise exponential (e^x).

**Parameters:**
- `t`: Tensor

**Returns:** Tensor

**Example:**
```nostos
result = tensorExp(fromList([0.0, 1.0, 2.0]))  # [1, e, e^2]
```

---

### `tensorLog(t)`
Element-wise natural logarithm.

**Parameters:**
- `t`: Tensor (values must be positive)

**Returns:** Tensor

**Example:**
```nostos
result = tensorLog(fromList([1.0, 2.718, 10.0]))
```

---

### `relu(t)`
ReLU activation: max(0, x).

**Parameters:**
- `t`: Tensor

**Returns:** Tensor

**Example:**
```nostos
result = relu(fromList([-1.0, 0.0, 1.0]))  # [0, 0, 1]
```

---

### `gelu(t)`
Gaussian Error Linear Unit activation.

**Parameters:**
- `t`: Tensor

**Returns:** Tensor

**Example:**
```nostos
result = gelu(fromList([-1.0, 0.0, 1.0]))
```

---

### `softmax(t, dim)`
Softmax function along a dimension. Converts logits to probabilities.

**Parameters:**
- `t`: Tensor
- `dim`: Integer, dimension along which to compute softmax

**Returns:** Tensor (values sum to 1 along the specified dimension)

**Example:**
```nostos
logits = fromList([[1.0, 2.0, 3.0]])
probs = softmax(logits, 1)  # Probabilities along dim 1
```

---

## Shape Operations

### `reshape(t, newShape)`
Reshape tensor to a new shape. Total elements must remain the same.

**Parameters:**
- `t`: Tensor
- `newShape`: List of integers

**Returns:** Tensor

**Example:**
```nostos
t = fromList([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
reshaped = reshape(t, [3, 2])  # Now 3x2
flat = reshape(t, [6])         # Flatten to 1D
```

---

### `swapDims(t, dim1, dim2)`
Transpose (swap) two dimensions.

**Parameters:**
- `t`: Tensor
- `dim1`: Integer, first dimension
- `dim2`: Integer, second dimension

**Returns:** Tensor

**Example:**
```nostos
t = fromList([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2
transposed = swapDims(t, 0, 1)  # Now 2x3
```

---

### `squeeze(t, dim)`
Remove a dimension of size 1.

**Parameters:**
- `t`: Tensor
- `dim`: Integer, dimension to remove (must have size 1)

**Returns:** Tensor

**Example:**
```nostos
t = fromList([[1.0, 2.0, 3.0]])  # Shape [1, 3]
squeezed = squeeze(t, 0)         # Shape [3]
```

---

### `unsqueeze(t, dim)`
Add a dimension of size 1.

**Parameters:**
- `t`: Tensor
- `dim`: Integer, position to insert the new dimension

**Returns:** Tensor

**Example:**
```nostos
t = fromList([1.0, 2.0, 3.0])  # Shape [3]
row = unsqueeze(t, 0)          # Shape [1, 3]
col = unsqueeze(t, 1)          # Shape [3, 1]
```

---

## Reductions

### `tensorSum(t)`
Sum all elements in the tensor.

**Parameters:**
- `t`: Tensor

**Returns:** Scalar tensor

**Example:**
```nostos
t = fromList([[1.0, 2.0], [3.0, 4.0]])
total = tensorSum(t)  # 10.0
```

---

### `tensorSumDim(t, dim)`
Sum along a specific dimension.

**Parameters:**
- `t`: Tensor
- `dim`: Integer, dimension to reduce

**Returns:** Tensor

**Example:**
```nostos
t = fromList([[1.0, 2.0], [3.0, 4.0]])
rowSums = tensorSumDim(t, 1)   # [[3], [7]]
colSums = tensorSumDim(t, 0)   # [[4, 6]]
```

---

### `tensorMean(t)`
Mean of all elements in the tensor.

**Parameters:**
- `t`: Tensor

**Returns:** Scalar tensor

**Example:**
```nostos
t = fromList([[1.0, 2.0], [3.0, 4.0]])
avg = tensorMean(t)  # 2.5
```

---

### `tensorMeanDim(t, dim)`
Mean along a specific dimension.

**Parameters:**
- `t`: Tensor
- `dim`: Integer, dimension to reduce

**Returns:** Tensor

**Example:**
```nostos
t = fromList([[1.0, 2.0], [3.0, 4.0]])
rowMeans = tensorMeanDim(t, 1)  # [[1.5], [3.5]]
```

---

### `argmax(t, dim)`
Index of maximum value along a dimension.

**Parameters:**
- `t`: Tensor
- `dim`: Integer, dimension to search

**Returns:** Tensor of indices

**Example:**
```nostos
scores = fromList([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
predictions = argmax(scores, 1)  # [[1], [0]]
```

---

## Tensor Info & Conversion

### `tensorShape(t)`
Get the shape of a tensor as a list.

**Parameters:**
- `t`: Tensor

**Returns:** List of integers

**Example:**
```nostos
t = fromList([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
shape = tensorShape(t)  # [2, 3]
```

---

### `toList(t)`
Convert a tensor to a nested list.

**Parameters:**
- `t`: Tensor

**Returns:** Nested list of floats

**Example:**
```nostos
t = zeros([2, 2])
data = toList(t)  # [[0, 0], [0, 0]]
```

---

### `tensorClone(t)`
Create a copy of a tensor.

**Parameters:**
- `t`: Tensor

**Returns:** Tensor (new copy)

**Example:**
```nostos
original = fromList([1.0, 2.0, 3.0])
copy = tensorClone(original)
```

---

## Direct Native Calls

If you prefer not to use the wrapper module, you can call native functions directly:

```nostos
# Without wrapper
t = __native__("Candle.zeros", [2, 3])
result = __native__("Candle.add", t1, t2)
shape = __native__("Candle.shape", t)
```

Available native functions:
- `Candle.zeros`, `Candle.ones`, `Candle.randn`, `Candle.fromList`
- `Candle.add`, `Candle.sub`, `Candle.mul`, `Candle.div`, `Candle.matmul`
- `Candle.exp`, `Candle.log`, `Candle.relu`, `Candle.gelu`, `Candle.softmax`
- `Candle.reshape`, `Candle.transpose`, `Candle.squeeze`, `Candle.unsqueeze`
- `Candle.sum`, `Candle.mean`, `Candle.argmax`
- `Candle.shape`, `Candle.toList`, `Candle.clone`
