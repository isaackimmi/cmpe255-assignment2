# 🧮 NumPy Foundations for Deep Learning

> **Core idea:** A neural network is a sequence of array operations. Shape reasoning makes those operations understandable and debuggable.

## 🎯 Purpose & learning objectives

Build the NumPy vocabulary needed to read a neural network implementation without relying on a framework. By the end, explain tensor shapes and dtypes; select batches and features; distinguish elementwise multiplication from matrix multiplication; use broadcasting and aggregation axes correctly; implement activations, losses, initialization, and a small trainable network.

## 🗺️ Conceptual map

**Arrays → indexing → elementwise functions → broadcasting → matrix multiplication → loss reduction → initialization → vectorization → XOR training.**

| Block | What it teaches | Why it matters |
| --- | --- | --- |
| Arrays, shapes, dtypes | 1D vectors, 2D matrices, image and text batches | Dimensions encode the meaning of data; dtype controls memory and precision |
| Indexing and slicing | Positive/negative indices, exclusive slice stops, Boolean masks, fancy indexing | Select minibatches, filter values, and look up embeddings |
| Activations | Sigmoid, tanh, ReLU, leaky ReLU, softmax | Apply nonlinear transformations using array expressions |
| Broadcasting | Combine a matrix with a bias vector | Add one bias per output feature across every sample |
| Matrix products | Inner dimensions must match | Compute all neurons for an entire batch |
| Aggregation | Mean, variance, reductions along an axis | Compute losses and feature statistics |
| Reshaping | Preserve element count, infer a dimension with `-1` | Flatten images while retaining the batch axis |
| Random initialization | Xavier, He, LeCun scaling | Set an appropriate initial scale for activations |
| Vectorization | Compare Python loops with compiled array operations | Reduce overhead in numerical workloads |
| Mini network | Forward pass, backpropagation, gradient updates | Combine the foundations into a working learner |

## 🔍 Important code, step by step

### 1. Read a shape before operating on it

A batch `(32, 28, 28)` contains 32 grayscale images. Slicing `dataset[start:end]` selects samples while retaining image dimensions. An embedding lookup with five token IDs into a `(1000, 128)` matrix returns `(5, 128)`.

```python
batch_images = dataset[start:end]
sentence_embeddings = embedding_matrix[sentence]
flattened = batch_images.reshape(4, -1)
```

The flattening example changes `(4, 28, 28)` into `(4, 784)`. It does not discard pixels. The shape exercise prints `(3, 4, 5)`, `(2,)`, and `(3, 3)`; the student comment claiming the second is `(1,)` is incorrect—the executed result is authoritative.

### 2. Build a dense layer from three operations

```python
self.z1 = X @ self.W1 + self.b1
self.a1 = self.sigmoid(self.z1)
```

`@` contracts the feature dimension. A `(4, 3)` batch times `(3, 2)` weights gives `(4, 2)` outputs; adding bias `(2,)` broadcasts over the four rows. `*` instead multiplies corresponding entries and cannot substitute for this product.

The hand-computed matrix example produces `[[58, 64], [139, 154]]`. The separate dense-layer example transforms `(3, 4)` inputs into `(3, 3)` activations.

### 3. Reduce the intended dimension

```python
np.sum(matrix, axis=0)
np.sum(matrix, axis=1)
return np.mean((y_true - y_pred) ** 2)
```

For the 1–9 matrix, reducing axis 0 gives column totals `[12, 15, 18]`; reducing axis 1 gives row totals `[6, 15, 24]`. The loss example reports MSE **0.0220** and binary cross-entropy **0.2027**. BCE clips predictions to avoid taking `log(0)`.

### 4. Initialize and train

```python
self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
self.W2 -= learning_rate * (self.a1.T @ delta2) / m
self.b2 -= learning_rate * np.mean(delta2, axis=0)
```

The initialization comparison finds standard deviations near **0.9998** for unscaled normal weights, **0.0439** for Xavier, and **0.0506** for He. The network uses sigmoid activations with He-style scaling for demonstration; this does not establish that He is optimal for sigmoid.

## 📊 Fresh execution results

- All **54/54 code cells** ran successfully in order with actual outputs saved.
- The XOR network has architecture **2 → 4 → 1** and runs **10,001 epochs**.
- MSE falls from **0.268272** to **0.001889**.
- Predictions are **0.0440, 0.9543, 0.9621, 0.0458** for inputs `(0,0), (0,1), (1,0), (1,1)`.
- Rounding gives all four training labels correctly: **100% training accuracy**.
- The loop/vectorization timing is measured on this machine, not a universal speed guarantee.

The loss curve shows optimization progress; the decision surface shows a nonlinear separation of XOR. Four correct training examples demonstrate the intended toy task, not generalization to a real dataset.

## ⚠️ Pitfalls & conceptual precision

A one-dimensional vector `(2,)` is not a `(1, 2)` row matrix. Broadcasting aligns trailing axes; it does not infer semantic meaning. Keep batch and feature axes explicit. Shuffling before splitting avoids an order-based split in the toy example. A raw `X.T @ X` is a Gram matrix; covariance additionally needs centering and scaling. The implementation's gradient uses an MSE-like objective with a constant factor absorbed into the learning rate. The color wording in the final decision-boundary title does not match the `RdYlBu` low/high colors reliably; interpret its colorbar and labeled examples.

## ✅ Run notes & practice

Executed locally from the user's copied notebook; no code adaptations, external downloads, or interactive inputs were needed. Original comments and teaching content remain intact. Some original plot labels contain literal escaped newlines or unsupported symbols; numerical outputs remain valid.

**Try next:** Predict output shapes without executing; change hidden width and learning rate; explain why a linear classifier cannot separate XOR; compare convergence across seeds.
