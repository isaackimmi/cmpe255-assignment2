# 🔢 Linear Algebra for Deep Learning

> **Core idea:** Shapes tell you which computations are valid; geometry explains what those computations do.

## 🎯 Purpose & learning objectives

Represent inputs, weights, batches, and embeddings as arrays. Explain vector addition/scaling, dot products, cosine similarity, elementwise products, matrix multiplication, transposes, inverses, determinants, eigenvectors, and norms. Trace a complete batched neural-network forward pass and recognize simplified attention scores.

## 🗺️ From a neuron to a network

| Block | Mathematical idea | Deep learning use |
| --- | --- | --- |
| Scalars/vectors/matrices/tensors | Rank, shape, axis meaning | Losses, embeddings, weights, image batches |
| Addition/scaling | Combine and resize vectors | Residual connections and gradient steps |
| Dot product/cosine | Weighted sum and directional alignment | Neurons and embedding comparison |
| Hadamard product | Coordinatewise multiplication | Gates and masks |
| Matrix products | Compose many dot products | Process a batch in one operation |
| Transpose | Exchange rows and columns | Attention scores and gradient shape alignment |
| Inverse/determinant | Undo a map; measure signed volume scaling | Solve systems and study invertibility |
| Eigenvalues/eigenvectors | Invariant directions and scaling | PCA and transformation analysis |
| Norms | Measure magnitude | Distances, regularization, constraints |

## 🔍 Important code, explained

### 1. A neuron is a weighted sum plus bias

```python
output = np.dot(inputs, weights) + bias
```

With inputs `[1.5, 2.0, 0.5]`, weights `[0.4, 0.6, -0.2]`, and bias `0.1`, the result is **1.80**. The signs indicate whether each feature increases or decreases the weighted sum. Dot-product magnitude depends on both angle and vector lengths; cosine similarity divides by the two norms to isolate direction.

The word-similarity example uses hand-created toy embeddings: queen has cosine similarity **0.9893** to king, versus **0.2896** for apple. This demonstrates the computation, not empirical evidence about a trained language model.

### 2. Keep elementwise and matrix multiplication distinct

```python
Y = X @ W + b
```

`*` multiplies matching coordinates; `@` sums products along the shared inner dimension. In the batch example, `X` is `(4, 3)`, `W` is `(3, 2)`, and `b` is `(2,)`, giving `Y` of shape `(4, 2)`. Bias broadcasts across rows. All four samples share weights while retaining separate outputs.

This notebook switches between single-column sample conventions (`W @ x`) and row-batched conventions (`X @ W`). Both are valid when the weight orientation matches the chosen convention. Follow shapes rather than memorizing one symbolic ordering.

### 3. Transpose creates all query–key comparisons

```python
scores = Q @ K.T
```

With four token vectors of dimension three, both Q and K have shape `(4, 3)`. Transposing K to `(3, 4)` produces scores `(4, 4)`: every query compared with every key. The heatmap contains raw dot-product scores, including negatives. It is not yet a probability distribution or a complete attention layer; scaled attention additionally divides by `sqrt(d_k)`, applies softmax (and often masking), then combines value vectors.

### 4. Inverse, determinant, and eigenvectors answer different questions

```python
A_inv = np.linalg.inv(A)
identity = A @ A_inv
eigenvalues, eigenvectors = np.linalg.eig(A)
```

For `[[4, 7], [2, 6]]`, the inverse is `[[0.6, -0.7], [-0.2, 0.4]]`, and multiplication gives the identity up to floating-point precision. The determinant plots show area scaling **4**, shear with area **1**, and a singular map with area **0**. Zero determinant means a square matrix loses rank and has no inverse.

For `A = [[3, 1], [1, 3]]`, eigenvalues are **4** and **2**. Both comparisons `A @ v == λ * v` pass via `np.allclose`. Eigenvectors are stored in **columns**. PCA uses eigenvectors of a centered covariance matrix, sorted by eigenvalue to identify directions of greatest variance.

**Precision:** A negative eigenvalue reverses an eigenvector's orientation while preserving its span. Spectral normalization controls the largest singular value, not generally an eigenvalue; for arbitrary matrices these quantities differ. The notebook's broad statement connecting spectral normalization to eigenvalues is an intuition, not the exact definition.

### 5. Norms quantify size and penalties

```python
l1 = np.linalg.norm(v, ord=1)
l2 = np.linalg.norm(v, ord=2)
linf = np.linalg.norm(v, ord=np.inf)
l2_reg = np.sum(W ** 2)
```

For `[3, -4, 0, 5]`, L1 is **12**, L2 is **sqrt(50) ≈ 7.07**, and L∞ is **5**. The unit-ball plots show the L1 diamond, L2 circle, and L∞ square. The boundary has norm one; the full unit ball includes points with norm at most one.

For the displayed weight matrix, L1 penalty is **4.500**, squared Frobenius penalty is **2.850**, and Frobenius norm is **1.688**. Matrix Frobenius norm is not generally the matrix spectral/operator 2-norm. L1 geometry can encourage exact zeros; L2 typically shrinks weights smoothly.

### 6. Complete forward pass

```python
Z1 = X @ W1 + b1
A1 = np.maximum(0, Z1)
Z2 = A1 @ W2 + b2
exp_Z2 = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
A2 = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)
```

The final architecture is **4 → 8 → 3** with five input samples: `(5,4) → (5,8) → (5,3)`. Each softmax row sums to one; `keepdims=True` preserves row-wise broadcast compatibility. The fresh predicted classes are **[1, 2, 1, 0, 2]**. This is a randomly initialized forward pass; there is no training loop or accuracy claim in this example.

## ✅ Run notes & practice

All **35/35 code cells** executed successfully in original order with actual outputs and plots saved. No code changes, external datasets, interactive inputs, or GPU were needed. Batched NumPy execution here runs locally on CPU; comments describing GPU parallelism explain a broader motivation, not the execution hardware.

**Try next:** Predict the shape of every intermediate array; verify the final probability row sums; replace `@` with `*` and explain why the operation changes; compute PCA with centered data; distinguish a norm from its square.
