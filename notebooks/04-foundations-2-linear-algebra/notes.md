# 📐 Foundations 2 — Linear Algebra

[Open my Colab](https://colab.research.google.com/drive/1mPXf5Z_xT7y6gIZpueyhoFCUR2N24qB8) · Notebook 04 of 16

> **Core idea:** A matrix is a transformation. Its columns tell us where basis vectors land; rank tells us how much independent information survives.

## 🎯 Purpose & learning objectives

Read matrix-vector multiplication geometrically, connect dot products to neurons and cosine similarity, distinguish shape from rank, interpret determinants and invertibility, compute eigenvectors, and assemble a network layer plus PCA from those pieces.

## 🗺️ Concept map

**Basis vectors → linear combinations → dot products → shape/rank → area and collapse → inverses → eigenvectors → PCA and neural layers.** This is the MODEL box of the four-part Foundations sequence.

## 📚 Walkthrough of all seven parts

| Part | Code and picture | Main lesson |
|---|---|---|
| 1 · A matrix moves space | Basis-vector products; grids under scale, rotation, shear, collapse; composition | Columns are landing spots; multiplication order matters |
| 2 · Dot product | Algebraic/geometric agreement, projection, cosine counterexample, spam neuron | Alignment and magnitude jointly determine the raw dot product |
| 3 · Changing dimension | Transformed cube, rectangular shapes, dependent columns, null direction | Output coordinate count and effective image dimension differ |
| 4 · Determinant | Signed area under stretch/shear/swap; determinant products | Zero means collapse for a square matrix; sign records orientation |
| 5 · Inverse and collapse | Hand inverse, caught singular inverse, redundant regression feature | Inverse recovery requires uniqueness; numerical conditioning matters |
| 6 · Eigenvectors | Characteristic polynomial checked with SymPy and NumPy; rotation and symmetric examples | Some directions are preserved up to scale; symmetric matrices have an orthogonal real eigenbasis |
| 7 · Capstone | A 6→3 affine layer; covariance eigendecomposition checked with sklearn PCA | Matrix products implement layers and expose variation in data |

## 💻 Important code blocks

### Weighted columns — cell 8

```python
A = np.array([[2.0, 0.0], [0.0, 3.0]])
x = np.array([3.0, 1.0])
A @ x
3 * A[:, 0] + 1 * A[:, 1]
```

Both routes produce **[6,3]**. The first column is the image of `[1,0]`; the second is the image of `[0,1]`. The transformed-grid figures make the same calculation visible across an entire plane.

### Cosine similarity — cell 28

```python
def cosine(u, v):
    return (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
```

For query `[1,0]`, candidates `[1,0]` and `[10,1]` have raw dots **1** and **10**, but cosines **1.0000** and **0.9950**. The longer candidate wins the dot-product comparison despite a less exact direction. Cosine requires nonzero vectors and is one possible similarity choice; some applications deliberately retain magnitude.

### PCA from covariance — cell 75

```python
centred = cloud - cloud.mean(axis=0)
cov = np.cov(centred, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
```

Center first, measure covariance, find orthogonal directions, then order by variance. The first component captures **95.6%**, the second **4.4%**. Eigenvalues **5.6520** and **0.2593** agree with sklearn's explained variances. Directions match up to sign, as they should: reversing an axis does not change its line.

## 🔎 Actual outputs & what they mean

- The 2×3 map sends both `[1,2,3]` and `[3,3,2]` to **[7,5]**. Their difference lies in the null space, so the original input cannot be uniquely recovered.
- A dependent 3×3 example has **rank 2**, determinant zero, and maps `[-1,-1,1]` to zero. Three output coordinates do not imply three independent dimensions.
- Stretch, shear and axis swap have determinants **6,1,-1**. Absolute value is the area factor; negative sign indicates orientation reversal.
- The hand inverse of `[[3,1],[2,4]]` is `[[0.4,-0.1],[-0.2,0.3]]`; multiplying by the original gives identity. A genuinely singular matrix raises a caught `LinAlgError`, as intended.
- Duplicating height in centimeters and meters makes a regression design redundant. The displayed condition number of `X.T@X` grows to approximately **9.7e19**, and explicit inversion returns a huge unstable residual. `lstsq` gives a coefficient norm around **0.969**. Exact error magnitudes are platform sensitive.
- `[[2,1],[0,3]]` has eigenvalues **2 and 3**. The 90-degree rotation instead has complex eigenvalues **+i and -i**, illustrating why a real matrix need not preserve a real direction.
- The 6→3 layer gives output approximately **[0.8125,-2.0179,1.2268]**, and explicit row dot products plus biases match the matrix calculation.

## ⚠️ Pitfalls & interpretation limits

Use `@` for matrix multiplication and `*` for elementwise multiplication. Matrix multiplication is generally noncommutative. A tall matrix raises the number of coordinates, not the intrinsic dimension beyond its input rank. Determinant/invertibility equivalences apply to square matrices. Determinants are poorly scaled numerical singularity tests; inspect singular values/rank/conditioning and prefer `solve`, `lstsq` or suitable regularization over forming an inverse. A very small determinant alone is not enough to diagnose numerical instability. This PCA example uses a **synthetic rotated Gaussian cloud**, despite the introductory wording about real data. PCA's high variance need not imply high predictive relevance; feature scaling matters.

## ✅ Run notes

All code cells were rerun without source adaptations, retaining seed 7 and explicit generator seeds. Static transformed grids, cube diagrams, eigenvector plots and PCA figures are embedded in the executed notebook. Expected caught singular-matrix errors are printed instructional results rather than notebook error outputs.

## 🧠 Quick self-check

Explain why a 3×2 matrix cannot create a third independent input direction, why a zero determinant prevents unique recovery, and why PCA directions may differ by a sign across libraries.
