# Intro to Linear Algebra — Zero to Hero for ML

> **The big idea** · Turn vector and matrix notation into concrete geometry and ML computation. A seeded dataset of 200 synthetic flowers and two measurements supports the entire progression from shapes to PCA.

## 🎯 Learning objectives

- Interpret vector shapes, arithmetic, norms, dot products, projections, and cosine similarity.
- Trace matrix multiplication, broadcasting, and geometric transformations.
- Solve systems and diagnose rank deficiency and ill-conditioning.
- Build batched layers and PCA; validate PCA through SVD and scikit-learn.

## 🧭 Conceptual map

The flowers → Vectors and shapes → Arithmetic and length → Dot products → Matrices → Matrix multiplication → Transformations → Systems and failure modes → Eigenvectors → Linear algebra becomes ML

## 📚 Walkthrough of every major section

### 0 · The flowers

Create 200 two-feature synthetic examples from two Gaussian species. Each row is one observation; each column is one measurement. These are simulated flowers, not an imported Iris dataset.

### 1 · Vectors and shapes

Read a vector as an arrow, coordinate list, or feature record. Compare (n,), (n,1), and (1,n); transposing a one-dimensional array does not create a column.

### 2 · Arithmetic and length

Add vectors tip-to-tail, subtract to form displacements, and scale length/direction. Compute Euclidean norms and pairwise distances; consider how units affect distance.

### 3 · Dot products

Derive the component formula and connect it to angles and projection. Normalize by lengths for cosine similarity. Raw dot products depend on magnitude; cosine similarity is unchanged by positive rescaling.

### 4 · Matrices

Use tables, indexing, axes, transpose, and broadcasting correctly. Mean-center the flower columns without changing pairwise distances. Shape guards catch accidental outer-shaped subtraction.

### 5 · Matrix multiplication

Each output entry is a row-column dot product. Interpret columns as transformed basis vectors. Check compatible dimensions, noncommutativity, associativity, and operation costs.

### 6 · Transformations

Construct scaling, shear, rotation, reflection, and projection from basis-vector destinations. Determinants track signed area; rank loss collapses dimensions. Composition order changes the resulting geometry.

### 7 · Systems and failure modes

Connect row/column pictures to elimination, solve, and inverse. Contrast singular and nearly singular matrices, rank and condition number. Use least squares when 200 observations cannot all lie on one fitted line.

### 8 · Eigenvectors

Find directions preserved up to scale, derive a 2x2 characteristic polynomial, and validate Av=lambda*v. Rotations may have no real eigenvectors. Real symmetric matrices admit an orthonormal eigenbasis; covariance eigenvectors reveal spread.

### 9 · Linear algebra becomes ML

Compare per-example loops with X@W.T+b and nonlinear layers. Center, form covariance, sort eigenpairs, project, and reconstruct PCA. Deliberately omit centering, then verify the correct implementation via SVD and sklearn.

## 💻 Important code, connected to its purpose

### Center features with broadcasting

Notebook cell **119** (one-based, including markdown).

```python
col_means = X.mean(axis=0)          # axis=0 -> average DOWN each column
Xc        = X - col_means           # broadcast: (200,2) - (2,) -> (200,2)
```

### Vectorized batch forward pass

Notebook cell **250** (one-based, including markdown).

```python
def forward_batch(X, W, b):
    """All 200 flowers in one matrix multiplication."""
    return X @ W.T + b
```

### Reconstruct from the first principal direction

Notebook cell **267** (one-based, including markdown).

```python
Xc_hat = np.outer(scores, pc1)      # score * direction  (Part 3's projection vector)
X_hat  = Xc_hat + mu                # undo the centring (Part 4)
```

## 🔬 Actual results from this execution

The excerpts below are retained stdout from the fresh local run. Figures are embedded in the executed notebook.

### Evidence · notebook cell 208

```text
A (from earlier)   det =   5.000000   cond =       2.6180
N (near-singular)  det =   0.001000   cond =   25008.0010

we change b from [3. 6.] to [3.    6.001]
relative change in b: 1.491e-04  (0.0149 %)

--- the well-conditioned matrix A ---
x before = [0.6 1.8]
x after  = [0.5998 1.8004]
relative change in x: 2.357e-04   -> amplification 1.58x   (cond = 2.62)

--- the near-singular matrix N ---
x before = [ 3. -0.]
x after  = [1. 1.]
relative change in x: 7.454e-01   -> amplification 5000.0x   (cond = 25008.0)

same nudge to b. With A the answer moved by 2.36e-04; with N it moved by 7.45e-01.
That is 3162 times more sensitive, from a matrix that looks perfectly ordinary.

--- and the same instability, in the inverse itself ---
inv(N) =
 [[ 4001. -2000.]
 [-2000.  1000.]]
largest entry of inv(N): 4,001.0    (largest entry of inv(A): 0.600)
Enormous entries with opposite signs cancel almost exactly when they
multiply b -- and cancellation is precisely where floating point loses
its digits. np.linalg.solve never builds this object, which is why it
is the routine to reach for.
```

### Evidence · notebook cell 271

```text
CHECK 1 -- np.linalg.svd on the centred data (no covariance matrix built)
  eigenvalues from eigh : [1.05249425 0.35434703]
  s^2/(n-1) from svd    : [1.05249425 0.35434703]
  max difference        : 2.220e-16
  |PC1_eigh . PC1_svd|  : 1.0000000000   (1 = same line; sign is arbitrary)

CHECK 2 -- scikit-learn's PCA
  sklearn explained_variance_       : [1.05249425 0.35434703]
  ours (eigenvalues)                : [1.05249425 0.35434703]
  sklearn explained_variance_ratio_ : [0.74812579 0.25187421]
  ours (EVR)                        : [0.74812579 0.25187421]
  |PC1_ours . PC1_sklearn|          : 1.0000000000
  scores agree (up to sign)         : True
  largest score disagreement        : 4.619e-14

sklearn cross-check performed: True

FINAL PCA SUMMARY
  PC1                        = [0.9947 0.1027]
  explained variance ratio   = 74.81% (PC1), 25.19% (PC2)
  reconstruction RMSE from 1D = 0.5938 cm
```

## ⚠️ Interpretation and common mistakes

- Use @ for matrix products and * for elementwise products.
- Check array shapes before trusting broadcasting; .T does nothing useful to a flat vector.
- Use solve instead of explicitly constructing an inverse, and inspect condition numbers.
- PCA directions have arbitrary sign; compare subspaces or absolute alignment.
- Explained variance is not classification accuracy, and this two-feature example retains about 75%, not all, of the variation.
- Synthetic class-centroid demonstrations are illustrative; they are not a held-out classifier evaluation.

## ✅ Run record

- Executed **82/82 code cells** in a fresh local Python kernel with actual outputs retained.
- Inputs are the notebook's seeded synthetic datasets; no manual input or external downloads are required.
- Original exercise prompts remain as optional practice; prose exercises are not reported as completed code cells.
- Execution details, environment versions, adaptations, and output inventory are in `execution.json`; rerun with `python reproduce.py`.
- Compatibility adaptation: use `numpy.linalg.eigh` for the real symmetric example and remove an exactly zero imaginary part from the earlier nonsymmetric eigenvector dot product before computing a real angle. The original run raised a TypeError under the installed NumPy version.

## 🎬 Recording plan

Use the linked video script for a representative 65–75 second walkthrough. The short video highlights the core idea and key evidence; these notes retain the broader notebook coverage.
