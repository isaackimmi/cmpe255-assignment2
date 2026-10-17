# 📉 Foundations 4 — Calculus, Loss & Update

[Open my Colab](https://colab.research.google.com/drive/1Fpjpzgz0kAecd5LT3RCb_rko76-oo8n3) · Notebook 06 of 16

> **Core idea:** Gradients tell a model how to change; backpropagation computes them efficiently. Integrals explain probabilities and the expected loss that sampled batches estimate.

## 🎯 Purpose & learning objectives

Derive and check derivatives, partials and gradients; understand steepest ascent and step-size stability; implement the chain rule including fan-out; train a NumPy neural network; verify against numerical gradients and PyTorch; connect integration with expected loss; and combine the whole Foundations series in a held-out capstone.

## 🗺️ Concept map

**Limit → local slope → vector gradient → descent → chain rule and accumulation → backpropagation → numerical/framework audit → integration and expectation → held-out learning.**

## 📚 Walkthrough of all seven parts

| Part | Important blocks | Meaning |
|---|---|---|
| 1 · Derivative | Secants converging to tangent; power/sum/product/chain rules; symbolic checks | A local rate of change supplies a direction |
| 2 · Many knobs | Partials of `x²+3y²`, direction sweep, contour tangent | Gradient direction maximizes the local directional derivative |
| 3 · Descent | Three steps by hand, stable/diverging runs, rate sweep | The minus sign chooses downhill; curvature limits the stable step |
| 4 · Chain rule | Composed function; two-path fan-out; sigmoid derivative; one-neuron loss | Multiply along paths and add where paths meet |
| 5 · Backprop | Finite-difference timing; transpose check; MLP class; all-parameter gradient check; torch referee; moons training; depth experiment | Cache forward values and reuse each returning delta |
| 6 · Integrals | Right/left/midpoint sums, fundamental theorem, Gaussian area, expected value, batch loss sampling | A density integrates to probability; empirical loss estimates population expectation |
| 7 · Capstone and scale | Gaussian classes, train/test network, Bayes ceiling, XOR, vectorization timing, power-law plots | Capacity, generalization, compute and scaling have distinct roles |

## 💻 Important code blocks

### Gradient and update — cells 20 and 32

```python
def grad2(p):
    return np.array([2 * p[0], 6 * p[1]])

p = p - eta * grad2(p)
```

For `f=x²+3y²` at `[2,1]`, gradient is **[4,6]**, magnitude **7.2111**. Rate 0.1 gives losses **7 → 3.04 → 1.7152 → 1.060864**. The y-coordinate multiplier is `1-6*eta`, so convergence requires **0 < eta < 1/3** for this bowl. At exactly 1/3 the y-coordinate oscillates without shrinking; above it the magnitude grows.

### Backpropagation — cell 71

```python
delta = (A[-1] - Y) / N
for l in reversed(range(len(self.W))):
    gW[l] = delta @ A[l].T
    gb[l] = delta.sum(axis=1, keepdims=True)
    if l > 0:
        below = self.W[l].T @ delta
        delta = below * (1 - A[l] ** 2)
```

These are the core lines of `Net.backward`: sigmoid plus BCE yields `a-y`; batch averaging occurs once. Each layer forms a weight gradient from its delta and cached input; transposed weights return gradients to the preceding layer. Hidden tanh contributes derivative `1-a²`.

### Integration and expected loss — cells 87 and 104

```python
def per_example_loss(x):
    return (1.5 * x - 2.0 * x) ** 2
```

When `x` is standard normal, expected loss is `0.25*E[x²]=0.25`. Numerical integration and SymPy both give **0.25**. Independent sampled batch means fluctuate around this value, with spread approximately proportional to `1/sqrt(N)`.

## 🔎 Actual outputs & how to interpret them

**Derivative checks:** For `x²` at 3, forward-difference slopes move **7,6.1,6.01,6.001** toward 6. A 3,601-direction search finds fastest climb at **56.30°**, agreeing with the gradient direction **56.31°**. Its contour-tangent dot product is about **8.88e-16**.

**Chain rule:** `(3w)²` has derivative **36** at `w=2`. The fan-out example `3u+u²` gives derivative **7** at `u=2`, by adding contributions 3 and 4. Sigmoid's derivative peaks at **0.25**, and numerical checks agree with its algebraic identity.

**Gradient audit and training:** All **59** parameters of the small check network pass with relative error **4.495e-10**. The larger network's handwritten gradients agree with PyTorch to worst displayed difference **2.776e-17**. The **2→16→16→1** network has **337 parameters** and fits 280 synthetic moon examples: BCE **0.694286 → 0.001653**, with **100% training accuracy**. No held-out moon score is claimed.

**Depth:** After 25 layers, median gradient norm is approximately **1.75e-16** for sigmoid/gain 1, **0.566** for ReLU/He, and **1.90e7** for ReLU/gain 8. These measurements show both activation derivatives and weight scaling matter. They are random-network statistics, not universal constants.

**Integration:** Four right-edge rectangles give **12.65625** for the integral of x² from 0 to 3; the exact answer is **9**. Midpoints converge faster for this smooth example. Standard-normal probabilities inside one/two/three standard deviations are **0.682689/0.954500/0.997300**. Expected die outcome is **3.5**; Uniform[0,2] expectation is **1**.

**Capstone:** The Gaussian-class model learns on 600 points and is evaluated separately. The theoretical Bayes accuracy ceiling is **90.41%**. This trained network scores **93.17%** on training data, **91.00%** on the 600-point test sample, and **90.22%** on 20,000 new examples. A finite sample can exceed the distribution-level ceiling through sampling variation. The displayed error bars are roughly **one standard error**, not 95% confidence intervals.

**Capacity and scale:** The hidden-layer XOR network reaches **100%** on its four examples; the flat model stays near probability 0.5. Vectorized and Python-loop matrix products agree within about **1.71e-13** in this run; timing speedups are machine dependent. The scaling-law plot contrasts a fitted illustrative power law approaching a floor with a constant loss decrease per decade; it is not an empirical claim that every model follows these chosen constants.

## ⚠️ Pitfalls & interpretation limits

A stationary point need not be a minimum. Descent is a local statement; a finite step can overshoot. Fan-out gradients must sum rather than take a maximum or average. Finite differences validate tested implementations numerically; they do not prove all cases. The tiny epsilon in `Net.loss` means its literal loss differs minutely from the unclipped BCE derivative, harmless at these tested values but relevant if extending to saturation. Backpropagation's cost also grows with model size; its advantage is a small multiple of a forward pass versus perturbing every parameter. The notebook's billion-parameter timing extrapolation holds a toy per-pass cost fixed and should not be read as a realistic large-model runtime forecast. Batch averaging is Monte Carlo estimation of an expectation, distinct from deterministic equal-width Riemann quadrature. The capstone plots test loss over epochs for instruction; practical epoch selection needs a validation split.

## ✅ Run notes

All 51 code cells were rerun with original seeds and experiment sizes. A plotting-label bug was fixed: five depth configurations had only four bar labels. The missing **sigmoid gain 128** label was inserted so each bar names the correct setup. Numerical training and checking logic are unchanged. Figures, text, genuine timings and model outputs are embedded.

## 🧠 Quick self-check

Explain the `1/3` stability boundary, derive the fan-out sum, identify why `W.T` appears, and explain why a 91% finite test score can coexist with a 90.41% population ceiling.
