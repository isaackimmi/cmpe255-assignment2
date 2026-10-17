# 📈 Calculus for Deep Learning

[Open my Colab](https://colab.research.google.com/drive/1c1yiFliuYpr7MF-qFAHIYwlhmmfqTz51) · Notebook 02 of 16

> **Core idea:** A derivative measures sensitivity; the chain rule carries that sensitivity backward through a network so its parameters can improve.

## 🎯 Purpose & learning objectives

Connect derivatives to neural-network training. Derive basic and activation derivatives, apply the chain rule, build gradients from partial derivatives, run gradient descent, implement backpropagation for XOR, and audit those gradients using finite differences and PyTorch.

## 🗺️ Concept map

**Derivative → chain rule → multivariable gradient → gradient descent → cached forward pass and backward pass → depth stability → numerical checks → autograd.**

## 📚 Walkthrough of all major sections

| Section | Main computation | Interpretation |
|---|---|---|
| Introduction | `simple_neuron(x, w) = w*x` with input 2 | Changing the weight changes output at rate 2 |
| 1 · Derivatives | Central differences versus `2*x`; SymPy identities; sigmoid/tanh/ReLU plots | Slopes describe local sensitivity; activation saturation limits gradient flow |
| 2 · Chain rule | `(3*x+2)**2`, then a tiny sigmoid network | Multiply local derivatives along a path |
| 3 · Partials and gradients | Partial derivatives, vector magnitude and contour arrows | The gradient points steepest uphill in Euclidean parameter space |
| 4 · Gradient descent | A quadratic bowl and a learning-rate comparison | Subtract the gradient; choose a finite step size carefully |
| 5 · Backpropagation | A NumPy 2→4→1 XOR network | Cache activations and reuse them in the reverse pass |
| 6 · Depth | Monte Carlo median gradient norms | Products through weights and activations can vanish or explode |
| 7 · Gradient checking | Every weight/bias versus central differences; intentional missing-factor bug | Audit implementation using relative error |
| 8 · Autograd | Copy the same weights into PyTorch and call `.backward()` | Automatic differentiation organizes the same chain rule |

## 💻 Important code blocks

### Numerical differentiation — cell 9

```python
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)
```

For `f(x)=x**2` at 3, numerical and analytical derivatives both print **6.000000**. SymPy then reduces the difference between its sigmoid derivative and `sigma*(1-sigma)` to zero. The chain-rule example at `x=2` gives **48**, matching the numerical check.

### Update rule — cell 28

The notebook applies the ordinary gradient-descent relationship `w_next = w - learning_rate * gradient`. On `L(w1,w2)=(w1-1)**2+(w2+0.5)**2`, the run starts at `[-1.5,1.5]` with loss **10.25** and reaches approximately **[0.9969, -0.4975]**, loss **0.000016**, after 30 steps at rate 0.1.

### Backpropagation audit — cell 43

```python
m = yb.shape[0]
a2 = net.forward(Xb)
dz2 = 2 * (a2 - yb) * net.sigmoid_derivative(a2)
gW2 = (net.a1.T @ dz2) / m
dz1 = (dz2 @ net.W2.T) * net.sigmoid_derivative(net.a1)
gW1 = (net.X.T @ dz1) / m
```

These are actual lines from `analytic_grads`. The output-layer error passes through sigmoid's derivative; hidden-layer errors additionally multiply the transposed output weights. Batch gradients are averaged once. The separate read-only function avoids the training method's parameter-update side effect while checking.

### Framework verification — cell 45

```python
ta1 = torch.sigmoid(tX @ tW1 + tb1)
ta2 = torch.sigmoid(ta1 @ tW2 + tb2)
tL = torch.mean((ta2 - ty) ** 2)
tL.backward()
```

This creates the same computation with differentiable tensors. PyTorch and NumPy loss both print **0.0006964604**; the largest displayed parameter-gradient difference is **5.42e-20**.

## 🔎 Actual outputs & what they mean

- XOR predictions are approximately **0.030, 0.975, 0.975, 0.025**, matching labels **0,1,1,0** after thresholding at 0.5. This demonstrates fitting four training examples, not held-out performance.
- The worst finite-difference relative error is **6.097e-09**, below the notebook's `1e-7` threshold. Deliberately halving `W1`'s gradient raises error to **0.3333**.
- After ten layers, median gradient norms are **1.67e-07** for sigmoid with ordinary weights, **10.4** with larger weights, and **0.853** for ReLU with He initialization. These measurements concern the stated random toy setup.
- Activation plots show sigmoid derivative peaking at **0.25**, tanh at **1**, and ReLU at 1 on its positive side. Weight matrices also influence the full network gradient.

## ⚠️ Pitfalls & interpretation limits

A falling training loss does not establish generalization. Finite differences provide a numerical implementation check, not a mathematical proof for every possible input. Choose `h` to balance truncation and cancellation; avoid ReLU's kink at zero. A zero gradient can indicate a stationary point other than a minimum. He initialization targets favorable gradient statistics; it does not guarantee identical gradient norms at every layer or in every trial. Very small gradients are not necessarily literally unrepresentable in floating point; the practical problem is the ineffective learning signal and update scale.

## ✅ Run notes

All executable sections, including PyTorch, were rerun from a fresh kernel. The only adaptation is choosing **DejaVu Sans** so subscript digits display correctly under the local plotting configuration. NumPy uses seed 42; the toy training and Monte Carlo setup retain their original inputs and iterations. Saved outputs include text and static plots. Extension exercises in the closing markdown are optional proposals rather than executed cells.

## 🧠 Quick self-check

Explain the minus sign in gradient descent, identify why `W2.T` appears in the hidden-layer derivative, and describe what the missing-factor check catches.
