# Intro to Calculus — Zero to Hero for ML

> **The big idea** · Build calculus from rates and accumulation, then use the chain rule to derive and verify a working neural network. One simulated car journey connects derivatives and integrals before the notebook moves to model training.

## 🎯 Learning objectives

- Read functions, domains, limits, continuity, derivatives, and curvature.
- Derive and numerically check differentiation rules, including the chain rule.
- Approximate integrals and explain the Fundamental Theorem of Calculus.
- Assemble partial derivatives into gradients; train models and verify backpropagation.

## 🧭 Conceptual map

One journey, two questions → Functions and average change → Limits → The derivative → Rules with independent checks → Shape and optimization → Integration → The Fundamental Theorem → Several variables → Calculus trains models

## 📚 Walkthrough of every major section

### 0 · One journey, two questions

Generate 121 position/speed readings across 120 seconds. Position is accumulated speed; speed is the local rate of position. Shaded slowing regions connect graph shape to physical units.

### 1 · Functions and average change

Distinguish domain/range, the vertical-line test, familiar function shapes, composition, and secant slopes. Average rate uses two distinct inputs, unlike the instantaneous derivative.

### 2 · Limits

Approach removable holes from both sides; contrast jumps, blow-ups, and oscillation. Explore continuity, limits at infinity, sigmoid saturation, and why excessively small finite differences lose precision.

### 3 · The derivative

Shrink secants toward tangents; derive powers from the difference quotient. Differentiate at many points to obtain a function. Corners such as absolute value/ReLU need explicit treatment; numerical differentiation recovers the car speed approximately.

### 4 · Rules with independent checks

Derive constant, sum, power, product, quotient, and chain rules with SymPy. Verify symbolic results against central differences. Obtain sigmoid derivative sigma*(1-sigma), then audit all 35 rules.

### 5 · Shape and optimization

Use first derivatives for increasing/decreasing regions and stationary points; use second derivatives for curvature and classify extrema carefully. Compare tangent approximations, optimization, and loss landscapes; a zero derivative alone does not prove a minimum.

### 6 · Integration

Compare left, right, and midpoint sums, signed area, interval additivity, and approximation error. Adaptive quadrature handles narrow peaks more reliably than a coarse fixed grid.

### 7 · The Fundamental Theorem

Differentiate accumulated area and integrate derivatives to recover net change. Separate definite integrals from antiderivatives with constants. Gaussian integrals introduce erf and probability-density normalization.

### 8 · Several variables

Hold other coordinates fixed for a partial derivative, assemble gradients, and take directional derivatives. Use multivariable chain rules and a finite-difference checker that catches a deliberately incorrect gradient.

### 9 · Calculus trains models

Explore learning-rate stability on bowls, fit a noisy line analytically and with gradient descent, derive nine network gradients, demonstrate vanishing sigmoid products, and train XOR for 6,000 epochs.

## 💻 Important code, connected to its purpose

### Central differences: an independent derivative measurement

Notebook cell **106** (one-based, including markdown).

```python
def num_deriv(f, a, step=1e-5):
    # Central difference: leading truncation error is proportional to step**2.
    return (f(a + step) - f(a - step)) / (2.0 * step)
```

### Backpropagation through the hidden sigmoid

Notebook cell **290** (one-based, including markdown).

```python
sig_prime = a1 * (1.0 - a1)                 # sigma'(z) = sigma(z)(1 - sigma(z))
dL_da1    = delta2 * W2
delta1    = dL_da1 * sig_prime
for j in (0, 1):
    print(f"dL/da1[{j}] = delta2 * W2[{j}]        = {delta2:+.6f} * {W2[j]:+.2f} "
```

### Parameter updates use the averaged gradients

Notebook cell **295** (one-based, including markdown).

```python
    A1 -= eta_net * gA1 / n                    # the update rule from 9.1
    B1 -= eta_net * gB1 / n
    A2 -= eta_net * gA2 / n
    B2 -= eta_net * gB2 / n
```

## 🔬 Actual results from this execution

The excerpts below are retained stdout from the fresh local run. Figures are embedded in the executed notebook.

### Evidence · notebook cell 5

```text
journey: 121 readings, 120 seconds
total distance : 1,145.6 m
top speed      : 13.96 m/s  (50.3 km/h)
average speed  : 9.55 m/s
```

### Evidence · notebook cell 281

```text
40 points generated from y = 2.3x + 1.7 plus noise of sd 2.0
(the model does not know those numbers, and noise means it should not
 recover them exactly -- it recovers the best line THROUGH THIS DATA.)

dL/dm = 0  ->   Eq(10.73*c + 75.71*m - 197.4, 0)
dL/dc = 0  ->   Eq(2.0*c + 10.73*m - 29.29, 0)

Two linear equations in two unknowns. Part 5 says solve them.

ANALYTIC   m = 2.218569752618   c = 2.739278244669   L = 4.628779869948
start          m = 0.0000000000   c = 0.0000000000   L = 263.677735
after     1    m = 3.9472214240   c = 0.5858375893   L = 82.423056
after    10    m = 2.5209855327   c = 0.5774177142   L = 5.746825
after   100    m = 2.3506544201   c = 1.8134344756   L = 4.833748
after  1000    m = 2.2185972721   c = 2.7390853480   L = 4.628780
after  4000    m = 2.2185697526   c = 2.7392782447   L = 4.628780

            slope m               intercept c
analytic    2.218569752618      2.739278244669
descent     2.218569752618      2.739278244670
difference  3.952e-14          2.358e-13

loss, analytic 4.628779869948   descent 4.628779869948
numpy's polyfit, as a third opinion: [2.21856975 2.73927824]
```

### Evidence · notebook cell 290

```text
BACKWARD PASS
==========================================================================
dL/do    = 2*(o - y)            = 2*(0.209255 - 1.0000) = -1.581490
delta2   = dL/do * do/dz2       = -1.581490 * 1 = -1.581490

dL/dW2[0] = delta2 * a1[0]        = -1.581490 * 0.336261 = -0.531794
dL/dW2[1] = delta2 * a1[1]        = -1.581490 * 0.586618 = -0.927730
dL/db2   = delta2               = -1.581490

dL/da1[0] = delta2 * W2[0]        = -1.581490 * +1.10 = -1.739639
sigma'(z1[0]) = a1[0]*(1-a1[0])   = 0.336261*0.663739 = 0.223190
delta1[0] = dL/da1[0] * sigma'   = -1.739639 * 0.223190 = -0.388269
dL/da1[1] = delta2 * W2[1]        = -1.581490 * -0.70 = +1.107043
sigma'(z1[1]) = a1[1]*(1-a1[1])   = 0.586618*0.413382 = 0.242497
delta1[1] = dL/da1[1] * sigma'   = +1.107043 * 0.242497 = +0.268455

dL/dW1[0,0] = delta1[0] * x[0]    = -0.388269 * 0.60 = -0.232962
dL/dW1[0,1] = delta1[0] * x[1]    = -0.388269 * 0.90 = -0.349442
dL/dW1[1,0] = delta1[1] * x[0]    = +0.268455 * 0.60 = +0.161073
dL/dW1[1,1] = delta1[1] * x[1]    = +0.268455 * 0.90 = +0.241610
dL/db1[0] = delta1[0]           = -0.388269
dL/db1[1] = delta1[1]           = +0.268455

sympy on sigma': 0  (zero means the identity we derived is exact)

parameter |  by hand (calculus) |  numerical (nudging) |  |difference| | verdict
------------------------------------------------------------------------------------
W1[0,0]   |     -0.232961602615 |      -0.232961602620 |    4.588e-12 | PASS
W1[0,1]   |     -0.349442403923 |      -0.349442403902 |    2.087e-11 | PASS
W1[1,0]   |     +0.161073000212 |      +0.161073000327 |    1.143e-10 | PASS
W1[1,1]   |     +0.241609500318 |      +0.241609500351 |    3.267e-11 | PASS
b1[0]     |     -0.388269337692 |      -0.388269337737 |    4.465e-11 | PASS
b1[1]     |     +0.268455000354 |      +0.268455000341 |    1.305e-11 | PASS
W2[0]     |     -0.531793801619 |      -0.531793801606 |    1.347e-11 | PASS
W2[1]     |     -0.927729685162 |      -0.927729685241 |    7.876e-11 | PASS
b2        |     -1.581489744774 |      -1.581489744718 |    5.634e-11 | PASS
------------------------------------------------------------------------------------
9 / 9 gradients agree to better than 1e-07   (largest disagreement 1.143e-10)
```

### Evidence · notebook cell 295

```text
epoch |    mean squared loss
------+---------------------
    0 | 3.36718311
    1 | 1.11249528
   10 | 0.32580029
  100 | 0.20564502
  500 | 0.00000000
 2000 | 0.00000000
 5999 | 0.00000000

loss fell from 3.367183 to 1.701e-30  (a factor of 1.98e+30)

 x1  x2 | target | prediction
--------+--------+-----------
0   0  |  0     | +0.000000
0   1  |  1     | +1.000000
1   0  |  1     | +1.000000
1   1  |  0     | +0.000000
```

## ⚠️ Interpretation and common mistakes

- A tiny difference step eventually worsens accuracy through subtraction cancellation.
- A stationary point can be a minimum, maximum, or saddle; local information does not establish global optimality.
- The sigmoid example uses unit weights; general vanishing/exploding gradients also depend on weight Jacobians.
- The car profile brakes to about 1 m/s, not precisely zero; the plotted formulas and actual outputs are authoritative.
- Finite-difference agreements test chosen points and tolerances, not a formal proof over every input.

## ✅ Run record

- Executed **79/79 code cells** in a fresh local Python kernel with actual outputs retained.
- Inputs are the notebook's seeded synthetic datasets; no manual input or external downloads are required.
- Original exercise prompts remain as optional practice; prose exercises are not reported as completed code cells.
- Execution details, environment versions, adaptations, and output inventory are in `execution.json`; rerun with `python reproduce.py`.

## 🎬 Recording plan

Use the linked video script for a representative 65–75 second walkthrough. The short video highlights the core idea and key evidence; these notes retain the broader notebook coverage.
