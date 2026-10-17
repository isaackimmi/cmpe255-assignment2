# 📊 Statistics for Deep Learning

> **Core idea:** Statistical ideas explain how models represent uncertainty, why losses work, and how sampling and regularization affect learning.

## 🎯 Purpose & learning objectives

Connect descriptive statistics to activation scaling; read probability distributions; relate conditional probability to sequence models; derive cross-entropy from likelihood; distinguish entropy, cross-entropy, and KL divergence; understand Monte Carlo estimation, dropout, covariance, and regularization priors.

## 🗺️ Conceptual map

| Block | Main idea | Demonstration |
| --- | --- | --- |
| Model outputs | Scores become a categorical distribution | PyTorch linear classifier and softmax |
| Mean/variance/std | Center and spread summarize activations | Manual calculations agree with NumPy |
| Standardization | Rescale generated activations | Before/after histograms |
| Conditional probability | Joint sequence probability factors by context | A six-word sentence |
| Distributions | Gaussian, uniform, Bernoulli, categorical, heavy tails | Distribution plots and softmax temperature |
| Initialization | Scale depends on fan-in/fan-out | Naive, Xavier, and He histograms |
| Expectation/variance | Averages concentrate as samples accumulate | Dice rolls and noisy batch gradients |
| Bayes/MLE | Priors update with evidence; losses fit likelihood | Rare-event test and true-class loss |
| Information theory | Confidence and distribution mismatch differ | Entropy, CE, KL comparisons |
| Sampling | Monte Carlo approximates expectations | π estimation and dropout masks |
| Relationships/priors | Correlation is linear; priors shape weights | Scatterplots and L1/L2 geometry |

## 🔍 Important blocks

### 1. From logits to probabilities

```python
logits = self.fc(x)
probabilities = F.softmax(logits, dim=-1)
return logits, probabilities
```

The fresh PyTorch example outputs approximately `[0.3728, 0.2701, 0.3571]`, summing to **1.000000**. `argmax` predicts class 0. This is a randomly initialized demonstration, not a trained classifier or measured predictive accuracy.

Softmax temperature rescales logits: a lower positive temperature makes the distribution sharper and a higher one makes it flatter. Probability normalization alone does not establish confidence calibration.

### 2. Statistics and normalization

```python
mean = np.sum(data) / n
variance = np.sum((data - mean)**2) / n
std = np.sqrt(variance)
x_normalized = (x - mean) / (std + eps)
```

For 1,000 synthetic Normal samples, manual and NumPy results agree: mean **5.0387**, variance **3.8316**, standard deviation **1.9575**. These use the population denominator `n`, not the unbiased sample-variance denominator `n-1`. The 68–95–99.7 rule is specific to approximately normal data.

Generated activations change from mean **10.0967**, std **4.8936** to approximately mean **0**, std **1**. This illustrates standardization. Full PyTorch BatchNorm uses feature-wise statistics, `sqrt(var + eps)`, trainable affine parameters, and running statistics; the notebook's statement that the simplified function is exactly BatchNorm is too strong. Standardization does not generally decorrelate distinct features.

### 3. Sequence probability and initialization

The chain rule multiplies context-conditioned probabilities for “The cat sat on the mat” to obtain **0.00000900**. Log probabilities convert this product into a sum and avoid underflow for long sequences. The probabilities are hand-specified educational values, not outputs from a language model.

Xavier standard deviation is `sqrt(2/(fan_in+fan_out))`, here **0.0510**; He uses `sqrt(2/fan_in)`, here **0.0625**. The correct scale depends on activation and architecture. The gradient-noise simulation illustrates variance **σ²/B** for an average of independent, identically distributed sample gradients, not a guarantee for correlated training data.

### 4. Training and information theory

```python
y_pred = np.clip(y_pred, 1e-10, 1.0)
return -np.sum(y_true * np.log(y_pred))
```

With a one-hot target, loss reduces to `-log(p_true)`. True-class probabilities **0.90, 0.50, 0.20, 0.05** yield losses **0.1054, 0.6931, 1.6094, 2.9957**. Minimizing negative log-likelihood is maximum-likelihood fitting under the chosen probability model.

```python
return -np.sum(p * np.log(q))
return np.sum(p * np.log(p / q))
```

These are cross-entropy and KL divergence. Their relationship is **CE(P,Q) = H(P) + KL(P||Q)**, so optimizing CE over Q also minimizes KL when P is fixed. Uniform four-class entropy is **1.3863 nats**, versus **0.4280** and **0.1677** for the more concentrated examples. KL is asymmetric and is not a distance metric.

### 5. Bayes, Monte Carlo, and dropout

The hypothetical rare-disease example gives posterior **16.1%** from a **1%** prior, **95%** sensitivity, and **5%** false-positive rate. This demonstrates the denominator's dependence on base rates.

```python
inside = (x**2 + y**2) <= 1
pi_estimate = 4 * np.mean(inside)
model.train()  # Dropout active
```

At one million samples, Monte Carlo estimates π as **3.141720** with absolute error **0.000127**. Individual sample-size errors need not decrease monotonically; the characteristic error scale falls like `1/sqrt(N)`.

Five passes through the same dropout model show different zeroed entries. PyTorch uses inverted dropout during training and disables it in ordinary evaluation; ordinary evaluation is not a literal average of every possible subnetwork. Monte Carlo dropout uncertainty requires deliberately sampling with dropout active at inference.

### 6. Correlation and regularization

The scatterplots compare positive, absent, negative, and nonlinear relationships. Near-zero Pearson correlation does not imply independence. Attention uses learned dot-product similarity; it is not automatically a Pearson correlation coefficient.

```python
l2_weights = noisy_weights / (1 + 0.5)
l1_weights = np.sign(noisy_weights) * np.maximum(np.abs(noisy_weights) - l1_threshold, 0)
```

These simulated shrinkage/soft-thresholding examples illustrate small versus sparse weights. A Gaussian weight prior leads to an L2 penalty and a Laplace prior to L1 under a MAP objective. Exact coefficients depend on loss scaling and prior parameterization. These plots are illustrative transformations, not a fitted regression experiment.

## ✅ Fresh execution & practice

All **16/16 code cells** ran successfully in original order with actual numerical, PyTorch, and graphical outputs saved. No source edits, external datasets, or interactive inputs were required. Original font settings produce some missing-symbol warnings; the computations complete.

**Try next:** Explain CE versus entropy using a confident wrong prediction; compare dropout under `train()` and `eval()`; use correlated samples to test the variance assumption; distinguish normalization from whitening.
