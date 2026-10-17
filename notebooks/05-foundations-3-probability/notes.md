# 🎲 Foundations 3 — Probability & Statistics

[Open my Colab](https://colab.research.google.com/drive/1TI8Yaj3erS-Js7BQTB1eJYNnAxUqqKpm) · Notebook 05 of 16

> **Core idea:** Conditioning changes the denominator. Bayes' rule combines evidence with base rates; reliable classifiers also need stable computations and meaningful evaluation.

## 🎯 Purpose & learning objectives

Derive conditional probability and Bayes' theorem, recognize base-rate mistakes, distinguish probability mass from density, identify common distributions, derive maximum-likelihood estimates, build a robust Naive Bayes classifier, and evaluate imbalanced classification using appropriate metrics.

## 🗺️ Concept map

**Counts → conditionals → Bayes and total probability → distributions → likelihood → Naive Bayes → smoothing/log-space → calibration → confusion matrix and thresholds.** This is the DATA box of the Foundations series.

## 📚 Walkthrough of all major sections

| Part | Important blocks | Learning outcome |
|---|---|---|
| 1 · Shrinking the world | Dog/leash populations and conditional counts | The conditioning event defines the relevant population |
| 2 · Symmetry trick | Two factorizations of the joint; total probability partitions | Derive Bayes by equating `P(A|B)P(B)` and `P(B|A)P(A)` |
| 3 · Rare-event example | Decimal calculation, natural-frequency tree, icon arrays, base-rate sweep | Sensitivity and posterior probability answer different questions |
| 4 · Uncertainty shapes | PMF/PDF; Bernoulli, binomial, Gaussian, uniform, Poisson | Choose distributions by the story and support, not only shape |
| 4 · Fitting distributions | Bent-coin and Gaussian MLE; likelihood and log-likelihood; BCE | Likelihood estimates parameters; BCE is mean Bernoulli negative log-likelihood |
| 5 · Naive Bayes | Worked spam score; zero-count trap; underflow; full class; sklearn comparison | Conditional independence makes estimation tractable; smoothing and logs make it usable |
| 5 · Calibration | Duplicate a correlated feature across 60,000 simulated emails | Correct labels do not ensure trustworthy probabilities |
| 6 · Evaluation | Fraud imbalance; TP/FN/FP/TN; precision, recall, F1, threshold sweep and PR curves | Choose metrics by mistake costs and inspect tradeoffs |
| 7 · Closing | Cheat sheets and 17 numerical self-checks | Recompute rather than memorize headline numbers |

## 💻 Important code blocks

### Bayes with a base rate — cell 23

```python
p_sick = 0.001
p_pos_if_sick = 0.99
p_pos_if_well = 0.05
true_pos_term = p_pos_if_sick * p_sick
false_pos_term = p_pos_if_well * (1 - p_sick)
p_pos = true_pos_term + false_pos_term
p_sick_given_pos = true_pos_term / p_pos
```

This hypothetical example yields **1.9435%** posterior probability. Among 100,000 hypothetical people there are **99** true positives and **4,995** false positives. The input is **99% sensitivity and 5% false-positive rate**, not a single claim of “99% overall accuracy.” The low base rate explains why most positive results are false alarms.

### Stable Naive Bayes — cell 87

```python
cnt = np.array([X[y == c].sum(axis=0) for c in self.classes_])
smoothed = cnt + self.alpha
self.log_lik_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
```

These lines from `fit` create Laplace-smoothed token probabilities. Prediction computes the joint log score:

```python
return np.asarray(X, dtype=float) @ self.log_lik_.T + self.log_prior_
```

A maximum-score class needs no evidence denominator because it is common to all classes. Probability recovery shifts scores by their row maximum before exponentiating, avoiding numerical overflow/underflow in normalization.

### Evaluation with explicit labels — cell 116

```python
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
```

For binary labels `[0,1]`, this is sklearn's default order. The notebook's positive-first diagram instead uses `labels=[1,0]`; label order is part of the interpretation.

## 🔎 Actual outputs & interpretation

**Conditioning:** `P(leash|dog)=20/30=0.6667`, whereas `P(dog|leash)=20/25=0.8`. They share a numerator but not a denominator. Both joint-probability factorizations give **0.20**.

**Distributions and MLE:** Four fair flips yield `P(two heads)=0.375`. A Uniform[0,0.5] density is **2**, yet its area is 1. Gaussian numerical mass inside one/two/three standard deviations is **0.6827/0.9545/0.9973**. A Poisson rate of 3 has computed mean and variance **3**. Seven heads in ten trials maximize the coin likelihood at **0.7**. Gaussian MLE on 5,000 samples gives mean about **169.9537** and standard deviation **7.1012**; its variance denominator is `n`, distinct from the unbiased `n-1` estimate. Mean BCE **0.273098** equals summed negative log-likelihood **2.730985/10**.

**Naive Bayes:** The small “free money lunch” calculation scores spam **0.0140** versus ham **0.0048**, giving normalized spam probability **0.744681**. With no smoothing, “free blockchain lunch” gives zero for both classes; smoothing resolves this. Direct multiplication of 200 small likelihoods underflows to zero while the log sum remains finite. The full classifier predicts spam/ham/ham for its three demonstration messages and matches sklearn within **1e-12** across checked quantities.

**Calibration:** Duplicating evidence changes synthetic accuracy from **0.8695** to **0.8619**, while worsening log loss from **0.3264** to **0.4346** and raising near-certain predictions from **40.72%** to **78.98%**. Classification and probability calibration are separate properties.

**Evaluation:** For `TP=8,FN=2,FP=5,TN=85`, accuracy is **93%**, precision **61.54%**, recall **80%**, and F1 **69.57%**. Thresholds 0.9/0.5/0.2 yield precision about **0.83/0.62/0.33** and recall **0.50/0.80/1.00**. The sampled best F1 is about **0.778** near threshold **0.70**. This demonstrates a tradeoff on the supplied synthetic scores, not a deployment threshold recommendation. All **17** closing numerical checks print agreement.

## ⚠️ Pitfalls

Never reverse conditional probabilities without base rates. Density height is not probability; area is. Naive Bayes assumes **conditional** independence given the class. Correlated evidence can exaggerate confidence; Laplace smoothing does not fix that assumption. A classifier that predicts every transaction legitimate can score 99% accuracy while catching no fraud. F1 omits true negatives and does not encode every mistake cost. Select thresholds on validation data, then evaluate on a separate test set. Precision need not increase monotonically at every threshold on a finite sample, though the broad tradeoff is instructive.

## ✅ Run notes

All 52 code cells ran without source adaptations. The population examples, texts, synthetic data, seeds and experiment sizes are preserved. Expected underflow and caught/illustrated metric edge cases remain visible as instructional results; there are no notebook error outputs.

## 🧠 Quick self-check

Recompute 99/(99+4995), explain why logs preserve the winning class, and identify which denominator changes between precision and recall.
