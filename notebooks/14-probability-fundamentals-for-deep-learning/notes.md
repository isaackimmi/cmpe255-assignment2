# 🎲 Probability Fundamentals for Deep Learning

> **Core idea:** Classification connects uncertainty, conditional probability, likelihood, and cross-entropy through one coherent mathematical story.

## 🎯 Purpose & learning objectives

Understand sample spaces and events; calculate unions, intersections, complements, and conditional probabilities; apply Bayes' theorem; distinguish discrete mass from continuous density; interpret expectation, variance, entropy, softmax, and likelihood in neural networks.

## 🧭 Reading guide

The provided source has an unusual physical order: its concluding pipeline, entropy, likelihood, normalization, distributions, and random-variable examples appear before the title and setup. Chapters 0–4 follow, and the notebook ends with the introduction to Chapter 5. All available code executes in its existing order because the early examples import what they need. **No missing chapters were invented or claimed as executed.** For study, use the conceptual order below, then revisit the opening synthesis.

| Concept | Demonstration | ML connection |
| --- | --- | --- |
| Probability and sample space | Repeated coin flips, die faces, two-dice grid | Output classes form a space of possible outcomes |
| Combining events | Card union and independent coin/die outcomes | Avoid double-counting and false independence assumptions |
| Conditioning | King given face card | A classifier estimates probabilities given its inputs |
| Bayes | Rare disease and imperfect test | Update a prior using evidence and base rates |
| Random variables/distributions | Dice PMF and height density, Gaussian curves | Discrete labels versus continuous measurements |
| Expectation/variance | Centering and scaling activations | Control the scale of intermediate values |
| MLE and information | Bernoulli likelihood, entropy | Explain classification losses and uncertainty |
| Complete pipeline | Logits → softmax → one-hot target → loss | Turn network scores into a training objective |

## 🔍 Key code & reasoning

### 1. Probability rules need the right assumptions

The two-dice grid contains **36 equally likely ordered outcomes**; six sum to seven, giving probability **1/6**. The card example applies `P(A ∪ B) = P(A) + P(B) - P(A ∩ B)`: a king of hearts belongs to both events and must not be counted twice. Independent events satisfy `P(A ∩ B) = P(A)P(B)`; dependent events require a conditional factor.

The 100,000-trial coin/die simulation produces **0.0828** for heads and six versus theoretical **0.0833**, and **0.5808** for heads or six versus **0.5833**. Sampling error explains the small discrepancies.

### 2. Conditioning changes the denominator

```python
p_king_given_face = p_king_and_face / p_face
```

`P(King) = 4/52`, but after learning that the card is a face card, the relevant set has twelve cards: `P(King | Face) = (4/52)/(12/52) = 1/3`. This is not the same question as the unconditional probability.

### 3. Bayes combines likelihood with the base rate

```python
def bayes_theorem(prior, likelihood, p_evidence):
    return (likelihood * prior) / p_evidence

P_positive = (P_positive_given_disease * P_disease +
              P_positive_given_healthy * (1 - P_disease))
```

In the notebook's hypothetical population of 10,000, a 1% disease rate gives 100 affected people. With sensitivity 95% and false-positive rate 5%, there are **95 true positives** and **495 false positives**. Of 590 positives, only **16.1%** are affected. The example illustrates base-rate reasoning; its parameters are invented teaching inputs.

Sequential updating uses the posterior as the next prior. For results `+, +, +, -, +`, the saved probabilities are **16.10%, 78.48%, 98.58%, 78.48%, 98.58%**. Multiplying the same fixed test likelihood at every step assumes tests are conditionally independent given disease status; correlated repeated tests would need a different model.

### 4. Convert logits into a probability mass function

```python
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-15))
```

Subtracting the maximum improves numerical stability without changing the resulting distribution. For logits `[2.5, 1.0, 0.5, -0.5]`, the first class receives about **71.0%**, all four probabilities sum to one, and the one-hot Cat target gives cross-entropy **0.3423**.

### 5. Connect likelihood, normalization, and information

The Bernoulli example contains six ones and four zeros, so its maximum-likelihood constant probability is **0.6**. Average cross-entropy is the negative *average* log-likelihood; the plotted summed log-likelihood differs by the sample-count factor, but the optimum is the same.

```python
normalized = (activations - mean) / std
return -np.sum(p * np.log2(p))
```

The first line standardizes generated activations. The entropy expression measures bits: a certain four-class distribution has entropy **0**, while a uniform four-class distribution has entropy **2 bits**. Low entropy means confidence, not guaranteed correctness or calibration. Cross-entropy elsewhere uses natural logs and is measured in nats.

## 📊 Outputs to inspect

Inspect the coin-flip convergence plot, the 6×6 dice grid, the conditioning card panels, the true/false-positive population view, the sequential posterior curve, and the four-panel classification pipeline. Discrete PMF bars sum to one; a continuous PDF is a density whose **area** over an interval gives probability. Its height is not an exact-point probability.

## ✅ Run notes & pitfalls

All **22/22 code cells** were freshly executed locally; actual plots and numerical outputs are saved. No source edits, external data, or interactive inputs were required. Source ordering was preserved. The opening standardization example is a simplified illustration; full batch normalization also uses an epsilon, learned scale/shift, feature-wise axes, and inference statistics.

**Try next:** Change the base rate while holding test quality constant; compare a confident wrong prediction to a uniform prediction; derive the dice PMF; explain why cross-entropy rises when the true class gets less probability.
