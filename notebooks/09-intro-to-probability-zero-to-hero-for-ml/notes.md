# Intro to Probability — Zero to Hero for ML

> **The big idea** · Build probability from counting and simulation, then connect Bayesian evidence, distributions, expectation, and joint structure to an interpretable spam classifier.

## 🎯 Learning objectives

- Compute event combinations, conditional probabilities, and Bayesian updates.
- Distinguish random variables, PMFs/PDFs/CDFs, expectation, and variance.
- Choose distributions from their generating assumptions; analyze covariance and dependence.
- Fit and evaluate Bernoulli Naive Bayes using log probabilities, smoothing, and calibration.

## 🧭 Conceptual map

A simulated inbox → Counting outcomes → Combining events → Conditional probability → Bayes → Random variables and expectations → Spread → Distribution families → Several variables → Probability becomes ML

## 📚 Walkthrough of every major section

### 0 · A simulated inbox

Generate 20,000 emails with a 30% spam prior and four word indicators. Word probabilities differ by class and are conditionally independent by construction.

### 1 · Counting outcomes

Define sample spaces/events, probability axioms, complements, and empirical frequencies. Seeded Monte Carlo estimates reveal sampling variation and convergence; short streaks do not change the next independent trial.

### 2 · Combining events

Use multiplication for independent AND, inclusion-exclusion for OR, and complements for at-least-one events. Birthday collisions illustrate why pair counts matter; staged experiments use probability trees.

### 3 · Conditional probability

Filter the inbox first, then count within the retained cases. P(A|B) differs from P(B|A). Distinguish exclusive events, marginal independence, conditional independence, and conditioning on impossible events. Total probability assembles conditional pieces.

### 4 · Bayes

Derive posterior=likelihood*prior/evidence and verify it by direct counts. A synthetic rare-condition test shows the base-rate effect. Sequential evidence updates work under the appropriate conditional-independence assumptions; duplicated evidence breaks them.

### 5 · Random variables and expectations

Map outcomes to numbers, construct PMFs, and distinguish density height from interval probability. Compute expected values and transformations, compare E[g(X)] with g(E[X]), and observe sample averages settling.

### 6 · Spread

Compute variance and standard deviation, transform/standardize variables, and inspect tail behavior. Similar means or standard deviations do not imply identical distributions.

### 7 · Distribution families

Derive or simulate Bernoulli, binomial, geometric, Poisson, uniform, normal, and exponential behavior. Match a family to trial/count/waiting assumptions; use scipy PMF/PDF/CDF/PPF consistently.

### 8 · Several variables

Build joint tables, sum marginals, normalize conditionals, and test factorization for independence. Derive covariance/correlation and covariance matrices. Nonlinear examples and Anscombe show why summaries cannot replace plots.

### 9 · Probability becomes ML

Optimize binomial likelihood and connect negative mean log likelihood to cross-entropy. Estimate Naive Bayes from 15,000 emails, score on 5,000 unseen emails, demonstrate duplicate-feature overconfidence and zero-count failure, then use Laplace smoothing and reliability diagrams.

## 💻 Important code, connected to its purpose

### Conditioning means filtering before counting

Notebook cell **69** (one-based, including markdown).

```python
def p_est(event, given=None):
    """Fraction of emails where `event` is True.

    event : boolean array over the 20,000 emails
    given : optional boolean array. If supplied, we FIRST discard every
            email where `given` is False, then take the fraction among
            what is left -- i.e. this returns P(event | given).
    """
    event = np.asarray(event)
    if given is None:
        return float(event.mean())
    given = np.asarray(given)
    n_kept = int(given.sum())
    if n_kept == 0:
        raise ZeroDivisionError(
            "cannot condition on an event that never happened -- there is "
            "nothing left to count (see the warning box in section 3.6)")
```

### Stable posterior normalization in log space

Notebook cell **263** (one-based, including markdown).

```python
post_true = np.exp(s_true - np.logaddexp(s_true, h_true))   # stable softmax of 2
```

### Laplace smoothing for Bernoulli features

Notebook cell **276** (one-based, including markdown).

```python
def laplace(Xc, alpha=1.0):
    """(count + alpha) / (n + 2*alpha) -- 2 because present/absent."""
    return (Xc.sum(axis=0) + alpha) / (Xc.shape[0] + 2 * alpha)
```

## 🔬 Actual results from this execution

The excerpts below are retained stdout from the fresh local run. Figures are embedded in the executed notebook.

### Evidence · notebook cell 106

```text
The two routes to a positive test, per 1 person:
  ill AND positive     = P(+|ill) P(ill)         = 0.99 x 0.0010   = 0.000990
  healthy AND positive = P(+|healthy) P(healthy) = 0.01 x 0.9990   = 0.009990
  -------------------------------------------------------------------
  P(+) = anyone positive                                = 0.010980

P(ill | +) = 0.000990 / 0.010980 = 0.090164   ->  9.0%

So a positive result on a 99%-accurate test leaves you about 9% likely to be ill,
and about 91% likely to be perfectly healthy.

Look at which of the two numerator lines is bigger. The false positives
outnumber the true positives 10.1 to 1 --
not because the test is bad, but because healthy people are 999 times
more numerous, and 1% of a huge group beats 99% of a tiny one.
```

### Evidence · notebook cell 271

```text
NAIVE BAYES on 5,000 unseen emails
    accuracy                     = 0.8962
    always-guess-the-common-class= 0.6940   <- the baseline
    errors it removed vs baseline= 0.2022

    recall on spam (caught)      = 0.7098
    precision on spam            = 0.9354
    real mail wrongly binned     = 75 of 3470
```

### Evidence · notebook cell 279

```text
bin       n   mean predicted   actually spam      gap
--------------------------------------------------------------
0.0 - 0.1   1,924           0.0164          0.0151  -0.0013
0.1 - 0.2       5           0.1036          0.2000   0.0964
0.2 - 0.3   1,791           0.2006          0.2083   0.0077
0.3 - 0.4      61           0.3036          0.2787  -0.0249
0.4 - 0.5      58           0.4842          0.4138  -0.0704
0.6 - 0.7      22           0.6746          0.8636   0.1891
0.8 - 0.9     511           0.8866          0.8924   0.0058
0.9 - 1.0     628           0.9741          0.9729  -0.0012

expected calibration error, honest model    = 0.0060
expected calibration error, duplicated word = 0.0605
ratio                                       = 10.0x worse

VERDICT on the honest model: WELL calibrated on this data.
And it should be. Part 0 generated the words to be conditionally
independent given the class, so naive Bayes' assumption is exactly true
here and its posterior is the real one. That is a luxury of simulated
data. The duplicated-word model is what real text looks like:
    accuracy differs by only 0.0102
    calibration error is     10.0x worse
Accuracy could not see the difference. Calibration could.
```

## ⚠️ Interpretation and common mistakes

- A likelihood is a function of candidate parameters with the data fixed; it is not a posterior over parameters.
- Density height is not the probability of one exact continuous value.
- Naive Bayes assumes feature independence conditional on the class, not marginal independence.
- Zero estimates from small samples are not proof of impossibility; smoothing prevents brittle zero likelihoods.
- Calibration depends on bins and sample support; sparse bins are noisy.
- The unsmoothed zero-count cell intentionally demonstrates non-finite log arithmetic; later smoothing repairs this pedagogical failure.

## ✅ Run record

- Executed **83/83 code cells** in a fresh local Python kernel with actual outputs retained.
- Inputs are the notebook's seeded synthetic datasets; no manual input or external downloads are required.
- Original exercise prompts remain as optional practice; prose exercises are not reported as completed code cells.
- Execution details, environment versions, adaptations, and output inventory are in `execution.json`; rerun with `python reproduce.py`.

## 🎬 Recording plan

Use the linked video script for a representative 65–75 second walkthrough. The short video highlights the core idea and key evidence; these notes retain the broader notebook coverage.
