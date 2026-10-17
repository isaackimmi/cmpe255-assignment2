# Intro to Statistics — Zero to Hero for ML

> **The big idea** · Learn what data can establish about an unseen population, quantify uncertainty, and avoid misleading model comparisons. A visible simulated population makes sampling claims directly testable.

## 🎯 Learning objectives

- Describe distributions robustly and distinguish population parameters from sample statistics.
- Explain sampling distributions, standard errors, CLT limitations, and confidence-interval coverage.
- Interpret hypothesis tests, power, pairing, effect sizes, regression, and multiple-comparison corrections.
- Evaluate ML using honest splits, uncertainty, cross-validation, paired tests, and selection-bias controls.

## 🧭 Conceptual map

Known population, limited sample → Describe before infer → Samples and estimators → Sampling distributions → Confidence intervals → Hypothesis tests → Compare groups → Regression → Statistical traps → Honest ML evaluation

## 📚 Walkthrough of every major section

### 0 · Known population, limited sample

Simulate 200,000 lognormal reaction times, then draw only 40 people. The visible population supplies ground truth for checking estimates; real experiments usually lack this luxury.

### 1 · Describe before infer

Implement mean, median, mode, variance, skewness, kurtosis, quantiles, and robust alternatives. Compare distributions with identical summaries and inject contamination to test sensitivity.

### 2 · Samples and estimators

Separate parameters from statistics; measure sampling variability, bias, and variance. Increasing sample size reduces random error but does not repair selection bias or dependence. Explore how sampling mechanisms fail.

### 3 · Sampling distributions

Repeatedly draw samples and plot their means. Distinguish data distribution from sampling distribution, derive SE=sigma/sqrt(n), and see CLT convergence plus failures involving heavy tails, dependence, or biased sampling.

### 4 · Confidence intervals

Build z intervals and measure repeated-study coverage, then account for unknown variance with t intervals. Compare percentile bootstrap intervals from resampling. Confidence is a property of the repeated procedure, with finite-sample approximations and assumptions.

### 5 · Hypothesis tests

Construct a null world and count statistics at least as extreme as observed. Compare simulation to a one-sample t test. Separate p-values from posterior null probabilities; measure false positives, false negatives, power, and sample-size effects.

### 6 · Compare groups

Derive mean-difference tests, use pairing where measurements share people, and compare Welch with equal-variance Student tests. Report effect sizes and intervals. Study nonparametric rank tests and why repeated pairwise tests across groups need care.

### 7 · Regression

Derive the least-squares line and correlation; interpret coefficients with units. Inspect residuals, R-squared limitations, slope intervals and paired bootstrap resampling. Demonstrate extrapolation risks and regression to the mean.

### 8 · Statistical traps

Simulate multiple comparisons, Bonferroni and BH, p-hacking, optional stopping, Simpson reversals, survivorship, confounding, and publication bias. Corrections address different error guarantees and trade power for error control.

### 9 · Honest ML evaluation

Train polynomial logistic models with training-only scaling. Compare train/test/oracle accuracy, Wald versus Wilson coverage, and hand-written five-fold CV. Preserve paired correctness for permutation/McNemar tests. Measure the winner's curse and separate validation selection from final test reporting.

## 💻 Important code, connected to its purpose

### Bootstrap resamples observations with replacement

Notebook cell **121** (one-based, including markdown).

```python
def bootstrap_ci(data, statistic=np.mean, B=B, conf=0.95, rng=RNG_B):
    """Percentile bootstrap CI for any statistic. No distributional assumption."""
    n    = len(data)
    idx  = rng.integers(0, n, size=(B, n))       # resample WITH replacement
    reps = statistic(data[idx], axis=1)          # the statistic, B times over
    a    = (1 - conf) / 2
    return np.quantile(reps, [a, 1 - a]), reps
```

### Fit preprocessing only on the training fold

Notebook cell **270** (one-based, including markdown).

```python
        Zt, mu_f, sd_f = design(X[~held], degree)          # fit scaling on TRAIN
        Zh, _, _       = design(X[held],  degree, mu_f, sd_f)
        w              = fit_logistic(Zt, y[~held], iters=iters)
        out.append(correct_flags(w, Zh, y[held]).mean())
```

### Preserve per-example pairing in model comparison

Notebook cell **279** (one-based, including markdown).

```python
    d       = c1 - c2                       # each entry is -1, 0, or +1
    obs     = d.mean()
    signs   = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null    = (signs * d).mean(axis=1)      # the null distribution, simulated
    p_value = float((np.abs(null) >= abs(obs) - 1e-12).mean())
```

## 🔬 Actual results from this execution

The excerpts below are retained stdout from the fresh local run. Figures are embedded in the executed notebook.

### Evidence · notebook cell 5

```text
population : 200,000 people
  true mean          mu =   290.99 ms   <- normally UNKNOWABLE
  true sd         sigma =    83.01 ms

one sample of 40:
  sample mean     x-bar =   280.85 ms
  sample sd           s =    74.36 ms
  off by                  -10.14 ms
```

### Evidence · notebook cell 100

```text
confidence level        = 95%
alpha (outside)         = 0.050  -> 0.025 in each tail
z critical value        = norm.ppf(0.975) = 1.959964
  (this is where '1.96' comes from -- it is 1.9600, rounded)

sample mean       x-bar =  280.850 ms
sample sd             s =   74.362 ms
standard error s/sqrt(n)=   11.758 ms   (n = 40)
margin of error 1.96*SE =   23.045 ms

95% CI = [257.805, 303.894]  ms      width = 46.089 ms

the truth  mu           =  290.989 ms
contained?                YES

We can print that last line only because we built the population.
A real study reaches this point and stops, never knowing.
```

### Evidence · notebook cell 166

```text
person-to-person spread (sd of 'before') :   84.54 ms
spread of the per-person CHANGES         :   14.23 ms  <- far smaller
mean change                              :   17.98 ms  (truth: 14 ms)

  UNPAIRED test (throws the pairing away) : p = 0.406645   NOT significant -- effect MISSED
  PAIRED test   (uses it)                 : p = 1.328e-07   significant -- effect FOUND

  paired t-test == one-sample t-test on the changes: p = 1.328e-07  (identical)

Same 60 measurements. Same real effect. Opposite conclusions.
The difference is entirely in the DESIGN, not in the data collection.
```

### Evidence · notebook cell 257

```text
model: logistic regression, degree-2 features, 6 weights, fitted by gradient descent

  training accuracy  (  400 examples) = 0.9350
  TEST accuracy      (  200 examples) = 0.8950   <- what gets reported
  TRUE accuracy      (40000 examples) = 0.9128   <- normally UNKNOWABLE

  test score minus truth = -0.0178

The test score missed the truth. Not by much, and not because anything
went wrong -- 200 examples is a sample, and samples wobble. Part 3.
```

### Evidence · notebook cell 270

```text
5-fold cross-validation on the 400 training examples
(each fold holds out 80 examples)

  fold 1:  0.9500  ################################################
  fold 2:  0.9250  ##############################################
  fold 3:  0.9375  ###############################################
  fold 4:  0.8875  ############################################
  fold 5:  0.9500  ################################################

  mean across folds     = 0.9300
  sd   across folds     = 0.0259
  worst fold            = 0.8875
  best  fold            = 0.9500
  spread (best - worst) = 0.0625

  single-split test score (Part 9.1) = 0.8950
  true accuracy                      = 0.9128
```

### Evidence · notebook cell 279

```text
==================================================================
 A vs B  -- the close call
==================================================================
  observed difference   = +1.50 percentage points
  disagreements         = 4 for A, 1 for B
  permutation p-value   = 0.3804
  effect size Cohen's dz= +0.095

==================================================================
 A vs C  -- the obvious case, as a control
==================================================================
  observed difference   = +8.00 percentage points
  disagreements         = 28 for A, 12 for C
  permutation p-value   = 0.0167
  effect size Cohen's dz= +0.181

------------------------------------------------------------------
 VERDICT on A vs B: NOT distinguishable from noise (p = 0.380).
 And yet in TRUTH A really is better than B, by 0.82 points.
 The difference is real. This test set is simply too small to see it.
 'Not significant' means 'not established', never 'not there'. Part 5.
------------------------------------------------------------------
```

### Evidence · notebook cell 284

```text
IDEALISED: 20 variants, all with true accuracy 90%, test set of 200
  score of ONE variant, on average          = 0.8999
  score of the BEST of 20, on average      = 0.9376
  optimism bias                             = +3.76 percentage points
  best-of-20 exceeded 90% in 100.0% of runs
  (every one of those 'wins' is noise: the variants are identical.)

REAL MODELS: 80 runs, 20 fitted variants each
  (percentage points of optimism, each with its own 95% Monte-Carlo
   uncertainty, so you can tell a real bias from simulation noise)

  select on the test set, report that same score :  +1.60 +/- 0.39
  select on validation, report the val score     :  +2.01 +/- 0.39
  select on validation, report the TEST score    :  +0.14 +/- 0.47   <- the fix

The first two lines estimate the SAME quantity by two routes -- in both,
the score reported is the score that was selected on -- so they ought to
agree, and their overlap is a check that this loop does what it claims.

  is the third line consistent with ZERO bias? True

The bias with real models is smaller than the idealised one, for two
honest reasons: the variants are not equally good (so some winning is
merit rather than luck), and their predictions are correlated, so their
scores cannot wander independently. Smaller is not zero.
```

## ⚠️ Interpretation and common mistakes

- A 95% frequentist confidence interval does not assign a 95% posterior probability to a fixed parameter after observation.
- A p-value is not the probability the null hypothesis is true, and non-significance does not establish equality.
- Statistical significance does not measure practical importance; report effect size and uncertainty.
- Dependent CV folds do not provide independent replicates for a naive sd/sqrt(k) interval.
- Preprocessing and model choice can leak test information even when the optimizer never trains on test rows.
- The 40-comparison simulation observed 88.6% with at least one false finding; the independent-test reference is 87.1%, and simulation/assumption details matter.

## ✅ Run record

- Executed **86/86 code cells** in a fresh local Python kernel with actual outputs retained.
- Inputs are the notebook's seeded synthetic datasets; no manual input or external downloads are required.
- Original exercise prompts remain as optional practice; prose exercises are not reported as completed code cells.
- Execution details, environment versions, adaptations, and output inventory are in `execution.json`; rerun with `python reproduce.py`.

## 🎬 Recording plan

Use the linked video script for a representative 65–75 second walkthrough. The short video highlights the core idea and key evidence; these notes retain the broader notebook coverage.
