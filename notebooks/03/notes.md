# 🧭 Foundations 1 — The Learning Machine

[Open my Colab](https://colab.research.google.com/drive/17VWQCKzuP78d5R2COfInh8PyWjeprddt) · Notebook 03 of 16

> **Core idea:** Learning is the repeated loop **DATA → MODEL → LOSS → UPDATE**. A useful result must also work beyond the examples used to fit it.

## 🎯 Purpose & learning objectives

Locate classical AI, machine learning and deep learning on one map; distinguish supervised, unsupervised and reinforcement feedback; build a complete learning loop with two parameters; and recognize overfitting by comparing training and held-out error.

## 🗺️ Concept map

**Who chooses the rules? → what feedback is available? → choose parameters → measure errors → follow gradients → test on unseen inputs.** The remaining Foundations notebooks unpack the model (linear algebra), data (probability), and loss/update (calculus) boxes.

## 📚 Walkthrough of all ten parts

| Part | Important block | What to learn |
|---|---|---|
| 1 · The map | Nested AI/ML/DL diagrams; fixed spam rule versus fitted logistic regression | Learning estimates parameters from examples; DL also learns useful representations |
| 2 · The fingerprint | Repeat logistic regression across seven sample sizes and 12 trials | A fixed rule cannot adapt; learning curves expose sample-size effects |
| 3 · Data shapes | Labeled points, unlabeled clusters, agent/reward diagrams | Feedback structure separates learning paradigms |
| 4 · A model is numbers | `predict(hours,w,b)` | Parameters define a family of possible rules |
| 5 · Loss | Squared error; cancellation of signed errors | Turn many mistakes into a nonnegative optimization objective |
| 6 · Update | Hand-derived gradients and three learning rates | A correct direction with a poor step size can still fail |
| 7 · Whole loop | Fit three students for 2,000 steps | Model, loss and update cooperate in a tiny complete training run |
| 8 · Generalization | Predict at new study-hour values | Parameters encode a reusable rule rather than a lookup table |
| 9 · Overfitting | Polynomial degrees 1,2,3,5,8 on nine noisy observations | Training error alone rewards memorization |
| 10 · Road ahead | Loop map, scale diagram and XOR limitation | The same loop supports richer architectures; capacity and resources matter |

## 💻 Important code blocks

### Model and loss — cells 31 and 34

```python
def predict(hours, w, b):
    return w * hours + b

def squared_loss(y_hat, y):
    return (y_hat - y) ** 2
```

At `w=5,b=20,x=4`, prediction is **40** versus target **70**, giving squared loss **900**. Squaring prevents positive and negative errors cancelling.

### The entire learning loop — cell 45

```python
for step in range(STEPS):
    y_hat = w * X + b
    errors = y_hat - Y
    loss = (errors ** 2).mean()
    history.append((w, b, loss))
    dL_dw = 2 * (errors * X).mean()
    dL_db = 2 * errors.mean()
    w -= eta * dL_dw
    b -= eta * dL_db
```

The input is `X=[2,4,6]`, targets `Y=[45,70,88]`, and rate `eta=0.02`. Each gradient averages the batch contribution. The learned line is approximately **10.750*x + 24.667**, with MSE **2.7222**, down from **856.333**. The residual floor is expected: these three targets are not collinear.

### Generalization and overfitting — cells 49 and 53

```python
coeffs = np.polyfit(x_train, y_train, d)
fitted = np.poly1d(coeffs)
mse_train = ((fitted(x_train) - y_train) ** 2).mean()
mse_test = ((fitted(x_test) - y_test) ** 2).mean()
```

The degree-8 fit drives training MSE effectively to zero, but its held-out MSE is approximately **694,210.6**. Degree 3 has held-out MSE approximately **9.2** in this particular draw. More flexibility is not automatically better.

## 🔎 Actual outputs & reading the plots

The fitted spam model scores **89.6%** on the fixed synthetic test inbox after 300 training examples; the handcrafted feature threshold scores **74.3%**. Learned link weight is about **1.820**, versus **0.221** for spam-word count: the model exploits the stronger synthetic signal. The learning curve improves from about **86.2%** at 20 examples to **89.6%** at 10,000.

For the single-student update, gradients are **-240** and **-60**. Rate 0.001 increases the parameters to **5.240** and **20.060**, lowering loss to about **839.8**. The three-rate plot then contrasts slow convergence, quick convergence and divergence; the too-large 0.075 rate ends near **2.89 million** loss after 12 steps.

The learned student line predicts about **78.4** at five hours, an unseen input. That shows interpolation by a fitted rule, but without a real unseen target at five hours it does not itself measure prediction accuracy. The separate noisy-polynomial experiment does include held-out targets.

## ⚠️ Pitfalls & interpretation limits

More data does not guarantee monotonic improvement in every finite run. A learning curve is evidence about a learner under the stated setup, not a universal definition of intelligence. The initial string-based spam rule and later numerical threshold are related demonstrations but are not identical implementations. Learning paradigms are determined by feedback, and unlabeled data can still support ML despite the simplified decision diagram. Reserve validation data for model selection; repeatedly selecting degree using the shown test set would turn it into a validation set. High-capacity fits can generalize in other settings, so the polynomial example demonstrates a failure mode rather than a universal law.

## ✅ Run notes

Seed 7 and each explicit generator seed were preserved. All code cells ran without source adaptations. Outputs retain the actual learning curves, training history and overfitting figures; the expected red cross on a deliberately inadequate spam rule is an instructional outcome, not an execution error.

## 🧠 Quick self-check

Name the four loop stages in the code, explain why the three-point MSE cannot reach zero for a line, and describe which curve reveals the degree-8 failure.
