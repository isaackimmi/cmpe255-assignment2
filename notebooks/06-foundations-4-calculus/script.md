# 🎬 Foundations 4 — Calculus — Video Script

**Target:** 65–75 seconds · **Spoken words:** 152 · **Estimated narration:** 73.0 seconds at 125 words/minute

**Recording approach:** A concise overview with representative executed blocks. The full section-by-section explanation is in the notebook notes. Pre-position these cells; do not wait for execution during recording.

## Screen cues — not spoken

- 0–12s: Title and sequence map.
- 12–27s: Stable/diverging gradient descent, cell 35.
- 27–41s: Fan-out diagram and backward method, cells 52/71.
- 41–56s: Gradient audit and moons training, cells 73/75/76.
- 56–73s: Expected loss and held-out capstone, cells 104/110.

## Spoken script

This notebook completes the Foundations series by opening the loss and update boxes. It connects derivatives, gradients, backpropagation, and integrals to an entire learning system.

Here, the gradient of the quadratic bowl points uphill, so the update subtracts it. The learning-rate comparison shows why the correct direction can still diverge when the step is too large.

Next, the chain rule multiplies derivatives along a path, and fan-out requires adding contributions from every path. Those rules become the backward method of this NumPy neural network.

Before training, the notebook checks every parameter in a small network against finite differences, then compares its gradients with PyTorch. The trained moon example reaches perfect training accuracy, which is not a held-out result.

The integral section explains expected loss and why batch averages fluctuate. Finally, the Gaussian capstone evaluates new examples against a theoretical accuracy ceiling. My main takeaway is to verify gradients and measure generalization separately.

## Recording check

Show the saved outputs at readable zoom. Briefly pause on the key result. Avoid reading every line of code or every section heading. This script is ready for the user to rehearse and record; no video has been recorded.
