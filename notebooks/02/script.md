# 🎬 Calculus for Deep Learning — Video Script

**Target:** 65–75 seconds · **Spoken words:** 152 · **Estimated narration:** 73.0 seconds at 125 words/minute

**Recording approach:** A concise overview with representative executed blocks. The full section-by-section explanation is in the notebook notes. Pre-position these cells; do not wait for execution during recording.

## Screen cues — not spoken

- 0–12s: Title and eight-section overview.
- 12–27s: Cells 9 and 11, derivative and symbolic checks.
- 27–43s: Cell 35 XOR predictions and cell 36 boundary plot.
- 43–58s: Cells 43 and 45 gradient checks.
- 58–72s: Cell 40 depth plot, point out log vertical axis.

## Spoken script

This notebook explains how calculus makes a neural network learn. I start with derivatives as local sensitivity, then connect the chain rule to gradients, gradient descent, and backpropagation.

Here, the numerical derivative of x squared at three matches the analytical answer, six. The symbolic check also confirms the sigmoid derivative. Those local derivatives are the pieces that get multiplied during the backward pass.

The main demonstration is this small network learning XOR. Its predictions approach zero, one, one, and zero, showing why a hidden layer helps with a pattern that one straight boundary cannot separate.

I’m also checking whether the gradients are trustworthy. The finite-difference comparison reports relative errors below the notebook’s threshold, and PyTorch independently reproduces the handwritten gradients.

Finally, the depth experiment shows gradients shrinking or growing depending on activation and weight scale. My takeaway is to inspect gradients and initialization, and remember that fitting XOR does not establish generalization.

## Recording check

Show the saved outputs at readable zoom. Briefly pause on the key result. Avoid reading every line of code or every section heading. This script is ready for the user to rehearse and record; no video has been recorded.
