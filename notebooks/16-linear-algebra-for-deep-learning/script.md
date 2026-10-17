# 🎬 Video Script — Linear Algebra for Deep Learning

**Target:** 65–75 seconds · **Spoken words:** 154 · Read around 125–135 words per minute, with brief screen pauses.

## Spoken script

In this notebook, I follow the linear algebra behind a neural network, starting with one neuron.

The first example takes a dot product and adds a bias, producing one point eight. That same operation scales up to batches through matrix multiplication.

Here, four samples with three features multiply a three by two weight matrix. Adding a two element bias gives four rows of two outputs. Checking those shapes helps catch mistakes before running the code.

The attention example uses a transpose so every query can be compared with every key. These are raw scores, not yet attention probabilities.

I then inspect eigenvectors, which preserve their direction under this example’s transformation, and norms, which measure vector size and support regularization.

The final forward pass combines matrix products, ReLU, and row wise softmax, producing five rows of three class probabilities.

My takeaway is that shapes explain what is valid, while geometry explains what the operations mean.

## Screen cues — do not read aloud

- 0–15s: Show single-neuron dot product and output 1.80.
- 15–33s: Show batch processing shapes and X @ W + b.
- 33–47s: Show Q @ K.T and 4×4 score heatmap.
- 47–59s: Show eigenvalue verification and norm unit balls.
- 59–73s: Show final 4→8→3 forward pass and probability table.

**Recording note:** Show the saved executed outputs. This is a concise representative walkthrough; the study notes cover the remaining blocks.
