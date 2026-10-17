# 🎬 Foundations 1 — The Learning Machine — Video Script

**Target:** 65–75 seconds · **Spoken words:** 154 · **Estimated narration:** 73.9 seconds at 125 words/minute

**Recording approach:** A concise overview with representative executed blocks. The full section-by-section explanation is in the notebook notes. Pre-position these cells; do not wait for execution during recording.

## Screen cues — not spoken

- 0–12s: Four-stage loop diagram, cell 28.
- 12–27s: Spam accuracy table and learning curve, cells 11/16.
- 27–52s: Training loop and final score, cell 45.
- 52–72s: Polynomial train/test error comparison, cells 53/54.

## Spoken script

This notebook introduces the learning machine through four stages: data, model, loss, and update. It also places machine learning inside AI, and deep learning inside machine learning.

I’m starting with the spam example. The handwritten rule stays fixed, while logistic regression learns weights from examples and performs better on this synthetic test inbox. The learning curve shows how its performance changes with training size.

Here is the complete training loop. The model predicts an exam score from study hours using a weight and bias. Squared error measures the mistake, and the gradients tell us how to update both parameters. After training, the loss falls substantially, although it cannot reach zero because these three points are not on one line.

The final comparison is the warning I want to remember: a flexible polynomial nearly memorizes the training data while its test error explodes. Successful learning means evaluating unseen examples, not celebrating a low training loss.

## Recording check

Show the saved outputs at readable zoom. Briefly pause on the key result. Avoid reading every line of code or every section heading. This script is ready for the user to rehearse and record; no video has been recorded.
