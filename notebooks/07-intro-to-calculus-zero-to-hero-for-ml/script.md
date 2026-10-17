# Video script · Intro to Calculus — Zero to Hero for ML

**Target:** 65–75 seconds · **Spoken words:** 151 · **Delivery:** approximately 129 words/minute for 70 seconds.

## Spoken script

In this notebook, I connect calculus to what happens when a model learns. I start with the simulated car journey: position tells me how far the car has travelled, and its slope tells me the speed. Shrinking the time interval turns an average rate into a derivative.

Next, I check differentiation rules against numerical estimates. The chain rule matters most because a neural network is a sequence of functions. Integrals reverse the perspective: they add small contributions, and the Fundamental Theorem links accumulation back to differentiation.

Here, I move to partial derivatives and gradients. This backward pass calculates how each network parameter affects the loss. All nine analytic gradients agree with independent finite differences within the stated tolerance.

Finally, the XOR training curve shows those gradients driving parameter updates and reducing the loss. My takeaway is that backpropagation supplies the gradient, while gradient descent uses it to take a learning step.

## Screen cues — do not read aloud

- 0–15s: cells 4–6, position/speed panels.
- 15–35s: cells 105 and 208, derivative checker and Fundamental Theorem.
- 35–55s: cell 289, nine-gradient PASS table.
- 55–72s: cell 294, XOR loss and prediction table.

## Recording reminder

Open the executed notebook in advance and rehearse the jumps. Read only the spoken script; screen cues and notes are supporting material. Video recording and upload are still to be done by you.
