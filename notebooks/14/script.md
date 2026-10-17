# 🎬 Video Script — Probability Fundamentals for Deep Learning

**Target:** 65–75 seconds · **Spoken words:** 148 · Read around 125–135 words per minute, with brief screen pauses.

## Spoken script

In this notebook, I connect basic probability rules to the way a classifier makes predictions.

The dice and card examples establish sample spaces, intersections, and conditional probability. Knowing a card is a face card changes the denominator, so the probability of a king becomes one third.

Next, the Bayes example combines evidence with the base rate. In the hypothetical population, there are ninety five true positives and four hundred ninety five false positives. That makes the probability of disease after a positive test about sixteen percent.

The classification example then turns raw logits into probabilities using softmax. The predicted Cat probability is about seventy one percent, and the one hot target produces cross entropy around zero point three four.

Entropy measures uncertainty, while cross entropy evaluates the probability assigned to the target.

My takeaway is that confidence, evidence, and correctness are related ideas, but they answer different questions.

## Screen cues — do not read aloud

- 0–20s: Begin near Chapter 1 dice grid, then Chapter 3 conditional card example.
- 20–42s: Show Chapter 3 true/false-positive population plot and Chapter 4 Bayes formula.
- 42–60s: Jump to opening complete classification pipeline; highlight softmax and saved loss.
- 60–71s: Show opening entropy comparison; close on pipeline.

**Recording note:** Show the saved executed outputs. This is a concise representative walkthrough; the study notes cover the remaining blocks.
