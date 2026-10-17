# Video script · Intro to Probability — Zero to Hero for ML

**Target:** 65–75 seconds · **Spoken words:** 153 · **Delivery:** approximately 131 words/minute for 70 seconds.

## Spoken script

In this notebook, I build probability around a simulated inbox containing twenty thousand emails. I begin by counting events, then combine them with complements, unions, and intersections. Conditional probability changes the denominator: I first keep only the emails matching the condition.

Bayes' theorem reverses that conditioning using the prior and likelihood. The rare-event example is useful because it shows why a strong detector can still produce many false alarms when the underlying event is uncommon.

The notebook then connects random variables, expectation, variance, and named distributions to machine learning. Here, I estimate a Naive Bayes spam filter from fifteen thousand training emails and evaluate it on five thousand unseen emails. Its accuracy is about eighty-nine point six percent, above the majority-class baseline.

Finally, I compare calibration. Duplicating one word barely changes accuracy, but substantially worsens probability estimates. That is my main takeaway: predicting the right label and reporting trustworthy confidence are different tasks.

## Screen cues — do not read aloud

- 0–15s: cells 4 and 68, inbox and conditional counts.
- 15–30s: cell 105, prior/likelihood/posterior arithmetic.
- 30–52s: cells 269–270, training estimates and confusion matrix.
- 52–73s: cell 278, reliability diagram and calibration error.

## Recording reminder

Open the executed notebook in advance and rehearse the jumps. Read only the spoken script; screen cues and notes are supporting material. Video recording and upload are still to be done by you.
