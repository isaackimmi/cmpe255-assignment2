# 🎬 Foundations 3 — Probability — Video Script

**Target:** 65–75 seconds · **Spoken words:** 151 · **Estimated narration:** 72.5 seconds at 125 words/minute

**Recording approach:** A concise overview with representative executed blocks. The full section-by-section explanation is in the notebook notes. Pre-position these cells; do not wait for execution during recording.

## Screen cues — not spoken

- 0–13s: Conditional-count table or Bayes derivation.
- 13–32s: Natural-frequency tree, cells 27/29.
- 32–52s: Smoothing, log scores and sklearn comparison, cells 79/83/88.
- 52–72s: Confusion matrix and threshold sweep, cells 103/110.

## Spoken script

This notebook connects probability to the data box of machine learning. It begins with conditional probability and derives Bayes’ rule by writing the same joint probability in two ways.

The rare-event example is the clearest demonstration. With the displayed base rate, sensitivity, and false-positive rate, there are ninety-nine true positives but nearly five thousand false positives. The posterior is therefore about two percent, showing why sensitivity is not the probability we actually wanted.

The notebook also covers distributions and maximum likelihood before building Naive Bayes. Here, smoothing prevents unseen words from forcing a class score to zero, while log scores prevent long products from underflowing. The implementation agrees with scikit-learn.

Finally, I’m showing the confusion matrix and threshold sweep. Precision asks how many alarms are correct; recall asks how many positives were caught. My takeaway is to consider base rates, calibration, and the cost of mistakes instead of trusting accuracy alone.

## Recording check

Show the saved outputs at readable zoom. Briefly pause on the key result. Avoid reading every line of code or every section heading. This script is ready for the user to rehearse and record; no video has been recorded.
