# Video script · Intro to Linear Algebra — Zero to Hero for ML

**Target:** 65–75 seconds · **Spoken words:** 153 · **Delivery:** approximately 131 words/minute for 70 seconds.

## Spoken script

In this notebook, I use two measurements from two hundred simulated flowers to make linear algebra concrete. One flower is a vector, and stacking all the flowers gives me the data matrix. I check shapes carefully because a flat array, a row, and a column can behave differently.

The dot product connects arithmetic to geometry: it measures alignment, and dividing by vector lengths gives cosine similarity. Matrix multiplication then combines those dot products to transform every observation. Here, the batch forward pass replaces a Python loop with one matrix operation.

The final example builds principal component analysis from scratch. I subtract the feature means, compute covariance, find its eigenvectors, and project onto the direction with the greatest variance. That first component retains about seventy-five percent of the variation.

I also reconstruct the flowers and measure the lost information. Independent SVD and scikit-learn checks agree, which gives me evidence that the implementation is correct.

## Screen cues — do not read aloud

- 0–15s: cell 4, data matrix and flower plot.
- 15–35s: dot product/projection diagrams and cell 249 batch code.
- 35–55s: cells 256–263, centered PCA directions.
- 55–73s: cells 266 and 270, reconstruction and independent checks.

## Recording reminder

Open the executed notebook in advance and rehearse the jumps. Read only the spoken script; screen cues and notes are supporting material. Video recording and upload are still to be done by you.
