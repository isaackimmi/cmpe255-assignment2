# 🎬 Foundations 2 — Linear Algebra — Video Script

**Target:** 65–75 seconds · **Spoken words:** 153 · **Estimated narration:** 73.4 seconds at 125 words/minute

**Recording approach:** A concise overview with representative executed blocks. The full section-by-section explanation is in the notebook notes. Pre-position these cells; do not wait for execution during recording.

## Screen cues — not spoken

- 0–13s: Matrix columns and basis output, cell 8.
- 13–27s: Transformation grids, cell 13.
- 27–42s: Dot/cosine table, cell 28.
- 42–55s: Rank/collapse example, cell 40 or 49.
- 55–73s: PCA table and synthetic cloud, cells 75/76.

## Spoken script

This notebook opens the model box using linear algebra. Its central idea is that a matrix moves space, and its columns tell me where the basis vectors land.

Here, multiplying the matrix by a vector agrees with taking a weighted sum of its columns. The grid plots make scaling, rotation, shear, and collapse visible.

Next, the cosine example separates direction from magnitude. A longer vector gets a larger dot product even though another vector matches the query direction perfectly. Dividing by the vector lengths gives cosine similarity.

The notebook then connects rank, determinants, null spaces, and invertibility. When different inputs collapse onto one output, an inverse cannot uniquely recover the original input.

I’m finishing with PCA on the synthetic data cloud. The covariance eigenvectors identify its principal directions, and the first component captures about ninety-six percent of the variance. The handwritten calculation matches scikit-learn, connecting matrix geometry to a practical dimensionality reduction method.

## Recording check

Show the saved outputs at readable zoom. Briefly pause on the key result. Avoid reading every line of code or every section heading. This script is ready for the user to rehearse and record; no video has been recorded.
