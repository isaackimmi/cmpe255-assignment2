# Reproducing the assignment notebooks

The root README is intentionally empty at the owner's request.

Each numbered, descriptively named directory contains the user's notebook copy, freshly executed code and saved outputs, study notes, a short spoken video script with screen cues, and an execution record. `manifest.json` maps each number to its title, instructor source, and user-owned Colab copy. All examples use generated or embedded data. No private datasets, credentials, GPU, or external data downloads are required. Notebook 01's two input prompts use the explicit sample `Machine learning`, printed in its outputs.

## Run locally

Use Python 3.12 and create an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python execute.py
```

Run one notebook or a subset with `python execute.py 1 12 16`. Each notebook starts in a fresh kernel using the active interpreter, executes in its own directory, and saves outputs only after a successful run. The runner stops on an unexpected error. Timings, random framework demonstrations, floating-point differences, and font rendering can vary by platform. The requirements record the environment used for this archive, including PyTorch and PyArrow. Some wheel versions depend on the platform.

## Inputs, outputs, and corrections

Every nonempty code cell was executed. Printed exceptions that are deliberately caught to illustrate incorrect operations are teaching results. Static plots are embedded in the `.ipynb` files; temporary export and file-format demonstrations clean up their scratch files. Re-running recreates them. Per-notebook `execution.json` and study notes document actual results and the small input/compatibility corrections in notebooks 01, 02, 06, 08, and 11.

Notebook 14 retains the supplied source's unusual chapter order and ends with the Chapter 5 introduction. Its available code all executes; no absent chapter material was invented. Optional prose exercises are study prompts, not claims of completed exercises.

## Recording

Each `script.md` separates spoken narration from screen directions. The sixteen spoken scripts contain 2,434 words in total, approximately 18.7 minutes at 130 words per minute. Screen changes and pauses add time. These short walkthroughs highlight the principal examples; `notes.md` provides the broader explanation across the notebook's major sections. Review the current Canvas rubric before recording if it requires a longer block-by-block treatment.
