# Video script · Matplotlib — Zero to Hero

**Target:** 65–75 seconds · **Spoken words:** 153 · **Delivery:** approximately 135 words/minute for 70 seconds.

## Spoken script

In this notebook, I improve a weather visualization step by step. The dataset has four stations, missing readings, and sensor outliers, so a default plot hides several problems.

I start with the object model. The Figure contains the output, each Axes is a plotting panel, and lines and labels are artists I can change. Using explicit object methods makes it clear which panel receives each edit.

Next, I choose chart types and encodings for the question. Lines show change over time, while small multiples make station comparisons easier. I label units, mark gaps and outliers, and control axes rather than accepting every default.

Color also carries meaning. A diverging map needs its neutral color at a meaningful reference, and redundant line styles help when colors are difficult to distinguish.

Finally, I verify the saved figure's pixel dimensions. My takeaway is that a good plot makes its evidence readable and its design choices intentional.

## Screen cues — do not read aloud

- 0–15s: cells 4 and 7, dataset defects and default plot.
- 15–35s: cells 12 and 36, Figure/Axes objects and explicit code.
- 35–55s: small multiples, centered colormap, annotations.
- 55–74s: cells 222 and 250, verified export and capstone.

## Recording reminder

Open the executed notebook in advance and rehearse the jumps. Read only the spoken script; screen cues and notes are supporting material. Video recording and upload are still to be done by you.
