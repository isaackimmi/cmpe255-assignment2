# Matplotlib — Zero to Hero

> **The big idea** · Learn to diagnose and repair misleading or unreadable plots by controlling Matplotlib objects, encodings, axes, layout, color, and export. One weather dataset is redesigned repeatedly.

## 🎯 Learning objectives

- Use Figure, Axes, Axis, and Artist objects through the explicit API.
- Choose chart types and make data defects, labels, scales, and comparisons legible.
- Use layout, colormaps, normalization, and redundant encoding intentionally.
- Export publication-sized figures and inspect actual pixels and metadata.

## 🧭 Conceptual map

Weather with deliberate defects → The object model → Choose the chart → Encoding and style → Axes control → Text and context → Layout → Color as data → Export → Redesign clinic

## 📚 Walkthrough of every major section

### 0 · Weather with deliberate defects

Generate 1,460 rows for four stations across 365 days, including 23 missing temperatures and two sensor outliers. An unlabeled default plot establishes the redesign problem.

### 1 · The object model

Identify Figure, Axes, Axis, spines, ticks, and artists. Compare implicit plt calls with explicit object methods, inspect returned Line2D objects, and demonstrate how the current figure can receive an unintended edit.

### 2 · Choose the chart

Match lines to ordered change, scatter to relationships, bars to categories, and histograms/boxplots to distributions. Show how connecting gaps, changing bins, or normalizing counts can change interpretation.

### 3 · Encoding and style

Control color cycles, line styles, markers, linewidth, alpha, and z-order. Use redundant encodings for grayscale or color-vision differences. Scope global styles and rcParams with context managers.

### 4 · Axes control

Inspect limits and autoscaling, choose locators/formatters for numbers and dates, and use log/symlog scales according to the data. Treat spines and grids as supporting elements. Manipulate twin axes to expose how arbitrary scales can imply a relationship.

### 5 · Text and context

Use meaningful titles, labels with units, explicit legend handles, direct labels, annotations, and reference marks. Compare data, axes-fraction, and figure-fraction coordinate systems; flag sensor gaps and outliers.

### 6 · Layout

Use shared axes, small multiples, GridSpec, insets, and figure-level decorations. Compare default spacing, constrained_layout, tight_layout, and manual adjustment, including measured bounding-box overlap.

### 7 · Color as data

Choose sequential, diverging, or categorical color schemes. Measure lightness behavior and simulate color-vision differences. Match Normalize, LogNorm, TwoSlopeNorm, and BoundaryNorm to meaning; center diverging scales on the semantic reference.

### 8 · Export

Verify pixels=inches*dpi, explore bounding-box cropping and text size, compare raster/vector formats and transparency, inspect embedded fonts and metadata, then read saved PNG dimensions back from disk. Temporary export files are removed by the notebook.

### 9 · Redesign clinic

Repair truncated bars, crowded series, twin-axis stories, near-equal pies, and incorrectly centered diverging maps. Finish with a weather dashboard using small multiples, uncertainty/context, explicit labeling, and an actual export.

## 💻 Important code, connected to its purpose

### Name the target Axes explicitly

Notebook cell **37** (one-based, including markdown).

```python
fig, ax = plt.subplots(figsize=(10, 4))

lines = ax.plot(wide.index, wide.values, lw=1.0)   # still a list, still 2-D input
for ln, name in zip(lines, wide.columns):
    ln.set_label(name)                              # one handle per station

```

### Give a diverging map its semantic midpoint

Notebook cell **182** (one-based, including markdown).

```python
im = heat(axes[0, 1], anom, "RdBu_r",
          norm=TwoSlopeNorm(vmin=anom.min(), vcenter=0.0, vmax=anom.max()),
          title="anomaly + DIVERGING, centred on 0 (RdBu_r)")
```

### Save an explicit Figure with controlled export settings

Notebook cell **251** (one-based, including markdown).

```python
fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
```

## 🔬 Actual results from this execution

The excerpts below are retained stdout from the fresh local run. Figures are embedded in the executed notebook.

### Evidence · notebook cell 5

```text
weather : 1,460 rows x 4 cols
wide    : 365 days x 4 stations
missing : 23 temperature readings
```

### Evidence · notebook cell 167

```text
measured overlap between panel bounding boxes (0 px = nothing collides)
  default                            worst panel-to-panel overlap:   28.9 px
  constrained_layout=True            worst panel-to-panel overlap:    0.0 px
  fig.tight_layout()                 worst panel-to-panel overlap:    0.0 px
  fig.subplots_adjust(...)           worst panel-to-panel overlap:    0.0 px

default overlap removed by constrained_layout: 28.9 px
default overlap removed by tight_layout      : 28.9 px
default overlap removed by subplots_adjust   : 28.9 px
```

### Evidence · notebook cell 223

```text
asked for : 3.25 x 2.4 in at 300 dpi = 975 x 720 px
got       : 975 x 720 px  (3.25 x 2.40 in at 300 dpi),  61 KB
match     : True
array     : shape (720, 975, 4), dtype float32, value range 0.00-1.00
fully transparent pixels: 0.0%
metadata  : Title        = Monthly mean temperature, two stations
metadata  : Author       = CMPE 258 matplotlib zero-to-hero
metadata  : Description  = Figure 1. Coastal and Alpine monthly means, 2024.
metadata  : Software     = matplotlib 3.11.2

scratch dir held 25 files, 18.0 MB total -- removed.
still exists: False
```

### Evidence · notebook cell 251

```text
figure size : 11.5 x 6.4 in
saved at    : dpi=200  ->  2300 x 1280 px nominal
file size   : 228 KB
temporary file cleaned up: True
```

## ⚠️ Interpretation and common mistakes

- Axes is the plotting panel; Axis is its ticks/scale object.
- Increasing DPI does not fix text that becomes too small after shrinking a physically large figure.
- A diverging palette needs a meaningful center, not just attractive colors.
- Twin y-axis overlap is controlled by arbitrary scale choices and does not establish a relationship.
- The color-vision simulation and sRGB-distance heuristic are educational approximations, not an accessibility certification.
- The export demo contains inherited CMPE 258 author metadata; that is source content, not a claim about this CMPE 255 assignment.

## ✅ Run record

- Executed **85/85 code cells** in a fresh local Python kernel with actual outputs retained.
- Inputs are the notebook's seeded synthetic datasets; no manual input or external downloads are required.
- Original exercise prompts remain as optional practice; prose exercises are not reported as completed code cells.
- Execution details, environment versions, adaptations, and output inventory are in `execution.json`; rerun with `python reproduce.py`.
- Compatibility adaptation: use an Agg renderer when a Figure canvas cannot provide `get_renderer`, so bounding-box overlap measurement works after display on a non-Agg inline canvas.

## 🎬 Recording plan

Use the linked video script for a representative 65–75 second walkthrough. The short video highlights the core idea and key evidence; these notes retain the broader notebook coverage.
