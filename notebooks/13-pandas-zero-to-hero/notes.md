# 🐼 Pandas — Zero to Hero

> **Core idea:** A trustworthy answer comes from understanding labels, missingness, data types, and the grain of each operation—not merely calling `head()`.

## 🎯 Purpose & learning objectives

Follow one intentionally messy coffee-chain dataset from raw records to a defensible business answer. Learn Series/DataFrame alignment, file loading, inspection, selection, cleaning, vectorized transformations, grouped summaries, reshaping, joins, time series, plotting, and performance measurement.

## 🗺️ The full journey

| Stage | Main tools | Question to ask |
| --- | --- | --- |
| Data structures | Series, DataFrame, index, columns | Are values aligned by labels or positions? |
| Loading/inspection | `read_csv`, parquet, `info`, `describe`, missingness | Did the file preserve identifiers and dtypes? |
| Selection | `loc`, `iloc`, Boolean masks, `query`, `at`, `iat` | Am I selecting by label, position, or condition? |
| Cleaning | `copy`, duplicates, missing values, numeric conversion | What does each invalid or absent value mean? |
| Transformations | arithmetic, `.str`, `.dt`, `assign`, `map`, `cut` | Can I express the rule across whole columns? |
| GroupBy | `agg`, `transform`, `apply`, `dropna`, `observed` | Should output reduce rows or remain aligned? |
| Reshaping/combining | pivot, melt, stack/unstack, merge, concat | What is the key and expected row count? |
| Time series | resample, asfreq, rolling, shift, time zones | Does the calendar contain gaps or partial periods? |
| Performance/capstone | plots, categories, NumExpr, chunked reads | Is the result correct before I optimize it? |

## 🧹 Dataset & cleaning decisions

The generator makes **900 orders**, appends **14 duplicates**, and deliberately adds uppercase store names, padded product names, missing ratings/quantities/regions, currency strings, and six impossible negative quantities. The resulting raw frame has **914 rows and 9 columns**.

The teaching pipeline retains the original `sales`, builds `clean`, then `tidy`. Removing duplicate rows and negative quantities leaves **894 rows**. Missing quantities are imputed by store/product median with a product-level fallback; missing regions become `Unassigned`; missing ratings remain missing because no response is not a zero-star review. Later string cleanup collapses **10 store labels to 5** and **12 product labels to 6**.

**Important distinction:** The final capstone deliberately starts again from raw `sales`, fills known regions from the store mapping, and drops non-positive/missing quantities rather than imputing them. Consequently, its regional winners and shares can differ from earlier intermediate summaries. This is a documented analytical choice, not an execution inconsistency.

## 🔍 Important blocks & code

### 1. Labels are part of the computation

```python
by_rating = sales.sort_values("rating", ascending=False)
lab = srt.loc[1005:1010]
pos = srt.iloc[5:10]
```

Sorting moves rows and retains their original labels. `.loc` uses labels and includes the stop label; `.iloc` uses positions and excludes its stop. Series arithmetic aligns matching labels, and `concat(axis=1)` also aligns indexes. The demonstrated mismatched-index concatenation gives eight rows and twelve missing values when four complete rows were intended.

### 2. Filtering and assignment

```python
mask = (sales.rating > 4.5) & (sales.store == "Fresno")
flagged.loc[flagged["quantity"] > 10, "bulk"] = 1
```

Use parentheses around each condition and `&`/`|` rather than Python `and`/`or`. Chained assignment can target a temporary copy; one `.loc[rows, columns]` assignment states the intended mutation. The notebook intentionally catches several exceptions to demonstrate mistakes; those printed examples are expected teaching outputs, not failed notebook cells.

### 3. Missing values require a policy

```python
by_store_product = clean.groupby(["store", "product"], observed=True)["quantity"].transform("median")
by_product = clean.groupby("product", observed=True)["quantity"].transform("median")
step1 = clean["quantity"].fillna(by_store_product)
step2 = step1.fillna(by_product)
```

`transform` returns row-aligned group statistics, which makes filling possible without a merge. Raw missingness totals **108 values**; the clean frame has **61 missing ratings** and no missing quantities or regions. Blindly filling every gap with zero changes means and invents observations.

### 4. Choose the right grouped output

```python
by_store = tidy.groupby("store")["revenue"].sum()
a_transform = tidy.groupby("store")["revenue"].transform("sum")
```

`agg`/reductions produce one result per group; `transform` produces one per original row. `size` counts rows, `count` counts non-missing values. Default grouping drops missing keys; `dropna=False` preserves them. Categorical `observed` settings determine whether unused category combinations appear.

### 5. Validate joins and calendars

```python
weekly = daily.resample("W").sum()
monthly = daily.resample("ME").sum()
roll7 = daily.rolling(7).mean()
```

A duplicate key in a left join grows **894 rows into 1,080** and invents **21.7% apparent revenue**. The `many_to_one` validation raises the expected `MergeError`, exposing the problem. Aggregation can silently hide missing days; count observations as well as summing revenue. A seven-row rolling window differs from seven calendar days when the index has gaps, and centered windows can use future observations.

### 6. Capstone: weekend revenue mix

```python
.groupby(["region", "product"])["revenue"].sum()
.unstack("product")
.fillna(0.0)
.pipe(lambda w: w.div(w.sum(axis=1), axis=0) * 100)
.round(1)
```

After the preceding raw-data cleanup and weekend filter, this returns a four-region by six-product table. Each row is normalized within its own region, so it compares product mix rather than total regional size.

## 📊 Fresh execution results

All **90/90 code cells** across **280 total cells** executed successfully. The final winners are **Central: cold brew (24.3%)**, **Coast: latte (30.6%)**, **North: latte (42.4%)**, and **South: latte (24.2%)**. Rounded row totals are 100%.

The intermediate daily analysis reports weekday mean revenue **126.14** versus weekend mean **171.91**, a **36.3% lift**. This is descriptive association in generated data, not a causal claim.

The chunked CSV example processes **107,280 rows in 22 chunks**, agrees with the full-file revenue total **2,984,070.00**, and measures about **5% of full-frame resident memory** for the largest chunk. It does not measure total process peak memory. The vectorized discount rule is checked against `apply` and `iterrows`; timings depend on hardware and runtime.

## ⚠️ Pitfalls, run notes & practice

CSV does not store the pandas schema: leading-zero IDs, categories, dates, and nullable integers need explicit loading rules. Parquet preserves more type information. `errors="coerce"` can silently introduce missing values; audit them. `qcut` creates rank-based bins, while `cut` uses fixed boundaries. Month totals can reflect incomplete months, and time-zone localization differs from conversion.

Executed locally with pandas, NumPy, Matplotlib, PyArrow, and NumExpr. No source code changes were needed. Temporary files created by teaching examples are cleaned up. No external data or user input is required.

**Try next:** Explain why the South winner changes between the intermediate and final pipelines; repair a duplicated merge key; compare `rolling(7)` with `rolling("7D")`; recompute the capstone using a clearly stated missing-quantity policy.
