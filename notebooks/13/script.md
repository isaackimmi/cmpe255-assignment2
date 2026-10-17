# 🎬 Video Script — Pandas — Zero to Hero

**Target:** 65–75 seconds · **Spoken words:** 153 · Read around 125–135 words per minute, with brief screen pauses.

## Spoken script

This notebook follows one messy coffee shop dataset from raw records to a useful business answer.

I begin with nine hundred fourteen rows. The inspection cells reveal duplicates, missing values, inconsistent text, and prices stored as strings. Looking only at the first few rows would hide those problems.

The cleaning pipeline removes duplicates and impossible quantities, converts prices, and handles missing values according to their meaning. Missing ratings stay missing, because an unanswered survey is not a zero rating.

I also compare label based selection with positional selection, then use groupby, reshaping, and validated joins. The duplicate key example is especially important: a left join can multiply rows and invent revenue.

The final capstone starts again from the raw data with an explicit policy, then calculates each region’s weekend product mix. Cold brew leads Central, while latte leads the other three regions.

My takeaway is to validate assumptions before trusting a polished summary.

## Screen cues — do not read aloud

- 0–16s: Show raw sales shape and missing-value inspection.
- 16–34s: Show Part 4 clean summary and retained missing ratings.
- 34–50s: Show loc versus iloc, then Part 7 duplicate-key merge and validation.
- 50–72s: Show final capstone chain and heatmap; point to regional winning products.

**Recording note:** Show the saved executed outputs. This is a concise representative walkthrough; the study notes cover the remaining blocks.
