#!/usr/bin/env python3
"""State contrasts for every population-level measure, paired within population.

Writes state_contrasts.csv: one row per (measure, state pair), giving the
direction count, the median paired difference with its interquartile range, and
a Wilcoxon signed-rank p. Every comparison is paired within a population, so a
population contributes one difference per pair; nothing is pooled before testing.

Reads multisession_50units_clauset.csv (notebook 07): 34 populations, each
subsampled to 50 units, with per-state recruitment and three summaries of the
eigenvalue spectrum.

Note on sign: alpha measures the steepness of the decay, so it runs OPPOSITE to
the two component counts. A positive Wake - NREM difference in alpha and a
negative one in D_pr are the same finding.
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

NB = "/Users/ilgaz/Desktop/data_share_20260429/notebooks/"
PAIRS = [("Wake", "NREM"), ("Wake", "REM"), ("NREM", "REM")]
MEASURES = [("alpha_conv", "converted tail index (steepness)"),
            ("D_pr", "participation ratio"),
            ("D_90", "components for 90 % of variance"),
            ("s", "mean active fraction")]

P = pd.read_csv(NB + "multisession_50units_clauset.csv")

rows = []
for q, lbl in MEASURES:
    for a, b in PAIRS:
        ca, cb = f"{q}_{a}", f"{q}_{b}"
        if ca not in P.columns or cb not in P.columns:
            continue
        d = (P[ca] - P[cb]).dropna()
        q1, q3 = d.quantile([0.25, 0.75])
        rows.append(dict(measure=lbl, contrast=f"{a} - {b}", n=len(d),
                         n_positive=int((d > 0).sum()),
                         median_difference=round(float(d.median()), 3),
                         iqr_low=round(float(q1), 3), iqr_high=round(float(q3), 3),
                         p_signed_rank=float(f"{wilcoxon(d).pvalue:.2e}")))

out = pd.DataFrame(rows)
out.to_csv(NB + "state_contrasts.csv", index=False)
print(out.to_string(index=False))

# the claim the Discussion makes about the sleep pair, as a ratio of margins
d90 = out[out.measure.str.startswith("components")].set_index("contrast")
ratio = abs(d90.loc["NREM - REM", "median_difference"] / d90.loc["Wake - NREM", "median_difference"])
print(f"\nsleep-pair margin as a fraction of the waking-to-non-REM margin "
      f"(components for 90 % of variance): {ratio:.2f}")
