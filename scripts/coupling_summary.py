#!/usr/bin/env python3
"""Coupling between recruitment and dimensionality across populations.

Writes coupling_summary.csv: one row per scope.
  within-*  : one Spearman rho per population across 120 s windows
              (from window_rho_by_state.csv, produced by window_rho_by_state.py),
              summarised by a sign count and a signed-rank test.
  across-*  : one Spearman rho over populations, each contributing its
              whole-state mean active fraction and converted tail index
              (from multisession_50units_clauset.csv, notebook 07).

Also reports the smallest |rho| an across-population test could detect at
p < 0.05, so a null result is read against the power that produced it.
"""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon, t as tdist

NB = "/Users/ilgaz/Desktop/data_share_20260429/notebooks/"
STATES = ("Wake", "NREM", "REM")
MIN_WINDOWS = 10          # a within-state rho needs at least this many windows

P = pd.read_csv(NB + "multisession_50units_clauset.csv")

def detectable_rho(n, alpha=0.05):
    """Smallest |rho| significant at the two-sided alpha for this n."""
    if n < 4:
        return np.nan
    tc = tdist.ppf(1 - alpha / 2, n - 2)
    return float(tc / np.sqrt(n - 2 + tc ** 2))

rows = []

def partial_spearman(x, y, z):
    """Spearman of x and y with the linear effect of rank(z) removed."""
    rx, ry, rz = (pd.Series(v).rank() for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return spearmanr(ex, ey)

MEASURES = [("D_pr", "participation ratio"), ("D_90", "components for 90 % variance"),
            ("alpha_conv", "converted tail index")]
for q, lbl in MEASURES:
    for s in STATES:
        d = P[[f"s_{s}", f"{q}_{s}", "n_units"]].dropna()
        r = spearmanr(d[f"s_{s}"], d[f"{q}_{s}"])
        pr = partial_spearman(d[f"s_{s}"], d[f"{q}_{s}"], d.n_units)
        rows.append(dict(scope=f"across populations, {s}: recruitment vs {lbl}",
                         n=len(d), negative="",
                         statistic=round(float(r.statistic), 3),
                         p=round(float(r.pvalue), 3),
                         detectable_rho=round(detectable_rho(len(d)), 2),
                         note=f"partial rho controlling recorded units {pr.statistic:+.2f} (p={pr.pvalue:.3f})"))
# the same relation within each region, in every state: a sleep null could mean no
# relation anywhere or regions that disagree and cancel, and these rows separate the two
for s_ in STATES:
    for g, d in P.groupby("group"):
        dd = d[[f"s_{s_}", f"D_pr_{s_}"]].dropna()
        if len(dd) < 5:
            continue
        r = spearmanr(dd[f"s_{s_}"], dd[f"D_pr_{s_}"])
        rows.append(dict(scope=f"across populations, {s_}, {g} only", n=len(dd), negative="",
                         statistic=round(float(r.statistic), 3), p=round(float(r.pvalue), 3),
                         detectable_rho=round(detectable_rho(len(dd)), 2),
                         note="recruitment vs participation ratio, region held fixed"))

out = pd.DataFrame(rows)
out.to_csv(NB + "coupling_summary.csv", index=False)
print(out.to_string(index=False))
