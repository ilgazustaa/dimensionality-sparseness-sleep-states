#!/usr/bin/env python3
"""Figure 4 and 5 panels: the multi-session replication and the coupling.

Run from the notebooks directory:

    python fig45_panels.py

Writes five PNGs into figures/. Titles carry no panel letters -- add those in
the LaTeX float so the figure file and the manuscript cannot disagree.

Where each number comes from
----------------------------
exponent and active fraction   rat_session_metrics.csv   (54 populations, the
                               set quoted in the text)
spectra curves                 recomputed from the data pack, seeded per
                               population so the draws are reproducible
goodness of fit                multisession_gof.csv      (35 populations with
                               >= 40 units; refitting costs ~45 min)
window coupling                recomputed from the pack, window stream seeded
                               independently of any fit

PAIRED_LINES draws one grey line per population on the two by-state panels.
Off by default: the medians and the count in the title carry the result, and
54 lines obscure the markers. Set it True to show the pairing.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                      # utils/ lives beside this file
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from utils.population_metrics import compute_pca
from dandi_data import load_pack

N_SUB, LO, HI, N_DRAWS, SEED, WIN = 25, 3, 20, 20, 1, 120
MIN_SPIKES_STATE = 10
STATES = ("Wake", "NREM", "REM")
COL = {"Wake": "#D55E00", "NREM": "#0072B2", "REM": "#009E73"}
PAIRED_LINES = False
EXAMPLE = "Achilles-10252013"
FIGS = "figures/"

mpl.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def rng_for(kind, tag):
    """Deterministic generator, independent per analysis kind and population.

    Seeding per tag rather than from one shared stream keeps a population's
    draws fixed when sessions are added or reordered; seeding per kind keeps
    the window analysis independent of the spectrum analysis, so changing one
    cannot silently move the other.
    """
    h = int.from_bytes(hashlib.sha256(f"{kind}|{tag}".encode()).digest()[:4], "big")
    return np.random.default_rng(SEED ^ h)


def ols(curve, lo=LO, hi=HI):
    """Exponent and intercept of log curve against log index over [lo, hi]."""
    k = np.arange(1, len(curve) + 1)
    b, a0 = np.polyfit(np.log(k[lo - 1:hi]), np.log(curve[lo - 1:hi]), 1)
    return -b, a0


def spectrum(M):
    M = M - M.mean(0)
    ev = np.linalg.svd(M, compute_uv=False) ** 2
    return ev / ev.sum()


def matched_spectrum(Xs, rng, draws=N_DRAWS):
    """Mean normalised spectrum over `draws` subsamples of N_SUB active units."""
    live = np.flatnonzero(Xs.sum(0) >= MIN_SPIKES_STATE)
    if len(live) < N_SUB:
        return None
    out = [spectrum(Xs[:, np.sort(rng.choice(live, N_SUB, replace=False))]) for _ in range(draws)]
    return np.mean(out, 0)


def alpha_matched(Xs, rng, draws=5):
    live = np.flatnonzero(Xs.sum(0) > 0)
    if len(live) < N_SUB or Xs.shape[0] < HI + 5:
        return np.nan
    vals = []
    for _ in range(draws):
        ev, slope = compute_pca(Xs[:, rng.choice(live, N_SUB, replace=False)], fit_range=(LO, HI))
        if np.all(np.asarray(ev)[LO - 1:HI] > 0) and np.isfinite(slope):
            vals.append(slope)
    return float(np.mean(vals)) if vals else np.nan


def load_spectra():
    """Full and matched spectra per population and state, from the data pack."""
    full, matched = {}, {}
    for s in load_pack(min_units=N_SUB):
        g, tg, X, lab = s["group"], s["tag"], s["X"], s["state"]
        rg = rng_for("spec", f"{g}|{tg}")
        for st in STATES:
            mk = lab == st
            if mk.sum() <= 25:
                continue
            Xs = X[mk][:, X[mk].sum(0) >= MIN_SPIKES_STATE]
            if Xs.shape[1] < N_SUB:
                continue
            full[(g, tg, st)] = spectrum(Xs)
            ms = matched_spectrum(X[mk], rg)
            if ms is not None:
                matched[(g, tg, st)] = ms
    return full, matched


def by_state_panel(values, ylabel, title, fname, fmt="{:.2f}"):
    """Three states on a categorical axis: median, interquartile range, pairing."""
    fig, ax = plt.subplots(figsize=(4.2, 3.5), constrained_layout=True)
    x = np.arange(3)
    if PAIRED_LINES:
        for row in values:
            ax.plot(x, row, "-", color="0.75", lw=.5, alpha=.55, zorder=1)
    for i, st in enumerate(STATES):
        v = values[:, i]
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.errorbar(i, med, yerr=[[med - q1], [q3 - med]], fmt="o", ms=7.5, color=COL[st],
                    capsize=4, lw=1.8, zorder=3, mec="white", mew=.8)
        ax.annotate(fmt.format(med), (i, med), textcoords="offset points", xytext=(13, -2),
                    fontsize=7.5, color=COL[st])
    ax.set_xticks(x)
    ax.set_xticklabels(STATES)
    ax.set_xlim(-.45, 2.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGS + fname)
    plt.close(fig)


def spectra_panel(matched, fname="fig4_spectra_by_state.png"):
    common = sorted({(g, tg) for (g, tg, st) in matched
                     if all((g, tg, s2) in matched for s2 in STATES)})
    kk = np.arange(1, N_SUB + 1)
    fig, ax = plt.subplots(figsize=(4.4, 3.6), constrained_layout=True)
    ax.axvspan(LO, HI, color="0.92", zorder=0)
    for st in STATES:
        med = np.median([matched[(g, tg, st)] for (g, tg) in common], 0)
        a_, i_ = ols(med)
        ax.loglog(kk, med, ls="none", marker="o", ms=4, color=COL[st], zorder=3)
        xf = np.linspace(LO, HI, 40)
        ax.loglog(xf, np.exp(i_) * xf ** (-a_), "-", lw=1.8, color=COL[st], zorder=2,
                  label=f"{st}   $\\alpha$ = {a_:.2f}")
    ax.set_xlabel("component index $k$")
    ax.set_ylabel("eigenvalue (fraction of variance)")
    ax.set_title(f"Median spectrum, {len(common)} populations", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGS + fname)
    plt.close(fig)


def gof_panel(full, gof, fname="fig4_goodness_of_fit.png"):
    """Every population's CCDF, with the median fitted power law per state.

    The thick line is not a fit to the thin ones: it is a power law drawn with
    the median cutoff and median tail index of that state's separate fits.
    Legends give the tail index and its rank-exponent equivalent 1/(a-1).
    """
    dash = {"Wake": (None, None), "NREM": (6, 2), "REM": (None, None)}
    fig, ax = plt.subplots(figsize=(4.4, 3.6), constrained_layout=True)
    for st in STATES:
        sub = gof[gof.state == st]
        for _, r in sub.iterrows():
            key = (r.group, r.tag, st)
            if key not in full:
                continue
            ev = np.sort(full[key])[::-1]
            ax.loglog(ev, np.arange(1, len(ev) + 1) / len(ev), "-", lw=.5,
                      color=COL[st], alpha=.18, zorder=2)
        xm, al = sub.xmin.median(), sub.alpha_mle.median()
        fx = np.geomspace(xm, 0.4, 100)
        ax.loglog(fx, (sub.n_tail.median() / sub.n_eig.median()) * (fx / xm) ** (1 - al), "-",
                  lw=2.2, color=COL[st], dashes=dash[st], zorder=3,
                  label=f"{st}   $\\hat{{\\alpha}}$ = {al:.2f}  ($\\alpha$ = {1/(al-1):.2f}),"
                        f"  D = {sub.ks_D.median():.3f}")
    ax.set_xlabel("eigenvalue $\\lambda$  (fraction of total variance)")
    ax.set_ylabel("fraction of eigenvalues $\\geq \\lambda$")
    ax.set_title("Fits above each population's own cutoff", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=6.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGS + fname)
    plt.close(fig)


def window_panel(fname="fig5_windows_example.png"):
    """Active fraction against exponent across 120 s windows of one recording.

    Windows are contiguous and must lie entirely within scored states. The
    active fraction uses the session's fixed population; the exponent uses a
    matched subsample, so the two are not computed on the same units.
    """
    sess = [s for s in load_pack(min_units=N_SUB) if EXAMPLE in s["tag"]][0]
    g, tg, X, lab = sess["group"], sess["tag"], sess["X"], sess["state"]
    rg = rng_for("win", f"{g}|{tg}")
    av, al, wst = [], [], []
    for i in range(0, X.shape[0] - WIN, WIN):
        sl = slice(i, i + WIN)
        if not np.isin(lab[sl], STATES).all():
            continue
        a = alpha_matched(X[sl], rg)
        if np.isfinite(a):
            av.append(float((X[sl] > 0).mean()))
            al.append(a)
            u, c = np.unique(lab[sl], return_counts=True)
            wst.append(u[c.argmax()])
    av, al, wst = np.array(av), np.array(al), np.array(wst)
    rho, p = spearmanr(av, al)

    fig, ax = plt.subplots(figsize=(4.6, 3.7), constrained_layout=True)
    for st in STATES:
        mk = wst == st
        if mk.sum() >= 15:
            r = spearmanr(av[mk], al[mk])
            lb = f"{st}   $\\rho$ = {r.statistic:+.2f}" + ("" if r.pvalue < .05 else " (n.s.)")
        else:
            lb = f"{st} (n = {mk.sum()})"
        ax.plot(av[mk], al[mk], "o", ms=4.2, mfc=COL[st], mec="white", mew=.5, alpha=.85, label=lb)
    b_, a0 = np.polyfit(av, al, 1)
    xs = np.linspace(av.min(), av.max(), 50)
    ax.plot(xs, a0 + b_ * xs, "-", color="0.35", lw=1.6, zorder=1)
    ax.text(0.97, 0.95, f"all windows   $\\rho$ = {rho:.2f}\nn = {len(av)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, linespacing=1.4)
    ax.set_xlabel("active fraction in window,  $\\langle s_t \\rangle$")
    ax.set_ylabel(f"exponent $\\alpha$  ({N_SUB} units, PCs {LO}-{HI})")
    ax.set_title(f"{EXAMPLE}, {WIN} s windows", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=7, handletextpad=.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGS + fname)
    plt.close(fig)
    return len(av), rho, p


def main():
    os.chdir(HERE)                            # tables and figures/ are relative to this file
    os.makedirs(FIGS, exist_ok=True)
    M = pd.read_csv("rat_session_metrics.csv")
    keep = M[[f"active_{s}" for s in STATES] + [f"a_ols_{s}" for s in STATES]].notna().all(1)
    M = M[keep]

    A = M[[f"a_ols_{s}" for s in STATES]].values
    n_wn = int((M.a_ols_Wake > M.a_ols_NREM).sum())
    by_state_panel(A, f"exponent $\\alpha$  ({N_SUB} units, PCs {LO}-{HI})",
                   f"Waking decays faster than non-REM in {n_wn} of {len(M)} populations",
                   "fig4_exponent_by_state.png")

    S = M[[f"active_{s}" for s in STATES]].values
    n_nw = int((M.active_NREM > M.active_Wake).sum())
    by_state_panel(S, "active fraction per bin,  $\\langle s_t \\rangle$",
                   f"Non-REM exceeds waking in {n_nw} of {len(M)} populations",
                   "fig4_sparseness_by_state.png")

    full, matched = load_spectra()
    spectra_panel(matched)
    gof_panel(full, pd.read_csv("multisession_gof.csv"))
    n_win, rho, p = window_panel()

    print(f"populations: {len(M)} | exponent Wake > NREM {n_wn} | recruitment NREM > Wake {n_nw}")
    print(f"example windows: {n_win}, rho {rho:.3f}, p {p:.1e}")
    print("written:", sorted(f for f in os.listdir(FIGS) if f.startswith(("fig4_", "fig5_"))))


if __name__ == "__main__":
    main()
