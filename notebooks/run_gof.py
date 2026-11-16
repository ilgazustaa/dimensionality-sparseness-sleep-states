import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath("../scripts"))
from analysis_config import BIN, THRESH, N_SUB, N_DRAWS, SEED, FIT_RANGE_MATCHED, FIT_RANGE_FULL
import dandi_data
import powerlaw

STATES = ["Wake", "NREM", "REM"]
COLORS = {"Wake": "#D55E00", "NREM": "#0072B2", "REM": "#009E73"}
REGCOL = {"rat CA1": "#7B3294", "rat hippocampus": "#008837",
          "rat amygdala": "#E66101", "rat frontal cortex": "#0571B0"}
GOF_REPS = 200          # bootstrap replicates for the goodness-of-fit p
MIN_UNITS_GOF = 40      # a likelihood fit below this is not reported
MIN_BINS = 300          # a state with fewer bins than this is not fitted

mpl.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight"})

print("powerlaw", powerlaw.__version__, "| bin", BIN, "s | tonic threshold", THRESH, "Hz")
print("matched population", N_SUB, "units over PCs", FIT_RANGE_MATCHED,
      "| full population over PCs", FIT_RANGE_FULL)

# Sessions come from dandi_data.load_pack, the same loader the rest of the rat
# analysis uses: it applies the tonic-rate exclusion, maps Quiet to Wake, and
# splits dandiset 000061 into its hippocampal and amygdalar populations.
rows, DATA = [], {}
for s in dandi_data.load_pack(min_units=N_SUB):
    key = (s["tag"], s["group"])
    DATA[key] = s
    counts = {st: int((s["state"] == st).sum()) for st in STATES}
    rows.append(dict(tag=s["tag"], group=s["group"], dandiset=s["tag"].split("_")[0],
                     n_units=s["n_units"], n_bins=int(len(s["state"])), **{f"bins_{k}": v for k, v in counts.items()}))

inv = pd.DataFrame(rows).sort_values(["group", "tag"]).reset_index(drop=True)
inv["gof_eligible"] = inv.n_units >= MIN_UNITS_GOF
inv.to_csv("multisession_inventory.csv", index=False)

print("populations:", len(inv), "from", inv.tag.nunique(), "sessions")
print(inv.groupby("group").agg(n=("tag", "size"), units_median=("n_units", "median"),
                               gof_eligible=("gof_eligible", "sum")).to_string())
short = inv[(inv[[f"bins_{s}" for s in STATES]] < MIN_BINS).any(axis=1)]
print("\npopulations with a state under", MIN_BINS, "bins:", len(short))
print(inv[[f"bins_{s}" for s in STATES]].describe().loc[["min", "50%", "max"]].round(0).to_string())

import zlib
from analysis_config import MIN_SPIKES_STATE, effective_range

def spectrum(M):
    """Eigenvalue spectrum as a fraction of total variance (columns centred)."""
    M = M - M.mean(0)
    ev = np.linalg.svd(M, compute_uv=False) ** 2
    return ev / ev.sum()

def ols_exponent(ev, fit_range):
    """OLS of log(lambda_k) on log(k). Returns (alpha, R2, range actually used)."""
    lo, hi = effective_range(fit_range, len(ev))
    k = np.arange(lo, hi + 1)
    y = np.log(np.sort(ev)[::-1][lo - 1:hi])
    a, b = np.polyfit(np.log(k), y, 1)
    resid = y - (a * np.log(k) + b)
    return -a, float(1 - resid.var() / y.var()), (lo, hi)

def session_rng(tag, group):
    """Deterministic per-population stream.

    zlib.crc32 rather than hash(): Python salts string hashing per process, so
    hash() would give a different subsample on every run and the matched
    exponents would not reproduce.
    """
    return np.random.default_rng(zlib.crc32(f"{tag}|{group}|{SEED}".encode()))

rows, SPEC = [], {}
for (tag, group), s in DATA.items():
    rng = session_rng(tag, group)
    for st in STATES:
        m = s["state"] == st
        if m.sum() < MIN_BINS:
            continue
        Xs = s["X"][m]
        active = Xs.sum(0) >= MIN_SPIKES_STATE          # a unit must fire in the state fitted
        Xs = Xs[:, active]
        if Xs.shape[1] < N_SUB:
            continue
        ev_full = spectrum(Xs)
        a_full, r2_full, rng_full = ols_exponent(ev_full, FIT_RANGE_FULL)

        draws = [spectrum(Xs[:, np.sort(rng.choice(Xs.shape[1], N_SUB, replace=False))])
                 for _ in range(N_DRAWS)]
        ev_match = np.mean(draws, axis=0)
        a_match, r2_match, rng_match = ols_exponent(ev_match, FIT_RANGE_MATCHED)

        SPEC[(tag, group, st)] = dict(full=ev_full, matched=ev_match)
        rows.append(dict(tag=tag, group=group, state=st, n_bins=int(m.sum()),
                         n_units_state=int(active.sum()),
                         a_full=a_full, r2_full=r2_full,
                         fit_lo_full=rng_full[0], fit_hi_full=rng_full[1],
                         a_matched=a_match, r2_matched=r2_match,
                         active_frac=float((Xs > 0).mean())))

EXP = pd.DataFrame(rows)
EXP.to_csv("multisession_exponents.csv", index=False)
np.savez_compressed("multisession_spectra.npz",
                    **{f"{k[2]}__{k[1].replace(' ', '-')}__{k[0]}__{w}": v[w]
                       for k, v in SPEC.items() for w in ("full", "matched")})

print("spectra:", len(EXP), "session-state fits |", EXP.tag.nunique(), "sessions")
print(EXP.groupby("state")[["a_matched", "a_full", "r2_matched", "n_units_state"]].median().round(3).to_string())
print("\nfull-population fit range truncated below 3-50 in",
      int((EXP.fit_hi_full < FIT_RANGE_FULL[1]).sum()), "of", len(EXP), "fits",
      "| median effective hi:", int(EXP.fit_hi_full.median()))

# Distributional fit and model comparison, full-population spectra only.
# Everything here comes from the `powerlaw` package: Fit() scans candidate
# cutoffs, fits the tail index by maximum likelihood above each and keeps the
# one minimising the Kolmogorov-Smirnov distance; distribution_compare()
# returns Vuong's normalised likelihood ratio for the non-nested alternatives
# and the classical nested test for the truncated power law.
#
# The bootstrap p is the Clauset semi-parametric test: synthetic datasets are
# drawn from the fitted law, the body resampled from the observed sub-cutoff
# values, and the whole cutoff scan re-run on each. It is reported for
# completeness; on data of this kind it is conservative, because its replicates
# are independent draws and vary more than eigenvalues of one covariance
# matrix do.

ALTS = ["lognormal", "exponential", "truncated_power_law"]

def bootstrap_p(ev, fit, reps, rng):
    xmin, alpha, D = fit.power_law.xmin, fit.power_law.alpha, fit.power_law.D
    n, body = len(ev), ev[ev < xmin]
    frac = float((ev >= xmin).mean())
    worse = 0
    for _ in range(reps):
        k = int((rng.random(n) < frac).sum())
        tail = xmin * (1 - rng.random(k)) ** (-1 / (alpha - 1))
        rest = rng.choice(body, n - k, replace=True) if len(body) and n - k else np.empty(0)
        synth = np.concatenate([tail, rest])
        worse += powerlaw.Fit(synth, discrete=False, verbose=False).power_law.D >= D
    return worse / reps

def fit_one(ev, reps, rng):
    f = powerlaw.Fit(ev, discrete=False, verbose=False)
    pl = f.power_law
    out = dict(xmin=pl.xmin, alpha_mle=pl.alpha, alpha_se=pl.sigma, ks_D=pl.D,
               n_tail=int(pl.n), n_eig=len(ev))
    for alt in ALTS:
        R, p = f.distribution_compare("power_law", alt, normalized_ratio=True)
        out[f"R_{alt}"], out[f"p_{alt}"] = R, p
    out["p_boot"] = bootstrap_p(ev, f, reps, rng)
    return out

gof_rows = []
for i, ((tag, group, st), sp) in enumerate(SPEC.items(), 1):
    if i % 25 == 0:
        print(f"  {i}/{len(SPEC)}", flush=True)
    e = EXP[(EXP.tag == tag) & (EXP.group == group) & (EXP.state == st)].iloc[0]
    if e.n_units_state < MIN_UNITS_GOF:
        continue
    rec = dict(tag=tag, group=group, state=st, n_bins=int(e.n_bins),
               n_units=int(e.n_units_state), a_ols_full=e.a_full, a_ols_matched=e.a_matched)
    rec.update(fit_one(sp["full"], GOF_REPS, session_rng(tag, group)))
    rec["a_converted"] = 1.0 / (rec["alpha_mle"] - 1.0)
    gof_rows.append(rec)

GOF = pd.DataFrame(gof_rows)
GOF.to_csv("multisession_gof.tmp.csv", index=False)
os.replace("multisession_gof.tmp.csv", "multisession_gof.csv")
print(len(GOF), "fits |", GOF.tag.nunique(), "sessions")
print(GOF.groupby("state")[["alpha_mle", "ks_D", "p_boot", "n_tail"]].median().round(3).to_string())