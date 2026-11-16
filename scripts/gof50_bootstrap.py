"""Goodness-of-fit for the matched 50-unit multi-session spectra.

For each population and state: average the 50-unit matched spectrum over 50
random subsamples, fit a power law above a KS-selected cutoff (powerlaw pkg),
and build a semi-parametric bootstrap p-value (each replicate refitted from
scratch, cutoff scan included).

Writes gof50_fits.csv and gof50_spectra.npz (the mean matched spectra, for the
cumulative-distribution panel).
"""
import sys, warnings, hashlib, numpy as np, pandas as pd, powerlaw
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/ilgaz/Desktop/data_share_20260429/scripts")
from dandi_data import load_pack

STATES, N_SUB, DRAWS, REPS, MIN_BINS, SEED = ("Wake", "NREM", "REM"), 50, 50, 250, 300, 1
OUT = "/Users/ilgaz/Desktop/data_share_20260429/notebooks/"


def spectrum(M):
    M = M - M.mean(0)
    ev = np.linalg.svd(M, compute_uv=False) ** 2
    return np.sort(ev / ev.sum())[::-1]


def population_rng(tag):
    h = int.from_bytes(hashlib.sha256(("pr50|" + tag).encode()).digest()[:4], "big")
    return np.random.default_rng(SEED ^ h)


def bootstrap_p(ev, pl, reps, rng):
    xmin, alpha, d_obs = pl.xmin, pl.alpha, pl.D
    ev = np.asarray(ev)
    body, n, frac = ev[ev < xmin], len(ev), pl.n / len(ev)
    worse = 0
    for _ in range(reps):
        k = int((rng.random(n) < frac).sum())
        tail = xmin * (1.0 - rng.random(k)) ** (-1.0 / (alpha - 1.0))
        rest = rng.choice(body, n - k, replace=True) if len(body) and n - k > 0 else np.array([])
        try:
            worse += powerlaw.Fit(np.concatenate([tail, rest]), discrete=False,
                                  verbose=False).power_law.D >= d_obs
        except Exception:
            pass
    return worse / reps


pops = [p for p in load_pack(min_units=N_SUB)
        if all((p["state"] == s).sum() >= MIN_BINS for s in STATES)]
print(f"{len(pops)} populations x {len(STATES)} states, {REPS} replicates each", flush=True)
rows, spectra = [], {}
for i, p in enumerate(pops, 1):
    rng = population_rng(f"{p['group']}|{p['tag']}")
    for s in STATES:
        M = p["X"][p["state"] == s]
        live = np.flatnonzero(M.sum(0) >= 10)
        if len(live) < N_SUB:
            continue
        ev = np.mean([spectrum(M[:, rng.choice(live, N_SUB, replace=False)])
                      for _ in range(DRAWS)], axis=0)
        pl = powerlaw.Fit(ev, discrete=False, verbose=False).power_law
        key = f"{p['group']}|{p['tag'].split('_ses-')[-1]}|{s}"
        spectra[key] = ev
        rows.append(dict(tag=p["tag"].split("_ses-")[-1], group=p["group"], state=s,
                         n_units=p["X"].shape[1], xmin=float(pl.xmin), alpha=float(pl.alpha),
                         alpha_conv=float(1.0 / (pl.alpha - 1.0)), n_tail=int(pl.n),
                         D=float(pl.D), p_boot=bootstrap_p(ev, pl, REPS, rng)))
    print(f"  {i}/{len(pops)}", flush=True)
pd.DataFrame(rows).round(4).to_csv(OUT + "gof50_fits.csv", index=False)
np.savez_compressed(OUT + "gof50_spectra.npz", **spectra)
print("DONE", len(rows), "fits", flush=True)
