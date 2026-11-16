"""Bootstrap p for the 50-unit fits, averaged over draws.

Matches the convention of the exponent panel: every quantity is computed on each
subsample and then averaged within a population, rather than computed once on an
averaged spectrum. The bootstrap is the expensive part, so it runs on the first
N_P of the 50 draws (same stream, so the draws are a subset of the ones the
exponent table uses).

Writes gof50_perdraw_p.csv: tag, group, state, p_boot (mean over draws).
"""
import sys, warnings, hashlib, numpy as np, pandas as pd, powerlaw
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/ilgaz/Desktop/data_share_20260429/scripts")
from dandi_data import load_pack

STATES, N_SUB, DRAWS, N_P, REPS, MIN_BINS, SEED = ("Wake", "NREM", "REM"), 50, 50, 10, 200, 300, 1
OUT = "/Users/ilgaz/Desktop/data_share_20260429/notebooks/gof50_perdraw_p.csv"


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
print(f"{len(pops)} populations, p averaged over the first {N_P} of {DRAWS} draws, "
      f"{REPS} replicates each", flush=True)
rows = []
for i, p in enumerate(pops, 1):
    rng = population_rng(f"{p['group']}|{p['tag']}")
    for s in STATES:
        M = p["X"][p["state"] == s]
        live = np.flatnonzero(M.sum(0) >= 10)
        if len(live) < N_SUB:
            continue
        ps = []
        for d in range(DRAWS):                      # consume the full stream, fit the first N_P
            ev = spectrum(M[:, rng.choice(live, N_SUB, replace=False)])
            if d < N_P:
                pl = powerlaw.Fit(ev, discrete=False, verbose=False).power_law
                ps.append(bootstrap_p(ev, pl, REPS, rng))
        rows.append(dict(tag=p["tag"].split("_ses-")[-1], group=p["group"], state=s,
                         p_boot=float(np.mean(ps)), p_sd=float(np.std(ps)), n_draws=len(ps)))
    print(f"  {i}/{len(pops)}", flush=True)
pd.DataFrame(rows).round(4).to_csv(OUT, index=False)
print("DONE", len(rows), flush=True)
