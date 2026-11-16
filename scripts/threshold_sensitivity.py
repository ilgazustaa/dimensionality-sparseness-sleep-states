"""Does the multi-recording result depend on the 4.71 Hz exclusion threshold?

The threshold is derived from a gap in ONE recording's rate distribution and
then applied to every recording. This re-runs the 50-unit analysis at several
thresholds, including no exclusion at all, and reports whether the state
orderings survive.
"""
import zlib
import numpy as np, pandas as pd, powerlaw, warnings
warnings.filterwarnings("ignore")

D = "/Users/ilgaz/Desktop/data_share_20260429/data/"
STATES = ("Wake", "NREM", "REM")
AMY = {"bla", "blv", "bmp", "cen", "ladl", "den", "ven"}
N_SUB, DRAWS, MIN_BINS = 50, 20, 300
z = np.load(D + "dandi_binned_1s_rat.npz", allow_pickle=True)
tags = sorted({k.split("__", 1)[1] for k in z.files if k.startswith("X__")})

def spectrum(M):
    Mc = M - M.mean(0)
    ev = np.linalg.svd(Mc, compute_uv=False) ** 2
    return np.sort(ev / ev.sum())[::-1]

def pops_at(th):
    for t in tags:
        ds = t.split("_")[0]
        reg = np.array([str(x).lower() for x in z[f"region__{t}"]])
        rate = z[f"rate__{t}"]
        groups = ([("ctx", np.ones(len(reg), bool))] if ds == "000041" else
                  [("ca1", np.ones(len(reg), bool))] if ds == "000044" else
                  [("hpc", reg == "hpc"), ("amy", np.isin(reg, list(AMY)))])
        lab = z[f"state__{t}"].copy(); lab[lab == "Quiet"] = "Wake"
        for nm, g in groups:
            sel = g & (rate <= th)
            if sel.sum() < N_SUB:
                continue
            X = z[f"X__{t}"][:, sel].astype(float)
            if any((lab == s).sum() < MIN_BINS for s in STATES):
                continue
            yield t, nm, X, lab

rows = []
for th in (2.5, 4.71, 8.0, 1e9):
    for t, nm, X, lab in pops_at(th):
        # zlib.crc32, not hash(): Python salts string hashing per process, so hash()
        # draws a different subsample on every run and the table does not reproduce
        rng = np.random.default_rng(zlib.crc32(f"{nm}|{t}".encode()))
        rec = dict(threshold=th, tag=t.split("_ses-")[-1], group=nm, n_units=X.shape[1])
        ok = True
        for s in STATES:
            M = X[lab == s]
            live = np.flatnonzero(M.sum(0) >= 10)
            if len(live) < N_SUB:
                ok = False; break
            pr, al, d90, sp = [], [], [], float((M > 0).mean())
            for _ in range(DRAWS):
                ev = spectrum(M[:, rng.choice(live, N_SUB, replace=False)])
                pr.append(float(ev.sum() ** 2 / (ev ** 2).sum()))
                # components carrying 90 % of the variance, the third dimensionality
                # measure the main analysis reports; untested here before 12 Nov
                d90.append(int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.90) + 1))
                f = powerlaw.Fit(ev, discrete=False, verbose=False)
                al.append(1.0 / (f.power_law.alpha - 1.0))
            rec[f"D_pr_{s}"], rec[f"a_{s}"], rec[f"s_{s}"] = np.mean(pr), np.mean(al), sp
            rec[f"D_90_{s}"] = float(np.mean(d90))
        if ok:
            rows.append(rec)
    print(f"threshold {th}: {sum(r['threshold'] == th for r in rows)} populations", flush=True)
P = pd.DataFrame(rows)
P.to_csv(D + "threshold_sensitivity.csv", index=False)
for th, sub in P.groupby("threshold"):
    print(f"\nthreshold {th} Hz | {len(sub)} populations | median units {int(sub.n_units.median())}")
    for q, nm in (("a", "exponent"), ("D_pr", "participation ratio"), ("D_90", "components 90 %"), ("s", "active fraction")):
        w_n = int((sub[f"{q}_Wake"] > sub[f"{q}_NREM"]).sum())
        n_r = int((sub[f"{q}_NREM"] > sub[f"{q}_REM"]).sum())
        print(f"   {nm:20s} Wake>NREM {w_n:2d}/{len(sub)} | NREM>REM {n_r:2d}/{len(sub)} | "
              f"medians {sub[f'{q}_Wake'].median():.3f} / {sub[f'{q}_NREM'].median():.3f} / {sub[f'{q}_REM'].median():.3f}")
print("DONE", flush=True)
