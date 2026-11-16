"""
population_metrics_extra.py
===========================

Diagnostics and figures. The two functions you cite in the thesis live in
population_metrics.py — these are the longer versions that additionally
report:

    compute_pca_detailed(X)         local_slope, r2, slope_se, participation
                                    ratio, dim_80/dim_90, optional z-scoring
    compute_sparseness_detailed(X)  Treves-Rolls sparseness + a summary dict
    plot_scree(res)                 3-panel scree figure
    plot_sparseness(sp, X=...)      4-panel sparseness figure
    coarsen(X, k)                   pool k adjacent time bins

`local_slope` is the one to reach for routinely: it is the running log-log
slope per PC rank, and the fitted alpha only means something over the ranks
where that curve is flat.
"""

import numpy as np

__all__ = [
    "compute_pca_detailed", "compute_sparseness_detailed",
    "coarsen", "plot_scree", "plot_sparseness",
]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _as_numpy(X):
    """numpy array from numpy array or torch tensor, float64, 2-D."""
    if hasattr(X, "detach"):                     # torch.Tensor
        X = X.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_bins, n_neurons), got shape {X.shape}")
    return X


def coarsen(X, k):
    """Sum k adjacent time bins -> activity matrix at k x the original bin size.

    Trailing bins that do not fill a full block are dropped.
    """
    X = _as_numpy(X)
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return X
    T = (X.shape[0] // k) * k
    return X[:T].reshape(T // k, k, X.shape[1]).sum(axis=1)


# ----------------------------------------------------------------------
# 1. PCA / eigenvalue spectrum
# ----------------------------------------------------------------------
def compute_pca_detailed(X, zscore=False, fit_range=None, tol_rel=1e-10,
                return_scores=False, smooth_local=3):
    """Eigenspectrum of the neuron x neuron covariance of X, with a power-law fit.

    Parameters
    ----------
    X : (n_bins, n_neurons) array_like
        Counts or rates. Columns are mean-centered (always).
    zscore : bool
        If True, divide each neuron by its SD after centering, i.e. eigen-
        decompose the CORRELATION matrix. Use this when you do not want the
        spectrum dominated by the few highest-rate cells; leave False when
        rate heterogeneity is part of the effect you are studying.
        Constant (silent) neurons are dropped in this case.
    fit_range : (n_min, n_max) or None
        Inclusive PC-rank window (1-based) used for the log-log fit.
        None -> all ranks with lambda_n > tol_rel * lambda_1, which is the
        numerically meaningful part of the spectrum. Fix this window
        explicitly when comparing spectra across conditions with different
        neuron counts, otherwise the fit windows differ between conditions.
    tol_rel : float
        Relative eigenvalue floor defining the usable rank.
    return_scores : bool
        Also return the projections of X onto the PCs (n_bins x rank).
    smooth_local : int
        Window (in ranks) for the local-slope estimate; >=2, odd is natural.

    Returns
    -------
    dict with keys
        eigvals      (n_kept,)  descending eigenvalues (variance per PC)
        eigvecs      (n_kept, n_kept)  columns = PCs, matching eigvals order
        evr, cum_evr fraction / cumulative fraction of variance explained
        rank         number of eigenvalues above the tolerance floor
        slope        alpha in lambda_n ~ n^(-alpha)  (positive)
        slope_se     standard error of alpha from the OLS fit
        intercept    log10 lambda at n = 1 from the fit
        r2           R^2 of the log-log fit
        fit_ranks    the ranks actually used
        local_slope  (n_kept,) -d log lambda / d log n, running estimate
        pr           participation ratio (sum l)^2 / sum l^2
        dim_80/90    #PCs needed for 80 % / 90 % of the variance
        n_bins, n_neurons, zscore, scores (optional)

    Notes
    -----
    Computed by SVD of the centered data matrix rather than by eigen-
    decomposition of X^T X: the small eigenvalues that set the tail of the
    power law are obtained to full precision instead of being squared into
    the noise floor.
    """
    X = _as_numpy(X)
    n_bins, n_neurons = X.shape
    if n_bins < 2:
        raise ValueError("need at least 2 time bins")

    Xc = X - X.mean(axis=0, keepdims=True)
    if zscore:
        sd = Xc.std(axis=0, ddof=1)
        keep = sd > 0
        Xc, sd = Xc[:, keep], sd[keep]
        Xc = Xc / sd
    else:
        keep = np.ones(n_neurons, dtype=bool)

    # eigenvalues of the covariance via SVD:  lambda_i = s_i^2 / (T - 1)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    eigvals = s ** 2 / (n_bins - 1)
    _, V = np.linalg.eigh(Xc.T @ Xc / (n_bins - 1))
    eigvecs = V[:, ::-1]                                  # descending order

    total = eigvals.sum()
    evr = eigvals / total
    cum = np.cumsum(evr)
    rank = int(np.sum(eigvals > tol_rel * eigvals[0]))

    # ---- power-law fit  log10 lambda = c - alpha * log10 n
    n = np.arange(1, len(eigvals) + 1)
    if fit_range is None:
        n_min, n_max = 1, rank
    else:
        n_min, n_max = int(fit_range[0]), int(min(fit_range[1], rank))
    m = (n >= n_min) & (n <= n_max) & (eigvals > 0)
    x, y = np.log10(n[m]), np.log10(eigvals[m])
    if m.sum() < 3:
        slope = slope_se = intercept = r2 = np.nan
    else:
        A = np.vstack([x, np.ones_like(x)]).T
        beta, res_ss, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = -beta[0], beta[1]
        yhat = A @ beta
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        dof = len(x) - 2
        sxx = float(np.sum((x - x.mean()) ** 2))
        slope_se = float(np.sqrt(ss_res / dof / sxx)) if dof > 0 and sxx > 0 else np.nan

    # ---- local slope: running OLS in log-log over a window of ranks
    w = max(2, int(smooth_local))
    ln, ll = np.log10(n[:rank]), np.log10(np.maximum(eigvals[:rank], 1e-300))
    local = np.full(len(eigvals), np.nan)
    half = w // 2
    for i in range(rank):
        lo, hi = max(0, i - half), min(rank, i + half + 1)
        if hi - lo >= 2:
            local[i] = -np.polyfit(ln[lo:hi], ll[lo:hi], 1)[0]

    out = dict(
        eigvals=eigvals, eigvecs=eigvecs, evr=evr, cum_evr=cum, rank=rank,
        slope=float(slope), slope_se=float(slope_se), intercept=float(intercept),
        r2=float(r2), fit_ranks=(n_min, n_max), local_slope=local,
        pr=float(total ** 2 / np.sum(eigvals ** 2)),
        dim_80=int(np.searchsorted(cum, 0.80) + 1),
        dim_90=int(np.searchsorted(cum, 0.90) + 1),
        n_bins=n_bins, n_neurons=n_neurons, n_used=int(keep.sum()),
        zscore=bool(zscore), neurons_kept=keep,
    )
    if return_scores:
        out["scores"] = Xc @ eigvecs
    return out


# ----------------------------------------------------------------------
# 2. Sparseness
# ----------------------------------------------------------------------
def compute_sparseness_detailed(X, threshold=0.0):
    """Population sparseness over time and lifetime sparseness over neurons.

    Two families of measure are returned, because they answer different
    questions and are usually reported together:

    (a) ACTIVITY FRACTION — binary, "how many neurons are active".
        active_frac_t[t] = (# neurons with X[t, i] > threshold) / N
        active_frac_n[i] = (# bins with X[t, i] > threshold) / T
        Directly interpretable, but strongly bin-width dependent: widening
        the bin can only raise it (see `coarsen`), so it is only comparable
        across conditions at a fixed bin size.

    (b) TREVES-ROLLS sparseness — graded, threshold-free, uses spike counts
        rather than just their presence:
            a = (sum_i r_i / N)^2 / (sum_i r_i^2 / N)   in [1/N, 1]
            S = (1 - a) / (1 - 1/N)                     in [0, 1]
        S = 0 when all neurons fire equally (dense), S -> 1 when activity is
        carried by one neuron (sparse). Computed across neurons per time bin
        (population sparseness) and across bins per neuron (lifetime
        sparseness). Undefined (nan) for a bin in which nothing fired.

    Parameters
    ----------
    X : (n_bins, n_neurons) array_like
    threshold : float
        A neuron counts as active in a bin if X > threshold. 0.0 = at least
        one spike.

    Returns
    -------
    dict with keys
        n_active_t     (T,) integer count of active neurons per bin
        active_frac_t  (T,) that count / N          <- "sparseness over time"
        active_frac_n  (N,) per-neuron active-bin fraction  <- "over neurons"
        S_pop_t        (T,) Treves-Rolls population sparseness per bin
        S_life_n       (N,) Treves-Rolls lifetime sparseness per neuron
        a_pop_t, a_life_n   the un-normalised a values
        mean_rate_n    (N,) mean of X per neuron (Hz if X is rates)
        summary        dict of scalars (means, medians, silent-bin count, ...)
        n_bins, n_neurons, threshold
    """
    X = _as_numpy(X)
    T, N = X.shape
    A = X > threshold

    n_active_t = A.sum(axis=1)
    active_frac_t = n_active_t / N
    active_frac_n = A.sum(axis=0) / T

    def _treves(M, axis):
        n = M.shape[axis]
        m1 = M.mean(axis=axis)
        m2 = (M ** 2).mean(axis=axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(m2 > 0, m1 ** 2 / m2, np.nan)
        S = (1.0 - a) / (1.0 - 1.0 / n)
        return a, S

    a_pop_t, S_pop_t = _treves(X, axis=1)      # across neurons, per bin
    a_life_n, S_life_n = _treves(X, axis=0)    # across bins, per neuron

    summary = dict(
        mean_active_frac=float(active_frac_t.mean()),
        median_active_frac=float(np.median(active_frac_t)),
        sd_active_frac=float(active_frac_t.std(ddof=1)),
        mean_n_active=float(n_active_t.mean()),
        frac_silent_bins=float(np.mean(n_active_t == 0)),
        n_silent_neurons=int(np.sum(active_frac_n == 0)),
        mean_S_pop=float(np.nanmean(S_pop_t)),
        median_S_pop=float(np.nanmedian(S_pop_t)),
        mean_S_life=float(np.nanmean(S_life_n)),
        # Fano-like index of how uneven the population activity is over time
        cv_n_active=float(n_active_t.std(ddof=1) / n_active_t.mean())
        if n_active_t.mean() > 0 else np.nan,
    )

    return dict(
        n_active_t=n_active_t, active_frac_t=active_frac_t,
        active_frac_n=active_frac_n,
        a_pop_t=a_pop_t, S_pop_t=S_pop_t,
        a_life_n=a_life_n, S_life_n=S_life_n,
        mean_rate_n=X.mean(axis=0),
        summary=summary, n_bins=T, n_neurons=N, threshold=float(threshold),
    )


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_scree(res, label=None, color="#1a679e", ref_slope=1.0, figsize=(7.2, 2.4)):
    """Three-panel scree figure: spectrum + power-law fit, cumulative variance,
    local slope. Returns (fig, axes)."""
    import matplotlib.pyplot as plt

    ev, evr, cum = res["eigvals"], res["evr"], res["cum_evr"]
    rank = res["rank"]
    n = np.arange(1, len(ev) + 1)
    a, se, r2 = res["slope"], res["slope_se"], res["r2"]
    n_min, n_max = res["fit_ranks"]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # (A) spectrum, log-log, with fit
    ax = axes[0]
    ax.loglog(n[:rank], evr[:rank], "o", ms=2.6, color=color, mew=0,
              label=label or "data")
    nf = np.array([n_min, n_max], dtype=float)
    ax.loglog(nf, 10 ** res["intercept"] / ev.sum() * nf ** (-a), "-",
              color="k", lw=1.2,
              label=rf"$\lambda_n \propto n^{{-{a:.2f}}}$" +
                    (rf" $\pm${se:.2f}" if np.isfinite(se) else ""))
    if ref_slope is not None:
        anchor = evr[0]
        ax.loglog(n[:rank], anchor * n[:rank] ** (-ref_slope), ":",
                  color="0.55", lw=1.0, label=rf"$n^{{-{ref_slope:g}}}$")
    ax.set_xlabel("PC rank $n$")
    ax.set_ylabel("variance fraction $\\lambda_n$")
    ax.set_title(f"Ranks {n_min}–{n_max}: $R^2$ = {r2:.3f}")
    ax.legend(frameon=False, loc="lower left")

    # (B) cumulative variance
    ax = axes[1]
    ax.semilogx(n[:rank], 100 * cum[:rank], "-", color=color, lw=1.4)
    for q, ls in ((0.80, "--"), (0.90, ":")):
        d = int(np.searchsorted(cum, q) + 1)
        ax.axhline(100 * q, color="0.7", lw=0.7, ls=ls)
        ax.plot([d], [100 * q], "o", ms=3.5, color="k", mew=0)
        ax.annotate(f"{int(q*100)} %: {d} PCs", (d, 100 * q),
                    textcoords="offset points", xytext=(6, -10), fontsize=6)
    ax.set_xlabel("PC rank $n$")
    ax.set_ylabel("cumulative variance (%)")
    ax.set_ylim(0, 104)
    ax.set_title(f"PR = {res['pr']:.1f} / {res['n_used']} neurons")

    # (C) local slope
    ax = axes[2]
    ax.semilogx(n[:rank], res["local_slope"][:rank], "-", color=color, lw=1.2)
    ax.axhline(a, color="k", lw=1.0, ls="-")
    ax.annotate(f"global fit {a:.2f}", (n[rank - 1], a),
                textcoords="offset points", xytext=(-4, 5),
                ha="right", fontsize=6)
    if ref_slope is not None:
        ax.axhline(ref_slope, color="0.55", lw=1.0, ls=":")
    ax.set_xlabel("PC rank $n$")
    ax.set_ylabel(r"local slope $-\mathrm{d}\log\lambda / \mathrm{d}\log n$")
    ax.set_title("Local slope")
    # the last few ranks sit on the numerical floor and can run away; keep the
    # y-range on the informative part of the curve
    body = res["local_slope"][:max(3, int(0.95 * rank))]
    body = body[np.isfinite(body)]
    if len(body):
        hi = max(np.percentile(body, 99), a * 1.6, ref_slope or 0)
        ax.set_ylim(0, hi * 1.15)

    for ax in axes:
        ax.margins(0.04)
    return fig, axes


def plot_sparseness(sp, X=None, bin_size=None, color="#1a679e",
                    coarsen_factors=(1, 2, 5, 10, 20, 50, 100),
                    figsize=(7.2, 4.6)):
    """Four-panel sparseness figure.

    A  histogram of the active fraction over time bins   ("over time")
    B  histogram of the per-neuron active fraction       ("over neurons")
    C  histogram of Treves-Rolls population sparseness per bin
    D  mean active fraction vs bin width, by pooling adjacent bins
       (needs X; skipped if X is None)

    bin_size : float or None
        Original bin width in seconds, used only for axis labels in D.
    Returns (fig, axes).
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    N, T = sp["n_neurons"], sp["n_bins"]
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()

    # (A) population activity fraction across time bins
    ax = axes[0]
    af = sp["active_frac_t"]
    ax.hist(af, bins=np.linspace(0, af.max() * 1.02, 60),
            color=color, edgecolor="none")
    ax.axvline(af.mean(), color="k", lw=1.0)
    ax.annotate(f"mean {af.mean():.3f}\n({sp['summary']['mean_n_active']:.1f} of {N} neurons)",
                (af.mean(), 0.95), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", va="top", fontsize=6)
    ax.set_xlabel("active fraction in a bin")
    ax.set_ylabel(f"# time bins (of {T:,})")
    ax.set_title("Most bins engage a small subset of the population")

    # (B) lifetime activity fraction across neurons
    ax = axes[1]
    fn = sp["active_frac_n"]
    pos = fn[fn > 0]
    if len(pos) and pos.min() > 0:
        bins = np.logspace(np.log10(pos.min()), np.log10(max(pos.max(), pos.min() * 10)), 30)
        ax.hist(pos, bins=bins, color=color, edgecolor="none")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.hist(fn, bins=30, color=color, edgecolor="none")
    ax.axvline(np.median(pos) if len(pos) else np.nan, color="k", lw=1.0)
    ax.annotate(f"median {np.median(pos):.3f}", (np.median(pos), 0.95),
                xycoords=("data", "axes fraction"), xytext=(6, 0),
                textcoords="offset points", va="top", fontsize=6)
    ax.set_xlabel("fraction of bins in which a neuron is active")
    ax.set_ylabel(f"# neurons (of {N})")
    ax.set_title("Participation spans orders of magnitude across cells")

    # (C) Treves-Rolls population sparseness per bin
    ax = axes[2]
    S = sp["S_pop_t"][np.isfinite(sp["S_pop_t"])]
    ax.hist(S, bins=np.linspace(0, 1, 60), color="#A64800", edgecolor="none")
    ax.axvline(np.median(S), color="k", lw=1.0)
    ax.annotate(f"median {np.median(S):.3f}", (np.median(S), 0.95),
                xycoords=("data", "axes fraction"), xytext=(-6, 0),
                textcoords="offset points", va="top", ha="right", fontsize=6)
    ax.set_xlabel("Treves–Rolls sparseness $S$ per bin")
    ax.set_ylabel("# time bins")
    ax.set_title("Threshold-free measure: 0 = dense, 1 = single-cell")

    # (D) bin-width dependence of the activity fraction
    ax = axes[3]
    if X is not None:
        ks, ms = [], []
        for k in coarsen_factors:
            Xc = coarsen(X, k)
            if Xc.shape[0] < 10:
                continue
            ks.append(k)
            ms.append(float((Xc > sp["threshold"]).mean()))
        xs = np.array(ks, dtype=float) * (bin_size if bin_size else 1.0)
        ax.semilogx(xs, ms, "o-", color=color, ms=3.5, lw=1.2)
        ax.set_xlabel("bin width (s)" if bin_size else "bin width (× original)")
        ax.set_ylabel("mean active fraction")
        ax.set_ylim(0, 1.02)
        ax.set_title("The activity fraction is a statement about bin width")
    else:
        ax.axis("off")

    for ax in axes:
        ax.margins(0.04)
    return fig, axes


# Alias: the functions are referred to as compute_PCA in some of the course
# material. Same object, either spelling works.
compute_PCA = compute_pca_detailed
