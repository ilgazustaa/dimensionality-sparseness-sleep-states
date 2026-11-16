"""Population metrics for a (time bins x neurons) activity matrix X."""

import numpy as np


def compute_pca(X, fit_range=None, zscore=False, strict=True):
    """Eigenvalue spectrum of the neuron covariance, and its power-law slope.

    fit_range is (first PC, last PC), inclusive and 1-indexed. With strict=True
    (the default) a range wider than the spectrum raises instead of being
    quietly truncated; pass strict=False to keep the old behaviour.
    """

    X = np.asarray(X, dtype=float)
    n_bins = X.shape[0]

    Xc = X - X.mean(axis=0)                          # center each neuron
    if zscore:
        sd = Xc.std(axis=0)
        assert (sd > 0).all(), "some neurons never fire — drop them before z-scoring"
        Xc = Xc / sd                                 # every neuron gets variance 1

    sv = np.linalg.svd(Xc, compute_uv=False)         # singular values
    eigvals = sv ** 2 / (n_bins - 1)                 # = variance per PC

    n = np.arange(1, len(eigvals) + 1)               # PC rank
    lo, hi = fit_range if fit_range else (1, len(eigvals))
    if strict and fit_range and hi > len(eigvals):
        # Silent truncation is how a "PCs 3-50" label came to mean PCs 3-25 on
        # 25-unit subsamples, which is a 60% different slope. Refuse it.
        raise ValueError(
            f"fit_range={fit_range} asks for PC {hi} but the spectrum has only "
            f"{len(eigvals)} components; pass fit_range=({lo}, {len(eigvals)}) "
            f"deliberately, or strict=False to accept the truncation")
    use = (n >= lo) & (n <= hi) & (eigvals > 0)
    slope, _ = np.polyfit(np.log10(n[use]), np.log10(eigvals[use]), 1)

    return eigvals, -slope


def compute_sparseness(X):
    """How sparse is the activity?

    Returns (over_time, over_neurons):
        over_time[t]    fraction of neurons active in bin t
        over_neurons[i] fraction of bins in which neuron i is active
    """
    active = np.asarray(X) > 0 #binary 0,1
    return active.mean(axis=1), active.mean(axis=0)


def rates_matrix(spike_time_dict, bin_size, t_start=0.0, t_stop=None):
    """Build X from a {neuron_id: spike_times_in_seconds} dict.

    Returns (X, neuron_ids, edges):
        X          (n_bins, n_neurons) spike counts; divide by bin_size for Hz
        neuron_ids the SORTED keys, so column i is the same neuron every run
        edges      bin edges in seconds; edges[:-1] are the bin start times
    """
    ids = sorted(spike_time_dict.keys())
    if t_stop is None:
        t_stop = float(np.ceil(max(t.max() for t in spike_time_dict.values() if len(t))))
    edges = np.arange(t_start, t_stop + bin_size, bin_size)
    X = np.zeros((len(edges) - 1, len(ids)), dtype=np.float32)
    for i, nid in enumerate(ids):
        X[:, i] = np.histogram(spike_time_dict[nid], bins=edges)[0]
    return X, ids, edges


