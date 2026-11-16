"""Analysis parameters, in one place.

Import these rather than re-declaring them in a notebook cell or a script
header. Every number in the thesis that depends on a choice made here should
be traceable to this file.

    from analysis_config import BIN, THRESH, N_SUB, FIT_RANGE, effective_range

Why a single module: the same constants were previously defined in each
notebook and each script, and they drifted -- the notebooks fitted PCs 3-50
while the multi-session scripts fitted 3-20. At a matched population of 25
units those are not two labels for one choice, they are two different
measurements (Achilles NREM: 0.58 at 3-20, 0.95 at the truncated 3-25), and
nothing in the code complained.
"""

BIN = 1.0               # bin width in seconds
THRESH = 4.71           # tonic-unit exclusion, spikes/s (Section 3)
MIN_SPIKES_STATE = 10   # a unit must fire at least this often in the state fitted
N_SUB = 25              # matched population size for cross-session comparison
N_DRAWS = 100            # subsample draws averaged per spectrum
WIN = 120               # window length in seconds for within-session correlations
SEED = 1

# Floors for the distributional (likelihood) fits. These do not apply to the
# matched-population exponent, which subsamples to N_SUB and needs only that
# many units.
#
# MIN_UNITS_GOF: the `powerlaw` cutoff scan becomes unstable on short spectra,
# since it must reserve the largest values and still fit a tail below them. 40
# is a convention, not a derived quantity; the sensitivity check in notebook 06
# repeats the comparisons at a floor of 60 and the median exponents move by at
# most 0.03, with waking steeper than non-REM in every population under both.
#
# MIN_BINS: a state shorter than this gives a covariance matrix estimated from
# too few observations for its spectrum to be meaningful. 300 bins is 5 minutes
# at BIN = 1 s.
MIN_UNITS_GOF = 40
MIN_BINS = 300

# Fit range, as (first PC, last PC), inclusive and 1-indexed.
#
# FIT_RANGE_MATCHED applies to matched-N spectra, where only N_SUB eigenvalues
# exist: the last few of those are the sampling floor, so the fit stops at 20.
# FIT_RANGE_FULL applies to full-population spectra (a single session with its
# whole retained population, typically 40-160 units), where PCs up to 50 carry
# signal.
FIT_RANGE_MATCHED = (3, 20)
FIT_RANGE_FULL = (3, 50)


def effective_range(fit_range, n_components):
    """The PC range that will actually be fitted, given how many PCs exist.

    compute_pca masks with (n >= lo) & (n <= hi), so a range wider than the
    spectrum is silently truncated. Call this before quoting a range in a
    caption, or use check_range to refuse the truncation outright.
    """
    lo, hi = fit_range
    return (lo, min(hi, int(n_components)))


def check_range(fit_range, n_components, tol=0):
    """Raise if `fit_range` cannot be honoured by a spectrum of this size.

    `tol` allows a stated slack (e.g. tol=5 permits fitting 3-45 under a 3-50
    label); the default refuses any truncation.
    """
    lo, hi = fit_range
    if n_components < lo + 1:
        raise ValueError(f"fit range {fit_range} needs at least {lo + 1} PCs, got {n_components}")
    if hi - min(hi, n_components) > tol:
        raise ValueError(
            f"fit range {fit_range} truncated to {effective_range(fit_range, n_components)} "
            f"by a {n_components}-component spectrum; state the effective range or raise N_SUB")
    return effective_range(fit_range, n_components)
