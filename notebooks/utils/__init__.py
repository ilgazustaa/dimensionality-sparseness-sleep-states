"""Analysis helpers for the hippocampal population project.

Re-exported here so notebooks can do:

    from utils import rates_matrix, compute_pca, compute_sparseness
"""

try:
    # these two need torch
    from .data_processing import compute_rates
    from .pca import pca
except ImportError:                      # pragma: no cover
    compute_rates = pca = None

# the two functions used in the thesis
from .population_metrics import (
    rates_matrix,
    compute_pca,
    compute_sparseness,
)

# diagnostics and ready-made figures
from .population_metrics_extra import (
    compute_pca_detailed,
    compute_sparseness_detailed,
    coarsen,
    plot_scree,
    plot_sparseness,
)

__all__ = [
    "compute_rates", "pca",
    "rates_matrix", "compute_pca", "compute_sparseness",
    "compute_pca_detailed", "compute_sparseness_detailed",
    "coarsen", "plot_scree", "plot_sparseness",
]
