
import torch


def pca(rates, zscore=False):

    n_time_bins, n_neurons = rates.shape

    # Center
    X = rates - rates.mean(dim=0, keepdim=True)

    # Optional z-score
    if zscore:
        X = X / X.std(dim=0, keepdim=True)

    # Covariance
    cov = (X.T @ X) / (n_time_bins - 1)

    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # Sort descending
    idx = torch.argsort(eigvals, descending=True)

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return eigvals, eigvecs