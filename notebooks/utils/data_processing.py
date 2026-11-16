import numpy as np
import torch

def compute_rates(spike_times_dict, bin_size=1.0):
    """
    Computes firing rates (Hz) from a dictionary of spike times.
    
    Args:
        spike_times_dict (dict): {neuron_id: [time1, time2, ...]}
        bin_size (float): Bin width in seconds (default: 1.0)
        
    Returns:
        torch.Tensor: Rate matrix of shape (n_bins, n_neurons)
    """
    # 1. Determine time boundaries automatically
    all_times = np.concatenate([t for t in spike_times_dict.values() if len(t) > 0])
    t_max = np.ceil(all_times.max())
    n_bins = int(np.ceil(t_max / bin_size))
    
    # 2. Get sorted list of neurons for consistent matrix columns
    neuron_ids = sorted(spike_times_dict.keys())
    n_neurons = len(neuron_ids)
    
    # 3. Initialize and compute rates
    rates = np.zeros((n_bins, n_neurons))
    for i, nid in enumerate(neuron_ids):
        # Bin the spikes
        counts, _ = np.histogram(spike_times_dict[nid], bins=n_bins, range=(0, t_max))
        # Convert to Rate (Hz)
        rates[:, i] = counts / bin_size
        
    return torch.tensor(rates, dtype=torch.float32)