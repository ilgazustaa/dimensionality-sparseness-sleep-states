#!/usr/bin/env python3
"""Does the 4.71 Hz exclusion separate cell classes?

The archive copy of Achilles_10252013 carries 137 units but no cell-type label.
The laboratory's own session file labels 120 units pyramidal and 17 interneuron,
but numbers its clusters differently. The two numbering spaces are matched by
spike count, which is unique enough here to give a one-to-one map, and the
exclusion is then cross-tabulated against the laboratory's labels.

    python check_celltypes.py
"""
import h5py
import numpy as np
import pandas as pd

DATA = "/Users/ilgaz/Desktop/data_share_20260429/data/"
SESSINFO = DATA + "Achilles_10252013_sessInfo.mat"
PACK = DATA + "dandi_binned_1s_rat.npz"
TAG = "000044_sub-Achilles_ses-Achilles-10252013"
THRESH = 4.71            # Hz, the exclusion threshold used throughout


def lab_units():
    """Per-cluster spike count and cell class from the laboratory's file."""
    with h5py.File(SESSINFO, "r") as f:
        sp = f["sessInfo"]["Spikes"] if "sessInfo" in f else f["Spikes"]
        ids = np.asarray(sp["SpikeIDs"]).ravel()
        pyr = set(np.asarray(sp["PyrIDs"]).ravel().astype(int))
        inter = set(np.asarray(sp["IntIDs"]).ravel().astype(int))
    clusters, counts = np.unique(ids.astype(int), return_counts=True)
    return pd.DataFrame(dict(
        cluster=clusters, n_spikes=counts,
        cell_class=["pyramidal" if c in pyr else "interneuron" if c in inter else "unlabelled"
                    for c in clusters]))


def archive_units():
    """Per-unit spike count and session-wide rate from the derived pack."""
    z = np.load(PACK, allow_pickle=True)
    X = z[f"X__{TAG}"]
    return pd.DataFrame(dict(unit=np.arange(X.shape[1]),
                             n_spikes=X.sum(0).astype(int),
                             rate_hz=z[f"rate__{TAG}"]))


def match_by_spike_count(lab, arc):
    """One-to-one map between the two numbering spaces, keyed on spike count."""
    merged = arc.merge(lab, on="n_spikes", how="inner", validate="one_to_one")
    assert len(merged) == len(arc) == len(lab), (
        f"matched {len(merged)} of {len(arc)} archive and {len(lab)} laboratory units")
    assert merged.cluster.nunique() == len(merged), "a cluster was matched twice"
    return merged


if __name__ == "__main__":
    lab, arc = lab_units(), archive_units()
    m = match_by_spike_count(lab, arc)
    m["excluded"] = m.rate_hz > THRESH
    table = pd.crosstab(m.cell_class, m.excluded.map({True: "excluded", False: "retained"}))
    print(table.to_string(), "\n")
    for cls in ("interneuron", "pyramidal"):
        sub = m[m.cell_class == cls]
        print(f"{cls:12s} n = {len(sub):3d} | excluded {int(sub.excluded.sum()):2d} "
              f"| rate range {sub.rate_hz.min():.2f}-{sub.rate_hz.max():.2f} Hz")
    kept = m[~m.excluded]
    print(f"\nretained population: {len(kept)} units "
          f"({int((kept.cell_class == 'pyramidal').sum())} pyramidal, "
          f"{int((kept.cell_class == 'interneuron').sum())} interneuron)")
    m.to_csv("celltype_mapping.csv", index=False)
