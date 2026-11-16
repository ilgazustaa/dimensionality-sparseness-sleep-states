"""Data access for the rat analysis.

Three entry points, in the order you will normally want them:

    load_pack()              the derived 1 s binned matrices (offline, fast)
    stream_session(tag)      the original NWB, read over HTTP from DANDI
    check_manifest()         re-query DANDI and confirm nothing upstream moved

Where the data comes from
-------------------------
Nothing is downloaded in full. Every session lives on the DANDI archive as a
5--12 GB NWB file; `stream_session` opens it with byte-range requests and
reads only the unit table and the sleep-state intervals, a few MB per
session. The binned matrices that the analysis actually consumes are built
once by `crawl_buzsaki.py extract` and stored in `data/*.npz`.

Why the version is pinned
-------------------------
`dandi_manifest_rat.csv` records, for every session used, the dandiset, the
*published* version, the asset id, and the file's SHA-256. Requests go to
those pinned versions rather than to `draft`. A draft is mutable -- the three
dandisets here have had their draft assets modified as recently as 2026 --
so a script written against `versions/draft/` can silently read different
bytes next month. `check_manifest()` re-queries the archive and compares the
recorded SHA-256 against what DANDI reports now; a mismatch means the
upstream file changed and any cached pack should be rebuilt.
"""
import json
import os
import urllib.request

import numpy as np
import pandas as pd

DANDI = "https://api.dandiarchive.org/api"
HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [os.path.join(HERE, "..", "data"), os.path.join(HERE, "data"), HERE]
DATA = next((p for p in _CANDIDATES
             if os.path.exists(os.path.join(p, "dandi_manifest_rat.csv"))), _CANDIDATES[0])
MANIFEST = os.path.join(DATA, "dandi_manifest_rat.csv")
PACK = "dandi_binned_1s_rat.npz"    # built by: python dandi_fetch.py extract
AMY = {"bla", "blv", "bmp", "cen", "ladl", "den", "ven"}
THRESH = 4.71                       # tonic-unit exclusion, Hz


def manifest():
    """The pinned session list: group, dandiset, version, asset id, sha256."""
    return pd.read_csv(MANIFEST, dtype={"dandiset": str})


def asset_url(tag_or_row):
    """Download URL for one session, at the pinned published version."""
    row = tag_or_row
    if isinstance(tag_or_row, str):
        mf = manifest()
        hit = mf[mf.tag == tag_or_row]
        if not len(hit):
            raise KeyError(f"{tag_or_row} is not in the manifest")
        row = hit.iloc[0]
    return (f"{DANDI}/dandisets/{row['dandiset']}/versions/{row['version']}"
            f"/assets/{row['asset_id']}/download/")


def stream_session(tag):
    """Spike times, unit columns and sleep-state intervals for one session.

    Reads the HDF5 datasets directly rather than through pynwb: the whole
    spike_times column arrives in one sequential range request instead of
    thousands of small ones.
    """
    import h5py
    import remfile
    import requests

    original = requests.adapters.HTTPAdapter.send      # remfile sets no timeout

    def send(self, request, **kw):
        kw["timeout"] = kw.get("timeout") or (15, 60)
        return original(self, request, **kw)

    requests.adapters.HTTPAdapter.send = send
    f = h5py.File(remfile.File(asset_url(tag)), "r")
    u = f["units"]
    idx = u["spike_times_index"][:]
    st = u["spike_times"][:]
    starts = np.concatenate([[0], idx[:-1]])
    spikes = [st[a:b].astype(np.float64) for a, b in zip(starts, idx)]
    txt = lambda ds: np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]])
    cols = {c: (txt(u[c]) if u[c].dtype.kind in "OSU" else u[c][:])
            for c in ("location", "region", "cell_type") if c in u}
    states = None
    p = f.get("processing/behavior/states")
    if p is not None:
        k0 = "start_time" if "start_time" in p else "start"
        k1 = "stop_time" if "stop_time" in p else "stop"
        states = dict(start=p[k0][:], stop=p[k1][:], label=txt(p["label"]))
    f.close()
    return spikes, cols, states


def load_pack(group=None, min_units=25):
    """Binned matrices from the local packs, units restricted by region group.

    Yields dicts with X (bins x units, float), state (label per bin), tag and
    group. `group` filters to one of: rat CA1, rat hippocampus, rat amygdala,
    rat frontal cortex. Mouse sessions are never returned.
    """
    mf = manifest()
    ds_of = dict(zip(mf.tag, mf.dandiset))
    z = np.load(os.path.join(DATA, PACK), allow_pickle=True)
    for tag in sorted({k.split("__", 1)[1] for k in z.files if k.startswith("X__")}):
        X = z[f"X__{tag}"].astype(np.float64)
        lab = z[f"state__{tag}"].copy()
        lab[lab == "Quiet"] = "Wake"
        reg = np.array([str(x).lower() for x in z[f"region__{tag}"]])
        keep = z[f"rate__{tag}"] <= THRESH
        ds = ds_of.get(tag, tag.split("_")[0])
        if ds == "000044":
            groups = [("rat CA1", keep)]
        elif ds == "000041":
            groups = [("rat frontal cortex", keep)]
        else:                                      # 000061: two regions in one file
            groups = [("rat hippocampus", keep & (reg == "hpc")),
                      ("rat amygdala", keep & np.isin(reg, list(AMY)))]
        for name, mask in groups:
            if mask.sum() >= min_units and (group is None or name == group):
                yield dict(tag=tag, group=name, X=X[:, mask], state=lab,
                           n_units=int(mask.sum()))


def check_manifest(verbose=True):
    """Re-query DANDI and compare recorded SHA-256 against the archive now.

    Returns the rows that changed. An empty frame means every pinned asset is
    byte-identical to what the analysis was run on.
    """
    mf = manifest()
    bad = []
    for _, r in mf.iterrows():
        url = (f"{DANDI}/dandisets/{r['dandiset']}/versions/{r['version']}"
               f"/assets/{r['asset_id']}/info/")
        try:
            with urllib.request.urlopen(url, timeout=45) as resp:
                md = json.load(resp)["metadata"]
            now = md.get("digest", {}).get("dandi:sha2-256", "")
        except Exception as exc:
            now = f"error: {type(exc).__name__}"
        if now != r["sha256"]:
            bad.append(dict(tag=r["tag"], recorded=r["sha256"][:16], found=str(now)[:16]))
    out = pd.DataFrame(bad)
    if verbose:
        print(f"{len(mf) - len(out)}/{len(mf)} pinned assets unchanged")
    return out
