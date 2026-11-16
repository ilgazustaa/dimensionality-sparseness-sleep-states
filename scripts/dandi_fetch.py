#!/usr/bin/env python3
"""Fetch pipeline for the rat analysis: DANDI API only, no web scraping.

    python dandi_fetch.py probe      # which assets have units + sleep scoring
    python dandi_fetch.py manifest   # pick the usable rat sessions, pin them
    python dandi_fetch.py extract    # stream them, bin at 1 s -> one npz

This replaces the earlier crawl_buzsaki.py for everything the analysis
depends on. That script additionally scraped the lab's session table at
buzsakilab.com to show that the catalogue lists ~350 recordings served only
through Globus; that is a provenance claim for the methods text, not a way to
obtain data, and nothing here needs it.

Version pinning
---------------
Requests go to *published* dandiset versions, never to `draft`. Drafts are
mutable -- the three dandisets used here have had draft assets modified as
recently as 2026, and some draft asset ids no longer exist in the releases --
so a pipeline written against `draft` is not reproducible. The manifest
records dandiset, version, asset id and SHA-256 for every session used, which
is the reproducible reference to the raw data (11 kB, belongs in version
control) as opposed to the 1.3 TB it points at.
"""
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DANDI = "https://api.dandiarchive.org/api"
# Published versions, pinned. Re-pin deliberately, never silently.
PINNED = {"000044": "0.250624.0426",    # Grosmark & Buzsaki, rat CA1
          "000041": "0.250624.0419",    # Watson et al., rat frontal cortex
          "000061": "0.250624.0426"}    # Girardeau et al., rat CA1 + amygdala
RAT = ("rattus", "brown rat")           # species filter, on the API's own label
BIN_SIZE = 1.0
TONIC_HZ = 4.71
MIN_UNITS = 25
MIN_STATE_S = 60
STATE_MAP = {"awake": "Wake", "wake": "Wake", "quiet": "Quiet",
             "nrem": "NREM", "non-rem": "NREM", "rem": "REM"}


def api(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def assets(ds, version=None):
    url = f"{DANDI}/dandisets/{ds}/versions/{version or PINNED[ds]}/assets/?page_size=200"
    out = []
    while url:
        page = api(url)
        out += page["results"]
        url = page.get("next")
    return out


def asset_info(ds, asset_id, version=None):
    return api(f"{DANDI}/dandisets/{ds}/versions/{version or PINNED[ds]}"
               f"/assets/{asset_id}/info/")["metadata"]


def download_url(ds, asset_id, version=None):
    return (f"{DANDI}/dandisets/{ds}/versions/{version or PINNED[ds]}"
            f"/assets/{asset_id}/download/")


def bound_http_timeouts(connect=15, read=60):
    """remfile calls requests.get with no timeout; an unresponsive file would
    otherwise park a worker for ever and stall the run without failing."""
    import requests
    if getattr(requests.adapters.HTTPAdapter, "_deadline_patched", False):
        return
    original = requests.adapters.HTTPAdapter.send

    def send(self, request, **kw):
        if kw.get("timeout") is None:
            kw["timeout"] = (connect, read)
        return original(self, request, **kw)

    requests.adapters.HTTPAdapter.send = send
    requests.adapters.HTTPAdapter._deadline_patched = True


def txt(ds_):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ds_[:]])


# ------------------------------------------------------------------ probe
def _probe_one(job):
    """Header-only read: unit count, region labels, seconds per sleep state."""
    import h5py
    import remfile
    ds, a = job
    rec = dict(dandiset=ds, version=PINNED[ds], path=a["path"], asset_id=a["asset_id"],
               size_gb=round(a["size"] / 1e9, 2))
    try:
        f = h5py.File(remfile.File(download_url(ds, a["asset_id"])), "r")
        u = f.get("units")
        rec["n_units"] = int(u["id"].shape[0]) if u is not None and "id" in u else 0
        if u is not None:
            for c in ("location", "region", "cell_type"):
                if c in u and u[c].dtype.kind in "OSU":
                    rec[c] = "|".join(sorted(set(txt(u[c]))))[:120]
        st = f.get("processing/behavior/states")
        if st is not None and "label" in st:
            k0 = "start_time" if "start_time" in st else "start"
            k1 = "stop_time" if "stop_time" in st else "stop"
            lab = txt(st["label"])
            rec["state_labels"] = "|".join(sorted(set(lab)))
            rec["state_secs"] = {s: round(float((st[k1][:][lab == s] - st[k0][:][lab == s]).sum()), 1)
                                 for s in sorted(set(lab))}
        f.close()
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:150]
    return rec


def stage_probe(out_path="probe_rat.jsonl", workers=10):
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["asset_id"] for l in open(out_path) if l.strip()}
    jobs = [(ds, a) for ds in PINNED for a in assets(ds) if a["asset_id"] not in done]
    print(f"probing {len(jobs)} assets across {len(PINNED)} dandisets", flush=True)
    bound_http_timeouts()
    with ThreadPoolExecutor(max_workers=workers) as ex, open(out_path, "a") as fh:
        for i, rec in enumerate(ex.map(_probe_one, jobs), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print("probe done", flush=True)


# --------------------------------------------------------------- manifest
def stage_manifest(probe_path="probe_rat.jsonl", out_path="dandi_manifest_rat.csv"):
    """Select the usable rat sessions and pin them with their checksums.

    Usable means: rat (per the API's species field), a spike-sorted table of
    at least MIN_UNITS units, and at least MIN_STATE_S seconds of each of
    Wake, NREM and REM, so every session can enter a three-state comparison.
    """
    rows = []
    for line in open(probe_path):
        r = json.loads(line)
        if r.get("error") or (r.get("n_units") or 0) < MIN_UNITS or not r.get("state_secs"):
            continue
        secs = {}
        for k, v in r["state_secs"].items():
            secs[STATE_MAP.get(k.lower(), "none")] = secs.get(STATE_MAP.get(k.lower(), "none"), 0) + v
        if min(secs.get(s, 0) for s in ("Wake", "NREM", "REM")) < MIN_STATE_S:
            continue
        md = asset_info(r["dandiset"], r["asset_id"])
        species = ((md.get("wasAttributedTo") or [{}])[0].get("species") or {}).get("name", "")
        if not any(k in species.lower() for k in RAT):
            continue
        rows.append(dict(
            tag=f"{r['dandiset']}_" + os.path.basename(r["path"]).split("_behavior")[0].split("_ecephys")[0],
            dandiset=r["dandiset"], version=r["version"], asset_id=r["asset_id"],
            path=r["path"], size_gb=r["size_gb"], species=species,
            sha256=md.get("digest", {}).get("dandi:sha2-256", ""),
            n_units=r["n_units"], locations=r.get("location", r.get("region", "")),
            **{f"sec_{s}": round(secs.get(s, 0), 1) for s in ("Wake", "NREM", "REM")}))
    rows.sort(key=lambda d: (d["dandiset"], d["path"]))
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"manifest: {len(rows)} rat sessions "
          f"({sum(r['size_gb'] for r in rows):.0f} GB remote) -> {out_path}", flush=True)


# ---------------------------------------------------------------- extract
def read_session(url):
    """Spike times, unit columns and sleep-state intervals from one NWB.

    Reads the HDF5 datasets directly rather than through pynwb, so the whole
    spike_times column arrives in one sequential range request.
    """
    import h5py
    import remfile
    f = h5py.File(remfile.File(url), "r")
    u = f["units"]
    idx = u["spike_times_index"][:]
    st = u["spike_times"][:]
    starts = np.concatenate([[0], idx[:-1]])
    spikes = [st[a:b].astype(np.float64) for a, b in zip(starts, idx)]
    cols = {c: (txt(u[c]) if u[c].dtype.kind in "OSU" else u[c][:])
            for c in ("location", "region", "cell_type") if c in u}
    p = f.get("processing/behavior/states")
    states = None
    if p is not None:
        k0 = "start_time" if "start_time" in p else "start"
        k1 = "stop_time" if "stop_time" in p else "stop"
        states = dict(start=p[k0][:], stop=p[k1][:], label=txt(p["label"]))
    f.close()
    return spikes, cols, states


def label_bins(states, n_bins):
    lab = np.full(n_bins, "none", dtype="<U5")
    for s0, s1, raw in zip(states["start"], states["stop"], states["label"]):
        lab[int(s0 // BIN_SIZE):int(s1 // BIN_SIZE)] = STATE_MAP.get(str(raw).strip().lower(), "none")
    return lab


def stage_extract(manifest_path="dandi_manifest_rat.csv", out_path="dandi_binned_1s_rat.npz"):
    """Stream every manifest session and save binned counts, states, regions.

    All units are kept and their mean rates stored alongside, so the tonic
    threshold (TONIC_HZ) can be re-derived downstream rather than baked in.
    """
    rows = list(csv.DictReader(open(manifest_path)))
    bound_http_timeouts()
    store, summary = {}, []
    for i, r in enumerate(rows, 1):
        t0 = time.time()
        try:
            spikes, cols, states = read_session(download_url(r["dandiset"], r["asset_id"]))
        except Exception as exc:
            print(f"  [{i}/{len(rows)}] FAIL {r['tag']}: {type(exc).__name__}", flush=True)
            continue
        if not spikes or states is None:
            continue
        t_stop = max((s.max() for s in spikes if len(s)), default=0.0)
        edges = np.arange(0.0, t_stop + BIN_SIZE, BIN_SIZE)
        X = np.stack([np.histogram(s, bins=edges)[0] for s in spikes], axis=1).astype(np.int16)
        rate = (X.sum(0) / ((len(edges) - 1) * BIN_SIZE)).astype(np.float32)
        lab = label_bins(states, X.shape[0])
        reg = cols.get("location", cols.get("region", np.array(["?"] * len(spikes))))
        tag = r["tag"]
        store[f"X__{tag}"] = X
        store[f"state__{tag}"] = lab
        store[f"region__{tag}"] = np.asarray(reg)
        store[f"rate__{tag}"] = rate
        summary.append(dict(tag=tag, dandiset=r["dandiset"], n_units=int(X.shape[1]),
                            n_below_tonic=int((rate <= TONIC_HZ).sum()),
                            duration_s=int(X.shape[0] * BIN_SIZE),
                            **{f"bins_{s}": int((lab == s).sum())
                               for s in ("Wake", "NREM", "REM", "Quiet")}))
        print(f"  [{i}/{len(rows)}] {tag[:46]}: {X.shape[1]}u x {X.shape[0]}bins, "
              f"{time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(out_path, **store)
    with open("dandi_session_summary_rat.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"extract: {len(summary)} sessions -> {out_path} "
          f"({os.path.getsize(out_path)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "manifest"
    {"probe": stage_probe, "manifest": stage_manifest, "extract": stage_extract}[stage](*sys.argv[2:])
