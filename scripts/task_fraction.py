#!/usr/bin/env python3
"""How much of each recording's waking time falls inside a task epoch?

Streams the epoch table and the sleep-state intervals of every analysed
recording from the two task datasets and writes task_fraction_of_waking.csv:
one row per recording with waking seconds, task-epoch waking seconds and the
fraction. Dandiset 000041 is omitted because its animals performed no task and
its archive files carry no epoch table.

Units differ between deposits and are NOT assumed: 000044 stores epoch bounds in
seconds, 000061 in milliseconds. The scale is detected per file by comparing the
last epoch bound with the last state interval, which must agree to within a few
per cent of the recording length.
"""
import os
import warnings

import h5py
import numpy as np
import pandas as pd
import remfile

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TASK_TAGS = ("maze", "run")          # 000044: MazeEpoch; 000061: prerun / run / postrun


def asset_url(row):
    return (f"https://api.dandiarchive.org/api/dandisets/{int(row.dandiset):06d}"
            f"/versions/{row.version}/assets/{row.asset_id}/download/")


def epoch_table(f):
    """Return (starts, stops, labels) in seconds, scale inferred from the states."""
    ep = f["intervals/epochs"]
    s0, s1 = ep["start_time"][:].astype(float), ep["stop_time"][:].astype(float)
    if "label" in ep:
        raw = ep["label"][:]
    else:
        raw = ep["tags"][:]
    lab = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
    st = f["processing/behavior/states"]
    last_state = float(st["stop_time"][:].max())
    scale = 1.0 if s1.max() < 2 * last_state else round(s1.max() / last_state, -2)
    return s0 / scale, s1 / scale, lab, scale


def waking_intervals(f):
    st = f["processing/behavior/states"]
    lab = [x.decode() if isinstance(x, bytes) else str(x) for x in st["label"][:]]
    keep = np.array([l.lower().startswith(("awake", "wake", "quiet")) for l in lab])
    return st["start_time"][:][keep], st["stop_time"][:][keep]


def overlap(a0, a1, b0, b1):
    """Total overlap in seconds between two sets of intervals."""
    tot = 0.0
    for x0, x1 in zip(a0, a1):
        tot += float(np.clip(np.minimum(x1, b1) - np.maximum(x0, b0), 0, None).sum())
    return tot


mf = pd.read_csv(os.path.join(DATA, "dandi_manifest_rat.csv"))
rows = []
for _, r in mf[mf.dandiset.isin([44, 61])].iterrows():
    try:
        f = h5py.File(remfile.File(asset_url(r)), "r")
        if "intervals/epochs" not in f:
            f.close()
            continue
        e0, e1, lab, scale = epoch_table(f)
        w0, w1 = waking_intervals(f)
        is_task = np.array([any(t in l.lower() for t in TASK_TAGS) for l in lab])
        wake_s = float((w1 - w0).sum())
        task_s = overlap(w0, w1, e0[is_task], e1[is_task])
        rows.append(dict(tag=r.tag, dandiset=int(r.dandiset),
                         wake_s=round(wake_s, 1), task_wake_s=round(task_s, 1),
                         task_fraction_of_wake=round(task_s / wake_s, 4) if wake_s else np.nan,
                         epoch_scale=scale, task_epochs=sum(is_task)))
        f.close()
        print(rows[-1]["tag"][:46], rows[-1]["task_fraction_of_wake"], flush=True)
    except Exception as exc:
        print("SKIP", r.tag[:46], type(exc).__name__, str(exc)[:60], flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "..", "notebooks", "task_fraction_of_waking.csv"), index=False)
print("\nrecordings:", len(out))
print(out.groupby("dandiset").task_fraction_of_wake.describe()[["count", "min", "50%", "max"]].round(3).to_string())
