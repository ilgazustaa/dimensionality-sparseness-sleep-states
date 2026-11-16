#!/usr/bin/env python3
"""Crawler for the Buzsaki lab public datasets, and extractor for the ones
that are actually downloadable.

Three stages, run independently:

    python crawl_buzsaki.py inventory   # what exists  -> buzsaki_session_inventory.csv
    python crawl_buzsaki.py probe       # what is usable -> probe_states.jsonl
    python crawl_buzsaki.py extract     # get the data -> dandi_binned_1s.npz

Why two portals
---------------
The lab's own page (buzsakilab.com/wp/2018/10/29/public-datasets/) links a
session table of ~350 recordings, but the files behind it are served through
Globus, which needs an interactive client and an account -- a script cannot
fetch them. The same recordings are mirrored, dataset by dataset, on the NWB
archive (DANDI), which has a plain HTTP API. So stage 1 crawls the lab page
for the authoritative *catalogue* (animal, date, brain region, project) and
the archive for what is *reachable*, then joins the two on animal + date.

Stages 2 and 3 read the archive files over HTTP with byte-range requests
(remfile + h5py) instead of downloading them. A session NWB is 5-12 GB on
disk; the unit table and the sleep-state intervals inside it are a few MB.
Nothing here ever reads the raw voltage traces.

Requires: numpy, h5py, remfile.
"""
import csv, glob, json, os, pickle, re, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

LAB_SESSIONS = "https://buzsakilab.com/wp/sessions/?frm-page-14315={page}"
DANDI = "https://api.dandiarchive.org/api"
# The lab's dandisets on the archive. 000044 = Grosmark CA1, 000041 = Watson
# frontal cortex; both carry sleep scoring, which is why they are extracted.
DANDISETS = ["000003", "000041", "000044", "000056", "000059", "000061",
             "000067", "000166", "000213", "000218", "000233", "000552", "000568"]
EXTRACT = {"000044": "CA1", "000041": "CTX"}
# Region label per dandiset, used when the unit table carries no location
# column (true of the mouse sets): the deposit's anatomy is then the only
# region information there is.
REGION_HINT = {"000003": "mouse hippocampus (DG/CA1)", "000041": "rat frontal cortex",
               "000044": "rat CA1", "000056": "mouse thalamus/postsubiculum",
               "000061": "rat hippocampus + amygdala"}
BIN_SIZE = 1.0          # seconds, as used throughout the thesis
TONIC_HZ = 4.71         # tonic-unit exclusion threshold (Section 3)


def get(url, timeout=90, method="GET"):
    req = urllib.request.Request(url, method=method,
                                 headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.geturl()


def api(url):
    return json.loads(get(url)[0].decode())


def txt(ds):
    """HDF5 string column -> array of str."""
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]])


# ---------------------------------------------------------------- stage 1
def crawl_lab(max_pages=25):
    """Scrape the paginated session table on buzsakilab.com.

    The table is rendered server-side by a WordPress form plugin; there is no
    JSON endpoint (the documented /databank REST route returns 403), so the
    rows are parsed out of the HTML. One row per recording session.
    """
    rows = []
    for page in range(1, max_pages + 1):
        html = get(LAB_SESSIONS.format(page=page))[0].decode("utf-8", "replace")
        table = re.search(r"<table[\s\S]*?</table>", html)
        if not table:
            break
        page_rows = []
        for tr in re.findall(r"<tr[^>]*>([\s\S]*?)</tr>", table.group(0))[1:]:
            cells = re.findall(r"<t[dh][^>]*>([\s\S]*?)</t[dh]>", tr)
            if len(cells) < 4:
                continue
            page_rows.append(parse_row(cells))
        if not page_rows:
            break
        rows += page_rows
        print(f"  page {page}: {len(page_rows)} rows (total {len(rows)})", flush=True)
    return rows


def strip_tags(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip(" ,")


def parse_row(cells):
    """One HTML table row -> one session record.

    Two cells are composite. The session cell holds the session name and a
    Globus link whose origin_path encodes owner/animal/session. The animal
    cell holds the animal name, then 'strain?, species, strain', then the
    project title after 'Projects:'.
    """
    sess_cell, animal_cell, date_cell = cells[0], cells[1], cells[2]
    session = strip_tags(re.search(r">([^<]+)</a>", sess_cell).group(1)) \
        if re.search(r">([^<]+)</a>", sess_cell) else strip_tags(sess_cell).split(",")[0]
    gp = re.search(r"origin_path=([^\"&]+)", sess_cell)
    globus = urllib.parse.unquote(gp.group(1)) if gp else ""
    owner = globus.strip("/").split("/")[0] if globus else ""

    anchors = re.findall(r">([^<]+)</a>", animal_cell)
    animal = strip_tags(anchors[0]) if anchors else strip_tags(animal_cell).split(" ")[0]
    project = strip_tags(anchors[1]) if len(anchors) > 1 else ""
    descr = strip_tags(re.sub(r"<a[\s\S]*?</a>", " ", animal_cell))
    parts = [p.strip() for p in descr.replace("Projects:", "").split(",") if p.strip()]
    species = next((p for p in parts if "Rattus" in p or "Mus " in p or "musculus" in p), "")
    strain = parts[-1] if len(parts) > 1 and parts[-1] != species else ""
    return dict(session=session, animal=animal, species=species, strain=strain,
                project=project, date=strip_tags(date_cell),
                paradigms=strip_tags(cells[3]) if len(cells) > 3 else "",
                manipulations=strip_tags(cells[4]) if len(cells) > 4 else "",
                owner=owner, globus_path=globus)


def parse_animal(cell):
    """'Achilles (Rattus norvegicus, Rat, male)' -> ('Achilles', 'Rattus norvegicus')."""
    name = cell.split(" ")[0] if cell else ""
    low = cell.lower()
    if "rattus" in low or " rat" in low:
        species = "Rattus norvegicus"
    elif "mus " in low or "mouse" in low:
        species = "Mus musculus"
    else:
        species = "not recorded"
    return name, species


def key_of(animal, date):
    """Join key: lowercase animal name + digits of the date."""
    return f"{animal.lower()}_{re.sub(r'[^0-9]', '', date or '')}"


def dandi_index():
    """Every asset of every lab dandiset, with its anatomy and species."""
    sets, assets = [], []
    for ds in DANDISETS:
        try:
            meta = api(f"{DANDI}/dandisets/{ds}/versions/draft/info/")
        except Exception as exc:
            print(f"  {ds}: {type(exc).__name__}", flush=True)
            continue
        m = meta.get("metadata", {})
        sets.append(dict(
            dandiset=ds, name=m.get("name", "")[:90],
            species="|".join(sorted({s.get("name", "") for s in m.get("about", [])
                                     if "species" in str(s.get("schemaKey", "")).lower()})),
            anatomy="|".join(sorted({a.get("name", "") for a in m.get("about", [])})),
            n_assets=meta.get("assetsSummary", {}).get("numberOfFiles", 0)))
        url = f"{DANDI}/dandisets/{ds}/versions/draft/assets/?page_size=200"
        while url:
            page = api(url)
            for a in page["results"]:
                sub = a["path"].split("/")[0].replace("sub-", "")
                date = re.search(r"(\d{6,8})", os.path.basename(a["path"]))
                assets.append(dict(dandiset=ds, asset_id=a["asset_id"], path=a["path"],
                                   size_gb=round(a["size"] / 1e9, 2), subject=sub,
                                   key=key_of(sub, date.group(1) if date else "")))
            url = page.get("next")
        print(f"  {ds}: {sum(x['dandiset'] == ds for x in assets)} assets", flush=True)
    return sets, assets


def norm_species(s):
    low = str(s).lower()
    if "mus" in low or "mouse" in low:
        return "Mus musculus"
    if "rattus" in low or "rat" in low:
        return "Rattus norvegicus"
    return "not recorded"


def stage_inventory():
    print("crawling buzsakilab.com session table", flush=True)
    rows = crawl_lab()
    print("indexing the archive", flush=True)
    sets, assets = dandi_index()
    anat = {s["dandiset"]: s["anatomy"] for s in sets}
    by_key = {}
    for a in assets:
        by_key.setdefault(a["key"], []).append(a)

    today = time.strftime("%Y-%m-%d")
    out, matched = [], 0
    for r in rows:
        hits = by_key.get(key_of(r["animal"], r["date"]), [])
        matched += bool(hits)
        out.append(dict(r, source="databank + archive" if hits else "databank (Globus only)",
                        dandiset=hits[0]["dandiset"] if hits else "",
                        anatomy=anat.get(hits[0]["dandiset"], "") if hits else "",
                        n_nwb=len(hits), nwb_gb=round(sum(h["size_gb"] for h in hits), 2) if hits else "",
                        crawl_date=today, species_norm=norm_species(r["species"])))
    seen = {key_of(r["animal"], r["date"]) for r in rows}
    for a in assets:                      # sessions only the archive has
        if a["key"] in seen:
            continue
        out.append(dict(session=os.path.basename(a["path"]), animal=a["subject"], species="",
                        strain="", project="", date="", paradigms="", manipulations="",
                        owner="", globus_path="", source="archive only",
                        dandiset=a["dandiset"], anatomy=anat.get(a["dandiset"], ""),
                        n_nwb=1, nwb_gb=a["size_gb"], crawl_date=today,
                        species_norm=norm_species(anat.get(a["dandiset"], ""))))
    write_csv("buzsaki_session_inventory.csv", out)
    write_csv("buzsaki_dandisets.csv", sets)
    print(f"inventory: {len(rows)} databank sessions, {len(assets)} archive assets, "
          f"{matched} matched, {len(out)} rows written", flush=True)


# ---------------------------------------------------------------- stage 2
def bound_http_timeouts(connect=15, read=60):
    """Give every range request a deadline.

    remfile calls requests.get with no timeout, and requests with timeout=None
    blocks for ever. One unresponsive file then parks a worker permanently,
    and once all workers are parked the probe stops writing without failing --
    which is exactly how the first run died silently at asset 268.
    """
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


def _probe_one(job):
    """Header-only read of one asset -> one record.

    Never raises: a failure is recorded as an 'error' field so the asset is
    not retried on the next resume.
    """
    import h5py, remfile
    ds, a = job
    rec = dict(dandiset=ds, path=a["path"], asset_id=a["asset_id"],
               size_gb=round(a["size"] / 1e9, 2))
    try:
        url = f"{DANDI}/dandisets/{ds}/versions/draft/assets/{a['asset_id']}/download/"
        f = h5py.File(remfile.File(url), "r")
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
            rec["duration"] = round(float(st[k1][:].max()), 1)
        rec["has_states"] = "state_labels" in rec
        f.close()
    except BaseException as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:150]
    return rec


def stage_probe(workers=10, out_path="probe_states.jsonl"):
    """Open each archive file's header and record: units? sleep states?

    Only small metadata datasets are read, so a 9 GB session costs seconds.
    One JSON line per asset, appended, so the run is resumable: re-running
    after an interruption picks up where it stopped.
    """
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            done = {json.loads(l)["asset_id"] for l in fh if l.strip()}
    jobs = []
    for ds in DANDISETS:
        url = f"{DANDI}/dandisets/{ds}/versions/draft/assets/?page_size=200"
        while url:
            page = api(url)
            jobs += [(ds, a) for a in page["results"] if a["asset_id"] not in done]
            url = page.get("next")
    print(f"probing {len(jobs)} assets", flush=True)
    bound_http_timeouts()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex, open(out_path, "a") as fh:
        for i, rec in enumerate(ex.map(_probe_one, jobs), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)}  ({(time.time()-t0)/i:.1f}s per asset)", flush=True)
    print("probe done", flush=True)


# ---------------------------------------------------------------- stage 3
def read_session(url):
    """Spike times, unit labels and sleep-state intervals from one NWB.

    Reads the HDF5 datasets directly rather than through pynwb: the whole
    spike_times column comes back in one sequential range request instead of
    thousands of small ones, which is the difference between a minute and an
    hour per session.
    """
    import h5py, remfile
    f = h5py.File(remfile.File(url), "r")
    u = f["units"]
    idx = u["spike_times_index"][:]
    st = u["spike_times"][:]
    starts = np.concatenate([[0], idx[:-1]])
    spikes = [st[a:b].astype(np.float64) for a, b in zip(starts, idx)]
    cols = {c: (txt(u[c]) if u[c].dtype.kind in "OSU" else u[c][:])
            for c in ("location", "region", "cell_type") if c in u}
    states = None
    p = f.get("processing/behavior/states")
    if p is not None:
        k1 = "stop_time" if "stop_time" in p else "stop"
        k0 = "start_time" if "start_time" in p else "start"
        states = dict(start=p[k0][:], stop=p[k1][:], label=txt(p["label"]))
    f.close()
    return spikes, cols, states


def bin_and_filter(spikes, dt=BIN_SIZE, tonic_hz=TONIC_HZ):
    """Spike times -> (bins x units) counts, tonic units dropped.

    Units firing above `tonic_hz` are excluded exactly as in Section 3: they
    are active in nearly every bin, so they carry no per-bin information and
    inflate the active fraction.
    """
    t_stop = max((s.max() for s in spikes if len(s)), default=0.0)
    edges = np.arange(0.0, t_stop + dt, dt)
    X = np.stack([np.histogram(s, bins=edges)[0] for s in spikes], axis=1)
    keep = (X.sum(0) / ((len(edges) - 1) * dt)) <= tonic_hz
    return X[:, keep], keep, t_stop


STATE_MAP = {"awake": "Wake", "wake": "Wake", "quiet": "Quiet",
             "nrem": "NREM", "non-rem": "NREM", "rem": "REM"}


def label_bins(states, n_bins):
    """Interval-coded sleep states -> one label per time bin.

    Label vocabularies differ between datasets ('Awake|Non-REM|REM' in the
    rat sets, 'awake|nrem|quiet|rem|transit' in the mouse set), so they are
    normalised here. 'transit' and anything unrecognised stay 'none' and are
    excluded from state comparisons.
    """
    lab = np.full(n_bins, "none", dtype="<U5")
    for s0, s1, raw in zip(states["start"], states["stop"], states["label"]):
        lab[int(s0 // BIN_SIZE):int(s1 // BIN_SIZE)] = STATE_MAP.get(str(raw).strip().lower(), "none")
    return lab


def usable_assets(dandisets, probe_path="probe2.jsonl", min_units=25, min_state_s=60):
    """Sessions worth streaming, taken from the probe output.

    Requires a spike-sorted unit table of at least `min_units` units and at
    least `min_state_s` seconds of each of Wake, NREM and REM, so that every
    extracted session can enter a three-state comparison.
    """
    out = []
    for line in open(probe_path):
        r = json.loads(line)
        if r["dandiset"] not in dandisets or r.get("error") or not r.get("state_secs"):
            continue
        if (r.get("n_units") or 0) < min_units:
            continue
        secs = {STATE_MAP.get(k.lower(), "none"): v for k, v in r["state_secs"].items()}
        if min(secs.get(s, 0) for s in ("Wake", "NREM", "REM")) < min_state_s:
            continue
        out.append(r)
    return out


def stage_extract(dandisets=None, out_path="dandi_binned_1s.npz", cache="dandi_cache"):
    """Stream the usable sessions of `dandisets` and save binned matrices.

    One pickle per session under `cache/`, so an interrupted run resumes
    where it stopped, then one npz holding per session: X__<tag> (bins x
    units, int16 counts), state__<tag> (label per bin), region__<tag> and
    rate__<tag> (mean firing rate per unit, Hz).

    Unlike the first pass, no unit is dropped at extraction time: the rates
    are saved alongside so the tonic-unit threshold can be re-derived per
    dataset in analysis. Mouse granule cells fire far more sparsely than rat
    CA1 pyramidal cells, and a threshold carried over from one to the other
    would not mean the same thing.
    """
    dandisets = dandisets or list(EXTRACT)
    os.makedirs(cache, exist_ok=True)
    bound_http_timeouts()
    todo = usable_assets(dandisets)
    print(f"extracting {len(todo)} sessions from {','.join(dandisets)}", flush=True)
    summary = []
    for i, r in enumerate(todo, 1):
        ds = r["dandiset"]
        name = re.sub(r"_(behavior\+)?ecephys\.nwb$", "", os.path.basename(r["path"]))
        tag = f"{ds}_{name}"
        pkl = os.path.join(cache, tag + ".pkl")
        if os.path.exists(pkl):
            continue
        t0 = time.time()
        try:
            spikes, cols, states = read_session(
                f"{DANDI}/dandisets/{ds}/versions/draft/assets/{r['asset_id']}/download/")
        except Exception as exc:
            print(f"  [{i}/{len(todo)}] FAIL {name}: {type(exc).__name__}: {exc}"[:130], flush=True)
            continue
        if not spikes or states is None:
            print(f"  [{i}/{len(todo)}] skip {name}: no units or no scoring", flush=True)
            continue
        t_stop = max((s.max() for s in spikes if len(s)), default=0.0)
        edges = np.arange(0.0, t_stop + BIN_SIZE, BIN_SIZE)
        X = np.stack([np.histogram(s, bins=edges)[0] for s in spikes], axis=1).astype(np.int16)
        rate = X.sum(0) / ((len(edges) - 1) * BIN_SIZE)
        lab = label_bins(states, X.shape[0])
        reg = cols.get("location", cols.get("region", np.array(["?"] * len(spikes))))
        reg = np.asarray(reg)
        if set(map(str, reg)) <= {"?", "", "nan"}:
            reg = np.array([REGION_HINT.get(ds, "?")] * X.shape[1])
        rec = dict(tag=tag, dandiset=ds, session=name, X=X, state=lab,
                   region=reg, rate=rate.astype(np.float32),
                   region_hint=REGION_HINT.get(ds, ""))
        with open(pkl, "wb") as fh:
            pickle.dump(rec, fh, protocol=4)
        print(f"  [{i}/{len(todo)}] {name}: {X.shape[1]}u x {X.shape[0]}bins, "
              f"{(rate <= TONIC_HZ).sum()} below {TONIC_HZ}Hz, {time.time()-t0:.0f}s", flush=True)

    store = {}
    for pkl in sorted(glob.glob(os.path.join(cache, "*.pkl"))):
        rec = pickle.load(open(pkl, "rb"))
        if "X" not in rec:              # first-pass cache files have another layout
            continue
        tag = rec["tag"]
        store[f"X__{tag}"] = rec["X"]
        store[f"state__{tag}"] = rec["state"]
        store[f"region__{tag}"] = rec["region"]
        store[f"rate__{tag}"] = rec["rate"]
        summary.append(dict(dandiset=rec["dandiset"], session=rec["session"],
                            region_hint=rec.get("region_hint", ""),
                            n_units=int(rec["X"].shape[1]),
                            n_below_tonic=int((rec["rate"] <= TONIC_HZ).sum()),
                            duration_s=int(rec["X"].shape[0] * BIN_SIZE),
                            **{f"bins_{s}": int((rec["state"] == s).sum())
                               for s in ("Wake", "NREM", "REM", "Quiet")}))
    np.savez_compressed(out_path, **store)
    write_csv("dandi_session_summary_hpc.csv", summary)
    print(f"extract: {len(summary)} sessions -> {out_path} "
          f"({os.path.getsize(out_path)/1e6:.1f} MB)", flush=True)


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "inventory"
    if stage == "extract":
        # extract [dandiset,dandiset,...] [out.npz] [cache_dir]
        dsets = sys.argv[2].split(",") if len(sys.argv) > 2 else None
        out = sys.argv[3] if len(sys.argv) > 3 else "dandi_binned_1s.npz"
        cache = sys.argv[4] if len(sys.argv) > 4 else "dandi_cache"
        stage_extract(dsets, out, cache)
    else:
        {"inventory": stage_inventory, "probe": stage_probe}[stage]()
