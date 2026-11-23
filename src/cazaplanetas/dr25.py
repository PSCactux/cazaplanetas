"""Our own Kepler DR25 dataset: every DR25 signal (TCE) turned into compact views.

Steps (see ``scripts/hito3_dr25.py``):

1. :func:`fetch_tce_table` downloads the DR25 TCE table (34,032 rows) from the
   NASA Exoplanet Archive into ``data/dr25/``.
2. :func:`process_star` downloads every long-cadence light curve of one star from
   the public MAST copy on AWS (``stpubdata``), checks each file's MD5 against the
   archive's ETag, reads it **in memory** (FITS files never touch the disk),
   detrends it and computes the views of every TCE of that star.
3. :func:`build` runs stars in batches (shards) with a process pool; each batch
   becomes one ``.npz`` written atomically, so the run can be stopped and resumed
   at any time. Stars that fail are listed with the reason.
4. :func:`write_manifest` records SHA-256 sums, sizes and counts.

The whole pipeline is written for this project; it uses the public view geometry
of Shallue & Vanderburg (2018) and the DR25 catalogue columns (Thompson et al. 2018).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import threading
import time
import traceback
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from cazaplanetas.env import DATA_DIR, use_local_cache

use_local_cache()  # before astropy

import numpy as np  # noqa: E402
import requests  # noqa: E402
from astropy.io import fits  # noqa: E402

from cazaplanetas import views as V  # noqa: E402

DR25_DIR = DATA_DIR / "dr25"
TCE_CSV = DR25_DIR / "q1_q17_dr25_tce.csv"
DR24_CSV = DR25_DIR / "q1_q17_dr24_tce.csv"
STELLAR_KEPMAG_CSV = DR25_DIR / "q1_q17_dr25_ks_kepmag.csv"  # milestone 5: Kepler magnitude only
SHARD_DIR = DR25_DIR / "views"
MANIFEST = DR25_DIR / "manifest.json"

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
S3_BUCKET = "https://stpubdata.s3.amazonaws.com"
S3_PREFIX = "kepler/public/lightcurves/{kic4}/{kic9}/"

# Kepler quality flags rejected (lightkurve's "default" bitmask): attitude tweak,
# safe mode, coarse point, Earth point, desaturation, manual exclude, detector
# anomaly, no data, thruster firing.
QUALITY_BITMASK = 1130799

DATASET_VERSION = "dr25-views-v1"


# --- Catalogue tables ----------------------------------------------------------


def _tap_csv(query: str, out: Path, retries: int = 3) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(TAP_URL, params={"query": query, "format": "csv"}, timeout=300)
            r.raise_for_status()
            tmp = out.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(out)
            return out
        except requests.RequestException as e:
            print(f"  TAP query failed (attempt {attempt}): {e}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"could not run TAP query for {out.name}")


def fetch_tce_table(out: Path = TCE_CSV) -> Path:
    """DR25 TCE table (all columns). Skipped if already downloaded."""
    if not out.exists():
        _tap_csv("select * from q1_q17_dr25_tce order by kepid, tce_plnt_num", out)
    return out


def fetch_stellar_kepmag(out: Path = STELLAR_KEPMAG_CSV) -> Path:
    """Kepler magnitude of every star of the DR25 stellar table (``q1_q17_dr25_ks``), and
    nothing else from that table (it also has KOI and planet counts, which are labels)."""
    if not out.exists():
        _tap_csv("select kepid, kepmag from q1_q17_dr25_ks order by kepid", out)
    return out


def fetch_dr24_table(out: Path = DR24_CSV) -> Path:
    """DR24 TCE ephemerides and Autovetter labels, to match DR25 signals with Astronet's."""
    if not out.exists():
        _tap_csv(
            "select kepid, tce_plnt_num, tce_period, tce_time0bk, tce_duration, av_training_set "
            "from q1_q17_dr24_tce order by kepid, tce_plnt_num",
            out,
        )
    return out


def read_csv(path: Path) -> dict[str, np.ndarray]:
    """CSV -> {column: array}; numeric columns become float (empty -> NaN)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    cols = {}
    for j, name in enumerate(header):
        raw = [r[j] for r in body]
        try:
            cols[name] = np.array([float(x) if x != "" else np.nan for x in raw])
        except ValueError:
            cols[name] = np.array(raw)
    for name in ("kepid", "tce_plnt_num"):
        if name in cols:
            cols[name] = cols[name].astype(np.int64)
    return cols


@dataclass(frozen=True)
class TCE:
    kepid: int
    tce_plnt_num: int
    period: float  # days
    t0: float  # BKJD
    duration: float  # days
    secondary_phase: float  # days after the primary (NaN if unknown)


def tces_by_star(table: dict[str, np.ndarray]) -> dict[int, list[TCE]]:
    stars: dict[int, list[TCE]] = {}
    for i in range(len(table["kepid"])):
        t = TCE(
            int(table["kepid"][i]),
            int(table["tce_plnt_num"][i]),
            float(table["tce_period"][i]),
            float(table["tce_time0bk"][i]),
            float(table["tce_duration"][i]) / 24.0,
            float(table["tce_maxmesd"][i]) if "tce_maxmesd" in table else float("nan"),
        )
        stars.setdefault(t.kepid, []).append(t)
    return dict(sorted(stars.items()))


# --- Download (in memory) -------------------------------------------------------

_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = _local.session = requests.Session()
    return s


def _get(url: str, retries: int = 6, deadline: float = 120.0, **kw) -> bytes:
    """GET with retries. Each attempt has a connect/read timeout *and* a total
    deadline, so a connection that trickles or stalls is dropped and retried."""
    last = None
    for attempt in range(retries):
        try:
            start = time.monotonic()
            with _session().get(url, timeout=(15, 30), stream=True, **kw) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                if r.status_code >= 500 or r.status_code == 429:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                chunks = []
                for chunk in r.iter_content(1 << 16):
                    chunks.append(chunk)
                    if time.monotonic() - start > deadline:
                        raise requests.Timeout(f"slower than {deadline:.0f} s")
                return b"".join(chunks)
        except FileNotFoundError:
            raise
        except requests.RequestException as e:
            last = e
            _local.session = None  # fresh connection next time
        time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"download failed after {retries} attempts: {last}")


def list_light_curves(kepid: int) -> list[tuple[str, int, str]]:
    """(key, size, md5) of every long-cadence file of a star in the MAST S3 bucket."""
    kic9 = f"{kepid:09d}"
    prefix = S3_PREFIX.format(kic4=kic9[:4], kic9=kic9)
    content = _get(f"{S3_BUCKET}/", params={"list-type": "2", "prefix": prefix})
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(content)
    out = []
    for c in root.findall("s3:Contents", ns):
        key = c.find("s3:Key", ns).text
        if key.endswith("_llc.fits"):
            out.append((key, int(c.find("s3:Size", ns).text), c.find("s3:ETag", ns).text.strip('"')))
    return sorted(out)


def download_file(key: str, size: int, md5: str) -> bytes:
    for attempt in range(3):
        data = _get(f"{S3_BUCKET}/{key}")
        if len(data) == size and hashlib.md5(data).hexdigest() == md5:
            return data
        time.sleep(2)
    raise RuntimeError(f"checksum mismatch for {key.rsplit('/', 1)[-1]}")


# --- One star -------------------------------------------------------------------


@dataclass
class Quarter:
    quarter: int
    time: np.ndarray
    flux: np.ndarray
    centr1: np.ndarray
    centr2: np.ndarray


def read_quarter(data: bytes) -> Quarter | None:
    """Good long-cadence points of one FITS file (PDCSAP flux and flux-weighted centroids)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(io.BytesIO(data), memmap=False) as hdul:
            q = int(hdul[0].header.get("QUARTER", -1))
            d = hdul[1].data
            t = np.asarray(d["TIME"], dtype=np.float64)
            f = np.asarray(d["PDCSAP_FLUX"], dtype=np.float64)
            qual = np.asarray(d["SAP_QUALITY"], dtype=np.int64)
            c1 = np.asarray(d["MOM_CENTR1"], dtype=np.float64)
            c2 = np.asarray(d["MOM_CENTR2"], dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(f) & ((qual & QUALITY_BITMASK) == 0)
    if ok.sum() < 50:
        return None
    order = np.argsort(t[ok], kind="stable")
    return Quarter(q, t[ok][order], f[ok][order], c1[ok][order], c2[ok][order])


def stitch(quarters: list[Quarter], tces: list[TCE], mask_transits: bool = True) -> tuple[V.LightCurve, float]:
    """Normalise, detrend (all TCE transits masked) and stitch the quarters of one star.
    The spline knot spacing is chosen once per star by BIC. Returns (light curve, spacing).
    ``mask_transits=False`` fits the spline through the transits, as Astronet did; it is
    only used to check that our views reproduce theirs."""
    quarters = sorted(quarters, key=lambda q: q.time[0])
    masks, rels = [], []
    for q in quarters:
        mask = np.zeros(len(q.time), bool)
        for tce in tces if mask_transits else []:
            m = V.transit_mask(q.time, tce.period, tce.t0, tce.duration)
            if m.mean() < 0.5:  # a signal covering most of the orbit cannot be masked
                mask |= m
        masks.append(mask)
        rels.append(q.flux / np.nanmedian(q.flux))
    spacing = V.choose_spacing([(q.time, r, m) for q, r, m in zip(quarters, rels, masks)])
    times, fluxes, cents = [], [], []
    for q, rel, mask in zip(quarters, rels, masks):
        flux = rel / V.detrend(q.time, rel, mask, spacing) - 1.0
        dist = np.full(len(q.time), np.nan)
        okc = np.isfinite(q.centr1) & np.isfinite(q.centr2)
        if okc.sum() > 50:
            tc = q.time[okc]
            dx = q.centr1[okc] - V.detrend(tc, q.centr1[okc], mask[okc], spacing)
            dy = q.centr2[okc] - V.detrend(tc, q.centr2[okc], mask[okc], spacing)
            dist[okc] = np.hypot(dx, dy) * V.KEPLER_ARCSEC_PER_PIXEL
        times.append(q.time)
        fluxes.append(flux)
        cents.append(dist)
    return V.LightCurve(np.concatenate(times), np.concatenate(fluxes), np.concatenate(cents)), spacing


_pool: ThreadPoolExecutor | None = None


def process_star(kepid: int, tces: list[TCE], threads: int = 3, mask_transits: bool = True) -> dict:
    """Download, detrend and compute the views of every TCE of one star.

    Returns {"kepid", "sources": [(file, size, md5)], "tces": [{..., "views", "diag"} or
    {..., "error"}], "error": str | None, "seconds"}."""
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(threads)
    start = time.perf_counter()
    out = {"kepid": kepid, "sources": [], "tces": [], "error": None}
    try:
        files = list_light_curves(kepid)
        if not files:
            raise RuntimeError("no long-cadence light curves in the archive")
        blobs = list(_pool.map(lambda f: download_file(*f), files))
        out["sources"] = [(k.rsplit("/", 1)[-1], s, m) for k, s, m in files]
        quarters = [q for q in (read_quarter(b) for b in blobs) if q is not None]
        del blobs
        if not quarters:
            raise RuntimeError("no usable data points")
        lc, out["spline_spacing"] = stitch(quarters, tces, mask_transits)
    except Exception as e:  # noqa: BLE001 - every failure is recorded with its reason
        out["error"] = f"{type(e).__name__}: {e}"
        out["seconds"] = time.perf_counter() - start
        return out
    for tce in tces:
        entry = {"tce_plnt_num": tce.tce_plnt_num}
        try:
            views, diag = V.tce_views(lc, tce.period, tce.t0, tce.duration, tce.secondary_phase)
            bad = [k for k in ("global_view", "local_view") if not np.isfinite(views[k]).all()]
            if bad:
                raise ValueError(f"no data in {', '.join(bad)}")
            # Odd/even/secondary/centroid views can be empty (e.g. only odd transits fall on
            # good data): they become flat zeros and the empty_* diagnostic says so.
            for k, v in views.items():
                if not np.isfinite(v).all():
                    views[k] = np.zeros_like(v)
            if diag["n_transits"] == 0:
                raise ValueError("no transit falls on good data")
            entry["views"], entry["diag"] = views, diag
        except Exception as e:  # noqa: BLE001
            entry["error"] = f"{type(e).__name__}: {e}"
            if not isinstance(e, ValueError):
                entry["error"] += " | " + traceback.format_exc(limit=1).strip().splitlines()[-1]
        out["tces"].append(entry)
    out["seconds"] = time.perf_counter() - start
    return out


# --- Batches ----------------------------------------------------------------------

DIAG_KEYS = (
    "n_points",
    "n_transits",
    "n_odd_transits",
    "n_even_transits",
    *(f"empty_{k}" for k in V.VIEW_NAMES),
)


def shard_path(i: int) -> Path:
    return SHARD_DIR / f"shard_{i:04d}.npz"


def write_shard(i: int, results: list[dict]) -> Path:
    """One batch -> shard_XXXX.npz (views + diagnostics), .sources.json.gz (files and MD5)
    and .failures.json. Written to temporary names first, then renamed."""
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    rows, failures, sources = [], [], {}
    for r in sorted(results, key=lambda r: r["kepid"]):
        sources[str(r["kepid"])] = r["sources"]
        if r["error"]:
            failures.append({"kepid": r["kepid"], "tce_plnt_num": None, "reason": r["error"]})
            continue
        for e in r["tces"]:
            if "error" in e:
                failures.append({"kepid": r["kepid"], "tce_plnt_num": e["tce_plnt_num"], "reason": e["error"]})
            else:
                rows.append((r["kepid"], e))
    arrays = {
        "kepid": np.array([k for k, _ in rows], dtype=np.int64),
        "tce_plnt_num": np.array([e["tce_plnt_num"] for _, e in rows], dtype=np.int64),
    }
    for name in V.VIEW_NAMES:
        arrays[name] = np.array([e["views"][name] for _, e in rows], dtype=np.float32).reshape(-1, V.VIEW_LENGTHS[name])
    for k in DIAG_KEYS:
        arrays[k] = np.array([e["diag"][k] for _, e in rows], dtype=np.int32)

    path = shard_path(i)
    base = path.with_suffix("")
    with gzip.open(f"{base}.sources.json.gz.part", "wt") as f:
        json.dump(sources, f)
    Path(f"{base}.failures.json.part").write_text(json.dumps(failures, indent=0))
    with open(f"{base}.npz.part", "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(f"{base}.sources.json.gz.part", f"{base}.sources.json.gz")
    os.replace(f"{base}.failures.json.part", f"{base}.failures.json")
    os.replace(f"{base}.npz.part", path)  # the .npz appears last: it marks the batch as done
    return path


def free_gb(path: Path = DATA_DIR) -> float:
    return shutil.disk_usage(path).free / 1e9


def _init_worker() -> None:
    use_local_cache()


def build(
    stars: dict[int, list[TCE]],
    batch_size: int = 500,
    workers: int = 10,
    threads: int = 3,
    min_free_gb: float = 30.0,
    limit_batches: int | None = None,
    log=print,
) -> dict:
    """Process every star in batches; already finished batches are skipped (resume)."""
    kepids = list(stars)
    batches = [kepids[i : i + batch_size] for i in range(0, len(kepids), batch_size)]
    stats = {"batches": len(batches), "done_before": 0, "min_free_gb": free_gb(), "seconds": 0.0}
    os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
    start = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
        ran = 0
        for i, batch in enumerate(batches):
            if shard_path(i).exists():
                stats["done_before"] += 1
                continue
            if limit_batches is not None and ran >= limit_batches:
                break
            free = free_gb()
            stats["min_free_gb"] = min(stats["min_free_gb"], free)
            if free < min_free_gb:
                raise SystemExit(f"only {free:.1f} GB free (< {min_free_gb} GB): stopping before batch {i}")
            t = time.perf_counter()
            futures = [ex.submit(process_star, k, stars[k], threads) for k in batch]
            results = [f.result() for f in as_completed(futures)]
            write_shard(i, results)
            n_fail = sum(1 for r in results if r["error"]) + sum(1 for r in results for e in r["tces"] if "error" in e)
            n_files = sum(len(r["sources"]) for r in results)
            mb = sum(s for r in results for _, s, _ in r["sources"]) / 1e6
            dt = time.perf_counter() - t
            ran += 1
            log(
                f"batch {i + 1}/{len(batches)}: {len(batch)} stars, {n_files} files, {mb:.0f} MB in {dt:.0f} s "
                f"({mb / dt:.1f} MB/s), {n_fail} failures, {free_gb():.1f} GB free"
            )
    stats["seconds"] = time.perf_counter() - start
    stats["min_free_gb"] = min(stats["min_free_gb"], free_gb())
    return stats


def redo_failures(stars: dict[int, list[TCE]], batch_size: int = 500, workers: int = 10, threads: int = 3, log=print) -> int:
    """Process again the stars with any failure (download errors are often transient, and
    the code may have been fixed) and rewrite their shards with the new results."""
    kepids = list(stars)
    redone = 0
    for path in shard_files():
        i = int(path.stem.split("_")[1])
        fails = json.loads(Path(f"{path.with_suffix('')}.failures.json").read_text())
        if not fails:
            continue
        batch = kepids[i * batch_size : (i + 1) * batch_size]
        failed = sorted({f["kepid"] for f in fails})
        with ProcessPoolExecutor(min(workers, len(failed)), initializer=_init_worker) as ex:
            new = {r["kepid"]: r for r in ex.map(process_star, failed, [stars[k] for k in failed], [threads] * len(failed))}
        old = _results_from_shard(i)
        results = [new.get(k, old.get(k)) for k in batch if k in new or k in old]
        write_shard(i, results)
        redone += len(failed)
        n_left = len(json.loads(Path(f"{path.with_suffix('')}.failures.json").read_text()))
        log(f"redo shard {i}: {len(failed)} stars processed again, {len(fails)} -> {n_left} failures")
    return redone


def _results_from_shard(i: int) -> dict[int, dict]:
    """Rebuild process_star-like results from a written shard (to rewrite it partially)."""
    path = shard_path(i)
    base = path.with_suffix("")
    with gzip.open(f"{base}.sources.json.gz", "rt") as f:
        sources = json.load(f)
    fails = json.loads(Path(f"{base}.failures.json").read_text())
    out: dict[int, dict] = {int(k): {"kepid": int(k), "sources": v, "tces": [], "error": None} for k, v in sources.items()}
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    for j, k in enumerate(arrays["kepid"].tolist()):
        out[k]["tces"].append({
            "tce_plnt_num": int(arrays["tce_plnt_num"][j]),
            "views": {name: arrays[name][j] for name in V.VIEW_NAMES},
            "diag": {d: int(arrays[d][j]) for d in DIAG_KEYS},
        })
    for f in fails:
        if f["tce_plnt_num"] is None:
            out[f["kepid"]]["error"] = f["reason"]
        else:
            out[f["kepid"]]["tces"].append({"tce_plnt_num": f["tce_plnt_num"], "error": f["reason"]})
    return out


# --- Reading the dataset back ------------------------------------------------------


def shard_files() -> list[Path]:
    return sorted(SHARD_DIR.glob("shard_*.npz"))


def load(views: tuple[str, ...] = V.VIEW_NAMES) -> dict[str, np.ndarray]:
    """Concatenate every shard: kepid, tce_plnt_num, the requested views and diagnostics."""
    parts: dict[str, list] = {}
    for p in shard_files():
        with np.load(p) as z:
            for k in ("kepid", "tce_plnt_num", *views, *DIAG_KEYS):
                parts.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def failures() -> list[dict]:
    out = []
    for p in sorted(SHARD_DIR.glob("shard_*.failures.json")):
        out += json.loads(p.read_text())
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(extra: dict | None = None, path: Path = MANIFEST) -> dict:
    files = [TCE_CSV, *sorted(SHARD_DIR.glob("shard_*"))]
    entries = [
        {"path": str(p.relative_to(DR25_DIR)), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in files
        if p.exists() and not p.name.endswith(".part")
    ]
    manifest = {
        "version": DATASET_VERSION,
        "created": time.strftime("%Y-%m-%d"),
        "views": {k: V.VIEW_LENGTHS[k] for k in V.VIEW_NAMES},
        "view_geometry": {
            "global": "whole orbit, 2001 bins of width P/2001, median per bin",
            "local": f"+-{V.LOCAL_DURATIONS:g} durations, 201 bins of width {V.LOCAL_BIN_WIDTH} durations, median per bin",
            "odd_even": "local view of odd / even transit numbers",
            "secondary": "local view centred on tce_maxmesd (P/2 if missing)",
            "centroid": "flux-weighted centroid distance from its spline trend, arcsec",
            "units": "relative flux minus 1 (transits negative); empty bins linearly interpolated",
        },
        "detrending": "cubic spline per quarter segment, knots every ~1.5 d, 3-sigma clipping, all TCE transits masked",
        "quality_bitmask": QUALITY_BITMASK,
        "source": f"{S3_BUCKET}/{S3_PREFIX} (MAST public Kepler data on AWS), MD5 checked against ETag",
        "total_bytes": sum(e["bytes"] for e in entries),
        "files": entries,
        **(extra or {}),
    }
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


# --- Matching with DR24 / Astronet -------------------------------------------------


def match_dr24(dr25: dict[str, np.ndarray], dr24: dict[str, np.ndarray], max_rel_period: float = 1e-3, max_phase_dur: float = 0.5) -> dict[tuple[int, int], tuple[int, int]]:
    """Map each DR24 signal (kepid, tce_plnt_num) to the DR25 signal of the same star with
    the same period (relative difference < ``max_rel_period``) and transit times that agree
    (DR24 epoch within ``max_phase_dur`` durations of a DR25 transit). Planet numbers
    are not stable between releases, so they are not used."""
    by_star: dict[int, list[int]] = {}
    for i, k in enumerate(dr25["kepid"]):
        by_star.setdefault(int(k), []).append(i)
    out = {}
    for j in range(len(dr24["kepid"])):
        k = int(dr24["kepid"][j])
        p24, t24, d24 = dr24["tce_period"][j], dr24["tce_time0bk"][j], dr24["tce_duration"][j] / 24
        best = None
        for i in by_star.get(k, []):
            p25, t25 = dr25["tce_period"][i], dr25["tce_time0bk"][i]
            rel = abs(p25 - p24) / p24
            dphase = abs(V.fold(np.array([t24]), p25, t25)[0])
            if rel < max_rel_period and dphase < max_phase_dur * d24 and (best is None or rel < best[0]):
                best = (rel, i)
        if best is not None:
            i = best[1]
            out[(k, int(dr24["tce_plnt_num"][j]))] = (k, int(dr25["tce_plnt_num"][i]))
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")
