"""Minimal light-curve pipeline: download -> clean -> BLS search -> fold.

This is the classical baseline (Box Least Squares, Kovacs et al. 2002) that the
learned model will later be compared against. Import :mod:`cazaplanetas.env` and
call ``use_local_cache()`` before importing this module so caches stay in ``data/``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from astropy.timeseries import BoxLeastSquares

import lightkurve as lk

# Kepler long cadence: 29.4 minutes.
KEPLER_LC_DAYS = 0.0204336


@dataclass
class BLSResult:
    period: float  # days
    t0: float  # BKJD (BJD - 2454833)
    duration: float  # days
    depth_ppm: float
    sde: float  # signal detection efficiency of the peak
    n_periods: int
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def download_kepler_long_cadence(target: str, download_dir: Path) -> lk.LightCurveCollection:
    """Download every Kepler long-cadence PDCSAP light curve of ``target`` from MAST."""
    search = lk.search_lightcurve(target, mission="Kepler", author="Kepler", exptime=1800)
    if len(search) == 0:
        raise RuntimeError(f"No Kepler long-cadence light curves found for {target!r}")
    return search.download_all(download_dir=str(download_dir), flux_column="pdcsap_flux")


def clean(collection: lk.LightCurveCollection, window_days: float = 2.0) -> lk.LightCurve:
    """Stitch quarters, drop NaNs, remove slow trends and upward outliers.

    The Savitzky-Golay window (``window_days``) must be much longer than the
    transits we want to keep (Kepler-10 b lasts ~1.8 h), otherwise the filter
    eats them.
    """
    lc = collection.stitch().remove_nans()
    window = int(round(window_days / KEPLER_LC_DAYS)) | 1  # odd number of cadences
    lc = lc.flatten(window_length=window)
    # Clip flares/cosmic rays upward only: dips are exactly what we are looking for.
    lc = lc.remove_outliers(sigma_upper=4, sigma_lower=np.inf)
    return lc


def period_grid(baseline_days: float, pmin: float, pmax: float, min_duration: float, oversample: float = 3.0) -> np.ndarray:
    """Log-uniform period grid fine enough that a transit never drifts out of phase.

    Over a baseline T, a period error dP shifts the last transit by (T/P)*dP.
    Keeping that below duration/oversample gives a constant step in ln(P).
    """
    step = min_duration / (baseline_days * oversample)
    n = int(np.ceil(np.log(pmax / pmin) / step))
    return np.exp(np.linspace(np.log(pmin), np.log(pmax), n))


def bls_search(
    lc: lk.LightCurve,
    pmin: float = 0.5,
    pmax: float = 20.0,
    durations: tuple[float, ...] = (0.04, 0.06, 0.08, 0.11, 0.15, 0.2, 0.3),
) -> tuple[BLSResult, np.ndarray, np.ndarray]:
    """Run BLS over a period range. Returns the best peak plus the periodogram."""
    t = np.asarray(lc.time.value, dtype=float)
    y = np.asarray(lc.flux.value, dtype=float)
    dy = np.asarray(lc.flux_err.value, dtype=float)
    periods = period_grid(t.max() - t.min(), pmin, pmax, min(durations))

    start = time.perf_counter()
    bls = BoxLeastSquares(t, y, dy)
    pg = bls.power(periods, list(durations), objective="snr")
    seconds = time.perf_counter() - start

    power = np.asarray(pg.power)
    i = int(np.nanargmax(power))
    sde = float((power[i] - np.nanmean(power)) / np.nanstd(power))
    result = BLSResult(
        period=float(pg.period[i]),
        t0=float(pg.transit_time[i]),
        duration=float(pg.duration[i]),
        depth_ppm=float(pg.depth[i] * 1e6),
        sde=sde,
        n_periods=len(periods),
        seconds=seconds,
    )
    return result, periods, power


def phase_hours(t: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Time from the nearest mid-transit, in hours."""
    return ((t - t0 + 0.5 * period) % period - 0.5 * period) * 24.0


def binned(x: np.ndarray, y: np.ndarray, width: float, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and standard error of ``y`` in bins of ``width`` over [lo, hi]."""
    edges = np.arange(lo, hi + width, width)
    idx = np.digitize(x, edges) - 1
    centers, means, errs = [], [], []
    for k in range(len(edges) - 1):
        sel = idx == k
        if sel.sum() < 3:
            continue
        centers.append(0.5 * (edges[k] + edges[k + 1]))
        means.append(y[sel].mean())
        errs.append(y[sel].std(ddof=1) / np.sqrt(sel.sum()))
    return np.array(centers), np.array(means), np.array(errs)
