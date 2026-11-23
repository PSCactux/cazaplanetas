"""Turn a Kepler light curve into the folded "views" a transit classifier reads.

Pure numpy/scipy, no I/O, so it can be tested offline with synthetic curves.

* :func:`detrend` removes slow stellar and instrumental trends with a robust
  cubic spline per continuous segment, fitted with the transits of every known
  signal on the star masked out (so the spline cannot eat them).
* :func:`binned_median` is a vectorised median in (possibly overlapping) bins.
* :func:`tce_views` folds the detrended flux and centroid on one signal's
  period and returns the global, local, odd, even, secondary and centroid views.

View geometry follows Shallue & Vanderburg (2018, Sec. 3): the global view
covers the whole orbit in 2001 bins of width P/2001; the local view covers
+-4 transit durations in 201 bins of width 0.16 durations (bins overlap).
Written from the paper's description; no code was copied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_lsq_spline

GLOBAL_BINS, LOCAL_BINS = 2001, 201
LOCAL_DURATIONS, LOCAL_BIN_WIDTH = 4.0, 0.16  # half-width and bin width, in transit durations
KEPLER_ARCSEC_PER_PIXEL = 3.98

VIEW_NAMES = (
    "global_view",
    "local_view",
    "odd_view",
    "even_view",
    "secondary_view",
    "centroid_global_view",
    "centroid_local_view",
)
VIEW_LENGTHS = {name: GLOBAL_BINS if name.endswith("global_view") else LOCAL_BINS for name in VIEW_NAMES}


# --- Detrending --------------------------------------------------------------


def segments(t: np.ndarray, max_gap: float = 0.75) -> list[slice]:
    """Split a sorted time array wherever consecutive points are more than ``max_gap`` days apart."""
    if len(t) == 0:
        return []
    cuts = np.flatnonzero(np.diff(t) > max_gap) + 1
    edges = np.r_[0, cuts, len(t)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def _fit_segment(t: np.ndarray, y: np.ndarray, good: np.ndarray, spacing: float, n_iter: int, clip: float) -> tuple[np.ndarray, int]:
    """Robust least-squares cubic spline of one segment. Returns (trend, number of coefficients)."""
    k = 3
    g = good.copy()
    trend = np.full(len(t), np.nanmedian(y[good]) if good.any() else np.nan)
    n_coef = 1
    for _ in range(n_iter + 1):
        tg, yg = t[g], y[g]
        if len(tg) < 2 * (k + 1):
            return trend, n_coef
        n_inner = int((tg[-1] - tg[0]) / spacing)
        # Knots at quantiles of the good points: every knot interval holds data
        # (Schoenberg-Whitney), even next to long masked transits.
        inner = np.unique(tg[np.linspace(0, len(tg) - 1, n_inner + 2)[1:-1].astype(int)]) if n_inner > 0 else np.array([])
        inner = inner[(inner > tg[0]) & (inner < tg[-1])]
        knots = np.r_[[tg[0]] * (k + 1), inner, [tg[-1]] * (k + 1)]
        try:
            spline = make_lsq_spline(tg, yg, knots, k=k, method="norm-eq")
        except (ValueError, np.linalg.LinAlgError):
            return trend, n_coef
        trend, n_coef = spline(t), len(knots) - k - 1
        resid = y - trend
        sigma = 1.4826 * np.median(np.abs(resid[g] - np.median(resid[g])))
        if not np.isfinite(sigma) or sigma == 0:
            break
        new = good & (np.abs(resid) < clip * sigma)
        if np.array_equal(new, g):
            break
        g = new
    return trend, n_coef


# Knot spacings tried by :func:`choose_spacing` (Shallue & Vanderburg 2018 chose
# among 0.5-20 days by the Bayesian information criterion).
SPACINGS = tuple(np.logspace(np.log10(0.5), np.log10(20.0), 20))


def _fit(t: np.ndarray, y: np.ndarray, mask: np.ndarray, spacing: float, n_iter: int, clip: float) -> tuple[np.ndarray, int]:
    trend = np.empty(len(t))
    n_coef = 0
    for s in segments(t):
        ys = y[s]
        good = ~mask[s] & np.isfinite(ys)
        if good.sum() < 20:  # transits cover the whole segment: fit without the mask
            good = np.isfinite(ys)
        trend[s], n = _fit_segment(t[s], ys, good, spacing, n_iter, clip)
        n_coef += n
    return trend, n_coef


def noise_level(y: np.ndarray) -> float:
    """Point-to-point scatter, robust to trends and outliers: 1.4826 MAD(diff) / sqrt(2)."""
    d = np.diff(y[np.isfinite(y)])
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if len(d) else float("nan")


def choose_spacing(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]], spacings=SPACINGS, n_iter: int = 3, clip: float = 3.0) -> float:
    """Knot spacing with the lowest BIC = chi^2 + (coefficients) ln(points), summed over
    the (time, flux, mask) ``parts`` of one star (e.g. its quarters)."""
    best, best_bic = spacings[0], np.inf
    for sp in spacings:
        chi2, n_coef, n_pts = 0.0, 0, 0
        for t, y, mask in parts:
            trend, n = _fit(t, y, mask, sp, n_iter, clip)
            ok = ~mask & np.isfinite(y)
            sigma = noise_level(y)
            if not np.isfinite(sigma) or sigma == 0 or ok.sum() == 0:
                continue
            chi2 += float(np.sum(((y[ok] - trend[ok]) / sigma) ** 2))
            n_coef += n
            n_pts += int(ok.sum())
        bic = chi2 + n_coef * np.log(max(n_pts, 2))
        if bic < best_bic:
            best, best_bic = sp, bic
    return float(best)


def detrend(t: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None, spacing: float = 1.5, n_iter: int = 3, clip: float = 3.0) -> np.ndarray:
    """Robust cubic-spline trend of ``y`` (knots every ~``spacing`` days, iterative
    ``clip``-sigma clipping), fitted per segment ignoring points where ``mask`` is True.
    Returns the trend evaluated at every ``t`` (sorted)."""
    mask = np.zeros(len(t), bool) if mask is None else mask
    return _fit(t, y, mask, spacing, n_iter, clip)[0]


def transit_mask(t: np.ndarray, period: float, t0: float, duration: float, width: float = 2.0) -> np.ndarray:
    """True within ``width`` x duration (total) of each transit centre."""
    phase = fold(t, period, t0)
    return np.abs(phase) < 0.5 * width * duration


# --- Folding and binning -----------------------------------------------------


def fold(t: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Time from the nearest event centre, in [-P/2, P/2)."""
    return np.mod(t - t0 + 0.5 * period, period) - 0.5 * period


def binned_median(x: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median of ``y`` for ``lo[i] <= x < hi[i]`` (bins may overlap). ``x`` must be sorted.
    Returns (medians with NaN for empty bins, number of points per bin)."""
    start = np.searchsorted(x, lo, "left")
    end = np.searchsorted(x, hi, "left")
    counts = end - start
    out = np.full(len(lo), np.nan)
    total = int(counts.sum())
    if total == 0:
        return out, counts
    first = np.cumsum(counts) - counts
    bin_id = np.repeat(np.arange(len(lo)), counts)
    idx = np.repeat(start, counts) + (np.arange(total) - np.repeat(first, counts))
    vals = y[idx]
    sorted_vals = vals[np.lexsort((vals, bin_id))]
    nz = counts > 0
    a = first[nz] + (counts[nz] - 1) // 2
    b = first[nz] + counts[nz] // 2
    out[nz] = 0.5 * (sorted_vals[a] + sorted_vals[b])
    return out, counts


def fill_empty(view: np.ndarray) -> np.ndarray:
    """Linear interpolation over empty (NaN) bins; constant beyond the ends."""
    ok = np.isfinite(view)
    if ok.all() or not ok.any():
        return view
    i = np.arange(len(view))
    return np.interp(i, i[ok], view[ok])


def global_bins(period: float, n: int = GLOBAL_BINS) -> tuple[np.ndarray, np.ndarray]:
    lo = -0.5 * period + period * np.arange(n) / n
    return lo, lo + period / n


def local_bins(period: float, duration: float, n: int = LOCAL_BINS) -> tuple[np.ndarray, np.ndarray]:
    t_min = max(-0.5 * period, -LOCAL_DURATIONS * duration)
    t_max = min(0.5 * period, LOCAL_DURATIONS * duration)
    width = LOCAL_BIN_WIDTH * duration
    lo = t_min + (t_max - t_min - width) * np.arange(n) / (n - 1)
    return lo, lo + width


def _view(phase: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, int]:
    ok = np.isfinite(y)
    phase, y = phase[ok], y[ok]
    order = np.argsort(phase, kind="stable")
    med, counts = binned_median(phase[order], y[order], lo, hi)
    return fill_empty(med), int((counts == 0).sum())


# --- Per-signal views --------------------------------------------------------


@dataclass
class LightCurve:
    """Detrended Kepler data of one star, all quarters stitched and sorted by time."""

    time: np.ndarray  # BKJD (BJD - 2454833), days
    flux: np.ndarray  # relative flux minus 1 (out of transit ~ 0, transits negative)
    centroid: np.ndarray  # centroid distance from its trend, arcsec (NaN where unavailable)

    def __len__(self) -> int:
        return len(self.time)


def tce_views(lc: LightCurve, period: float, t0: float, duration: float, secondary_phase: float | None = None) -> tuple[dict, dict]:
    """All views of one signal. ``period``, ``t0`` (BKJD) and ``duration`` in days;
    ``secondary_phase`` is the time of the secondary event after the primary (days),
    P/2 if not given. Returns (views, diagnostics)."""
    phase = fold(lc.time, period, t0)
    epoch = np.floor((lc.time - t0) / period + 0.5).astype(np.int64)
    odd = epoch % 2 == 1
    in_transit = np.abs(phase) < 0.5 * duration
    if secondary_phase is None or not np.isfinite(secondary_phase):
        secondary_phase = 0.5 * period
    phase2 = fold(lc.time, period, t0 + secondary_phase)

    g_lo, g_hi = global_bins(period)
    l_lo, l_hi = local_bins(period, duration)
    views, empty = {}, {}
    views["global_view"], empty["global_view"] = _view(phase, lc.flux, g_lo, g_hi)
    views["local_view"], empty["local_view"] = _view(phase, lc.flux, l_lo, l_hi)
    views["odd_view"], empty["odd_view"] = _view(phase[odd], lc.flux[odd], l_lo, l_hi)
    views["even_view"], empty["even_view"] = _view(phase[~odd], lc.flux[~odd], l_lo, l_hi)
    views["secondary_view"], empty["secondary_view"] = _view(phase2, lc.flux, l_lo, l_hi)
    views["centroid_global_view"], empty["centroid_global_view"] = _view(phase, lc.centroid, g_lo, g_hi)
    views["centroid_local_view"], empty["centroid_local_view"] = _view(phase, lc.centroid, l_lo, l_hi)
    diag = {
        "n_points": len(lc),
        "n_transits": int(len(np.unique(epoch[in_transit]))),
        "n_odd_transits": int(len(np.unique(epoch[in_transit & odd]))),
        "n_even_transits": int(len(np.unique(epoch[in_transit & ~odd]))),
        **{f"empty_{k}": v for k, v in empty.items()},
    }
    return views, diag


def normalize(views: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Astronet-style scaling for training: subtract the median and divide by the
    depth (|min|) of the global (global view) or local (local, odd, even, secondary)
    view, so odd/even/secondary keep their depth *relative* to the primary transit.
    Centroid views are left in arcsec, minus their median. Works on (n, bins) arrays."""

    def centred(v):
        return v - np.median(v, axis=-1, keepdims=True)

    def scale(v):
        s = np.abs(v.min(axis=-1, keepdims=True))
        return np.where(s > 0, s, 1.0)

    out = {}
    g = centred(views["global_view"])
    out["global_view"] = g / scale(g)
    loc = centred(views["local_view"])
    s = scale(loc)
    out["local_view"] = loc / s
    for k in ("odd_view", "even_view", "secondary_view"):
        if k in views:
            out[k] = centred(views[k]) / s
    for k in ("centroid_global_view", "centroid_local_view"):
        if k in views:
            out[k] = centred(views[k])
    return out
