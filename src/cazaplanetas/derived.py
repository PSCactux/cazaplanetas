"""Model inputs derived from the DR25 tables and the stored views, without light curves
(milestone 5, step 2).

* :func:`sibling_features`: the other signal of the same star with the same period
  (the primary or secondary eclipse of a binary detected as a separate signal), as
  ExoMiner's "data correction" of the secondary test does (Valizadegan et al. 2022,
  Sec. 3.5): does it exist, how much stronger and deeper it is, and its phase.
* :func:`ephemeris_matches`: a signal of *another* star, nearby on the sky, with the
  same period (or twice or half of it) and the same epoch: the usual sign that the
  light of one star contaminates the other (Coughlin et al. 2014). Coordinates are
  only used to measure the distance between two stars, never as an input.
* :func:`sibling_view` and :func:`half_period_view`: a local view at the phase of the
  sibling signal and the global view folded at half the period, both rebuilt from the
  stored global view (2001 bins over the whole orbit).

Every function reads only the arrays it is given; :data:`SOURCE_COLUMNS` lists the
DR25 TCE columns they use. ``kepid``, ``ra`` and ``dec`` are used to pair signals
(same star; distance between stars) and never reach a model. The outputs are listed
in :data:`DERIVED_COLUMNS`, the allow-list of derived inputs (``features``).
"""

from __future__ import annotations

import numpy as np

from cazaplanetas import views as V

KEPLER_BASELINE_DAYS = 1460.0  # Q1-Q17: about four years
EPHEMERIS_RADIUS_DEG = 1.0  # search radius for ephemeris matches (fixed before looking at the data)

# DR25 TCE columns read by the derived inputs (``kepid``, ``ra``, ``dec``: pairing only).
SOURCE_COLUMNS = ("kepid", "ra", "dec", "tce_period", "tce_time0bk", "tce_duration", "tce_depth", "tce_max_mult_ev")

# Derived inputs -> what they are. Missing values (no sibling, no match) are NaN and
# become the training median after standardisation; the "*_has" flags tell them apart.
DERIVED_COLUMNS: dict[str, str] = {
    "sib_has": "another signal of the same star has the same period (within 1 %)",
    "sib_mes_ratio": "log10 of the sibling's detection statistic over the signal's",
    "sib_depth_ratio": "log10 of the sibling's depth over the signal's",
    "sib_phase": "phase of the sibling after the signal, folded to [0, 0.5]",
    "sib_dur_ratio": "log10 of the sibling's duration over the signal's",
    "kepmag": "Kepler magnitude of the star (DR25 stellar table); bright stars saturate",
    "em_has": "a signal of another star within 1 degree matches period and epoch",
    "em_log_sep": "log10 of the distance to that star, arcsec",
    "em_depth_ratio": "log10 of the matching signal's depth over the signal's",
    "em_mes_ratio": "log10 of the matching signal's detection statistic over the signal's",
    "em_dkepmag": "Kepler magnitude of the other star minus the signal's star",
}
SIBLING_COLUMNS = ("sib_has", "sib_mes_ratio", "sib_depth_ratio", "sib_phase", "sib_dur_ratio")
EPHEMERIS_COLUMNS = ("em_has", "em_log_sep", "em_depth_ratio", "em_mes_ratio", "em_dkepmag")


def _log_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((a > 0) & (b > 0), np.log10(np.where(a > 0, a, 1.0) / np.where(b > 0, b, 1.0)), np.nan)


def sibling_features(kepid, period, t0, duration, depth, mes, tolerance: float = 0.01) -> dict[str, np.ndarray]:
    """For each signal, the strongest other signal of the same star whose period is the
    same within ``tolerance`` (relative). ``duration`` in any unit (only ratios are used).

    Returns the :data:`SIBLING_COLUMNS` plus ``sib_index`` (row of the sibling, -1 if
    none) and ``sib_offset`` (days from the signal's transit to the sibling's, in
    (-P/2, P/2]), used to rebuild the sibling's view."""
    kepid = np.asarray(kepid)
    period, t0 = np.asarray(period, float), np.asarray(t0, float)
    mes = np.asarray(mes, float)
    n = len(kepid)
    sib = np.full(n, -1, dtype=np.int64)
    order = np.argsort(kepid, kind="mergesort")
    starts = np.flatnonzero(np.r_[True, np.diff(kepid[order]) != 0, True])
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < 2:
            continue
        idx = order[a:b]
        p = period[idx]
        close = np.abs(p[:, None] / p[None, :] - 1.0) < tolerance
        np.fill_diagonal(close, False)
        strength = np.where(close, np.nan_to_num(mes[idx], nan=-np.inf)[None, :], -np.inf)
        best = np.argmax(strength, axis=1)
        has = close.any(axis=1)
        sib[idx[has]] = idx[best[has]]
    has = sib >= 0
    j = np.where(has, sib, 0)
    frac = np.mod((t0[j] - t0) / period, 1.0)
    offset = np.where(frac > 0.5, frac - 1.0, frac) * period
    nan = np.full(n, np.nan)
    return {
        "sib_has": has.astype(float),
        "sib_mes_ratio": np.where(has, _log_ratio(mes[j], mes), nan),
        "sib_depth_ratio": np.where(has, _log_ratio(np.asarray(depth, float)[j], depth), nan),
        "sib_phase": np.where(has, np.minimum(frac, 1.0 - frac), nan),
        "sib_dur_ratio": np.where(has, _log_ratio(np.asarray(duration, float)[j], duration), nan),
        "sib_index": sib,
        "sib_offset": np.where(has, offset, 0.0),
    }


def _unit_vectors(ra_deg, dec_deg) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)


def ephemeris_matches(kepid, ra, dec, period, t0, duration_days, depth, mes, kepmag=None,
                      radius_deg: float = EPHEMERIS_RADIUS_DEG, baseline: float = KEPLER_BASELINE_DAYS) -> dict[str, np.ndarray]:
    """For each signal, the nearest signal of another star within ``radius_deg`` whose
    ephemeris matches: periods equal or in a 2:1 ratio (the accumulated drift over the
    mission, ``|P_long - k P_short| * baseline / P_long``, below the longer of the two
    durations) and transits at the same time (epochs within half that duration, modulo
    the shorter period). Returns the :data:`EPHEMERIS_COLUMNS` and ``em_index``."""
    kepid = np.asarray(kepid)
    period, t0 = np.asarray(period, float), np.asarray(t0, float)
    dur = np.asarray(duration_days, float)
    n = len(kepid)
    xyz = _unit_vectors(ra, dec)
    cos_r = np.cos(np.radians(radius_deg))
    order = np.argsort(period)
    ps = period[order]
    best_sep = np.full(n, np.inf)
    best = np.full(n, -1, dtype=np.int64)
    max_rel = 2.0 * np.nanmax(dur) / baseline  # loose pre-filter on the period ratio
    for k in (1.0, 2.0, 0.5):
        lo = np.searchsorted(ps, period * k * (1 - max_rel), side="left")
        hi = np.searchsorted(ps, period * k * (1 + max_rel), side="right")
        for i in np.flatnonzero(hi > lo):
            js = order[lo[i]:hi[i]]
            js = js[kepid[js] != kepid[i]]
            if not len(js):
                continue
            cos_sep = xyz[js] @ xyz[i]
            near = cos_sep >= cos_r
            js, cos_sep = js[near], cos_sep[near]
            if not len(js):
                continue
            p_short, p_long = np.minimum(period[i], period[js]), np.maximum(period[i], period[js])
            ratio = np.round(p_long / p_short)
            d_max = np.maximum(dur[i], dur[js])
            drift = np.abs(p_long - ratio * p_short) * baseline / p_long
            dt = np.mod(t0[i] - t0[js], p_short)
            dt = np.minimum(dt, p_short - dt)
            ok = (ratio <= 2) & (drift < d_max) & (dt < 0.5 * d_max)
            if not ok.any():
                continue
            sep = np.degrees(np.arccos(np.clip(cos_sep[ok], -1.0, 1.0))) * 3600.0
            m = int(np.argmin(sep))
            if sep[m] < best_sep[i]:
                best_sep[i], best[i] = sep[m], js[ok][m]
    has = best >= 0
    j = np.where(has, best, 0)
    nan = np.full(n, np.nan)
    out = {
        "em_has": has.astype(float),
        "em_log_sep": np.where(has, np.log10(np.maximum(best_sep, 0.1)), nan),
        "em_depth_ratio": np.where(has, _log_ratio(np.asarray(depth, float)[j], depth), nan),
        "em_mes_ratio": np.where(has, _log_ratio(np.asarray(mes, float)[j], mes), nan),
        "em_dkepmag": nan.copy(),
        "em_index": best,
    }
    if kepmag is not None:
        km = np.asarray(kepmag, float)
        out["em_dkepmag"] = np.where(has, km[j] - km, nan)
    return out


# --- Views rebuilt from the stored global view ----------------------------------------


def _periodic_mean(g: np.ndarray, lo: np.ndarray, hi: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Mean of the global view ``g`` (n, B) over the phase intervals [lo, hi) (n, m), in
    units of global bins (bin b covers [b, b + 1)), wrapping around the orbit. Where an
    interval is narrower than a bin, the view is interpolated linearly at its centre."""
    n, b = g.shape
    if n > chunk:  # bounded memory: the cumulative sums are (rows, 3B) in float64
        return np.concatenate([_periodic_mean(g[i:i + chunk], lo[i:i + chunk], hi[i:i + chunk]) for i in range(0, n, chunk)])
    g = np.asarray(g, dtype=np.float64)
    c = np.concatenate([np.zeros((n, 1)), np.cumsum(np.concatenate([g, g, g], axis=1), axis=1)], axis=1)
    lo_i, hi_i = lo + b, hi + b  # into the middle copy
    rows = np.arange(n)[:, None]

    def integral(x):  # integral of the step function from 0 to x (x in [0, 3B])
        k = np.clip(np.floor(x).astype(np.int64), 0, 3 * b - 1)
        return c[rows, k] + (x - k) * g[rows, k % b]

    width = hi_i - lo_i
    mean = (integral(hi_i) - integral(lo_i)) / np.where(width > 0, width, 1.0)
    centre = 0.5 * (lo + hi) - 0.5  # bin centres sit at b + 0.5
    k0 = np.floor(centre).astype(np.int64)
    w = centre - k0
    rows = np.arange(n)[:, None]
    interp = (1 - w) * g[rows, np.mod(k0, b)] + w * g[rows, np.mod(k0 + 1, b)]
    return np.where(width >= 1.0, mean, interp)


def sibling_view(global_view: np.ndarray, period, duration_days, offset_days, has=None) -> np.ndarray:
    """Local view (201 bins, same window as ``local_view``) centred ``offset_days`` after
    the transit, rebuilt from the raw global view (n, 2001). Rows without a sibling
    (``has`` False) are left at zero."""
    g = np.asarray(global_view, np.float32)
    n, b = g.shape
    period = np.asarray(period, float)
    out = np.zeros((n, V.LOCAL_BINS), dtype=np.float32)
    rows = np.arange(n) if has is None else np.flatnonzero(has)
    if not len(rows):
        return out
    lo = np.empty((len(rows), V.LOCAL_BINS))
    hi = np.empty_like(lo)
    for r, i in enumerate(rows):
        l_lo, l_hi = V.local_bins(period[i], float(duration_days[i]))
        lo[r], hi[r] = l_lo + offset_days[i], l_hi + offset_days[i]
    to_bins = (b / period[rows])[:, None]
    lo_b = (lo + 0.5 * period[rows][:, None]) * to_bins
    hi_b = (hi + 0.5 * period[rows][:, None]) * to_bins
    shift = np.floor(lo_b / b) * b  # bring each interval into [0, B)
    out[rows] = _periodic_mean(g[rows], lo_b - shift, hi_b - shift).astype(np.float32)
    return out


def half_period_view(global_view: np.ndarray) -> np.ndarray:
    """The global view folded at half the period: bin m of the result covers the same
    fraction of P/2 that bin m of the input covers of P, and averages the two halves
    of the orbit. If the true period is half of the detected one, the transit shows up
    with both halves stacked; if it is an eclipsing binary, primary and secondary mix."""
    g = np.asarray(global_view, np.float32)
    b = g.shape[1]
    # Output bins are half as wide as the input ones, so the input is interpolated
    # linearly (periodically) at the centre of each output bin, in each half of the orbit.
    centre = (np.arange(b) + 0.5) / 2.0 + 0.25 * b  # phase -P/4..P/4 of the input, in input bins

    def interp(x):
        x = x - 0.5  # bin k is centred at k + 0.5
        k0 = np.floor(x).astype(np.int64)
        w = (x - k0).astype(np.float32)
        return (1 - w) * g[:, np.mod(k0, b)] + w * g[:, np.mod(k0 + 1, b)]

    return (0.5 * (interp(centre) + interp(centre + 0.5 * b))).astype(np.float32)
