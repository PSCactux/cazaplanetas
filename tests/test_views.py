"""Offline checks for the DR25 view builder, the star split and the threshold choice.

Synthetic light curves only: no download, a few seconds on CPU.
"""

import numpy as np
import pytest

from cazaplanetas import splits, training
from cazaplanetas import views as V

KEPLER_LC = 0.0204336  # days


def synthetic_star(period=3.7, t0=1.3, duration=0.15, depth=1e-3, odd_extra=0.0, secondary=0.0, noise=1e-4, seed=0):
    """~200 days of Kepler-like cadence with a slow trend, box transits and (optionally)
    deeper odd transits and a secondary eclipse at phase 0.5."""
    rng = np.random.default_rng(seed)
    t = np.arange(100.0, 300.0, KEPLER_LC)
    t = t[(t < 180) | (t > 182)]  # a data gap
    trend = 1 + 0.01 * np.sin(2 * np.pi * t / 15.0)
    phase = V.fold(t, period, t0)
    epoch = np.floor((t - t0) / period + 0.5).astype(int)
    dip = np.where(np.abs(phase) < duration / 2, depth + odd_extra * (epoch % 2 == 1), 0.0)
    dip += np.where(np.abs(V.fold(t, period, t0 + period / 2)) < duration / 2, secondary, 0.0)
    flux = trend * (1 - dip) + rng.normal(0, noise, len(t))
    return t, flux, (period, t0, duration)


def test_binned_median_matches_a_loop():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 10, 500))
    y = rng.normal(size=500)
    lo = np.linspace(-1, 9.5, 40)
    hi = lo + 0.8  # overlapping bins, some empty
    med, counts = V.binned_median(x, y, lo, hi)
    for i in range(len(lo)):
        sel = (x >= lo[i]) & (x < hi[i])
        assert counts[i] == sel.sum()
        if sel.any():
            assert med[i] == pytest.approx(np.median(y[sel]))
        else:
            assert np.isnan(med[i])


def test_bin_geometry():
    lo, hi = V.global_bins(10.0)
    assert len(lo) == V.GLOBAL_BINS
    assert lo[0] == pytest.approx(-5.0) and hi[-1] == pytest.approx(5.0)
    lo, hi = V.local_bins(10.0, 0.2)
    assert len(lo) == V.LOCAL_BINS
    assert lo[0] == pytest.approx(-0.8) and hi[-1] == pytest.approx(0.8)  # +-4 durations
    assert (hi - lo) == pytest.approx(np.full(V.LOCAL_BINS, 0.032))  # 0.16 durations


def _views(t, flux, ephem, **kw):
    period, t0, duration = ephem
    mask = V.transit_mask(t, period, t0, duration) | V.transit_mask(t, period, t0 + period / 2, duration)
    spacing = V.choose_spacing([(t, flux, mask)])
    detr = flux / V.detrend(t, flux, mask, spacing) - 1
    lc = V.LightCurve(t, detr, np.zeros(len(t)))
    return V.tce_views(lc, period, t0, duration, **kw)


def test_detrending_keeps_the_transit_and_removes_the_trend():
    t, flux, ephem = synthetic_star()
    views, diag = _views(t, flux, ephem)
    loc = views["local_view"]
    centre = loc[95:106].mean()
    edges = np.r_[loc[:40], loc[-40:]]
    assert centre == pytest.approx(-1e-3, rel=0.1)  # full depth recovered
    assert abs(np.median(edges)) < 5e-5  # flat out of transit: the 1 % trend is gone
    assert np.argmin(views["global_view"]) in range(990, 1011)
    assert diag["n_transits"] > 40
    assert all(np.isfinite(v).all() for v in views.values())


def test_odd_even_and_secondary_views():
    t, flux, ephem = synthetic_star(odd_extra=1e-3, secondary=5e-4)
    views, diag = _views(t, flux, ephem)
    odd, even = views["odd_view"][95:106].mean(), views["even_view"][95:106].mean()
    assert odd == pytest.approx(-2e-3, rel=0.1)
    assert even == pytest.approx(-1e-3, rel=0.1)
    assert views["secondary_view"][95:106].mean() == pytest.approx(-5e-4, rel=0.15)
    assert abs(diag["n_odd_transits"] - diag["n_even_transits"]) <= 1

    n = V.normalize({k: v[None] for k, v in views.items()})
    assert n["local_view"].min() == pytest.approx(-1.0)
    assert n["odd_view"][0, 95:106].mean() < n["even_view"][0, 95:106].mean()  # relative depths kept


def test_bic_prefers_short_knots_for_fast_variability():
    rng = np.random.default_rng(2)
    t = np.arange(0, 60, KEPLER_LC)
    quiet = 1 + rng.normal(0, 1e-4, len(t))
    wiggly = quiet + 5e-3 * np.sin(2 * np.pi * t / 1.2)
    no_mask = np.zeros(len(t), bool)
    assert V.choose_spacing([(t, wiggly, no_mask)]) < V.choose_spacing([(t, quiet, no_mask)])


def test_split_by_star_has_no_leakage_and_right_sizes():
    rng = np.random.default_rng(3)
    kepid = rng.integers(0, 3000, 10_000)  # many stars with several signals
    part = splits.split_by_star(kepid, (0.8, 0.1, 0.1), seed=0)
    train, val, test = (kepid[part == i] for i in range(3))
    assert splits.shared_stars(train, test) == 0
    assert splits.shared_stars(train, val) == 0
    assert splits.shared_stars(val, test) == 0
    assert len(train) + len(val) + len(test) == len(kepid)
    for got, want in zip((len(train), len(val), len(test)), (8000, 1000, 1000)):
        assert abs(got - want) < 30
    np.testing.assert_array_equal(part, splits.split_by_star(kepid, (0.8, 0.1, 0.1), seed=0))  # deterministic
    assert not np.array_equal(part, splits.split_by_star(kepid, (0.8, 0.1, 0.1), seed=1))
    folds = splits.star_folds(kepid, k=5)
    assert set(np.unique(folds)) == set(range(5))


def test_choose_threshold_maximises_f1():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    p = np.array([0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.9])
    thr = training.choose_threshold(y, p)
    # Best F1 marks the top 5 (4 planets, 1 false alarm): F1 = 8/9, better than the
    # top 3 (6/7). The threshold falls between the 5th score (0.3) and the 6th (0.2).
    assert 0.2 < thr < 0.3
    m = training.classification_metrics(y, p, thr)
    assert (m["tp"], m["fp"], m["fn"]) == (4, 1, 0)
    # A perfectly separable case puts the threshold between the classes.
    assert 0.4 < training.choose_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])) < 0.6
