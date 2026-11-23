"""Inputs derived from the DR25 tables and the stored views (milestone 5, step 2)."""

import numpy as np
import pytest

from cazaplanetas import derived as D
from cazaplanetas import features as F
from cazaplanetas import views as V


def test_sibling_is_the_strongest_same_period_signal_of_the_star():
    kepid = np.array([1, 1, 1, 2, 2, 3])
    period = np.array([10.0, 10.05, 3.0, 5.0, 7.0, 10.0])
    t0 = np.array([100.0, 105.0, 101.0, 100.0, 100.0, 100.0])
    dur = np.array([0.2, 0.2, 0.1, 0.1, 0.1, 0.2])
    depth = np.array([100.0, 1000.0, 50.0, 10.0, 10.0, 100.0])
    mes = np.array([10.0, 100.0, 8.0, 9.0, 9.0, 10.0])
    s = D.sibling_features(kepid, period, t0, dur, depth, mes)
    np.testing.assert_array_equal(s["sib_has"], [1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(s["sib_index"], [1, 0, -1, -1, -1, -1])
    assert s["sib_mes_ratio"][0] == pytest.approx(1.0) and s["sib_mes_ratio"][1] == pytest.approx(-1.0)
    assert s["sib_depth_ratio"][0] == pytest.approx(1.0)
    assert s["sib_phase"][0] == pytest.approx(0.5)  # 5 days after, on a 10-day orbit
    assert abs(s["sib_offset"][0]) == pytest.approx(5.0)
    assert np.isnan(s["sib_mes_ratio"][2:]).all()


def test_ephemeris_match_needs_another_star_nearby_same_period_and_epoch():
    # 0 and 1: other stars, 10" apart, same period and epoch -> match.
    # 2: same ephemeris but 2 degrees away -> no match. 3: same star as 0 -> not a match.
    # 4: near 0, same period, transits half an orbit later -> no match. 5: twice the period -> match with 0.
    kepid = np.array([1, 2, 3, 1, 4, 5])
    ra = np.array([290.0, 290.0 + 10 / 3600, 292.0, 290.0, 290.0, 290.0])
    dec = np.array([45.0, 45.0, 45.0, 45.0, 45.0 + 20 / 3600, 45.0 - 30 / 3600])
    period = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 8.0])
    t0 = np.array([100.0, 100.01, 100.0, 100.0, 102.0, 104.0])
    dur = np.full(6, 0.1)
    depth = np.array([100.0, 10000.0, 100.0, 100.0, 100.0, 50.0])
    mes = np.array([10.0, 500.0, 10.0, 10.0, 10.0, 9.0])
    kepmag = np.array([15.0, 12.0, 14.0, 15.0, 13.0, 16.0])
    m = D.ephemeris_matches(kepid, ra, dec, period, t0, dur, depth, mes, kepmag)
    assert m["em_index"][0] == 1 and m["em_index"][1] in (0, 3)
    assert m["em_index"][2] == -1 and m["em_index"][4] == -1
    assert m["em_index"][5] in (0, 3)
    assert 10 ** m["em_log_sep"][0] == pytest.approx(10 * np.cos(np.radians(45)), rel=1e-3)
    assert m["em_depth_ratio"][0] == pytest.approx(2.0) and m["em_dkepmag"][0] == pytest.approx(-3.0)
    # a period that drifts by more than a duration over the mission does not match
    period2 = period.copy()
    period2[1] = 4.0 * (1 + 1e-3)
    assert D.ephemeris_matches(kepid, ra, dec, period2, t0, dur, depth, mes)["em_index"][0] != 1


def _global_with_dips(n, period, dips):
    """Raw global views (n, 2001) with box dips at the given phases (days from the transit)."""
    t = (-0.5 + (np.arange(V.GLOBAL_BINS) + 0.5) / V.GLOBAL_BINS) * period
    g = np.zeros((n, V.GLOBAL_BINS), np.float32)
    for centre, depth, half_width in dips:
        d = np.abs((t - centre + 0.5 * period) % period - 0.5 * period)
        g[:, d < half_width] -= depth
    return g


def test_sibling_view_shows_the_sibling_eclipse():
    period, dur = 10.0, 0.4
    g = _global_with_dips(2, period, [(0.0, 1.0, 0.2), (5.0, 10.0, 0.2)])  # shallow transit, deep eclipse at P/2
    v = D.sibling_view(g, [period] * 2, [dur] * 2, [5.0, 0.0], has=np.array([True, False]))
    assert v.shape == (2, V.LOCAL_BINS)
    assert v[0, V.LOCAL_BINS // 2] == pytest.approx(-10.0, abs=0.5)  # the deep eclipse, centred
    assert abs(v[0, 0]) < 0.5 and abs(v[0, -1]) < 0.5  # out of eclipse at the edges
    assert (v[1] == 0).all()
    # an offset of -5 days is the same place on the orbit (wraps around)
    w = D.sibling_view(g[:1], [period], [dur], [-5.0])
    np.testing.assert_allclose(w[0], v[0], atol=1e-5)
    # a narrow local window (finer than a global bin) is interpolated, not left empty
    narrow = D.sibling_view(_global_with_dips(1, 400.0, [(200.0, 10.0, 1.0)]), [400.0], [0.3], [200.0])
    assert narrow[0, V.LOCAL_BINS // 2] < -5


def test_half_period_view_stacks_the_two_halves():
    period = 10.0
    g = _global_with_dips(1, period, [(0.0, 1.0, 0.1), (5.0, 1.0, 0.1)])  # true period is P/2
    h = D.half_period_view(g)
    assert h.shape == g.shape
    assert h[0, V.GLOBAL_BINS // 2] == pytest.approx(-1.0, abs=0.05)  # both transits stacked at the centre
    assert np.abs(h[0, :800]).max() < 0.05
    one = D.half_period_view(_global_with_dips(1, period, [(0.0, 1.0, 0.1)]))  # a single transit is halved
    assert one[0, V.GLOBAL_BINS // 2] == pytest.approx(-0.5, abs=0.05)


def test_derived_inputs_are_allowed_and_read_no_forbidden_column():
    F.assert_allowed([*D.DERIVED_COLUMNS, *F.DERIVED_VIEWS])
    for c in (*D.DERIVED_COLUMNS, *F.DERIVED_VIEWS):
        assert not F.is_forbidden(c), c
    # The DR25 columns they read: identifiers and coordinates only to pair signals.
    pairing = {"kepid", "ra", "dec"}
    for c in D.SOURCE_COLUMNS:
        assert c in pairing or c in F.SCALAR_COLUMNS or c in ("tce_time0bk",), c
    # Every column of the sibling and ephemeris groups is documented in the allow-list.
    assert set(D.SIBLING_COLUMNS) | set(D.EPHEMERIS_COLUMNS) | {"kepmag"} == set(D.DERIVED_COLUMNS)


def test_derived_inputs_do_not_see_the_label():
    """The functions take arrays, not tables: a label hidden in the identifiers' order,
    or in any other column, cannot change them unless it changes those arrays."""
    rng = np.random.default_rng(0)
    n = 200
    kepid = rng.integers(0, 60, n)
    period = rng.choice([3.0, 5.0, 7.0], n) * (1 + rng.normal(0, 1e-5, n))
    t0 = rng.uniform(100, 110, n)
    dur = np.full(n, 0.1)
    depth, mes = rng.lognormal(5, 1, n), rng.lognormal(2, 0.5, n)
    a = D.sibling_features(kepid, period, t0, dur, depth, mes)
    perm = rng.permutation(n)
    b = D.sibling_features(kepid[perm], period[perm], t0[perm], dur[perm], depth[perm], mes[perm])
    for k in D.SIBLING_COLUMNS:
        np.testing.assert_array_equal(np.nan_to_num(a[k][perm], nan=-9), np.nan_to_num(b[k], nan=-9))
