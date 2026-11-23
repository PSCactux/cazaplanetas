"""Offline checks for the DR25 dataset code: FITS reading, shards, resume and matching.

A tiny synthetic Kepler-like FITS file is built in memory, so nothing is downloaded.
"""

import io
import json

import numpy as np
import pytest
from astropy.io import fits

from cazaplanetas import dr25
from cazaplanetas import views as V


def fake_kepler_fits(n=3000, quarter=5, bad_quality_rows=(10, 11), nan_rows=(20,), period=2.5, t0=401.0, depth=2e-3):
    t = 400.0 + np.arange(n) * 0.0204336
    flux = 1e4 * (1 - depth * (np.abs(V.fold(t, period, t0)) < 0.05))
    flux[list(nan_rows)] = np.nan
    qual = np.zeros(n, dtype=np.int32)
    qual[list(bad_quality_rows)] = 32  # desaturation event: rejected by the default bitmask
    qual[30] = 512  # impulsive outlier: kept (not in the bitmask)
    cols = [
        fits.Column("TIME", "D", array=t),
        fits.Column("PDCSAP_FLUX", "E", array=flux),
        fits.Column("SAP_QUALITY", "J", array=qual),
        fits.Column("MOM_CENTR1", "D", array=np.full(n, 500.0)),
        fits.Column("MOM_CENTR2", "D", array=np.full(n, 300.0)),
    ]
    primary = fits.PrimaryHDU()
    primary.header["QUARTER"] = quarter
    buf = io.BytesIO()
    fits.HDUList([primary, fits.BinTableHDU.from_columns(cols)]).writeto(buf)
    return buf.getvalue(), n


def test_read_quarter_applies_quality_and_nan_masks():
    data, n = fake_kepler_fits()
    q = dr25.read_quarter(data)
    assert q.quarter == 5
    assert len(q.time) == n - 3  # two flagged rows and one NaN removed, the 512 flag kept
    assert np.all(np.diff(q.time) > 0)


def test_stitch_and_views_from_fits():
    data, _ = fake_kepler_fits(n=4000)
    tce = dr25.TCE(1, 1, 2.5, 401.0, 0.1, float("nan"))
    lc, spacing = dr25.stitch([dr25.read_quarter(data)], [tce])
    assert spacing in V.SPACINGS
    views, diag = V.tce_views(lc, tce.period, tce.t0, tce.duration)
    assert views["local_view"][100] == pytest.approx(-2e-3, rel=0.05)
    assert np.allclose(views["centroid_local_view"], 0, atol=1e-9)  # constant centroid: no shift
    assert diag["n_transits"] > 10


def _result(kepid, plnts, error=None):
    tces = []
    for p in plnts:
        if p < 0:
            tces.append({"tce_plnt_num": -p, "error": "ValueError: no transit falls on good data"})
        else:
            views = {k: np.full(V.VIEW_LENGTHS[k], float(p), dtype=np.float32) for k in V.VIEW_NAMES}
            diag = {k: p for k in dr25.DIAG_KEYS}
            tces.append({"tce_plnt_num": p, "views": views, "diag": diag})
    return {"kepid": kepid, "sources": [("f.fits", 1, "abc")], "tces": [] if error else tces, "error": error}


def test_shards_roundtrip_failures_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(dr25, "SHARD_DIR", tmp_path / "views")
    dr25.write_shard(0, [_result(7, [1, 2]), _result(3, [1, -2])])
    dr25.write_shard(1, [_result(9, [], error="RuntimeError: no long-cadence light curves in the archive")])
    d = dr25.load()
    assert d["kepid"].tolist() == [3, 7, 7]  # sorted by star inside a shard
    assert d["tce_plnt_num"].tolist() == [1, 1, 2]
    assert d["global_view"].shape == (3, V.GLOBAL_BINS) and d["global_view"].dtype == np.float32
    reasons = {(f["kepid"], f["tce_plnt_num"]): f["reason"] for f in dr25.failures()}
    assert set(reasons) == {(3, 2), (9, None)}
    assert not list((tmp_path / "views").glob("*.part"))

    # Resume: batches whose shard exists are skipped without touching the network.
    calls = []
    monkeypatch.setattr(dr25, "process_star", lambda k, *a, **kw: calls.append(k) or _result(k, [1]))
    monkeypatch.setattr(dr25, "ProcessPoolExecutor", _InlineExecutor)
    stars = {3: [None], 7: [None], 9: [None], 11: [None]}
    stats = dr25.build(stars, batch_size=2, workers=1, min_free_gb=0, log=lambda *_: None)
    # 4 stars in batches of 2 -> batches 0 and 1 already exist, nothing to do.
    assert stats["done_before"] == 2 and calls == []
    stars[12] = [None]
    dr25.build(stars, batch_size=2, workers=1, min_free_gb=0, log=lambda *_: None)
    assert calls == [12]
    json.loads((tmp_path / "views" / "shard_0002.failures.json").read_text())


class _InlineExecutor:
    """Runs submitted work immediately, in-process (stands in for the process pool)."""

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kw):
        from concurrent.futures import Future

        f = Future()
        f.set_result(fn(*args, **kw))
        return f


def test_build_stops_when_disk_is_low(tmp_path, monkeypatch):
    monkeypatch.setattr(dr25, "SHARD_DIR", tmp_path / "views")
    monkeypatch.setattr(dr25, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(dr25, "free_gb", lambda *a: 12.0)
    with pytest.raises(SystemExit, match="GB free"):
        dr25.build({1: [None]}, batch_size=1, min_free_gb=30, log=lambda *_: None)


def test_match_dr24_uses_period_and_epoch_not_planet_number():
    dr25_table = {
        "kepid": np.array([5, 5, 6]),
        "tce_plnt_num": np.array([1, 2, 1]),
        "tce_period": np.array([10.0, 3.0, 7.0]),
        "tce_time0bk": np.array([131.0, 132.0, 140.0]),
    }
    dr24_table = {
        "kepid": np.array([5, 5, 6, 8]),
        "tce_plnt_num": np.array([1, 2, 1, 1]),
        "tce_period": np.array([3.0001, 10.0, 7.0, 4.0]),  # planet numbers swapped between releases
        "tce_time0bk": np.array([135.0, 141.0, 143.5, 130.0]),  # last-but-one: same period, transits 3.5 d apart
        "tce_duration": np.array([3.0, 5.0, 4.0, 2.0]),
    }
    m = dr25.match_dr24(dr25_table, dr24_table)
    assert m == {(5, 1): (5, 2), (5, 2): (5, 1)}


def test_redo_failures_rewrites_only_failed_stars(tmp_path, monkeypatch):
    monkeypatch.setattr(dr25, "SHARD_DIR", tmp_path / "views")
    monkeypatch.setattr(dr25, "ProcessPoolExecutor", _MapExecutor)
    dr25.write_shard(0, [_result(3, [1, -2]), _result(7, [1]), _result(9, [], error="RuntimeError: download failed")])
    calls = []
    monkeypatch.setattr(dr25, "process_star", lambda k, *a, **kw: calls.append(k) or _result(k, [1, 2]))
    stars = {3: [None, None], 7: [None], 9: [None, None]}
    assert dr25.redo_failures(stars, batch_size=10, log=lambda *_: None) == 2
    assert sorted(calls) == [3, 9]  # star 7 was fine and is not downloaded again
    d = dr25.load()
    assert list(zip(d["kepid"].tolist(), d["tce_plnt_num"].tolist())) == [(3, 1), (3, 2), (7, 1), (9, 1), (9, 2)]
    assert dr25.failures() == []


class _MapExecutor(_InlineExecutor):
    def map(self, fn, *iterables):
        return map(fn, *iterables)
