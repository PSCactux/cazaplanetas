"""The BLS baseline must recover known periods.

* ``test_synthetic_box_transit`` runs offline on an injected signal.
* ``test_kepler10b_period`` downloads Kepler-10 (about 6 MB, cached in data/raw)
  and checks the recovered period against the published one.
"""

import numpy as np
import pytest

from cazaplanetas import pipeline
from cazaplanetas.env import RAW_DIR

# NASA Exoplanet Archive (pscomppars), Dumusque et al. 2014.
KEPLER10B_PERIOD = 0.8374907  # days


def test_synthetic_box_transit():
    import lightkurve as lk

    rng = np.random.default_rng(0)
    t = np.arange(0, 90, pipeline.KEPLER_LC_DAYS)
    period, t0, duration, depth = 3.217, 1.3, 0.12, 400e-6
    flux = 1 + rng.normal(0, 150e-6, t.size)
    flux[np.abs(pipeline.phase_hours(t, period, t0)) < duration * 12] -= depth
    lc = lk.LightCurve(time=t, flux=flux, flux_err=np.full(t.size, 150e-6))

    res, _, _ = pipeline.bls_search(lc, pmin=1.0, pmax=10.0)

    assert res.period == pytest.approx(period, rel=1e-3)
    assert res.depth_ppm == pytest.approx(depth * 1e6, rel=0.25)


@pytest.mark.network
def test_kepler10b_period():
    lc = pipeline.clean(pipeline.download_kepler_long_cadence("Kepler-10", RAW_DIR))
    # 0.4-2 d covers the P/2 and 2P aliases, so this also checks we pick the right one.
    res, _, _ = pipeline.bls_search(lc, pmin=0.4, pmax=2.0)

    assert res.period == pytest.approx(KEPLER10B_PERIOD, rel=1e-4)
    assert res.sde > 10
    assert 100 < res.depth_ppm < 250  # published depth is about 150 ppm
