"""Model inputs for the DR25 signals, from an explicit list of allowed columns.

The DR25 TCE table has 158 columns. Some of them carry the answer, or a proxy of
it, and must never reach a model:

* ``av_*`` (``av_training_set``, ``av_pred_class``, vote fractions, posteriors): the
  Autovetter's training labels and predictions. Empty in DR25, filled in DR24.
* ``tce_nkoi``: number of KOIs linked to the TCE's star. ``tce_nkoi = 0`` means the
  star has no KOI, so the TCE is an NTP by construction (21,943 TCEs).
* ``*_prov`` and the stellar parameter errors: where the stellar parameters come
  from (spectroscopy, asteroseismology...). Follow-up spectroscopy was done mostly
  for stars with planet candidates, so the provenance hints at the label.
* ``tce_rogue_flag`` (used to drop rogue TCEs), identifiers (``kepid``,
  ``tce_plnt_num``, ``ra``, ``dec``) and bookkeeping (``tce_delivname``,
  ``tce_datalink_*``, ``rowupdate``).
* Anything from the KOI or FPWG tables (``koi_*``, ``fpwg_*``) or from the
  Robovetter (``robovetter_*``, dispositions, scores, flags): those are the labels,
  or the competitor's answer.

:data:`SCALAR_COLUMNS` is the explicit list of allowed DR25 TCE columns, each with
the transform applied before standardisation. :func:`scalar_matrix` reads *only*
those columns, after :func:`assert_allowed` has checked them; the tests poison the
forbidden columns with the label and check that the inputs do not change.
"""

from __future__ import annotations

import re

import numpy as np

# Allowed scalar inputs: DR25 TCE column -> transform before standardisation.
#   "log": log10(x) (x <= 0 -> missing); "slog": sign(x) log10(1 + |x|); "lin": as is;
#   "fap": -log10(max(x, 1e-50)) with -1 (not computed) -> missing.
# Pipeline (DV) diagnostics, the transit fit and the stellar parameters of the KIC/DR25
# stellar table. None of them was computed from the dispositions.
SCALAR_COLUMNS: dict[str, str] = {
    # Transit fit
    "tce_period": "log",
    "tce_duration": "log",
    "tce_depth": "slog",
    "tce_prad": "log",
    "tce_ror": "log",
    "tce_impact": "lin",
    "tce_dor": "log",
    "tce_model_snr": "slog",
    "tce_max_mult_ev": "log",
    "tce_max_sngle_ev": "slog",
    "tce_num_transits": "slog",
    "tce_chisq1": "slog",
    "tce_chisq2": "slog",
    # Secondary eclipse
    "tce_maxmes": "slog",
    "wst_robstat": "slog",
    "wst_depth": "slog",
    "tce_albedo_stat": "slog",
    "tce_ptemp_stat": "slog",
    # Odd / even depths
    "tce_bin_oedp_stat": "slog",
    # Centroid motion and offsets
    "tce_fwm_stat": "slog",
    "tce_dikco_msky": "slog",
    "tce_dikco_msky_err": "slog",
    "tce_dicco_msky": "slog",
    "tce_dicco_msky_err": "slog",
    # Other DV diagnostics: bootstrap false-alarm probability, ghost (core / halo
    # aperture correlation) and rolling-band counts
    "boot_fap": "fap",
    "tce_cap_stat": "slog",
    "tce_hap_stat": "slog",
    "tce_rb_tcount0": "slog",
    # Star
    "tce_steff": "lin",
    "tce_slogg": "lin",
    "tce_smet": "lin",
    "tce_sradius": "log",
}

# The allowed scalars by the diagnostic test they belong to (used by the ablations of
# milestone 5). Every allowed column is in exactly one group (checked by the tests).
SCALAR_GROUPS: dict[str, tuple[str, ...]] = {
    "transit_fit": ("tce_period", "tce_duration", "tce_depth", "tce_prad", "tce_ror", "tce_impact", "tce_dor", "tce_model_snr",
                    "tce_max_mult_ev", "tce_max_sngle_ev", "tce_num_transits", "tce_chisq1", "tce_chisq2"),
    "secondary": ("tce_maxmes", "wst_robstat", "wst_depth", "tce_albedo_stat", "tce_ptemp_stat"),
    "odd_even": ("tce_bin_oedp_stat",),
    "centroid": ("tce_fwm_stat", "tce_dikco_msky", "tce_dikco_msky_err", "tce_dicco_msky", "tce_dicco_msky_err"),
    "other_dv": ("boot_fap", "tce_cap_stat", "tce_hap_stat", "tce_rb_tcount0"),
    "star": ("tce_steff", "tce_slogg", "tce_smet", "tce_sradius"),
}

# The allowed views by diagnostic test.
VIEW_GROUPS: dict[str, tuple[str, ...]] = {
    "global_flux": ("global_view",),
    "local_flux": ("local_view",),
    "odd_even": ("odd_view", "even_view"),
    "secondary": ("secondary_view",),
    "centroid": ("centroid_global_view", "centroid_local_view"),
}

# Views (from our own light-curve processing, scripts/hito3_dr25.py) that may be fed.
VIEW_INPUTS = (
    "global_view",
    "local_view",
    "odd_view",
    "even_view",
    "secondary_view",
    "centroid_global_view",
    "centroid_local_view",
)

# Views rebuilt from the stored global view (milestone 5, step 2; ``derived``): the local
# view at the phase of the same-period sibling signal, and the global view folded at P/2.
DERIVED_VIEWS = ("sibling_view", "half_global_view")

# Scalars derived from allowed DR25 columns and the DR25 stellar table (``derived``).
from cazaplanetas.derived import DERIVED_COLUMNS  # noqa: E402

# Columns, or patterns of columns, that must never be a model input.
FORBIDDEN_PATTERNS = (
    r"^av_",  # Autovetter labels and predictions: av_training_set, av_pred_class, ...
    r"training_set",
    r"pred_class",
    r"^koi_",  # KOI tables: dispositions, scores, flags
    r"^fpwg_",  # Certified False Positive table
    r"^robovetter",  # the competitor's answer
    r"disp",  # any disposition
    r"score",
    r"^label",
    r"^tce_nkoi$",  # number of KOIs on the star: 0 means NTP by construction
    r"^tce_rogue_flag$",
    r"_prov$",  # stellar-parameter provenance (follow-up spectroscopy hints at the label)
    r"^tce_(steff|slogg|smet|sradius)_err$",  # same: errors shrink with follow-up
    r"^(kepid|tce_plnt_num|kepoi_name|kepler_name|ra|dec)$",  # identifiers
    r"^tce_delivname$",
    r"^tce_datalink",
    r"^rowupdate$",
)


class ForbiddenColumnError(ValueError):
    pass


def is_forbidden(column: str) -> bool:
    return any(re.search(p, column) for p in FORBIDDEN_PATTERNS)


def assert_allowed(columns) -> None:
    """Raise if any column is forbidden, or is not in the explicit allow-list."""
    allowed = set(SCALAR_COLUMNS) | set(VIEW_INPUTS) | set(DERIVED_VIEWS) | set(DERIVED_COLUMNS)
    bad = [c for c in columns if is_forbidden(c)]
    if bad:
        raise ForbiddenColumnError(f"forbidden model inputs: {bad}")
    unknown = [c for c in columns if c not in allowed]
    if unknown:
        raise ForbiddenColumnError(f"columns not in the allow-list: {unknown}")


def _transform(x: np.ndarray, kind: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "log":
            return np.where(x > 0, np.log10(np.where(x > 0, x, 1.0)), np.nan)
        if kind == "slog":
            return np.sign(x) * np.log10(1.0 + np.abs(x))
        if kind == "fap":
            return np.where(x < 0, np.nan, -np.log10(np.maximum(x, 1e-50)))
        if kind == "lin":
            return x.copy()
    raise ValueError(f"unknown transform {kind!r}")


def scalar_matrix(table: dict[str, np.ndarray], rows: np.ndarray | None = None, columns: dict[str, str] = SCALAR_COLUMNS) -> np.ndarray:
    """(n, k) matrix of the allowed scalar columns, transformed (NaN where missing).

    Reads nothing but ``columns`` from ``table``; raises if any of them is forbidden."""
    assert_allowed(list(columns))
    cols = []
    for name, kind in columns.items():
        v = np.asarray(table[name], dtype=np.float64)
        cols.append(_transform(v if rows is None else v[rows], kind))
    return np.stack(cols, axis=1)


class Standardizer:
    """Standardisation fitted on training rows only, clipped to +-``clip``; missing values
    become 0 (the training median).

    ``kind="robust"`` (default): (x - median) / IQR-sigma. ``kind="zscore"``: (x - mean) /
    standard deviation, as ExoMiner does (its Algorithm 3, with a clip of 20)."""

    def __init__(self, clip: float = 8.0, kind: str = "robust"):
        if kind not in ("robust", "zscore"):
            raise ValueError(f"unknown standardisation {kind!r}")
        self.clip, self.kind = clip, kind

    def fit(self, x: np.ndarray) -> "Standardizer":
        std = np.nanstd(x, axis=0)
        if self.kind == "zscore":
            self.median = np.nanmean(x, axis=0)
            self.scale = np.where(std > 0, std, 1.0)
            # missing values -> the training median, i.e. (median - mean) / std
            self.fill = (np.nanmedian(x, axis=0) - self.median) / self.scale
            return self
        self.median = np.nanmedian(x, axis=0)
        q75, q25 = np.nanpercentile(x, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349
        self.scale = np.where(scale > 0, scale, np.where(std > 0, std, 1.0))
        self.fill = np.zeros_like(self.median)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = np.clip((x - self.median) / self.scale, -self.clip, self.clip)
        return np.where(np.isnan(z), np.broadcast_to(self.fill, z.shape), z).astype(np.float32)


def centroid_significance(view: np.ndarray, clip: float = 20.0) -> np.ndarray:
    """Centroid view (arcsec, median already removed) divided by its own robust scatter,
    so that a shift reads as "how many sigma" whatever the star's brightness."""
    med = np.median(view, axis=-1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(view - med), axis=-1, keepdims=True)
    floor = 1e-4  # arcsec; flat (empty) views stay 0
    return np.clip((view - med) / np.maximum(mad, floor), -clip, clip).astype(np.float32)


def view_inputs(views: dict[str, np.ndarray], names=VIEW_INPUTS) -> dict[str, np.ndarray]:
    """Normalised views ready for the network (see views.normalize), only the allowed ones."""
    from cazaplanetas import views as V

    assert_allowed(list(names))
    norm = V.normalize({k: views[k] for k in names})
    out = {}
    for k in names:
        v = norm[k]
        out[k] = centroid_significance(v) if k.startswith("centroid") else np.asarray(v, dtype=np.float32)
    return out
