"""Tools for the milestone 5 diagnosis: why the pocket model stays below ExoMiner.

* :data:`ABLATIONS` and :func:`mask_inputs`: input ablations. An ablated input is
  replaced by zeros (a flat view, or the training median of a standardised scalar) in
  training and in evaluation, so the model, its parameter count, its initial weights
  and its batch order stay exactly those of the full model with the same seed.
* :func:`best_recall_with_fixes` and :func:`fixes_needed`: how many objects (negatives
  ranked too high, planets ranked too low) stand between a ranking and a target
  recall at a given precision.
* :func:`variance_components`: how much of the spread of a metric comes from the seed
  and how much from the fold.
* :func:`interpolated_pr_auc`: the PR AUC estimator of ``tf.keras.metrics.AUC`` (the one
  ExoMiner's code uses), to compare like with like.

Pure numpy; no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cazaplanetas import features


@dataclass(frozen=True)
class Ablation:
    """Inputs removed from the full pocket model: view groups and scalar groups
    (keys of ``features.VIEW_GROUPS`` and ``features.SCALAR_GROUPS``)."""

    views: tuple[str, ...] = ()
    scalars: tuple[str, ...] = ()
    label: str = ""

    def view_names(self) -> tuple[str, ...]:
        return tuple(v for g in self.views for v in features.VIEW_GROUPS[g])

    def scalar_columns(self) -> tuple[str, ...]:
        return tuple(c for g in self.scalars for c in features.SCALAR_GROUPS[g])


_ALL_VIEWS = tuple(features.VIEW_GROUPS)
_ALL_SCALARS = tuple(features.SCALAR_GROUPS)

# In order of priority: the first ten (``MAIN_ABLATIONS``) answer "views only, scalars
# only, and each group of inputs removed", with one group per diagnostic test (its views
# and its scalars together). The rest split the three tests into views and scalars.
ABLATIONS: dict[str, Ablation] = {
    "full": Ablation(label="Todo (referencia)"),
    "views_only": Ablation(scalars=_ALL_SCALARS, label="Solo vistas (sin escalares)"),
    "scalars_only": Ablation(views=_ALL_VIEWS, label="Solo escalares (sin vistas)"),
    # One group of inputs out
    "no_test_centroid": Ablation(views=("centroid",), scalars=("centroid",), label="Sin la prueba de centroide (2 vistas + 5 escalares)"),
    "no_test_secondary": Ablation(views=("secondary",), scalars=("secondary",), label="Sin la prueba del secundario (1 vista + 5 escalares)"),
    "no_test_oddeven": Ablation(views=("odd_even",), scalars=("odd_even",), label="Sin la prueba de impares/pares (2 vistas + 1 escalar)"),
    "no_view_global": Ablation(views=("global_flux",), label="Sin vista global de flujo"),
    "no_view_local": Ablation(views=("local_flux",), label="Sin vista local de flujo"),
    "no_sc_transit": Ablation(scalars=("transit_fit",), label="Sin escalares del ajuste del tránsito (13)"),
    "no_sc_other": Ablation(scalars=("other_dv",), label="Sin otros diagnósticos DV (4)"),
    "no_sc_star": Ablation(scalars=("star",), label="Sin parámetros de la estrella (4)"),
    # The three tests, views and scalars apart
    "no_view_centroid": Ablation(views=("centroid",), label="Sin vistas de centroide"),
    "no_sc_centroid": Ablation(scalars=("centroid",), label="Sin escalares de centroide"),
    "no_view_secondary": Ablation(views=("secondary",), label="Sin vista del secundario"),
    "no_sc_secondary": Ablation(scalars=("secondary",), label="Sin escalares del secundario"),
    "no_view_oddeven": Ablation(views=("odd_even",), label="Sin vistas de impares y pares"),
    "no_sc_oddeven": Ablation(scalars=("odd_even",), label="Sin escalar de impares/pares"),
}
MAIN_ABLATIONS = tuple(ABLATIONS)[:11]  # "full" and the ten main ablations


def mask_inputs(inputs: dict, ablation: Ablation, scalar_columns=tuple(features.SCALAR_COLUMNS)) -> dict:
    """Copy of ``inputs`` (name -> array or tensor, first axis = example) with the ablated
    views and scalar columns set to zero. Works with numpy arrays and torch tensors."""
    out = dict(inputs)
    for name in ablation.view_names():
        if name in out:
            out[name] = out[name] * 0
    cols = [list(scalar_columns).index(c) for c in ablation.scalar_columns()]
    if cols and "scalars" in out:
        s = out["scalars"].clone() if hasattr(out["scalars"], "clone") else out["scalars"].copy()
        s[:, cols] = 0
        out["scalars"] = s
    return out


# --- Ranking arithmetic --------------------------------------------------------------


def _curve(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tp, fp, score) at every distinct threshold, from the highest score down."""
    y = np.asarray(y).astype(bool)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    s, yy = np.asarray(score)[order], y[order]
    tp, fp = np.cumsum(yy), np.cumsum(~yy)
    last = np.r_[np.diff(s) != 0, True]
    return tp[last], fp[last], s[last]


def ap_loss_per_positive(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """What each signal takes away from the area under the precision-recall curve: the
    step-wise area is the mean, over the positives, of the precision at each positive's
    threshold, so ``1 - AUC-PR`` splits exactly into ``(1 - that precision) / positives``
    for each positive (0 for the negatives). Ties share their threshold's precision."""
    y = np.asarray(y).astype(bool)
    score = np.asarray(score)
    order = np.argsort(-score, kind="mergesort")
    s, yy = score[order], y[order]
    tp, seen = np.cumsum(yy), np.arange(1, len(yy) + 1)
    group_end = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]  # last index of each tie group
    end = np.repeat(group_end, np.diff(np.r_[-1, group_end]))
    precision = tp[end] / seen[end]
    out = np.zeros(len(y))
    out[order] = np.where(yy, (1.0 - precision) / max(int(y.sum()), 1), 0.0)
    return out


def operating_point(y: np.ndarray, score: np.ndarray, precision: float = 0.99) -> dict:
    """The threshold that gives the highest recall with precision >= ``precision``:
    its score, true and false positives and recall (all zero if none exists)."""
    tp, fp, s = _curve(y, score)
    ok = tp / (tp + fp) >= precision
    n_pos = int(np.asarray(y).astype(bool).sum())
    if not ok.any():
        return {"threshold": float("inf"), "tp": 0, "fp": 0, "recall": 0.0, "n_pos": n_pos}
    i = int(np.flatnonzero(ok)[np.argmax(tp[ok])])
    return {"threshold": float(s[i]), "tp": int(tp[i]), "fp": int(fp[i]), "recall": float(tp[i] / n_pos), "n_pos": n_pos}


def best_recall_with_fixes(y: np.ndarray, score: np.ndarray, max_fixes: int, precision: float = 0.99) -> np.ndarray:
    """``out[k]`` = highest recall at precision >= ``precision`` if ``k`` objects could be
    fixed, for k = 0..max_fixes. Fixing an object means giving it the score it should
    have: a negative above the threshold drops out, or a planet below it moves up.

    With the threshold at a given rank (tp planets and fp negatives above it), fixing
    ``a`` negatives and ``b`` planets leaves tp + b planets and fp - a negatives above
    it; it is valid if fp - a <= (tp + b) (1 - precision) / precision."""
    tp, fp, _ = _curve(y, score)
    tp = np.r_[0, tp]  # allow an empty selection
    fp = np.r_[0, fp]
    n_pos = int(np.asarray(y).astype(bool).sum())
    ratio = (1.0 - precision) / precision
    out = np.zeros(max_fixes + 1)
    for k in range(max_fixes + 1):
        best = 0
        for b in range(k + 1):  # planets moved up
            a = k - b  # negatives removed
            t = np.minimum(tp + b, n_pos)
            valid = np.maximum(fp - a, 0) <= np.floor(t * ratio + 1e-9)
            valid &= (b <= n_pos - tp) & (t > 0)
            if valid.any():
                best = max(best, int(t[valid].max()))
        out[k] = best / n_pos
    return np.maximum.accumulate(out)


def fixes_needed(y: np.ndarray, score: np.ndarray, target_recall: float, precision: float = 0.99, max_fixes: int = 200) -> int:
    """Smallest number of fixed objects for which the recall at ``precision`` reaches
    ``target_recall`` (see :func:`best_recall_with_fixes`); -1 if more than ``max_fixes``."""
    r = best_recall_with_fixes(y, score, max_fixes, precision)
    hit = np.flatnonzero(r >= target_recall - 1e-12)
    return int(hit[0]) if len(hit) else -1


def fixes_for_mean(per_fold_curves: list[np.ndarray], target_mean: float) -> tuple[int, list[int]]:
    """Fewest fixes, shared among the folds, for which the *mean over folds* of the
    recall reaches ``target_mean``. ``per_fold_curves[f][k]`` is the recall of fold f
    with k fixes (from :func:`best_recall_with_fixes`). Returns (total, fixes per fold);
    (-1, []) if it cannot be reached. Exact (dynamic programming over the folds)."""
    n = len(per_fold_curves)
    kmax = sum(len(c) - 1 for c in per_fold_curves)
    # best[j] = highest sum of recalls using j fixes in the folds seen so far
    best = np.full(kmax + 1, -np.inf)
    best[0] = 0.0
    choice: list[np.ndarray] = []
    for c in per_fold_curves:
        new = np.full(kmax + 1, -np.inf)
        pick = np.zeros(kmax + 1, dtype=int)
        for k, r in enumerate(c):
            cand = np.full(kmax + 1, -np.inf)
            cand[k:] = best[: kmax + 1 - k] + r
            better = cand > new
            new[better], pick[better] = cand[better], k
        best = new
        choice.append(pick)
    hit = np.flatnonzero(best >= target_mean * n - 1e-9)
    if not len(hit):
        return -1, []
    total = int(hit[0])
    alloc, j = [], total
    for pick in reversed(choice):
        alloc.append(int(pick[j]))
        j -= int(pick[j])
    return total, alloc[::-1]


def recall_at_rounded_precision(y: np.ndarray, score: np.ndarray, target: float = 0.99, decimals: int = 2) -> float:
    """Highest recall among the thresholds whose precision, *rounded to* ``decimals``
    places, is >= ``target``. With two decimals, "precision 0.99" means precision >= 0.985
    and "precision 0.95" means >= 0.945, while 0.975 stays exact. This is the reading
    that reproduces the Recall@p columns of ExoMiner's Table 5 from its published
    scores (the exact reading, ``training.recall_at_precision``, does not)."""
    tp, fp, _ = _curve(y, score)
    ok = np.round(tp / (tp + fp), decimals) >= target - 1e-12
    return float(tp[ok].max() / np.asarray(y).astype(bool).sum()) if ok.any() else 0.0


# --- Spread: seed against fold -------------------------------------------------------


def variance_components(values: np.ndarray) -> dict:
    """One-way random-effects decomposition of a (folds, seeds) table of a metric.

    ``seed_sd`` is the spread among the seeds of a fold (pooled within-fold standard
    deviation); ``fold_sd`` is the spread of the fold itself (the part of the variance
    of the fold means that the seeds do not explain). ``fold_share`` is the share of the
    variance of a single run that comes from the fold."""
    v = np.asarray(values, dtype=float)
    n_folds, n_seeds = v.shape
    within = v.var(axis=1, ddof=1).mean() if n_seeds > 1 else 0.0
    means = v.mean(axis=1)
    between = max(means.var(ddof=1) - within / n_seeds, 0.0) if n_folds > 1 else 0.0
    total = within + between
    return {
        "mean": float(v.mean()),
        "seed_sd": float(np.sqrt(within)),
        "fold_sd": float(np.sqrt(between)),
        "fold_means_sd": float(means.std(ddof=1)) if n_folds > 1 else 0.0,
        "fold_share": float(between / total) if total > 0 else 0.0,
        "n_folds": int(n_folds),
        "n_seeds": int(n_seeds),
    }


# --- Keras-style PR AUC ---------------------------------------------------------------


def interpolated_pr_auc(y: np.ndarray, score: np.ndarray, num_thresholds: int = 1000) -> float:
    """PR AUC as ``tf.keras.metrics.AUC(curve="PR", summation_method="interpolation")``
    computes it: ``num_thresholds`` evenly spaced thresholds in [0, 1] and the
    interpolation of Davis & Goadrich (2006) between them. Written from the Keras
    documentation; scores that pile up near 0 and 1 fall into few buckets, so this
    estimate can differ from the exact step-wise area (``training.average_precision``)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=float)
    eps = 1e-7
    thr = np.r_[-eps, (np.arange(1, num_thresholds - 1) / (num_thresholds - 1)), 1.0 + eps]
    pos, neg = np.sort(s[y]), np.sort(s[~y])
    tp = (len(pos) - np.searchsorted(pos, thr, side="right")).astype(float)
    fp = (len(neg) - np.searchsorted(neg, thr, side="right")).astype(float)
    fn = len(pos) - tp
    dtp = tp[:-1] - tp[1:]
    p = tp + fp
    dp = p[:-1] - p[1:]
    slope = np.divide(dtp, np.maximum(dp, 0), out=np.zeros_like(dtp), where=dp > 0)
    intercept = tp[1:] - slope * p[1:]
    both = (p[:-1] > 0) & (p[1:] > 0)
    ratio = np.where(both, np.divide(p[:-1], np.maximum(p[1:], 0), out=np.ones_like(dtp), where=p[1:] > 0), 1.0)
    denom = np.maximum(tp[1:] + fn[1:], 0)
    area = np.divide(slope * (dtp + intercept * np.log(ratio)), denom, out=np.zeros_like(dtp), where=denom > 0)
    return float(area.sum())


# --- Likely cause of an error, from public catalogues -----------------------------------
#
# Rules, not eyes: each object gets the first cause of the list that applies. The fields
# come from the DR25 TCE table, the cumulative KOI table, the Certified False Positive
# table (FPWG) and the public output of the DR25 Robovetter. They are used here only to
# describe the errors; none of them is a model input.

NEGATIVE_CAUSES = {
    "dubious_label": "Etiqueta dudosa: falso positivo certificado que el archivo da hoy como candidato",
    "secondary_tce": "Eclipse secundario de una binaria, detectado como señal aparte en la misma estrella",
    "ephemeris_match": "Contaminación de otra estrella (mismo periodo y época que otra señal)",
    "centroid_offset": "Binaria de fondo: el centro de luz se desplaza en el tránsito",
    "eclipsing_binary": "Binaria eclipsante en la propia estrella (secundario, impares/pares o forma en V)",
    "low_snr": "Falsa alarma con poca señal (MES < 10)",
    "other_false_alarm": "Otra falsa alarma (no parecida a un tránsito para el Robovetter)",
}
PLANET_CAUSES = {
    "wrong_period": "Periodo mal: el del catálogo de señales es el doble o la mitad del real",
    "controversial": "Etiqueta en disputa: planeta marcado como controvertido en el archivo",
    "saturated": "Estrella saturada o muy brillante: la prueba de centroide no es fiable",
    "secondary": "Secundario visible (ocultación de un júpiter caliente, u otro planeta en esa fase)",
    "centroid": "Prueba de centroide o de reflejos marcada",
    "few_transits": "Pocos tránsitos (5 o menos)",
    "low_snr": "Poca señal (MES < 12)",
    "not_transit_like": "El Robovetter tampoco la ve como tránsito",
    "giant": "Tránsito profundo o planeta gigante (> 10 radios terrestres): se parece a una binaria",
    "short_period": "Periodo muy corto (< 1 día)",
    "unclear": "Sin causa clara en los catálogos",
}


def negative_cause(row: dict) -> str:
    """Likely reason why a negative (AFP or NTP) gets a high score. ``row`` needs:
    label_class, koi_disposition, rv_flags (4 characters: not transit-like, stellar
    eclipse, centroid offset, ephemeris match), rv_comment, fpwg_eb, fpwg_offset,
    fpwg_perep (0/1 or None), same_period_sibling (bool) and mes."""
    flags = row.get("rv_flags") or "0000"
    comment = row.get("rv_comment") or ""
    if row.get("label_class") == "AFP" and row.get("koi_disposition") == "CANDIDATE":
        return "dubious_label"
    if "IS_SEC_TCE" in comment or (row.get("same_period_sibling") and row.get("label_class") == "NTP"):
        return "secondary_tce"
    if row.get("fpwg_perep") == 1 or flags[3] == "1":
        return "ephemeris_match"
    if row.get("fpwg_offset") == 1 or flags[2] == "1":
        return "centroid_offset"
    if row.get("fpwg_eb") == 1 or flags[1] == "1":
        return "eclipsing_binary"
    if row.get("mes") is not None and row["mes"] < 10:
        return "low_snr"
    return "other_false_alarm"


def planet_cause(row: dict) -> str:
    """Likely reason why a confirmed planet gets a low score. ``row`` needs: tce_period,
    koi_period, controversial (bool), kepmag, rv_flags, rv_comment, n_transits, mes, prad
    (Earth radii) and depth (ppm)."""
    flags = row.get("rv_flags") or "0000"
    comment = row.get("rv_comment") or ""
    kp, tp = row.get("koi_period"), row.get("tce_period")
    if kp and tp and np.isfinite(kp) and abs(tp / kp - 1.0) > 0.01:
        return "wrong_period"
    if row.get("controversial"):
        return "controversial"
    if "CENT_SATURATED" in comment or (row.get("kepmag") is not None and row["kepmag"] < 11.5):
        return "saturated"
    if flags[1] == "1" or "MOD_SEC" in comment or "PLANET_OCCULT" in comment or "HAS_SEC_TCE" in comment:
        return "secondary"
    if flags[2] == "1" or "OFFSET" in comment or "GHOST" in comment:
        return "centroid"
    if row.get("n_transits") is not None and row["n_transits"] <= 5:
        return "few_transits"
    if row.get("mes") is not None and row["mes"] < 12:
        return "low_snr"
    if flags[0] == "1":
        return "not_transit_like"
    if (row.get("prad") is not None and row["prad"] > 10) or (row.get("depth") is not None and row["depth"] > 5000):
        return "giant"
    if tp and tp < 1.0:
        return "short_period"
    return "unclear"


def same_period_siblings(kepid: np.ndarray, period: np.ndarray, tolerance: float = 0.01, strength: np.ndarray | None = None) -> np.ndarray:
    """True for each signal that shares its star with another signal of the same period
    (within ``tolerance``, relative): the usual signature of an eclipsing binary whose
    primary and secondary eclipses were detected as two signals. From the TCE table only.

    With ``strength`` (for instance the detection statistic), True only where one of those
    siblings is stronger than the signal itself: the signal is the weaker event of the pair."""
    kepid, period = np.asarray(kepid), np.asarray(period, dtype=float)
    out = np.zeros(len(kepid), dtype=bool)
    order = np.argsort(kepid, kind="mergesort")
    starts = np.flatnonzero(np.r_[True, np.diff(kepid[order]) != 0, True])
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < 2:
            continue
        idx = order[a:b]
        p = period[idx]
        close = np.abs(p[:, None] / p[None, :] - 1.0) < tolerance
        np.fill_diagonal(close, False)
        if strength is not None:
            w = np.asarray(strength, dtype=float)[idx]
            close &= w[None, :] > w[:, None]  # column: the sibling; row: the signal
        out[idx] = close.any(axis=1)
    return out
