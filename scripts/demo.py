"""Score a few Kepler DR25 signals with the published ensembles (no download, no training).

    uv run python scripts/demo.py

The inputs of 19 signals are in release/demo_signals.npz, already prepared and normalised
for the fold in which each signal was in the test part: the ten models of that fold
(release/weights/) never saw it, nor any other signal of the same star. For each signal the
script runs the ten models on the CPU, averages their scores (the ensemble of the verdict)
and compares the ensemble with a cut chosen on that fold's validation part for 99 %
precision (from release/predictions_hito5_veredicto.npz). It then checks that every score is
the one recorded when the models were trained, and that the files match release/SHA256SUMS.
"""

from __future__ import annotations

import sys

from cazaplanetas.env import use_local_cache

use_local_cache()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hito5_receta as R  # noqa: E402
import hito5_veredicto as V  # noqa: E402
from cazaplanetas import models, release, training  # noqa: E402

TOLERANCE = 1e-4  # recorded on Apple's GPU, recomputed here on the CPU
CLASSES = {"PC": "planet", "AFP": "false positive", "NTP": "not transit-like"}


def cut_for_precision(y: np.ndarray, score: np.ndarray, target: float = 0.99) -> float:
    """Lowest score s such that the signals scored >= s have precision >= ``target``."""
    order = np.argsort(-score, kind="mergesort")
    s, yy = score[order], y[order].astype(bool)
    tp, fp = np.cumsum(yy), np.cumsum(~yy)
    last = np.r_[np.diff(s) != 0, True]
    ok = np.flatnonzero(last & (tp / (tp + fp) >= target))
    return float(s[ok[-1]]) if len(ok) else float("inf")


def validation_cuts() -> dict[int, float]:
    pred = release.load_predictions()
    fold = V.folds_of(pred["kepid"])
    ens_val = pred[f"{V.RECIPE}/val"].mean(0)
    cuts = {}
    for f in range(release.FOLDS):
        vi = fold == (f + 1) % release.FOLDS
        cuts[f] = cut_for_precision(pred["label"][vi], ens_val[vi])
    return cuts


def main() -> int:
    bad = release.verify_checksums()
    if bad:
        print(f"files that do not match release/SHA256SUMS: {', '.join(bad)}")
        return 1
    table, inputs = release.load_demo_signals()
    recipe = R.RECIPES[V.RECIPE]
    n = len(table["kepid"])
    scores = np.zeros((n, release.SEEDS), np.float32)
    n_params = None
    for f in sorted(set(table["fold"].tolist())):
        sel = np.flatnonzero(table["fold"] == f)
        x = training.Tensors.from_arrays({k: v[sel] for k, v in inputs.items()}, table["label"][sel], torch.device("cpu"))
        for s, sd in release.load_ensemble(f).items():
            model = recipe.make_model()
            model.load_state_dict(sd)
            n_params = models.count_parameters(model)
            scores[sel, s] = training.predict(model, x)
    ens = scores.mean(1)
    cuts = validation_cuts()
    print(f"Ensembles of 10 models with {n_params:,} parameters each (10 folds x 10 seeds published).")
    print("Score: mean of the 10 models of the fold that never saw the signal. Cut: 99 % precision on that fold's validation part.\n")
    print(f"{'signal':<20} {'name':<14} {'DR25 label':<17} {'score':>6} {'10 models':>13}  {'cut':>5}  flagged")
    group = None
    for i in range(n):
        if table["group"][i] != group:
            group = str(table["group"][i])
            print(f"-- {group}")
        sig = table["kepoi_name"][i] or f"KIC {table['kepid'][i]}.{table['tce_plnt_num'][i]}"
        f = int(table["fold"][i])
        flagged = "planet" if ens[i] >= cuts[f] else "-"
        print(f"{sig:<20} {table['kepler_name'][i] or '-':<14} {CLASSES.get(str(table['label_class'][i]), table['label_class'][i]):<17} "
              f"{ens[i]:6.3f} {scores[i].min():6.3f}-{scores[i].max():.3f}  {cuts[f]:5.3f}  {flagged}")
    worst = float(np.abs(scores - table["recorded_scores"]).max())
    ok = worst < TOLERANCE
    print(f"\nLargest difference from the scores recorded at training time: {worst:.1e} ({'ok' if ok else 'MISMATCH'}).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
