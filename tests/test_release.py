"""The files published in release/: checksums, weights and the scores they give."""

import sys

import numpy as np
import pytest
import torch

from cazaplanetas import models, release, splits, training
from cazaplanetas.env import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import hito5_receta as R  # noqa: E402
import hito5_veredicto as V  # noqa: E402

pytestmark = pytest.mark.skipif(not release.SHA256SUMS.exists(), reason="release/ not present")


def test_checksums_match():
    assert release.verify_checksums() == []
    names = set(release.read_checksums())
    assert {"predictions_hito5_veredicto.npz", "demo_signals.npz"} <= names
    assert {f"weights/fold{f}.npz" for f in range(release.FOLDS)} <= names


def test_save_and_load_ensemble_round_trip(tmp_path):
    torch.manual_seed(0)
    sds = {s: models.PocketBranches(4, branches={"flux_global": ("global_view",)}).state_dict() for s in (0, 1)}
    path = tmp_path / "w.npz"
    release.save_ensemble(sds, path)
    back = release.load_ensemble(0, path)
    assert list(back) == [0, 1]
    for s in sds:
        assert list(back[s]) == list(sds[s])
        for k, v in sds[s].items():
            assert back[s][k].dtype == v.dtype and torch.equal(back[s][k], v)


def test_published_weights_are_the_final_recipe():
    recipe = R.RECIPES[V.RECIPE]
    for f in range(release.FOLDS):
        ens = release.load_ensemble(f)
        assert list(ens) == list(range(release.SEEDS))
        model = recipe.make_model()
        model.load_state_dict(ens[0])  # strict: same layout as the recipe
        assert models.count_parameters(model) == 25_745


def test_predictions_cover_every_signal_once_per_seed():
    z = release.load_predictions()
    n = len(z["kepid"])
    fold = splits.star_folds(z["kepid"], release.FOLDS, seed=int(z["fold_seed"]))
    assert int(z["fold_seed"]) == V.VERDICT_FOLD_SEED
    assert set(np.unique(z["label"])) == {0, 1}
    for name in V.RECIPES:
        for part in ("test", "val"):
            p = z[f"{name}/{part}"]
            assert p.shape == (release.SEEDS, n) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        assert int(z[f"{name}/parameters"]) == 25_745
    assert np.bincount(fold, minlength=release.FOLDS).min() > 3000


def test_published_weights_give_the_recorded_scores():
    table, inputs = release.load_demo_signals()
    recipe = R.RECIPES[V.RECIPE]
    pred = release.load_predictions()
    keys = {(int(k), int(p)): i for i, (k, p) in enumerate(zip(pred["kepid"], pred["tce_plnt_num"]))}
    fold_all = V.folds_of(pred["kepid"])
    for f in sorted(set(table["fold"].tolist())):
        sel = np.flatnonzero(table["fold"] == f)
        rows = [keys[(int(table["kepid"][i]), int(table["tce_plnt_num"][i]))] for i in sel]
        assert (fold_all[rows] == f).all()  # each demo signal is scored by the models that never saw it
        x = training.Tensors.from_arrays({k: v[sel] for k, v in inputs.items()}, table["label"][sel], torch.device("cpu"))
        for s, sd in release.load_ensemble(f).items():
            model = recipe.make_model()
            model.load_state_dict(sd)
            p = training.predict(model, x)
            np.testing.assert_allclose(p, table["recorded_scores"][sel, s], atol=1e-4)
            np.testing.assert_allclose(p, pred[f"{V.RECIPE}/test"][s, rows], atol=1e-4)
