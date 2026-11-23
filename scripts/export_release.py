"""Write the files of release/ from a full local run of milestone 5, step 4.

    uv run python scripts/export_release.py

Needs what ``hito5_veredicto.py run`` leaves under data/ (the DR25 views and the 200 runs
of data/hito5/veredicto/runs/). Writes, under release/ (see ``cazaplanetas.release``):

* the validation and test scores of every run (the input of ``hito5_veredicto.py report``);
* the weights of the 100 models of the final recipe (10 folds x 10 seeds);
* the prepared inputs of a few signals for ``scripts/demo.py``: well-known Kepler planets,
  the planets the ensemble scores lowest, the false positives it scores highest and a few
  ordinary false positives, each with the models of the fold that never saw it;
* SHA256SUMS.

Before writing, it checks that the saved weights give back the recorded scores of the
demo signals on the CPU.
"""

from __future__ import annotations

import sys

from cazaplanetas.env import use_local_cache

use_local_cache()

import hito4_dr25_cv as h4  # noqa: E402
import hito5_diagnostico as h5  # noqa: E402
import hito5_receta as R  # noqa: E402
import hito5_veredicto as V  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from cazaplanetas import release, training  # noqa: E402

FAMOUS = ("Kepler-10 b", "Kepler-22 b", "Kepler-62 f", "Kepler-69 c", "Kepler-90 h", "Kepler-186 f",
          "Kepler-442 b", "Kepler-452 b", "Kepler-11 d", "Kepler-16 b")
N_MISSED, N_HARDEST_FP, N_FP = 3, 3, 4


def export_predictions(data: dict, fold: np.ndarray) -> dict[str, np.ndarray]:
    n = len(data["label"])
    out = {"kepid": data["kepid"], "tce_plnt_num": data["tce_plnt_num"], "label": data["label"].astype(np.int8),
           "fold_seed": np.int64(V.VERDICT_FOLD_SEED)}
    for name in V.RECIPES:
        test, val = np.full((release.SEEDS, n), np.nan, np.float32), np.full((release.SEEDS, n), np.nan, np.float32)
        secs, epoch = np.zeros((release.FOLDS, release.SEEDS)), np.zeros((release.FOLDS, release.SEEDS), np.int64)
        params = set()
        for f in range(release.FOLDS):
            for s in range(release.SEEDS):
                with np.load(V.run_path(name, f, s)) as z:
                    assert np.array_equal(z["test_index"], np.flatnonzero(fold == f))
                    assert np.array_equal(z["val_index"], np.flatnonzero(fold == (f + 1) % release.FOLDS))
                    test[s, z["test_index"]], val[s, z["val_index"]] = z["test"], z["val"]
                    secs[f, s], epoch[f, s] = float(z["train_seconds"]), int(z["selected_epoch"])
                    params.add(int(z["parameters"]))
        assert np.isfinite(test).all() and np.isfinite(val).all() and len(params) == 1
        out.update({f"{name}/test": test, f"{name}/val": val, f"{name}/train_seconds": secs,
                    f"{name}/selected_epoch": epoch, f"{name}/parameters": np.int64(params.pop())})
    return out


def pick_demo(ctx: dict, ensemble: np.ndarray) -> list[tuple[int, str]]:
    """(row, group) of the demo signals."""
    y, name = ctx["label"], ctx["kepler_name"]
    rows = [(int(np.flatnonzero(name == n)[0]), "well-known planet") for n in FAMOUS if (name == n).any() and y[name == n][0] == 1]
    planets, fps = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rows += [(int(i), "planet scored lowest") for i in planets[np.argsort(ensemble[planets])[:N_MISSED]]]
    rows += [(int(i), "false positive scored highest") for i in fps[np.argsort(-ensemble[fps])[:N_HARDEST_FP]]]
    taken = {r for r, _ in rows}
    ordinary = [i for i in np.random.default_rng(0).permutation(fps) if i not in taken][:N_FP]
    rows += [(int(i), "false positive (random)") for i in ordinary]
    return rows


def export_demo(data: dict, ctx: dict, fold: np.ndarray, preds: dict) -> dict[str, np.ndarray]:
    recipe = R.RECIPES[V.RECIPE]
    ensemble = preds[f"{V.RECIPE}/test"].mean(0)
    rows = pick_demo(ctx, ensemble)
    idx = np.array([r for r, _ in rows])
    inputs: dict[str, np.ndarray] = {}
    for f in sorted(set(fold[idx].tolist())):
        sel = np.flatnonzero(fold[idx] == f)
        parts = h4.fold_parts(data["kepid"], f, fold_seed=V.VERDICT_FOLD_SEED)
        dev = R.device_tensors(data, recipe, parts, torch.device("cpu"))
        sub = dev.subset(idx[sel])
        for k, v in sub.inputs.items():
            inputs.setdefault(k, np.zeros((len(idx), *v.shape[1:]), np.float32))[sel] = v.numpy()
        del dev
    table = {
        "kepid": data["kepid"][idx], "tce_plnt_num": data["tce_plnt_num"][idx], "kepoi_name": ctx["kepoi_name"][idx].astype(str),
        "kepler_name": ctx["kepler_name"][idx].astype(str), "label": data["label"][idx].astype(np.int8),
        "label_class": np.asarray(data["label_class"][idx]).astype(str), "group": np.array([g for _, g in rows]),
        "fold": fold[idx], "recorded_scores": preds[f"{V.RECIPE}/test"][:, idx].T.copy(),
    }
    return {**table, **{release.INPUT_PREFIX + k: v for k, v in inputs.items()}}


def check_demo(demo: dict) -> float:
    """Largest difference between the recorded scores and those of the saved weights on the CPU."""
    table, inputs = release.load_demo_signals()
    recipe = R.RECIPES[V.RECIPE]
    worst = 0.0
    for f in sorted(set(table["fold"].tolist())):
        sel = np.flatnonzero(table["fold"] == f)
        x = training.Tensors.from_arrays({k: v[sel] for k, v in inputs.items()}, table["label"][sel], torch.device("cpu"))
        for s, sd in release.load_ensemble(f).items():
            model = recipe.make_model()
            model.load_state_dict(sd)
            p = training.predict(model, x)
            worst = max(worst, float(np.abs(p - table["recorded_scores"][sel, s]).max()))
    return worst


def main() -> int:
    data = R.load_dataset(need_centroid_raw=True)
    ctx = h5.load_context()
    assert np.array_equal(ctx["kepid"], data["kepid"]) and np.array_equal(ctx["tce_plnt_num"], data["tce_plnt_num"])
    fold = V.folds_of(data["kepid"])
    preds = export_predictions(data, fold)
    release.RELEASE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(release.PREDICTIONS, **preds)
    files = [release.PREDICTIONS]
    for f in range(release.FOLDS):
        sds = {s: torch.load(V.run_path(V.RECIPE, f, s).with_suffix(".pt"), map_location="cpu", weights_only=True)
               for s in range(release.SEEDS)}
        release.save_ensemble(sds, release.weights_path(f))
        files.append(release.weights_path(f))
    demo = export_demo(data, ctx, fold, preds)
    np.savez_compressed(release.DEMO_SIGNALS, **demo)
    files.append(release.DEMO_SIGNALS)
    worst = check_demo(demo)
    print(f"demo: {len(demo['kepid'])} signals; largest difference from the recorded scores on the CPU {worst:.2e}")
    assert worst < 1e-4, worst
    release.write_checksums(files)
    for p in files:
        print(f"{p.relative_to(release.RELEASE_DIR)}: {p.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
