"""Milestone 4: the pocket model against the Robovetter and Astronet on Kepler DR25,
with ExoMiner's protocol: 10-fold cross-validation by star.

    uv run python scripts/hito4_dr25_cv.py run       # train everything (resumable; hours)
    uv run python scripts/hito4_dr25_cv.py report    # tables in docs/hito4.md, JSON, figure
    uv run python scripts/hito4_dr25_cv.py run --quick   # smoke test: 2 folds, 1 seed, 1 epoch

Labels (``cazaplanetas.labels``): ExoMiner's rules on today's archive tables, i.e.
confirmed planets of the cumulative KOI table (PC) against certified false positives
(AFP) and not-transit-like signals plus certified false alarms (NTP); rogue TCEs and
unconfirmed KOIs left out.

Folds: ``splits.star_folds`` (10 folds by star). For fold f the test part is fold f,
the validation part is fold f+1 and the other 8 folds train. Every model sees the
same parts. The decision threshold is the one with the best F1 on the validation
part, per model, fold and seed; the test fold is only used to measure.

Models:

* ``robovetter``: the DR25 Robovetter's published score and PC/FP disposition. Not
  trained here; note that its rules were tuned knowing these labels.
* ``gbm``: gradient-boosted trees (scikit-learn) on the allowed scalar columns only.
* ``pocket_gl``: the 36k-parameter pocket model of milestones 2-3 (global + local views).
* ``pocket_full``: the 38k-parameter pocket model with every view (odd, even,
  secondary, centroid) and the allowed scalars.
* ``astronet``: Astronet re-implemented (global + local views), milestone 2-3 recipe.

Model inputs come only from ``features.SCALAR_COLUMNS`` and ``features.VIEW_INPUTS``
(checked by ``features.assert_allowed`` at run time and by tests/test_features.py).

Each (model, fold, seed) writes its validation and test predictions to
data/hito4/preds/, so ``run`` can be stopped and resumed and ``report`` only reads them.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time

from cazaplanetas.env import DATA_DIR, REPO_ROOT, use_local_cache

use_local_cache()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cazaplanetas import dr25, features, labels, models, splits, training  # noqa: E402
from cazaplanetas import views as V  # noqa: E402

DOCS = REPO_ROOT / "docs"
RUN_DIR = DATA_DIR / "hito4"
K_FOLDS = 10
FOLD_SEED = 0

POCKET_RECIPE = dict(epochs=20, batch_size=64, lr=1e-3, weight_decay=1e-4, optimizer="adamw", schedule="cosine", select="best_val",
                     select_metric="auc_pr")  # milestone 3 recipe; early stopping on validation AUC-PR
CONFIGS = {
    "astronet": training.TrainConfig(epochs=50, batch_size=64, lr=1e-5, optimizer="adam", select="last"),
    "pocket_gl": training.TrainConfig(**POCKET_RECIPE),
    "pocket_full": training.TrainConfig(**POCKET_RECIPE),
}
# max_features: each tree split sees a random 80 % of the columns, so seeds give different models.
GBM_PARAMS = dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
                  max_features=0.8, early_stopping=False)
NEURAL = ("pocket_full", "pocket_gl", "astronet")
ALL_MODELS = ("robovetter", "gbm", *NEURAL)
LABEL = {
    "robovetter": "Robovetter (DR25, publicado)",
    "gbm": "Árboles (escalares)",
    "pocket_gl": "Bolsillo, vistas global y local",
    "pocket_full": "Bolsillo completo",
    "astronet": "Astronet reentrenado",
}
METRICS = ("auc", "auc_pr", "recall_at_p099", "precision", "recall")
EXOMINER = {  # Valizadegan et al. 2022, Tables 4-5: whole dataset (fold mean ± 99 % interval)
    "recall_at_p099": (0.936, 0.939, 0.030),
    "auc_pr": (0.995, 0.995, 0.002),
    "precision": (0.968, 0.968, 0.011),
    "recall": (0.974, 0.974, 0.000),
    "auc": (1.000, 1.000, 0.000),
}
EXOMINER_ROBOVETTER = {"recall_at_p099": (0.01, 0.364, 0.484), "auc_pr": (0.958, 0.958, 0.019), "auc": (0.994, 0.994, 0.003)}
EXOMINER_ASTRONET = {"recall_at_p099": (0.0012, 0.198, 0.174), "auc_pr": (0.925, 0.927, 0.014), "auc": (0.993, 0.993, 0.003)}
# ExoMiner's parameter count is not in the paper. Counted by hand (not by running its code) from the
# tuned architecture of its Figure 9 (kernel 6, filters 16-32-64, branch FC-4, 4 x FC-128, PReLU, no
# batch norm, 301/31-bin views) and the model class of the public repository at that time
# (CNN1dPlanetFinderv2, nasa/ExoMiner commit 551bb1f): 252,647. Range 252,193-268,519 depending on
# pooling padding and on the stellar/DV scalar layers added to the code after the paper.
EXOMINER_PARAMETERS = (252_647, 252_193, 268_519)


def model_inputs(name: str) -> tuple[str, ...]:
    return {"pocket_full": models.PocketExo.INPUTS, "pocket_gl": models.PocketAstronet.INPUTS,
            "astronet": models.AstronetCNN.INPUTS, "gbm": ("scalars",)}.get(name, ())


def make_model(name: str, n_scalars: int) -> torch.nn.Module:
    if name == "pocket_full":
        return models.PocketExo(n_scalars)
    return models.MODELS["pocket" if name == "pocket_gl" else "astronet"]()


# --- Data ------------------------------------------------------------------------


def load_dataset() -> dict:
    """Labelled DR25 signals with their normalised views and raw (transformed) scalars."""
    labels.fetch_all()
    tce = dr25.read_csv(dr25.TCE_CSV)
    lab = labels.build(tce)
    rows = np.flatnonzero(lab["label"] >= 0)
    built = dr25.load(V.VIEW_NAMES)
    index = {(int(k), int(p)): i for i, (k, p) in enumerate(zip(built["kepid"], built["tce_plnt_num"]))}
    vrow = np.array([index[(int(lab["kepid"][i]), int(lab["tce_plnt_num"][i]))] for i in rows])
    views = features.view_inputs({k: built[k][vrow] for k in features.VIEW_INPUTS})
    del built
    return {
        "kepid": lab["kepid"][rows],
        "tce_plnt_num": lab["tce_plnt_num"][rows],
        "label": lab["label"][rows],
        "label_class": lab["label_class"][rows],
        "robovetter_score": lab["robovetter_score"][rows],
        "robovetter_pc": lab["robovetter_pc"][rows],
        "views": views,
        "scalars": features.scalar_matrix(tce, rows),
        "counts": labels.counts(lab),
    }


def fold_parts(kepid: np.ndarray, fold: int, fold_seed: int = FOLD_SEED) -> dict[str, np.ndarray]:
    """Test = fold ``fold``, validation = the next fold, training = the other 8, with the
    folds of ``splits.star_folds(kepid, 10, seed=fold_seed)``."""
    folds = splits.star_folds(kepid, K_FOLDS, seed=fold_seed)
    test, val = folds == fold, folds == (fold + 1) % K_FOLDS
    parts = {"train": np.flatnonzero(~test & ~val), "val": np.flatnonzero(val), "test": np.flatnonzero(test)}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert splits.shared_stars(kepid[parts[a]], kepid[parts[b]]) == 0
    return parts


def pred_path(model: str, fold: int, seed: int, quick: bool):
    return RUN_DIR / ("quick" if quick else "preds") / f"{model}_fold{fold}_seed{seed}.npz"


# --- Training ----------------------------------------------------------------------


def run_gbm(data: dict, parts: dict, seed: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    x, y = data["scalars"], data["label"]
    load0 = os.getloadavg()[0]
    t = time.perf_counter()
    clf = HistGradientBoostingClassifier(**GBM_PARAMS, random_state=seed)
    clf.fit(x[parts["train"]], y[parts["train"]])
    seconds = time.perf_counter() - t
    # Number of boosting rounds chosen on the validation fold (AUC-PR), like early stopping.
    val_stages = [p[:, 1] for p in clf.staged_predict_proba(x[parts["val"]])]
    scores = [training.average_precision(y[parts["val"]], p) for p in val_stages]
    best = int(np.argmax(scores))
    test_p = None
    for i, p in enumerate(clf.staged_predict_proba(x[parts["test"]])):
        if i == best:
            test_p = p[:, 1]
            break
    nodes = sum(len(pred.nodes) for it in clf._predictors[: best + 1] for pred in it)
    return {"val": val_stages[best], "test": test_p, "train_seconds": seconds, "selected_epoch": best + 1, "parameters": nodes,
            "load": 0.5 * (load0 + os.getloadavg()[0])}


def run_neural(name: str, data: dict, device_data: training.Tensors, parts: dict, fold: int, seed: int, quick: bool) -> dict:
    cfg = CONFIGS[name]
    if quick:
        cfg = training.TrainConfig(**{**cfg.to_dict(), "epochs": 1})
    sub = {k: device_data.subset(v) for k, v in parts.items()}
    training.seed_everything(seed)
    model = make_model(name, data["scalars"].shape[1])
    load0 = os.getloadavg()[0]
    fit = training.train(model, sub["train"], sub["val"], cfg, seed=seed, log_every=0)
    out = {
        "val": training.predict(model, sub["val"]),
        "test": training.predict(model, sub["test"]),
        "train_seconds": fit["train_seconds"],
        "selected_epoch": fit["selected_epoch"],
        "parameters": models.count_parameters(model),
        "load": 0.5 * (load0 + os.getloadavg()[0]),
    }
    if fold == 0 and seed == 0:  # cost of one prediction, once per model
        device = device_data.label.device
        out["inference_device"] = training.time_inference(model, sub["test"], device, 1024)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cpu = torch.device("cpu")
        out["inference_cpu1"] = training.time_inference(model.to(cpu), sub["test"].to(cpu), cpu, batch_size=1, n=200)
        torch.set_num_threads(threads)
        out["macs"] = models.count_macs(model.to(cpu))
        model.to(device)
    if name != "astronet" and not quick:  # pocket weights are tiny; keep them
        torch.save(model.state_dict(), pred_path(name, fold, seed, quick).with_suffix(".pt"))
    return out


def device_tensors(name: str, data: dict, scalars: np.ndarray, device) -> training.Tensors:
    """Every labelled signal on the device, with only the inputs this model reads."""
    names = model_inputs(name)
    features.assert_allowed([n for n in names if n != "scalars"])
    inputs = {n: data["views"][n] for n in names if n != "scalars"}
    if "scalars" in names:
        inputs["scalars"] = scalars
    return training.Tensors.from_arrays(inputs, data["label"], device)


def run(args) -> None:
    device = training.pick_device(args.device)
    print(f"device: {device}  torch {torch.__version__}  load {os.getloadavg()[0]:.1f}", flush=True)
    data = load_dataset()
    print(f"labelled signals: {len(data['label'])} ({int(data['label'].sum())} planets), {data['counts']}", flush=True)
    features.assert_allowed(list(features.SCALAR_COLUMNS))
    y = data["label"]
    for name in args.models:
        if name == "robovetter":
            continue
        n_seeds = min(args.seeds, args.astronet_seeds) if name == "astronet" else args.seeds
        dev = None
        order = range(args.folds)[::-1] if args.reverse else range(args.folds)
        for fold in order:
            todo = [s for s in range(n_seeds) if not pred_path(name, fold, s, args.quick).exists()]
            if not todo:
                continue
            parts = fold_parts(data["kepid"], fold)
            scalars = features.Standardizer().fit(data["scalars"][parts["train"]]).transform(data["scalars"])
            if name in NEURAL:
                if dev is None:
                    dev = device_tensors(name, data, scalars, device)
                if "scalars" in dev.inputs:  # standardised with this fold's training part
                    dev.inputs["scalars"] = torch.from_numpy(scalars).to(device)
            for seed in todo:
                if pred_path(name, fold, seed, args.quick).exists():  # done meanwhile by another process
                    continue
                t = time.perf_counter()
                r = run_gbm(data, parts, seed) if name == "gbm" else run_neural(name, data, dev, parts, fold, seed, args.quick)
                thr = training.choose_threshold(y[parts["val"]], r["val"])
                m = training.classification_metrics(y[parts["test"]], r["test"], thr)
                path = pred_path(name, fold, seed, args.quick)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, **{k: np.asarray(v) for k, v in r.items()}, threshold=thr,
                         val_index=parts["val"], test_index=parts["test"])
                print(f"[{name}] fold {fold} seed {seed}: AUC-PR {m['auc_pr']:.4f}  R@P.99 {m['recall_at_p099']:.3f}  "
                      f"P {m['precision']:.3f} R {m['recall']:.3f}  ({r['train_seconds']:.0f} s, epoch {r['selected_epoch']}, "
                      f"load {os.getloadavg()[0]:.0f}, {time.perf_counter() - t:.0f} s total)", flush=True)
        del dev
        if device.type == "mps":
            torch.mps.empty_cache()


# --- Report ------------------------------------------------------------------------


def _t95(n: int) -> float:
    from scipy.stats import t

    return float(t.ppf(0.975, n - 1)) if n > 1 else float("nan")


def _summary(per_fold: list[float]) -> dict:
    a = np.asarray(per_fold, dtype=float)
    half = _t95(len(a)) * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
    return {"mean": float(a.mean()), "ci95": float(half), "folds": a.tolist()}


def _metrics(y, p, thr=None, pred=None) -> dict:
    m = training.classification_metrics(y, p, 0.5 if thr is None else thr)
    if pred is not None:  # a fixed operating point (the Robovetter's PC disposition)
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        m.update(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), tp=tp, fp=fp, fn=fn)
    return m


def evaluate(name: str, data: dict, args) -> dict | None:
    """Per (fold, seed) metrics, their fold means with a 95 % interval over the 10 folds,
    the ensemble of the seeds of each fold and the pooled (whole-dataset) numbers."""
    y, cls = data["label"], data["label_class"]
    folds = range(args.folds)
    per_run, fold_avg, ens_fold = {}, {k: [] for k in METRICS}, {k: [] for k in METRICS}
    pooled_idx, pooled_p, pooled_pred, costs = [], [], [], {}
    seeds_done = None
    by_seed: dict[int, list] = {}
    for fold in folds:
        parts = fold_parts(data["kepid"], fold)
        yt = y[parts["test"]]
        if name == "robovetter":
            score, pc = data["robovetter_score"][parts["test"]], data["robovetter_pc"][parts["test"]]
            m = _metrics(yt, score, pred=pc)
            runs = {0: m}
            ens_p, ens_pred = score, pc
        else:
            files = sorted(f for f in (RUN_DIR / ("quick" if args.quick else "preds")).glob(f"{name}_fold{fold}_seed*.npz"))
            if not files:
                return None
            runs, val_ps, test_ps = {}, [], []
            for f in files:
                seed = int(re.search(r"seed(\d+)", f.name).group(1))
                z = dict(np.load(f))
                assert np.array_equal(z["test_index"], parts["test"]) and np.array_equal(z["val_index"], parts["val"])
                runs[seed] = _metrics(yt, z["test"], float(z["threshold"]))
                runs[seed]["train_seconds"] = float(z["train_seconds"])
                runs[seed]["selected_epoch"] = int(z["selected_epoch"])
                runs[seed]["load"] = float(z["load"]) if "load" in z else float("nan")
                by_seed.setdefault(seed, []).append((parts["test"], z["test"], z["test"] > float(z["threshold"])))
                costs.setdefault("parameters", int(z["parameters"]))
                for k in ("inference_device", "inference_cpu1", "macs"):
                    if k in z:
                        costs[k] = float(z[k])
                val_ps.append(z["val"])
                test_ps.append(z["test"])
            ens_val, ens_p = np.mean(val_ps, axis=0), np.mean(test_ps, axis=0)
            thr = training.choose_threshold(y[parts["val"]], ens_val)
            ens_pred = ens_p > thr
            m = _metrics(yt, ens_p, thr)
        seeds_done = len(runs) if seeds_done is None else min(seeds_done, len(runs))
        per_run[fold] = runs
        for k in METRICS:
            fold_avg[k].append(float(np.mean([r[k] for r in runs.values()])))
            ens_fold[k].append(_metrics(yt, ens_p, pred=ens_pred)[k] if name == "robovetter" else
                               _metrics(yt, ens_p, thr)[k])
        pooled_idx.append(parts["test"])
        pooled_p.append(ens_p)
        pooled_pred.append(ens_pred)
    idx, p, pred = np.concatenate(pooled_idx), np.concatenate(pooled_p), np.concatenate(pooled_pred)
    yp = y[idx]
    pooled = _metrics(yp, p, pred=pred)
    pooled["missed_planets"] = int((~pred & (yp == 1)).sum())
    pooled["false_alarms_afp"] = int((pred & (cls[idx] == "AFP")).sum())
    pooled["false_alarms_ntp"] = int((pred & (cls[idx] == "NTP")).sum())
    pooled["n_planets"] = int(yp.sum())
    seed_pooled = {}
    for seed, chunks in by_seed.items():
        if len(chunks) == args.folds:
            ii = np.concatenate([c[0] for c in chunks])
            pp = np.concatenate([c[1] for c in chunks])
            seed_pooled[seed] = {"auc_pr": training.average_precision(y[ii], pp), "recall_at_p099": training.recall_at_precision(y[ii], pp)}
    all_runs = [r for runs in per_run.values() for r in runs.values()]
    if name != "robovetter":
        costs["train_seconds"] = {"mean": float(np.mean([r["train_seconds"] for r in all_runs])),
                                  "std": float(np.std([r["train_seconds"] for r in all_runs], ddof=1)) if len(all_runs) > 1 else 0.0,
                                  "max": float(np.max([r["train_seconds"] for r in all_runs]))}
        costs["load"] = float(np.nanmean([r["load"] for r in all_runs]))
        costs["selected_epoch"] = float(np.mean([r["selected_epoch"] for r in all_runs]))
    return {
        "folds": args.folds,
        "seeds": seeds_done,
        "runs": len(all_runs),
        "single": {k: _summary(v) for k, v in fold_avg.items()},
        "single_all_runs_mean": {k: float(np.mean([r[k] for r in all_runs])) for k in METRICS},
        "ensemble": {k: _summary(v) for k, v in ens_fold.items()},
        "pooled_ensemble": pooled,
        "pooled_single_by_seed": seed_pooled,
        "pr_curve": _pr_curve(yp, p),
        "costs": costs,
    }


def _pr_curve(y, p, n: int = 400) -> dict:
    order = np.argsort(-p, kind="mergesort")
    yy = y[order].astype(bool)
    tp, fp = np.cumsum(yy), np.cumsum(~yy)
    keep = np.unique(np.linspace(0, len(p) - 1, n).astype(int))
    return {"recall": (tp[keep] / yy.sum()).round(5).tolist(), "precision": (tp[keep] / (tp[keep] + fp[keep])).round(5).tolist()}


def _es(text: str) -> str:
    return re.sub(r"(\d)\.(\d)", r"\1,\2", text)


def _int(n) -> str:
    return f"{int(round(n)):,}".replace(",", ".")


def _ci(s: dict, d: int = 3) -> str:
    return _es(f"{s['mean']:.{d}f} ± {s['ci95']:.{d}f}")


def tables(out: dict) -> str:
    r = out["results"]
    c = out["labels"]
    lines = [
        f"Generado por `{out['command']}` el {out['generated']} ({out['environment']['device']}, PyTorch "
        f"{out['environment']['torch']}, scikit-learn {out['environment']['sklearn']}). Validación cruzada por estrella "
        f"en {out['folds']} particiones: {_int(out['n_signals'])} señales con etiqueta ({_int(out['n_planets'])} planetas) en "
        f"{_int(out['n_stars'])} estrellas. «± » es el intervalo del 95 % (t de Student) sobre las {out['folds']} particiones.",
        "",
        "**Etiquetas** (reglas de ExoMiner con las tablas del archivo de hoy):",
        "",
        "| Clase | Nosotros | ExoMiner (2022) |",
        "|---|---|---|",
        f"| PC (planeta confirmado) | {_int(c.get('PC', 0))} | 2.291 |",
        f"| AFP (falso positivo certificado) | {_int(c.get('AFP', 0))} | 3.538 |",
        f"| NTP (no transitorio) | {_int(c.get('NTP', 0))} | 24.779 |",
        f"| Total con etiqueta | {_int(c.get('PC', 0) + c.get('AFP', 0) + c.get('NTP', 0))} | 30.609 |",
        f"| Sin etiqueta (candidatos sin confirmar y casos dudosos) | {_int(c.get('UNK', 0))} | — |",
        f"| TCE «rogue» quitados | {_int(c.get('ROGUE', 0))} | 1.498 |",
        "",
        "**Calidad.** Modelo único: media de las semillas en cada partición, y media ± intervalo sobre las particiones. "
        "Precisión y recall con el umbral elegido en la validación (mejor F1).",
        "",
        "| Modelo | Semillas × particiones | AUC-PR | R@P0,99 | AUC | Precisión | Recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ALL_MODELS:
        if name not in r:
            continue
        e = r[name]
        sp = "—" if name == "robovetter" else f"{e['seeds']} × {e['folds']}"
        lines.append(f"| {LABEL[name]} | {sp} | {_ci(e['single']['auc_pr'])} | {_ci(e['single']['recall_at_p099'])} | "
                     f"{_ci(e['single']['auc'], 4)} | {_ci(e['single']['precision'])} | {_ci(e['single']['recall'])} |")
        if name != "robovetter" and e["seeds"] > 1:
            lines.append(f"| {LABEL[name]}, media de {e['seeds']} | — | {_ci(e['ensemble']['auc_pr'])} | "
                         f"{_ci(e['ensemble']['recall_at_p099'])} | {_ci(e['ensemble']['auc'], 4)} | "
                         f"{_ci(e['ensemble']['precision'])} | {_ci(e['ensemble']['recall'])} |")
    lines += [
        "| *ExoMiner (publicado, media de 10 por partición)* | 10 × 10 | *0,995 ± 0,002* | *0,939 ± 0,030* | *1,000* | *0,968* | *0,974* |",
        "| *Robovetter según ExoMiner* | — | *0,958 ± 0,019* | *0,364 ± 0,484* | *0,994* | *0,950* | *0,975* |",
        "| *Astronet según ExoMiner* | 10 × 10 | *0,927 ± 0,014* | *0,198 ± 0,174* | *0,993* | *0,862* | *0,885* |",
        "",
        "Las filas en cursiva son las cifras publicadas por Valizadegan y otros (2022, tablas 4 y 5): media sobre "
        "las 10 particiones ± intervalo del **99 %**, con umbral 0,5.",
        "",
        "**Todas las señales a la vez** (como la cifra principal de ExoMiner: cada señal puntuada por los modelos de "
        "la partición en la que estaba en prueba; media de las semillas):",
        "",
        "| Modelo | AUC-PR | R@P0,99 | Precisión | Recall | Planetas perdidos | Falsas alarmas (AFP + NTP) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ALL_MODELS:
        if name not in r:
            continue
        pe = r[name]["pooled_ensemble"]
        lines.append(f"| {LABEL[name]} | {_es(format(pe['auc_pr'], '.3f'))} | {_es(format(pe['recall_at_p099'], '.3f'))} | "
                     f"{_es(format(pe['precision'], '.3f'))} | {_es(format(pe['recall'], '.3f'))} | "
                     f"{_int(pe['missed_planets'])} de {_int(pe['n_planets'])} | {_int(pe['false_alarms_afp'])} + {_int(pe['false_alarms_ntp'])} |")
    lines += ["| *ExoMiner (publicado)* | *0,995* | *0,936* | *0,968* | *0,974* | — | — |", "",
              "**Coste.** Entrenamiento: media por modelo en un Mac con M3 Max (14 núcleos) que tenía otros trabajos en marcha "
              "(carga media indicada). Inferencia: por señal, en MPS por lotes de 1.024 y en CPU con un hilo y de una en una.",
              "",
              "| Modelo | Parámetros | Multiplicaciones-sumas por señal | Entrenamiento | Carga media | Inferencia (MPS · CPU 1 hilo) |",
              "|---|---|---|---|---|---|"]
    for name in ("gbm", "pocket_gl", "pocket_full", "astronet"):
        if name not in r:
            continue
        co = r[name]["costs"]
        params = f"{_int(co['parameters'])} nodos" if name == "gbm" else _int(co["parameters"])
        macs = _int(co["macs"]) if "macs" in co else "—"
        tr = _es(f"{co['train_seconds']['mean']:.0f} ± {co['train_seconds']['std']:.0f} s (máx. {co['train_seconds']['max']:.0f} s)")
        inf = (_es(f"{co['inference_device'] * 1e6:.0f} µs · {co['inference_cpu1'] * 1e3:.2f} ms") if "inference_cpu1" in co else "—")
        load = _es(f"{co['load']:.0f}") if np.isfinite(co.get("load", np.nan)) else "—"
        lines.append(f"| {LABEL[name]} | {params} | {macs} | {tr} | {load} | {inf} |")
    lines += [f"| *ExoMiner (contados a mano de su artículo)* | *≈ {_int(EXOMINER_PARAMETERS[0])}* "
              f"(*{_int(EXOMINER_PARAMETERS[1])}–{_int(EXOMINER_PARAMETERS[2])}*) | — | — | — | — |", ""]
    v = out.get("criterion")
    if v:
        lines += [f"**Criterio del hito 4:** {v['text']}", ""]
    return "\n".join(lines)


def criterion(out: dict) -> dict | None:
    r = out["results"]
    if "pocket_full" not in r:
        return None
    pf = r["pocket_full"]
    single, ens = pf["single"]["recall_at_p099"], pf["ensemble"]["recall_at_p099"]
    lo = EXOMINER["recall_at_p099"][1] - EXOMINER["recall_at_p099"][2]
    n = pf["costs"]["parameters"]
    ratio = EXOMINER_PARAMETERS[0] / n
    ratio_lo = EXOMINER_PARAMETERS[1] / n
    ratio_astronet = r["astronet"]["costs"]["parameters"] / n if "astronet" in r else float("nan")
    minutes = pf["costs"]["train_seconds"]["max"] / 60
    ok_quality = single["mean"] >= lo
    ok_params, ok_time = ratio_lo >= 50, minutes < 10
    ok = ok_quality and ok_params and ok_time
    missing = [w for w, good in (("calidad", ok_quality), ("parámetros", ok_params), ("tiempo", ok_time)) if not good]
    hi = EXOMINER["recall_at_p099"][1] + EXOMINER["recall_at_p099"][2]
    parts = [
        f"R@P0,99 del modelo de bolsillo completo = {_ci(single)} (un modelo) y {_ci(ens)} (media de {pf['seeds']}), "
        f"frente al intervalo publicado de ExoMiner {_es(f'{lo:.3f}')}–{_es(f'{hi:.3f}')} (0,939 ± 0,030): "
        + ("dentro o por encima." if ok_quality else
           f"por debajo; la distancia a la media de ExoMiner es {_es(format(EXOMINER['recall_at_p099'][1] - single['mean'], '.3f'))} "
           f"en R@P0,99 y {_es(format(EXOMINER['auc_pr'][1] - pf['single']['auc_pr']['mean'], '.3f'))} en AUC-PR."),
        f"Parámetros: {_int(n)}, {_es(f'{ratio:.1f}')} veces menos que ExoMiner (≈ {_int(EXOMINER_PARAMETERS[0])}, contados "
        f"a mano de su artículo) y {_es(f'{ratio_astronet:.0f}')} veces menos que Astronet: "
        + ("cumple" if ok_params else "no llega") + " a las 50 veces frente a ExoMiner.",
        f"Entrenamiento más largo: {_es(f'{minutes:.1f}')} min (" + ("menos" if ok_time else "más") + " de 10).",
        "**Se cumple.**" if ok else f"**No se cumple** ({', '.join(missing)}); se publican la curva calidad–coste y las distancias.",
    ]
    return {"met": bool(ok), "quality_met": bool(ok_quality), "parameters_met": bool(ok_params), "time_met": bool(ok_time),
            "parameter_ratio_vs_exominer": ratio, "parameter_ratio_vs_exominer_low": ratio_lo,
            "parameter_ratio_vs_astronet": ratio_astronet, "max_train_minutes": minutes, "text": " ".join(parts)}


def figure(out: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    r = out["results"]
    color = {"pocket_full": "#eb6834", "pocket_gl": "#eb6834", "astronet": "#2a78d6", "gbm": "#1baf7a", "robovetter": "#52514e"}
    ink, muted, grid, surface = "#1a1a19", "#52514e", "#e2e1dc", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    lo = EXOMINER["recall_at_p099"][1] - EXOMINER["recall_at_p099"][2]
    hi = EXOMINER["recall_at_p099"][1] + EXOMINER["recall_at_p099"][2]
    ax.axhspan(lo, hi, color="#d9d8d2", alpha=0.6, lw=0)
    ax.text(1.6e3, hi + 0.008, "ExoMiner publicado (0,939 ± 0,030, intervalo 99 %)", fontsize=8.5, color=muted)
    for name in ("gbm", "pocket_gl", "pocket_full", "astronet"):
        if name not in r:
            continue
        e = r[name]
        n = e["costs"]["parameters"]
        s, en = e["single"]["recall_at_p099"], e["ensemble"]["recall_at_p099"]
        hollow = name == "pocket_gl"
        ax.errorbar([n], [s["mean"]], yerr=[s["ci95"]], fmt="o", ms=9, color=color[name], mfc="white" if hollow else color[name],
                    mew=2, capsize=3, lw=2, zorder=3)
        if e["seeds"] > 1:
            ax.errorbar([n * e["seeds"]], [en["mean"]], yerr=[en["ci95"]], fmt="D", ms=7, color=color[name],
                        mfc="white" if hollow else color[name], mew=2, capsize=3, lw=1.5, zorder=3)
            ax.plot([n, n * e["seeds"]], [s["mean"], en["mean"]], color=color[name], lw=1, ls=":", zorder=2)
        label = LABEL[name] + (" (nodos)" if name == "gbm" else "")
        ax.annotate(label, (n, s["mean"]), xytext=(8, -14), textcoords="offset points", fontsize=8.5, color=ink)
    ax.errorbar([EXOMINER_PARAMETERS[0]], [EXOMINER["recall_at_p099"][1]], yerr=[EXOMINER["recall_at_p099"][2]], fmt="*", ms=14,
                color=ink, capsize=3, lw=1.5, zorder=3)
    ax.annotate("ExoMiner (parámetros contados a mano)", (EXOMINER_PARAMETERS[0], EXOMINER["recall_at_p099"][1]), xytext=(10, -14),
                textcoords="offset points", fontsize=8.5, color=ink)
    if "robovetter" in r:
        rv = r["robovetter"]["single"]["recall_at_p099"]["mean"]
        ax.axhline(rv, color=muted, lw=1.2, ls="--")
        ax.text(1.6e3, rv + 0.01, "Robovetter (reglas, sin parámetros entrenados)", fontsize=8.5, color=muted)
    ax.set_xscale("log")
    ax.set_xlim(1e3, 3e8)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("Parámetros (escala logarítmica)")
    ax.set_ylabel("R@P0,99 (planetas recuperados con precisión 0,99)")
    ax.set_title("Calidad frente a coste: ● un modelo, ◆ media de las semillas", fontsize=10)
    ax = axes[1]
    for name in ("robovetter", "gbm", "pocket_gl", "astronet", "pocket_full"):
        if name not in r:
            continue
        pr = r[name]["pr_curve"]
        ax.plot(pr["recall"], pr["precision"], color=color[name], lw=2, ls=":" if name == "pocket_gl" else ("--" if name == "robovetter" else "-"),
                label=LABEL[name])
    ax.plot([EXOMINER["recall_at_p099"][0]], [0.99], marker="*", ms=14, color=ink, ls="none", label="ExoMiner publicado (R@P0,99)")
    ax.axhline(0.99, color=grid, lw=1)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.8, 1.002)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 0.99])
    ax.set_xlabel("Recall (planetas encontrados)")
    ax.set_ylabel("Precisión")
    ax.set_title("Precisión–recall con todas las señales (media de las semillas)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    comma = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(color=grid, lw=0.8)
        a.set_axisbelow(True)
        a.yaxis.set_major_formatter(comma)
    axes[1].xaxis.set_major_formatter(comma)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


def update_doc(doc, text: str) -> None:
    start, end = "<!-- tabla:inicio -->", "<!-- tabla:fin -->"
    content = doc.read_text()
    before, rest = content.split(start, 1)
    _, after = rest.split(end, 1)
    doc.write_text(f"{before}{start}\n{text}\n{end}{after}")


def report(args) -> None:
    import sklearn

    data = load_dataset()
    out = {
        "generated": time.strftime("%Y-%m-%d"),
        "command": "uv run python scripts/hito4_dr25_cv.py run && uv run python scripts/hito4_dr25_cv.py report",
        "environment": {"device": str(training.pick_device(args.device)), "torch": torch.__version__, "sklearn": sklearn.__version__,
                        "python": platform.python_version(), "machine": platform.machine(), "system": platform.system()},
        "labels": data["counts"],
        "label_rules": labels.__doc__.split("\n\n")[1],
        "n_signals": int(len(data["label"])),
        "n_planets": int(data["label"].sum()),
        "n_stars": int(len(np.unique(data["kepid"]))),
        "folds": args.folds,
        "fold_rule": "splits.star_folds(kepid, 10, seed=0); test = fold f, validation = fold f+1, training = the other 8",
        "threshold_rule": "max F1 on the validation fold, per model, fold and seed",
        "inputs": {"scalars": features.SCALAR_COLUMNS, "views": list(features.VIEW_INPUTS),
                   "forbidden_patterns": list(features.FORBIDDEN_PATTERNS),
                   "per_model": {m: list(model_inputs(m)) for m in ALL_MODELS if m != "robovetter"}},
        "configs": {**{k: v.to_dict() for k, v in CONFIGS.items()}, "gbm": GBM_PARAMS},
        "exominer_published": {"model": EXOMINER, "robovetter": EXOMINER_ROBOVETTER, "astronet": EXOMINER_ASTRONET,
                               "parameters_counted_by_hand": {"best": EXOMINER_PARAMETERS[0], "low": EXOMINER_PARAMETERS[1],
                                                              "high": EXOMINER_PARAMETERS[2]},
                               "format": "metric: (whole dataset, fold mean, 99 % half-interval)"},
        "results": {},
    }
    for name in args.models:
        res = evaluate(name, data, args)
        if res is not None:
            out["results"][name] = res
            print(f"{name}: {res['runs']} runs  AUC-PR {res['single']['auc_pr']['mean']:.4f}  R@P.99 "
                  f"{res['single']['recall_at_p099']['mean']:.3f} ± {res['single']['recall_at_p099']['ci95']:.3f}", flush=True)
    out["criterion"] = criterion(out)
    text = tables(out)
    print(text)
    if args.quick:
        return
    (DOCS / "hito4_results.json").write_text(json.dumps(out, indent=1) + "\n")
    update_doc(DOCS / "hito4.md", text)
    figure(out, DOCS / "hito4_calidad_coste.png")
    print("wrote docs/hito4_results.json, docs/hito4.md, docs/hito4_calidad_coste.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=list(ALL_MODELS))
    ap.add_argument("--folds", type=int, default=K_FOLDS)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--astronet-seeds", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--reverse", action="store_true", help="folds from last to first (to share the work with a second process)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.folds, args.seeds = min(args.folds, 2), 1
    if args.command == "run":
        run(args)
    else:
        report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
