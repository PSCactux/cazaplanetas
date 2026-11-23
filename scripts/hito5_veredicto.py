"""Milestone 5, step 4: the final model and the verdict against ExoMiner.

    uv run python scripts/hito5_veredicto.py run      # 200 models on the new folds (resumable; hours)
    uv run python scripts/hito5_veredicto.py report   # docs/hito5_veredicto.md, its JSON and the quality-cost figure (es, en)
    uv run python scripts/hito5_veredicto.py run --quick   # smoke test: 1 fold, 1 seed, 1 epoch

Everything here was fixed and committed before any model was trained on these folds:

* **Folds.** A new assignment of stars to 10 folds, ``splits.star_folds(kepid, 10,
  seed=VERDICT_FOLD_SEED)``, never used before (every earlier step used seed 0). Same
  protocol as milestone 4: test = fold f, validation = fold f+1, training = the other 8.
* **Recipe.** ``combo`` of step 2 (docs/hito5_receta.md), unchanged: 25,745 parameters.
  "One model" is the mean of the per-fold metrics over its 10 seeds; "ensemble" is the
  mean score of the 10 seeds; "distilled" is one model of the same recipe trained with
  target 0.5 x label + 0.5 x the ensemble's score on its training part (the ensemble of
  each fold never saw that fold's validation or test part). Seeds 0-9 for both.
* **ExoMiner.** Its published per-signal cross-validation scores (Tables 3 and 13 of its
  paper), measured on the same new test parts with our labels and the same code
  (``hito5_diagnostico.measure``), and its published fold means (0.939 as it counts
  R@P0.99, 0.920 with the exact count, 0.995 AUC-PR; its labels and folds).
* **Verdict** (:func:`verdict`): see ``RULE``.

Each (recipe, fold, seed) saves its validation, test and all-signal predictions and its
weights under data/hito5/veredicto/runs/, so ``run`` can be stopped and resumed and
several processes can share the work with ``--shard i/n``. Without those runs, ``report``
reads the copy of their validation and test scores published in release/, so the verdict
can be recomputed on a fresh clone (it downloads about 50 MB of public catalogue tables).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from cazaplanetas.env import DATA_DIR, REPO_ROOT, use_local_cache

use_local_cache()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hito4_dr25_cv as h4  # noqa: E402
import hito5_diagnostico as h5  # noqa: E402
import hito5_receta as R  # noqa: E402
from cazaplanetas import exominer_tables as ex  # noqa: E402
from cazaplanetas import models, release, splits, training  # noqa: E402

DOCS = REPO_ROOT / "docs"
DOC = DOCS / "hito5_veredicto.md"
RUN_DIR = DATA_DIR / "hito5" / "veredicto"
VERDICT_FOLD_SEED = 20261122  # the date the recipe was closed; any seed other than 0 would do
SEEDS = 10
RECIPE = "combo"
DISTILLED = "distilled_combo"
RECIPES = {
    RECIPE: R.RECIPES[RECIPE],
    DISTILLED: R.Recipe(**{**{k: getattr(R.RECIPES[RECIPE], k) for k in R.Recipe.__dataclass_fields__},
                           "label": "Destilado del conjunto de 10 de la receta final (objetivo: ½ etiqueta + ½ media del conjunto)",
                           "parent": RECIPE, "teacher": RECIPE,
                           "train": tuple({**dict(R.RECIPES[RECIPE].train), "distill_alpha": 0.5}.items())}),
}

# ExoMiner as published (Valizadegan et al. 2022, Table 4, mean of its 10 folds, its labels): the
# bar of the plan's definition. The exact count is the same scores measured with precision >= 0.99
# (docs/hito5_diagnostico.md); its fold mean of the AUC-PR is 0.995 as published.
PUBLISHED = {"recall_paper": 0.939, "recall_exact": 0.920, "auc_pr": 0.995}
EXOMINER_ENSEMBLE_PARAMETERS = 10 * h4.EXOMINER_PARAMETERS[0]
RULE = ("«Supera» (definición del plan, hito 5): la media de las 10 particiones es igual o mayor que la de ExoMiner en "
        "R@P0,99 y en AUC-PR, con menos parámetros y ≤ 50.000 por modelo. Contra lo publicado: R@P0,99 contado como ExoMiner "
        "≥ 0,939, R@P0,99 exacto ≥ 0,920 (la misma cuenta a los dos lados) y AUC-PR ≥ 0,995. Contra la medida directa: "
        "las mismas tres medias ≥ las de las puntuaciones publicadas de ExoMiner en las mismas partes de prueba nuevas, con "
        "nuestras etiquetas y el mismo código. Se comparan el conjunto de 10 con su conjunto de 10 y, aparte, un modelo "
        "(y el destilado) con su conjunto de 10 (ExoMiner no publica modelos sueltos).")


def run_path(name: str, fold: int, seed: int, quick: bool = False):
    return RUN_DIR / ("quick" if quick else "runs") / f"{name}_fold{fold}_seed{seed}.npz"


def folds_of(kepid: np.ndarray) -> np.ndarray:
    return splits.star_folds(kepid, h4.K_FOLDS, seed=VERDICT_FOLD_SEED)


# --- Training ------------------------------------------------------------------------------


def run(args) -> None:
    device = training.pick_device(args.device)
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    print(f"device: {device}  torch {torch.__version__}  load {os.getloadavg()[0]:.1f}  shard {shard}/{n_shards}  pid {os.getpid()}", flush=True)
    data = R.load_dataset(need_centroid_raw=True)
    y = data["label"]
    folds, seeds = (range(h4.K_FOLDS), range(SEEDS)) if not args.quick else ((0,), (0,))
    names = [RECIPE] if args.quick else [RECIPE, DISTILLED]
    tasks = [(name, fold) for name in names for fold in folds]
    tasks = [t for i, t in enumerate(tasks) if i % n_shards == shard]
    for name, fold in tasks:
        recipe = RECIPES[name]
        todo = [s for s in seeds if not run_path(name, fold, s, args.quick).exists()]
        if not todo:
            continue
        parts = h4.fold_parts(data["kepid"], fold, fold_seed=VERDICT_FOLD_SEED)
        dev = R.device_tensors(data, recipe, parts, device)
        if recipe.teacher:
            while not all(run_path(recipe.teacher, fold, s).exists() for s in range(SEEDS)):  # teacher still training
                time.sleep(60)
            teacher = np.mean([np.load(run_path(recipe.teacher, fold, s))["all"] for s in range(SEEDS)], axis=0)
            dev.soft = torch.from_numpy(teacher.astype(np.float32)).to(device)
        sub = {k: dev.subset(v) for k, v in parts.items()}
        cfg = recipe.train_config(args.quick)
        for seed in todo:
            path = run_path(name, fold, seed, args.quick)
            if path.exists():  # done meanwhile by another process
                continue
            t = time.perf_counter()
            training.seed_everything(seed)
            model = recipe.make_model()
            n_params = models.count_parameters(model)
            assert n_params <= R.PARAMETER_CAP, f"{name}: {n_params} parameters"
            fit = training.train(model, sub["train"], sub["val"], cfg, seed=seed, log_every=0)
            val, test, every = training.predict(model, sub["val"]), training.predict(model, sub["test"]), training.predict(model, dev)
            thr = training.choose_threshold(y[parts["val"]], val)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part.npz")
            np.savez(tmp, val=val, test=test, all=every.astype(np.float32), threshold=thr, val_index=parts["val"], test_index=parts["test"],
                     train_seconds=fit["train_seconds"], selected_epoch=fit["selected_epoch"], parameters=n_params,
                     load=os.getloadavg()[0])
            tmp.replace(path)
            if not args.quick:
                torch.save(model.state_dict(), path.with_suffix(".pt"))
            # Only the validation AUC-PR is printed while training: the test parts are read by ``report``.
            print(f"[{name}] fold {fold} seed {seed}: val AUC-PR {training.average_precision(y[parts['val']], val):.4f}  "
                  f"({fit['train_seconds']:.0f} s, epoch {fit['selected_epoch']}, {n_params} params, load {os.getloadavg()[0]:.0f}, "
                  f"{time.perf_counter() - t:.0f} s total)", flush=True)
        del dev, sub
        if device.type == "mps":
            torch.mps.empty_cache()


# --- Report: numbers -----------------------------------------------------------------------


def load_runs(name: str, kepid: np.ndarray, plnt: np.ndarray) -> dict | None:
    """Test and validation scores of the 10 folds x 10 seeds of ``name``: from the local runs
    under data/hito5/veredicto/runs/ if they are all there, otherwise from the copy published in
    release/ (``cazaplanetas.release``; written from those runs by scripts/export_release.py)."""
    n = len(kepid)
    if not all(run_path(name, f, s).exists() for f in range(h4.K_FOLDS) for s in range(SEEDS)):
        return _runs_from_release(name, kepid, plnt)
    test = np.full((SEEDS, n), np.nan)
    val, secs, params = {}, [], None
    for f in range(h4.K_FOLDS):
        for s in range(SEEDS):
            z = np.load(run_path(name, f, s))
            test[s, z["test_index"]] = z["test"]
            val[(f, s)] = (z["val_index"], z["val"])
            secs.append(float(z["train_seconds"]))
            params = int(z["parameters"])
    return {"test": test, "val": val, "seconds": secs, "parameters": params, "folds": list(range(h4.K_FOLDS)), "seeds": list(range(SEEDS))}


def _runs_from_release(name: str, kepid: np.ndarray, plnt: np.ndarray) -> dict | None:
    if not release.PREDICTIONS.exists():
        return None
    z = release.load_predictions()
    if f"{name}/test" not in z:
        return None
    if not (np.array_equal(z["kepid"], kepid) and np.array_equal(z["tce_plnt_num"], plnt)):
        raise RuntimeError(f"{release.PREDICTIONS.name}: its signals are not the labelled DR25 signals built here")
    fold = folds_of(kepid)
    test = z[f"{name}/test"].astype(np.float64)
    val = {}
    for f in range(h4.K_FOLDS):
        vi = np.flatnonzero(fold == (f + 1) % h4.K_FOLDS)
        for s in range(SEEDS):
            val[(f, s)] = (vi, z[f"{name}/val"][s, vi])
    return {"test": test, "val": val, "seconds": z[f"{name}/train_seconds"].ravel().tolist(), "parameters": int(z[f"{name}/parameters"]),
            "folds": list(range(h4.K_FOLDS)), "seeds": list(range(SEEDS)), "source": "release"}


def progress() -> dict:
    local = {name: sum(run_path(name, f, s).exists() for f in range(h4.K_FOLDS) for s in range(SEEDS)) for name in RECIPES}
    if release.PREDICTIONS.exists() and not all(v == h4.K_FOLDS * SEEDS for v in local.values()):
        with np.load(release.PREDICTIONS) as z:
            return {name: h4.K_FOLDS * SEEDS if f"{name}/test" in z.files else local[name] for name in RECIPES}
    return local


def exominer_published(ctx: dict) -> dict:
    """ExoMiner's scores on its own labels and folds (the published protocol), with our code."""
    t3, _ = ctx["exominer_tables"]
    their_fold = ex.read_folds()
    f3 = np.array([their_fold.get((int(k), int(p)), -1) for k, p in zip(t3["kepid"], t3["tce_plnt_num"])])
    return h5.measure(t3["label"], f3, t3["exominer"])


def split_check(ctx: dict, fold: np.ndarray) -> dict:
    """The new folds against those of every earlier step: sizes, stars shared, signals that move."""
    old = splits.star_folds(ctx["kepid"], h4.K_FOLDS, seed=h4.FOLD_SEED)
    y = ctx["label"]
    same_test_mates = []  # for each signal, the share of its test-part companions that were also with it before
    for f in range(h4.K_FOLDS):
        m = fold == f
        counts = np.bincount(old[m], minlength=h4.K_FOLDS)
        same_test_mates.append(float(counts.max() / m.sum()))
    leaks = 0
    for f in range(h4.K_FOLDS):
        parts = h4.fold_parts(ctx["kepid"], f, fold_seed=VERDICT_FOLD_SEED)
        leaks += sum(splits.shared_stars(ctx["kepid"][parts[a]], ctx["kepid"][parts[b]])
                     for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    return {"fold_seed": VERDICT_FOLD_SEED, "signals": [int((fold == f).sum()) for f in range(h4.K_FOLDS)],
            "planets": [int(((fold == f) & (y == 1)).sum()) for f in range(h4.K_FOLDS)],
            "shared_stars_between_parts": int(leaks),
            "largest_share_from_one_old_fold": max(same_test_mates),
            "signals_in_same_fold_number": float(np.mean(old == fold))}


def verdict(ours: dict, exo_direct: dict) -> dict:
    """The rule fixed before measuring (see RULE). ``ours`` and ``exo_direct``: outputs of
    ``h5.measure`` on the new folds."""
    keys = ("recall_paper", "recall_exact", "auc_pr")
    published = {k: {"ours": ours[k]["mean"], "theirs": PUBLISHED[k], "gap": ours[k]["mean"] - PUBLISHED[k],
                     "ok": bool(ours[k]["mean"] >= PUBLISHED[k])} for k in keys}
    direct = {}
    for k in keys:
        d = h5._stat(np.subtract(ours[k]["folds"], exo_direct[k]["folds"]))
        direct[k] = {"ours": ours[k]["mean"], "theirs": exo_direct[k]["mean"], "gap": d["mean"], "ci95": d["ci95"],
                     "folds_ahead": int(np.sum(np.asarray(d["folds"]) >= 0)), "ok": bool(ours[k]["mean"] >= exo_direct[k]["mean"])}
    return {"published": published, "published_ok": all(v["ok"] for v in published.values()),
            "direct": direct, "direct_ok": all(v["ok"] for v in direct.values())}


def picked(runs: dict, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """In each fold, the seed with the best validation AUC-PR (as in step 2)."""
    out = np.full(len(y), np.nan)
    for f in runs["folds"]:
        best = max(runs["seeds"], key=lambda s: training.average_precision(y[runs["val"][(f, s)][0]], runs["val"][(f, s)][1]))
        out[fold == f] = runs["test"][best][fold == f]
    return out


def compute(ctx: dict) -> dict:
    y = ctx["label"]
    n = len(y)
    fold = folds_of(ctx["kepid"])
    exo_score = ctx["exominer_score"]
    missing = ~np.isfinite(exo_score)
    exo = np.where(missing, 0.0, exo_score)  # 3 signals without a published score count as 0, as in step 2
    res = {"generated": time.strftime("%Y-%m-%d"), "rule": RULE, "published_bar": PUBLISHED, "seeds": SEEDS,
           "recipe": {"name": RECIPE, "label": RECIPES[RECIPE].label, "train": dict(RECIPES[RECIPE].train_config().to_dict())},
           "split": split_check(ctx, fold), "progress": progress(),
           "exominer_missing_scores": {"signals": int(missing.sum()), "planets": int((missing & (y == 1)).sum())},
           "rows": {}}
    rows = res["rows"]
    rows["exominer_published"] = {"label": "*ExoMiner, conjunto de 10, publicado (sus etiquetas y particiones)*",
                                  "parameters": EXOMINER_ENSEMBLE_PARAMETERS, "m": exominer_published(ctx)}
    rows["exominer_direct"] = {"label": "*ExoMiner, conjunto de 10, sus puntuaciones en nuestras particiones nuevas*",
                               "parameters": EXOMINER_ENSEMBLE_PARAMETERS, "m": h5.measure(y, fold, exo)}
    combo, dist = (load_runs(name, ctx["kepid"], ctx["tce_plnt_num"]) for name in (RECIPE, DISTILLED))
    if combo is not None:
        p = combo["parameters"]
        rows["single"] = {"label": "**Un modelo** (media de las 10 semillas)", "parameters": p, "m": h5.measure(y, fold, combo["test"]),
                          "train_seconds": float(np.mean(combo["seconds"]))}
        rows["picked"] = {"label": "Un modelo elegido con la validación entre 10", "parameters": p, "m": h5.measure(y, fold, picked(combo, y, fold))}
        rows["ensemble"] = {"label": "**Conjunto de 10**", "parameters": 10 * p, "m": h5.measure(y, fold, combo["test"].mean(0))}
    if dist is not None:
        p = dist["parameters"]
        rows["distilled"] = {"label": "**Destilado del conjunto** (media de las 10 semillas)", "parameters": p, "m": h5.measure(y, fold, dist["test"]),
                             "train_seconds": float(np.mean(dist["seconds"]))}
        rows["distilled_picked"] = {"label": "Destilado elegido con la validación entre 10", "parameters": p, "m": h5.measure(y, fold, picked(dist, y, fold))}
    exo_d = rows["exominer_direct"]["m"]
    res["verdicts"] = {}
    for key, label in (("ensemble", "Conjunto de 10 contra su conjunto de 10"), ("single", "Un modelo contra su conjunto de 10"),
                       ("distilled", "Destilado (un modelo) contra su conjunto de 10")):
        if key in rows:
            res["verdicts"][key] = {"label": label, "parameters": rows[key]["parameters"], **verdict(rows[key]["m"], exo_d)}
    step2 = DOCS / "hito5_receta.json"
    if step2.exists():  # the same recipe on the folds of every earlier step, for reference
        old = {r["key"]: r for r in json.loads(step2.read_text())["final_rows"]}
        res["step2"] = {k: old[k]["m"] for k in ("exominer", "combo_single", "combo_ens", "distilled_single", "base_single", "base_ens10") if k in old}
    return res


# --- Report: text and figure -----------------------------------------------------------------

_f, _pm, _int = h5._f, h5._pm, h4._int


def _sv(x: float, d: int = 3) -> str:
    return h4._es(f"{x:+.{d}f}").replace("-", "−")


def text_table(res: dict) -> str:
    rows = res["rows"]
    lines = ["Media de las 10 particiones nuevas ± intervalo del 95 % (entre paréntesis, ± intervalo del 99 %, el que usa ExoMiner). "
             "Todo con nuestras etiquetas y el mismo código de medida, salvo la primera fila, que es la cifra publicada de ExoMiner "
             "(sus etiquetas y sus particiones, medida con sus puntuaciones).", "",
             "| | Parámetros al predecir | R@P0,99 exacto | R@P0,99 como lo mide ExoMiner | AUC-PR |", "|---|---|---|---|---|"]
    for key, r in rows.items():
        m = r["m"]
        par = "≈ 2,5 millones" if key.startswith("exominer") else _int(r["parameters"])
        lines.append(f"| {r['label']} | {par} | {_pm(m['recall_exact'])} ({_f(m['recall_exact']['ci99'])}) | "
                     f"{_pm(m['recall_paper'])} ({_f(m['recall_paper']['ci99'])}) | {_pm(m['auc_pr'], 4)} ({_f(m['auc_pr']['ci99'], 4)}) |")
    missing = [k for k in ("single", "ensemble", "distilled") if k not in rows]
    if missing:
        p = res["progress"]
        lines += ["", f"*Faltan modelos: {p[RECIPE]}/100 de la receta y {p[DISTILLED]}/100 del destilado.*"]
    old = res.get("step2")
    if old:
        lines += ["", "La misma receta con las particiones de los pasos anteriores (paso 2, `docs/hito5_receta.md`), como referencia: "
                  f"un modelo {_f(old['combo_single']['recall_exact']['mean'])} · {_f(old['combo_single']['recall_paper']['mean'])} · "
                  f"{_f(old['combo_single']['auc_pr']['mean'], 4)}; conjunto de 10 {_f(old['combo_ens']['recall_exact']['mean'])} · "
                  f"{_f(old['combo_ens']['recall_paper']['mean'])} · {_f(old['combo_ens']['auc_pr']['mean'], 4)}; ExoMiner en aquellas "
                  f"particiones {_f(old['exominer']['recall_exact']['mean'])} · {_f(old['exominer']['recall_paper']['mean'])} · "
                  f"{_f(old['exominer']['auc_pr']['mean'], 4)}."]
    return "\n".join(lines)


NAMES = {"recall_paper": "R@P0,99 como ExoMiner", "recall_exact": "R@P0,99 exacto", "auc_pr": "AUC-PR"}


def text_verdict(res: dict) -> str:
    lines = [res["rule"], ""]
    if not res["verdicts"]:
        return "\n".join(lines + ["*Pendiente: faltan modelos.*"])
    lines += ["| Comparación | Parámetros (nuestros · suyos) | Medida | Nosotros | Publicado | Distancia | Directa: ExoMiner | Distancia, partición a partición (95 %) | Particiones igual o por delante |",
              "|---|---|---|---|---|---|---|---|---|"]
    for v in res["verdicts"].values():
        first = True
        for k in ("recall_paper", "recall_exact", "auc_pr"):
            d = 4 if k == "auc_pr" else 3
            pub, dr = v["published"][k], v["direct"][k]
            head = f"{v['label']} | {_int(v['parameters'])} · ≈ 2,5 millones" if first else " | "
            lines.append(f"| {head} | {NAMES[k]} | {_f(pub['ours'], d)} | {_f(pub['theirs'], d)} | {_sv(pub['gap'], d)} {'✔' if pub['ok'] else '✘'} | "
                         f"{_f(dr['theirs'], d)} | {_sv(dr['gap'], d)} ± {_f(dr['ci95'], d)} {'✔' if dr['ok'] else '✘'} | {dr['folds_ahead']}/10 |")
            first = False
    lines.append("")
    keys = ("recall_paper", "recall_exact", "auc_pr")
    for v in res["verdicts"].values():
        if v["published_ok"] and v["direct_ok"]:
            lines.append(f"* {v['label']}: **supera** a ExoMiner, contra sus cifras publicadas y contra sus puntuaciones en las mismas "
                         "particiones, en las tres medias.")
            continue
        sides = []
        for side, name in (("published", "contra sus cifras publicadas"), ("direct", "contra sus puntuaciones en las mismas particiones")):
            fails = [NAMES[k] for k in keys if not v[side][k]["ok"]]
            sides.append(f"{name}, falla en {' y '.join(fails)}" if fails else f"{name}, cumple las tres")
        lines.append(f"* {v['label']}: **no supera** a ExoMiner ({'; '.join(sides)}).")
    return "\n".join(lines)


def text_split(res: dict) -> str:
    s = res["split"]
    return (f"`splits.star_folds(kepid, 10, seed={s['fold_seed']})`: {_int(min(s['signals']))}–{_int(max(s['signals']))} señales y "
            f"{min(s['planets'])}–{max(s['planets'])} planetas por partición; {s['shared_stars_between_parts']} estrellas compartidas entre "
            f"entrenamiento, validación y prueba en las 10 particiones. Frente al reparto de todos los pasos anteriores (semilla 0), "
            f"cada parte de prueba nueva toma como mucho el {100 * s['largest_share_from_one_old_fold']:.0f} % de sus señales de una misma "
            f"parte antigua, y el {100 * s['signals_in_same_fold_number']:.0f} % de las señales cae en la partición del mismo número "
            f"(lo esperable al azar es el 10 %). ExoMiner no tiene puntuación publicada para {res['exominer_missing_scores']['signals']} "
            f"señales ({res['exominer_missing_scores']['planets']} planetas); cuentan como 0, como en el paso 2.")


FIGURE_TEXT = {
    "es": {"published": "ExoMiner\npublicado\n{v}", "exominer": "ExoMiner,\nconjunto de 10", "m4_single": "Hito 4, un modelo",
           "m4_ens": "Hito 4, conjunto de 10", "single": "Un modelo", "distilled": "Destilado", "ensemble": "Conjunto de 10",
           "xlabel": "Parámetros al predecir (escala logarítmica)", "recall_paper": "R@P0,99 (como lo cuenta ExoMiner)", "auc_pr": "AUC-PR",
           "title_recall": "Planetas recuperados con precisión 0,99 (media de 10 particiones, ± 95 %)",
           "title_auc": "Área bajo la curva precisión–recall",
           "note": "Naranja: receta final en el reparto nuevo. Azul hueco: hito 4 en el reparto antiguo. Estrella: ExoMiner medido "
                   "con sus puntuaciones publicadas en el reparto nuevo."},
    "en": {"published": "ExoMiner\npublished\n{v}", "exominer": "ExoMiner,\nensemble of 10", "m4_single": "Milestone 4, one model",
           "m4_ens": "Milestone 4, ensemble of 10", "single": "One model", "distilled": "Distilled", "ensemble": "Ensemble of 10",
           "xlabel": "Parameters at prediction (log scale)", "recall_paper": "R@P0.99 (as ExoMiner counts it)", "auc_pr": "AUC-PR",
           "title_recall": "Planets recovered at 0.99 precision (mean of 10 folds, ± 95 %)",
           "title_auc": "Area under the precision-recall curve",
           "note": "Orange: final recipe on the new folds. Hollow blue: milestone 4 on the old folds. Star: ExoMiner measured with "
                   "its published scores on the new folds."},
}


def figure(res: dict, path, lang: str = "es") -> None:
    """Quality against cost: R@P0.99 (as ExoMiner counts it) and AUC-PR against parameters at prediction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1a1a19", "#52514e", "#d9d8d2"
    ours, before = "#eb6834", "#2a78d6"
    rows, old = res["rows"], res.get("step2", {})
    tx = FIGURE_TEXT[lang]

    def num(text: str) -> str:
        return h4._es(text) if lang == "es" else text

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, k, d, (lo, hi) in ((axes[0], "recall_paper", 3, (0.84, 1.0)), (axes[1], "auc_pr", 4, (0.986, 0.999))):
        ax.axhline(PUBLISHED[k], color=muted, lw=1.2, ls="--", zorder=1)
        ax.text(1.01, PUBLISHED[k], tx["published"].format(v=num(f"{PUBLISHED[k]:.{d}f}")), transform=ax.get_yaxis_transform(),
                fontsize=8, color=muted, va="center")

        def point(x, m, marker, color, label, hollow=False, dx=8, dy=-14, ha="left"):
            ax.errorbar([x], [m[k]["mean"]], yerr=[m[k]["ci95"]], fmt=marker, ms=9, color=color, mfc="white" if hollow else color,
                        mew=2, capsize=3, lw=1.5, zorder=3)
            ax.annotate(label, (x, m[k]["mean"]), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=ink, ha=ha, va="center",
                        bbox={"fc": "white", "ec": "none", "pad": 0.5})

        if "exominer_direct" in rows:
            point(EXOMINER_ENSEMBLE_PARAMETERS, rows["exominer_direct"]["m"], "*", ink, tx["exominer"], dx=-12, dy=0, ha="right")
        if "base_single" in old:
            point(38_441, old["base_single"], "o", before, tx["m4_single"], hollow=True, dy=0, dx=10)
            point(384_410, old["base_ens10"], "D", before, tx["m4_ens"], hollow=True, dy=0, dx=10)
        for key, marker, dx, dy, ha in (("single", "o", -10, 0, "right"), ("distilled", "s", 10, -14, "left"), ("ensemble", "D", -10, 0, "right")):
            if key in rows:
                x = rows[key]["parameters"] * (1.25 if key == "distilled" else 1.0)
                point(x, rows[key]["m"], marker, ours, tx[key], dx=dx, dy=dy, ha=ha)
        if "single" in rows and "ensemble" in rows:
            ax.plot([rows["single"]["parameters"], rows["ensemble"]["parameters"]], [rows["single"]["m"][k]["mean"], rows["ensemble"]["m"][k]["mean"]],
                    color=ours, lw=1, ls=":", zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(8e3, 6e6)
        ax.set_ylim(lo, hi)
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.set_xlabel(tx["xlabel"])
        ax.set_ylabel(tx[k])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_title(tx["title_recall"], fontsize=10)
    axes[1].set_title(tx["title_auc"], fontsize=10)
    fig.text(0.01, 0.005, tx["note"], fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(args) -> None:
    ctx = h5.load_context()
    res = compute(ctx)
    blocks = {"tabla": text_table(res), "veredicto": text_verdict(res), "reparto": text_split(res)}
    for text in blocks.values():
        print(text, "\n")
    if args.quick:
        return
    slim = json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (DOCS / "hito5_veredicto.json").write_text(json.dumps(slim, indent=1, ensure_ascii=False) + "\n")
    h5.update_doc(DOC, blocks)
    figure(res, DOCS / "hito5_calidad_coste.png")
    figure(res, DOCS / "quality_cost.png", lang="en")
    print("wrote docs/hito5_veredicto.md, docs/hito5_veredicto.json, docs/hito5_calidad_coste.png and docs/quality_cost.png (English)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--shard", default="0/1", help="i/n: do only the tasks whose index is i modulo n")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="run: 1 fold, 1 seed, 1 epoch; report: print without writing files")
    args = ap.parse_args()
    if args.command == "run":
        run(args)
    else:
        report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
