"""Milestone 5, step 1: diagnosis of the gap between the pocket model and ExoMiner on
Kepler DR25. No light curve is downloaded; everything comes from the dataset of
milestone 3, the predictions of milestone 4 and small public tables.

    uv run python scripts/hito5_diagnostico.py ablate          # input ablations (resumable; hours)
    uv run python scripts/hito5_diagnostico.py ablate --quick  # smoke test: 1 fold, 1 seed, 1 epoch
    uv run python scripts/hito5_diagnostico.py occlude         # same inputs zeroed on the saved models (minutes)
    uv run python scripts/hito5_diagnostico.py report          # docs/hito5_diagnostico.md, JSON, CSV and figure

``ablate`` trains the full pocket model of milestone 4 (same recipe, folds and seeds)
with groups of inputs replaced by zeros (``cazaplanetas.diagnosis.ABLATIONS``). Each
(ablation, fold, seed) writes its predictions under data/hito5/ablation/, so it can be
stopped and resumed, and several processes can share the work with ``--shard i/n``.

``report`` reads those predictions and the ones of milestone 4 and writes:

* ExoMiner's published scores (Tables 3 and 13 of Valizadegan et al. 2022, from the
  arXiv source of the paper) measured with our signals, labels and metrics;
* the objects that break precision 0.99 and the planets that are lost, by likely cause;
* how much of the spread comes from the seed and how much from the fold;
* the ablation table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time

from cazaplanetas.env import DATA_DIR, REPO_ROOT, use_local_cache

use_local_cache()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hito4_dr25_cv as h4  # noqa: E402  (same folder: dataset, folds and recipe of milestone 4)
from cazaplanetas import diagnosis, dr25, features, labels, models, splits, training  # noqa: E402
from cazaplanetas import exominer_tables as ex  # noqa: E402

DOCS = REPO_ROOT / "docs"
RUN_DIR = DATA_DIR / "hito5"
ABLATION_SEEDS = 3
MODEL = "pocket_full"


def ablation_path(name: str, fold: int, seed: int, quick: bool = False):
    return RUN_DIR / ("ablation_quick" if quick else "ablation") / f"{name}_fold{fold}_seed{seed}.npz"


# --- Ablations ---------------------------------------------------------------------


def ablate(args) -> None:
    device = training.pick_device(args.device)
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    print(f"device: {device}  torch {torch.__version__}  load {os.getloadavg()[0]:.1f}  shard {shard}/{n_shards}  pid {os.getpid()}", flush=True)
    data = h4.load_dataset()
    y = data["label"]
    cfg = h4.CONFIGS[MODEL]
    if args.quick:
        cfg = training.TrainConfig(**{**cfg.to_dict(), "epochs": 1})
    names = [n for n in args.configs if n != "full"]  # "full" is milestone 4's pocket_full, already trained
    # Ablation by ablation, so that whatever is finished is a complete row of the table.
    tasks = [(fold, name) for name in names for fold in range(args.folds)]
    if args.reverse:  # from the last task to the first, to meet processes that started from the front
        tasks = tasks[::-1]
    tasks = [t for i, t in enumerate(tasks) if i % n_shards == shard]
    base = None
    last_fold = None
    for fold, name in tasks:
        todo = [s for s in range(args.seeds) if not ablation_path(name, fold, s, args.quick).exists()]
        if not todo:
            continue
        if fold != last_fold:
            parts = h4.fold_parts(data["kepid"], fold)
            scalars = features.Standardizer().fit(data["scalars"][parts["train"]]).transform(data["scalars"])
            if base is None:
                base = h4.device_tensors(MODEL, data, scalars, device)
            base.inputs["scalars"] = torch.from_numpy(scalars).to(device)
            last_fold = fold
        ablation = diagnosis.ABLATIONS[name]
        dev = training.Tensors(diagnosis.mask_inputs(base.inputs, ablation), base.label)
        sub = {k: dev.subset(v) for k, v in parts.items()}
        del dev
        for seed in todo:
            path = ablation_path(name, fold, seed, args.quick)
            if path.exists():  # done meanwhile by another process
                continue
            t = time.perf_counter()
            training.seed_everything(seed)
            model = models.PocketExo(data["scalars"].shape[1])
            fit = training.train(model, sub["train"], sub["val"], cfg, seed=seed, log_every=0)
            val, test = training.predict(model, sub["val"]), training.predict(model, sub["test"])
            thr = training.choose_threshold(y[parts["val"]], val)
            m = training.classification_metrics(y[parts["test"]], test, thr)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part.npz")
            np.savez(tmp, val=val, test=test, threshold=thr, val_index=parts["val"], test_index=parts["test"],
                     train_seconds=fit["train_seconds"], selected_epoch=fit["selected_epoch"],
                     parameters=models.count_parameters(model))
            tmp.replace(path)
            print(f"[{name}] fold {fold} seed {seed}: AUC-PR {m['auc_pr']:.4f}  R@P.99 {m['recall_at_p099']:.3f}  "
                  f"({fit['train_seconds']:.0f} s, epoch {fit['selected_epoch']}, load {os.getloadavg()[0]:.0f}, "
                  f"{time.perf_counter() - t:.0f} s total)", flush=True)
        del sub
        if device.type == "mps":
            torch.mps.empty_cache()


# --- Occlusion (no retraining) -----------------------------------------------------


def occlude(args) -> None:
    """Score the test part of every fold with the saved pocket models of milestone 4 while
    one group of inputs is replaced by zeros (the models are not retrained). It tells
    which inputs the trained model leans on; data/hito5/occlusion.npz."""
    device = training.pick_device(args.device)
    data = h4.load_dataset()
    n = len(data["label"])
    out = {name: np.full((args.occlusion_seeds, n), np.nan, dtype=np.float32) for name in diagnosis.ABLATIONS}
    worst = 0.0
    base = None
    for fold in range(args.folds):
        parts = h4.fold_parts(data["kepid"], fold)
        scalars = features.Standardizer().fit(data["scalars"][parts["train"]]).transform(data["scalars"])
        if base is None:
            base = h4.device_tensors(MODEL, data, scalars, device)
        base.inputs["scalars"] = torch.from_numpy(scalars).to(device)
        test = base.subset(parts["test"])
        for seed in range(args.occlusion_seeds):
            model = models.PocketExo(data["scalars"].shape[1])
            model.load_state_dict(torch.load(h4.pred_path(MODEL, fold, seed, False).with_suffix(".pt"), map_location="cpu"))
            model.to(device)
            for name, ablation in diagnosis.ABLATIONS.items():
                out[name][seed, parts["test"]] = training.predict(model, training.Tensors(diagnosis.mask_inputs(test.inputs, ablation), test.label))
            saved = np.load(h4.pred_path(MODEL, fold, seed, False))["test"]
            worst = max(worst, float(np.abs(out["full"][seed, parts["test"]] - saved).max()))
        print(f"fold {fold}: done (largest difference with the predictions saved in milestone 4 so far: {worst:.2e})", flush=True)
    if worst > 1e-3:
        raise SystemExit(f"the saved weights do not reproduce the saved predictions (difference {worst:.3f})")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUN_DIR / "occlusion.npz", **out)
    print(f"wrote {RUN_DIR / 'occlusion.npz'}")


# --- Report: data ------------------------------------------------------------------

TARGETS = (0.909, 0.939)  # intermediate step and ExoMiner's published mean
PUBLISHED_RECALL_AT_P = {  # Valizadegan et al. 2022, Table 5, first row of each model (whole dataset)
    "astronet": {0.99: 0.001, 0.975: 0.333, 0.95: 0.643},
    "exonet": {0.99: 0.506, 0.975: 0.619, 0.95: 0.803},
    "exominer": {0.99: 0.936, 0.975: 0.966, 0.95: 0.986},
    "exominer_tce": {0.99: 0.949, 0.975: 0.960, 0.95: 0.990},
}
PUBLISHED_FOLD_MEAN = {0.99: (0.939, 0.030), 0.975: (0.964, 0.009), 0.95: (0.988, 0.000)}  # ExoMiner, mean ± 99 % interval
EXOMINER_SELF_VALIDATED = ("Valizadegan et al. 2022", "Valizadegan et al. 2023")


def load_context() -> dict:
    """The labelled DR25 signals with everything the report needs (no views): labels,
    folds, a few TCE columns, catalogue notes and ExoMiner's published scores."""
    labels.fetch_all()
    labels.fetch_cumulative_details()
    labels.fetch_kepler_planets()
    ex.fetch_tables()
    ex.fetch_folds()
    tce = dr25.read_csv(dr25.fetch_tce_table())
    lab = labels.build(tce)
    rows = np.flatnonzero(lab["label"] >= 0)
    kepid, plnt = lab["kepid"][rows], lab["tce_plnt_num"][rows]
    keys = list(zip(kepid.tolist(), plnt.tolist()))
    ctx = {
        "kepid": kepid, "tce_plnt_num": plnt, "label": lab["label"][rows], "label_class": lab["label_class"][rows],
        "kepoi_name": lab["kepoi_name"][rows], "fold": splits.star_folds(kepid, h4.K_FOLDS, seed=h4.FOLD_SEED),
        "period": tce["tce_period"][rows], "mes": tce["tce_max_mult_ev"][rows], "depth": tce["tce_depth"][rows],
        "prad": tce["tce_prad"][rows], "n_transits": tce["tce_num_transits"][rows],
        "same_period_sibling": diagnosis.same_period_siblings(tce["kepid"], tce["tce_period"])[rows],
        "weaker_of_pair": diagnosis.same_period_siblings(tce["kepid"], tce["tce_period"], strength=tce["tce_max_mult_ev"])[rows],
        "counts": labels.counts(lab),
    }
    cum = dr25.read_csv(labels.CUMULATIVE_DETAILS_CSV)
    ci = {str(n): i for i, n in enumerate(cum["kepoi_name"])}
    pick = np.array([ci.get(str(n), -1) for n in ctx["kepoi_name"]])

    def from_cum(col, missing):
        return np.array([cum[col][i] if i >= 0 else missing for i in pick])

    ctx["kepler_name"] = from_cum("kepler_name", "")
    ctx["koi_disposition"] = from_cum("koi_disposition", "")
    ctx["koi_period"] = from_cum("koi_period", np.nan)
    ctx["kepmag"] = from_cum("koi_kepmag", np.nan)
    ps = dr25.read_csv(labels.KEPLER_PLANETS_CSV)
    pi = {str(n): i for i, n in enumerate(ps["pl_name"])}
    prow = [pi.get(str(n), -1) for n in ctx["kepler_name"]]
    ctx["disc_ref"] = np.array([re.sub(r"<[^>]+>", "", str(ps["disc_refname"][i])).replace("&ntilde;", "ñ").replace("&eacute;", "é").strip() if i >= 0 else "" for i in prow])
    ctx["disc_pubdate"] = np.array([str(ps["disc_pubdate"][i]) if i >= 0 else "" for i in prow])
    ctx["controversial"] = np.array([bool(ps["pl_controv_flag"][i] == 1) if i >= 0 else False for i in prow])
    fpwg = labels.read_fpwg_details()

    def flag(name, col):
        v = fpwg.get(str(name), {}).get(col, "")
        return int(v) if v in ("0", "1") else None

    ctx["fpwg_eb"] = [flag(n, "fpwg_disp_eb") for n in ctx["kepoi_name"]]
    ctx["fpwg_offset"] = [flag(n, "fpwg_disp_offst") for n in ctx["kepoi_name"]]
    ctx["fpwg_perep"] = [flag(n, "fpwg_disp_perep") for n in ctx["kepoi_name"]]
    rv = labels.read_robovetter_flags()
    ctx["rv_flags"] = np.array([rv.get(k, ("0000", ""))[0] for k in keys])
    ctx["rv_comment"] = np.array([rv.get(k, ("0000", ""))[1] for k in keys])
    ctx["rv_score"] = lab["robovetter_score"][rows]
    # ExoMiner: its labels and cross-validation scores (Table 3), its scores of the KOIs it left
    # unlabelled (Table 13) and its folds.
    t3, t13 = ex.read_labelled(), ex.read_unlabelled()
    ctx["exominer_label"] = ex.align(kepid, plnt, t3, "label", missing=-1).astype(int)
    s3, s13 = ex.align(kepid, plnt, t3, "exominer"), ex.align(kepid, plnt, t13, "exominer")
    ctx["exominer_score"] = np.where(np.isfinite(s3), s3, s13)
    ctx["in_table3"], ctx["in_table13"] = np.isfinite(s3), np.isfinite(s13)
    ctx["exominer_tables"] = (t3, t13)
    return ctx


def load_test_predictions(directory, prefix: str, n: int, folds: int = h4.K_FOLDS) -> np.ndarray:
    """(seeds, signals) test predictions of ``prefix`` (NaN where a run is missing): every
    signal is scored by the model of the fold in which it was in the test part."""
    files = sorted(directory.glob(f"{prefix}_fold*_seed*.npz"))
    seeds = sorted({int(re.search(r"seed(\d+)", f.name).group(1)) for f in files})
    out = np.full((len(seeds), n), np.nan, dtype=np.float64)
    for f in files:
        m = re.search(r"_fold(\d+)_seed(\d+)", f.name)
        if int(m.group(1)) >= folds:
            continue
        z = np.load(f)
        out[seeds.index(int(m.group(2))), z["test_index"]] = z["test"]
    return out


# --- Report: numbers ---------------------------------------------------------------

EXACT, PAPER = "exact", "paper"  # precision >= 0.99, and precision rounded to two decimals >= 0.99


def recall_at(y, p, precision=0.99, how=EXACT) -> float:
    if how == PAPER:
        return diagnosis.recall_at_rounded_precision(y, p, precision)
    return training.recall_at_precision(y, p, precision)


def _interval(values, level: float) -> float:
    from scipy.stats import t

    a = np.asarray(values, dtype=float)
    return float(t.ppf(0.5 + level / 2, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")


def _stat(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "ci95": _interval(a, 0.95), "ci99": _interval(a, 0.99), "folds": [round(float(v), 4) for v in a]}


def measure(y, fold, score, sel=None, n_folds: int = h4.K_FOLDS) -> dict:
    """Metrics of ``score`` on the signals ``sel``: per fold (mean and intervals) and with
    all the signals at once. ``score`` is (signals,) or (seeds, signals); with several
    seeds the per-fold value is the mean over the seeds (a single model)."""
    sel = np.ones(len(y), bool) if sel is None else sel
    scores = np.atleast_2d(score)
    out = {"n": int(sel.sum()), "n_pos": int(y[sel].sum()), "seeds": int(scores.shape[0])}
    for how in (EXACT, PAPER):
        per_fold = [np.mean([recall_at(y[sel & (fold == f)], s[sel & (fold == f)], how=how) for s in scores]) for f in range(n_folds)]
        out[f"recall_{how}"] = _stat(per_fold)
        out[f"recall_{how}_pooled"] = float(np.mean([recall_at(y[sel], s[sel], how=how) for s in scores]))
    per_fold = [np.mean([training.average_precision(y[sel & (fold == f)], s[sel & (fold == f)]) for s in scores]) for f in range(n_folds)]
    out["auc_pr"] = _stat(per_fold)
    out["auc_pr_pooled"] = float(np.mean([training.average_precision(y[sel], s[sel]) for s in scores]))
    return out


def direct_comparison(ctx: dict, pocket: np.ndarray) -> dict:
    """ExoMiner's published scores and ours under the same signals, labels, folds and metric."""
    y, fold = ctx["label"], ctx["fold"]
    t3, _ = ctx["exominer_tables"]
    their_fold = ex.read_folds()
    f3 = np.array([their_fold.get((int(k), int(p)), -1) for k, p in zip(t3["kepid"], t3["tce_plnt_num"])])
    ens = pocket.mean(axis=0)
    ey, es = ctx["exominer_label"], ctx["exominer_score"]
    theirs = ctx["in_table3"] & (ey >= 0)  # their labelled set that is also labelled here
    have = np.isfinite(es)
    self_validated = np.isin(ctx["disc_ref"], EXOMINER_SELF_VALIDATED) & (y == 1)
    rows = {
        "exominer_own": {"who": "ExoMiner", "labels": "suyas (2020)", "folds": "suyas", **measure(t3["label"], f3, t3["exominer"])},
        "exominer_own_ourfolds": {"who": "ExoMiner", "labels": "suyas (2020)", "folds": "nuestras", **measure(ey, fold, es, theirs)},
        "exominer_ours": {"who": "ExoMiner", "labels": "nuestras (2026)", "folds": "nuestras", **measure(y, fold, es, have)},
        "exominer_ours_noself": {"who": "ExoMiner", "labels": "nuestras, sin los planetas que validó el propio ExoMiner", "folds": "nuestras",
                                 **measure(y, fold, es, have & ~self_validated)},
        "pocket_ens_ours": {"who": "Bolsillo, media de 10", "labels": "nuestras (2026)", "folds": "nuestras", **measure(y, fold, ens)},
        "pocket_ens_noself": {"who": "Bolsillo, media de 10", "labels": "nuestras, sin los planetas que validó el propio ExoMiner", "folds": "nuestras",
                              **measure(y, fold, ens, ~self_validated)},
        "pocket_ens_theirs": {"who": "Bolsillo, media de 10", "labels": "suyas (2020)", "folds": "nuestras", **measure(ey, fold, ens, theirs)},
        "pocket_single_ours": {"who": "Bolsillo, un modelo", "labels": "nuestras (2026)", "folds": "nuestras", **measure(y, fold, pocket)},
        "pocket_single_theirs": {"who": "Bolsillo, un modelo", "labels": "suyas (2020)", "folds": "nuestras", **measure(ey, fold, pocket, theirs)},
    }
    # Which reading of "Recall@p" reproduces the paper's Table 5 from its own scores
    definition = []
    for model, published in PUBLISHED_RECALL_AT_P.items():
        for p, value in published.items():
            definition.append({"model": model, "precision": p, "published": value,
                               "exact": training.recall_at_precision(t3["label"], t3[model], p),
                               "rounded": diagnosis.recall_at_rounded_precision(t3["label"], t3[model], p)})
    fold_check = {}
    for p, (mean, half) in PUBLISHED_FOLD_MEAN.items():
        for how in (EXACT, PAPER):
            vals = [recall_at(t3["label"][f3 == f], t3["exominer"][f3 == f], p, how) for f in range(10)]
            fold_check[f"{p}_{how}"] = {"published_mean": mean, "published_ci99": half, **_stat(vals)}
    m05 = training.classification_metrics(t3["label"], t3["exominer"], 0.5)
    # Label parity
    ours_all = {(int(k), int(p)): c for k, p, c in zip(ctx["kepid"], ctx["tce_plnt_num"], ctx["label_class"])}
    same = sum(ours_all.get((int(k), int(p))) == c for k, p, c in zip(t3["kepid"], t3["tce_plnt_num"], t3["label_class"]))
    new_pc = ctx["in_table13"] & (y == 1)
    refs = {}
    for r in ctx["disc_ref"][new_pc]:
        refs[r or "(sin referencia en la tabla de planetas)"] = refs.get(r or "(sin referencia en la tabla de planetas)", 0) + 1
    return {
        "rows": rows,
        "definition": definition,
        "definition_matches": {"exact": sum(abs(round(d["exact"], 3) - d["published"]) < 5e-4 for d in definition),
                               "rounded": sum(abs(round(d["rounded"], 3) - d["published"]) < 5e-4 for d in definition), "of": len(definition)},
        "fold_check": fold_check,
        "exominer_at_0.5": {"precision": m05["precision"], "recall": m05["recall"], "false_positives": m05["fp"], "missed_planets": m05["fn"],
                            "auc_pr_exact": training.average_precision(t3["label"], t3["exominer"]),
                            "auc_pr_keras_1000_thresholds": diagnosis.interpolated_pr_auc(t3["label"], t3["exominer"]),
                            "auc_roc": training.roc_auc(t3["label"], t3["exominer"])},
        "label_parity": {"their_labelled": int(len(t3["label"])), "same_class_here": int(same),
                         "our_new_planets": int(new_pc.sum()), "our_new_planets_by_reference": dict(sorted(refs.items(), key=lambda kv: -kv[1])),
                         "our_other_extra": int((~ctx["in_table3"] & ~new_pc).sum()),
                         "self_validated_in_our_positives": int(self_validated.sum()),
                         "new_planets_exominer_above_0.99": int((es[new_pc] > 0.99).sum()),
                         "new_planets_pocket_above_0.99": int((ens[new_pc] > 0.99).sum())},
    }


def _row(ctx: dict, i: int) -> dict:
    return {k: (ctx[k][i].item() if hasattr(ctx[k][i], "item") else ctx[k][i]) for k in (
        "kepid", "tce_plnt_num", "label_class", "kepoi_name", "kepler_name", "koi_disposition", "koi_period", "kepmag", "period", "mes",
        "depth", "prad", "n_transits", "same_period_sibling", "fpwg_eb", "fpwg_offset", "fpwg_perep", "rv_flags", "rv_comment",
        "controversial", "disc_ref", "disc_pubdate", "fold")} | {"tce_period": float(ctx["period"][i])}


def error_analysis(ctx: dict, pocket: np.ndarray) -> dict:
    """The objects that decide R@P0.99 for the mean of the 10 seeds, with all the signals
    at once: negatives ranked too high and planets ranked too low, by likely cause."""
    y, fold = ctx["label"], ctx["fold"]
    ens = pocket.mean(axis=0)
    n_pos = int(y.sum())
    op = diagnosis.operating_point(y, ens)
    planets_desc = np.sort(ens[y == 1])[::-1]
    out = {"operating_point": op, "targets": {}}
    max_fixes = 60
    curve = diagnosis.best_recall_with_fixes(y, ens, max_fixes)
    curve_paper = diagnosis.best_recall_with_fixes(y, ens, max_fixes, precision=0.985)
    fold_curves = [diagnosis.best_recall_with_fixes(y[fold == f], ens[fold == f], 25) for f in range(h4.K_FOLDS)]
    single_curves = [diagnosis.best_recall_with_fixes(y, s, 150) for s in pocket]
    single_fold_curves = [[diagnosis.best_recall_with_fixes(y[fold == f], s[fold == f], 40) for f in range(h4.K_FOLDS)] for s in pocket]

    def first(c, target):
        hit = np.flatnonzero(np.asarray(c) >= target - 1e-12)
        return int(hit[0]) if len(hit) else -1

    for target in TARGETS:
        need = int(np.ceil(target * n_pos))
        cut = float(planets_desc[need - 1])
        total, alloc = diagnosis.fixes_for_mean(fold_curves, target)
        single_fold = [diagnosis.fixes_for_mean(c, target)[0] for c in single_fold_curves]
        out["targets"][str(target)] = {
            "planets_needed": need, "cut_score": cut, "negatives_above_cut": int((ens[y == 0] >= cut).sum()),
            "negatives_allowed": int(np.floor(need * 0.01 / 0.99 + 1e-9)),
            "fixes_pooled_ensemble": first(curve, target), "fixes_pooled_ensemble_paper_metric": first(curve_paper, target),
            "fixes_fold_mean_ensemble": total, "fixes_fold_mean_ensemble_per_fold": alloc,
            "fixes_pooled_single": [first(c, target) for c in single_curves],
            "fixes_fold_mean_single": single_fold,
        }
    out["recall_by_fixes_pooled_ensemble"] = [round(float(v), 4) for v in curve[:31]]
    cut = out["targets"][str(TARGETS[-1])]["cut_score"]
    # Negatives: everything above the cut of the highest target, plus the next ones down to 0.5
    neg = np.flatnonzero((y == 0) & (ens >= 0.5))
    neg = neg[np.argsort(-ens[neg])]
    negatives = []
    for rank, i in enumerate(neg, 1):
        r = _row(ctx, i)
        cause = diagnosis.negative_cause(r)
        negatives.append({**r, "rank": rank, "score": float(ens[i]), "seeds_above_threshold": int((pocket[:, i] >= op["threshold"]).sum()),
                          "seed_sd": float(pocket[:, i].std()), "exominer_score": float(ctx["exominer_score"][i]),
                          "above_cut": bool(ens[i] >= cut), "above_cut_909": bool(ens[i] >= out["targets"][str(TARGETS[0])]["cut_score"]),
                          "cause": cause, "label_same_as_exominer": bool(ctx["exominer_label"][i] == 0)})
    out["negatives"] = negatives
    by_cause = {}
    for c in diagnosis.NEGATIVE_CAUSES:
        members = [n for n in negatives if n["cause"] == c and n["above_cut"]]
        if not members:
            continue
        fixed = ens.copy()
        fixed[[int(np.flatnonzero((ctx["kepid"] == n["kepid"]) & (ctx["tce_plnt_num"] == n["tce_plnt_num"]))[0]) for n in negatives if n["cause"] == c]] = -1.0
        by_cause[c] = {"above_cut": len(members), "above_cut_909": sum(n["above_cut_909"] for n in members),
                       "exominer_also_high": sum(n["exominer_score"] >= 0.5 for n in members),
                       "exominer_below_0.1": sum(n["exominer_score"] < 0.1 for n in members),
                       "afp": sum(n["label_class"] == "AFP" for n in members),
                       "recall_if_fixed_pooled": training.recall_at_precision(y, fixed),
                       "recall_if_fixed_fold_mean": float(np.mean([training.recall_at_precision(y[fold == f], fixed[fold == f]) for f in range(h4.K_FOLDS)])),
                       "recall_if_fixed_fold_mean_paper": float(np.mean([diagnosis.recall_at_rounded_precision(y[fold == f], fixed[fold == f]) for f in range(h4.K_FOLDS)]))}
    out["negatives_by_cause"] = by_cause
    # Planets: lost at today's operating point; the ones below the cut stay lost even at the target
    pos = np.flatnonzero((y == 1) & (ens < op["threshold"]))
    pos = pos[np.argsort(ens[pos])]
    planets = []
    for rank, i in enumerate(pos, 1):
        r = _row(ctx, i)
        planets.append({**r, "rank": rank, "score": float(ens[i]), "seeds_above_threshold": int((pocket[:, i] >= op["threshold"]).sum()),
                        "seed_sd": float(pocket[:, i].std()), "exominer_score": float(ctx["exominer_score"][i]),
                        "below_cut": bool(ens[i] < cut), "cause": diagnosis.planet_cause(r),
                        "confirmed_after_exominer_download": bool(ctx["in_table13"][i])})
    out["planets"] = planets
    pc = {}
    all_pos = np.flatnonzero(y == 1)
    all_causes = [diagnosis.planet_cause(_row(ctx, i)) for i in all_pos]
    for c in diagnosis.PLANET_CAUSES:
        low = [p for p in planets if p["cause"] == c and p["below_cut"]]
        mid = [p for p in planets if p["cause"] == c and not p["below_cut"]]
        total = sum(1 for a in all_causes if a == c)
        if total:
            pc[c] = {"below_cut": len(low), "between_cut_and_threshold": len(mid), "all_planets": total,
                     "exominer_also_low": sum(p["exominer_score"] < 0.5 for p in low),
                     "confirmed_after_2020": sum(p["confirmed_after_exominer_download"] for p in low)}
    out["planets_by_cause"] = pc
    out["planets_below_cut"] = sum(p["below_cut"] for p in planets)
    out["planets_lost_now"] = len(planets)
    # ExoMiner's own top negatives with our labels, for reference
    es = ctx["exominer_score"]
    have = np.isfinite(es)
    out["exominer_operating_point"] = diagnosis.operating_point(y[have], es[have])
    out["shared_negatives_above_cut"] = sum(n["above_cut"] and n["exominer_score"] >= 0.5 for n in negatives)
    # What a rule with no labels would do: score 0 for the weaker signal of a same-star, same-period pair
    weak = ctx["weaker_of_pair"]
    ruled = np.where(weak, 0.0, ens)
    ruled_single = np.where(weak[None, :], 0.0, pocket)
    out["sibling_rule"] = {
        "signals_zeroed": int(weak.sum()), "planets_zeroed": int((weak & (y == 1)).sum()),
        "planets_zeroed_above_threshold": int((weak & (y == 1) & (ens >= op["threshold"])).sum()),
        "secondary_tce_negatives_above_cut_zeroed": sum(1 for n in negatives if n["cause"] == "secondary_tce" and n["above_cut"]
                                                         and weak[(ctx["kepid"] == n["kepid"]) & (ctx["tce_plnt_num"] == n["tce_plnt_num"])][0]),
        "ensemble": measure(y, fold, ruled), "single": measure(y, fold, ruled_single),
        "fixes_to_939_pooled_ensemble": diagnosis.fixes_needed(y, ruled, TARGETS[-1]),
    }
    out["sibling_feature"] = {
        "negatives_with_sibling": int(((y == 0) & ctx["same_period_sibling"]).sum()),
        "planets_with_sibling": int(((y == 1) & ctx["same_period_sibling"]).sum()),
        "is_sec_tce": int(np.array(["IS_SEC_TCE" in c for c in ctx["rv_comment"]]).sum()),
        "is_sec_tce_with_sibling": int((np.array(["IS_SEC_TCE" in c for c in ctx["rv_comment"]]) & ctx["same_period_sibling"]).sum()),
    }
    return out


def spread(ctx: dict, pocket: np.ndarray) -> dict:
    """How much of the spread of R@P0.99 and AUC-PR comes from the seed and how much from the fold."""
    y, fold = ctx["label"], ctx["fold"]
    out = {}
    tables = {
        "recall_exact": lambda yy, s: training.recall_at_precision(yy, s),
        "recall_paper": lambda yy, s: diagnosis.recall_at_rounded_precision(yy, s),
        "auc_pr": training.average_precision,
    }
    for name, fn in tables.items():
        table = np.array([[fn(y[fold == f], s[fold == f]) for s in pocket] for f in range(h4.K_FOLDS)])
        out[name] = {**diagnosis.variance_components(table), "table": np.round(table, 4).tolist(),
                     "ensemble_folds": [round(float(fn(y[fold == f], pocket.mean(0)[fold == f])), 4) for f in range(h4.K_FOLDS)]}
    out["planets_per_fold"] = [int(y[fold == f].sum()) for f in range(h4.K_FOLDS)]
    # Ensembles of k seeds: mean over the folds of R@P0.99, averaged over disjoint groups of seeds
    sizes = {}
    for k in (1, 2, 3, 5, 10):
        groups = [list(range(a, a + k)) for a in range(0, pocket.shape[0] - k + 1, k)]
        vals = [np.mean([training.recall_at_precision(y[fold == f], pocket[g].mean(0)[fold == f]) for f in range(h4.K_FOLDS)]) for g in groups]
        vals_paper = [np.mean([diagnosis.recall_at_rounded_precision(y[fold == f], pocket[g].mean(0)[fold == f]) for f in range(h4.K_FOLDS)]) for g in groups]
        sizes[str(k)] = {"groups": len(groups), "recall_exact": float(np.mean(vals)), "recall_paper": float(np.mean(vals_paper))}
    out["ensemble_size"] = sizes
    return out


def cheap_paths(ctx: dict, pocket: np.ndarray) -> dict:
    """The cheap fixes of section 6 combined, on the saved milestone 4 predictions (nothing
    is trained): the seed picked with the validation part, averaging k seeds and the rule
    of the weaker signal of a same-period pair. Then where the AUC-PR still lost lies."""
    y, fold = ctx["label"], ctx["fold"]
    n = len(y)
    rule = ctx["weaker_of_pair"]

    def ruled(s):
        s = np.array(s, dtype=float, copy=True)
        s[..., rule] = 0.0
        return s

    picked = np.zeros(n)  # per fold, the seed with the best validation AUC-PR (as in section 4)
    for f in range(h4.K_FOLDS):
        zs = [np.load(h4.pred_path(MODEL, f, s, False)) for s in range(pocket.shape[0])]
        z = zs[int(np.argmax([training.average_precision(y[z["val_index"]], z["val"]) for z in zs]))]
        picked[z["test_index"]] = z["test"]
    exo = np.where(np.isfinite(ctx["exominer_score"]), ctx["exominer_score"], 0.0)
    groups = {k: np.stack([pocket[i:i + k].mean(axis=0) for i in range(0, pocket.shape[0] - k + 1, k)]) for k in (2, 3, 5)}
    ens = pocket.mean(axis=0)
    rows = [
        ("exominer", "ExoMiner, media de 10 modelos (nuestras etiquetas y particiones)", 10, exo),
        ("single", "Bolsillo, un modelo (una semilla cualquiera)", 1, pocket),
        ("single_rule", "Bolsillo, un modelo + regla", 1, ruled(pocket)),
        ("picked", "Bolsillo, un modelo elegido con la validación", 1, picked),
        ("picked_rule", "Bolsillo, un modelo elegido con la validación + regla", 1, ruled(picked)),
        *[(f"mean{k}_rule", f"Bolsillo, media de {k} + regla", k, ruled(g)) for k, g in groups.items()],
        ("mean10", "Bolsillo, media de 10", 10, ens),
        ("mean10_rule", "Bolsillo, media de 10 + regla", 10, ruled(ens)),
    ]
    out = {"rows": {key: {"label": label, "models": k, **measure(y, fold, s)} for key, label, k, s in rows}}
    # AUC-PR lost, all signals at once, split by planet and by likely cause.
    planets = np.flatnonzero(y == 1)
    cause = np.array([diagnosis.planet_cause(_row(ctx, i)) for i in planets])
    losses = {"exominer": exo, "mean10_rule": ruled(ens), "picked_rule": ruled(picked)}
    losses = {k: diagnosis.ap_loss_per_positive(y, s)[planets] for k, s in losses.items()}
    out["loss_total"] = {k: float(v.sum()) for k, v in losses.items()}
    out["loss_worst"] = {str(m): {k: float(np.sort(v)[::-1][:m].sum()) for k, v in losses.items()} for m in (25, 50, 100)}
    out["loss_by_cause"] = {c: {"planets": int((cause == c).sum()), **{k: float(v[cause == c].sum()) for k, v in losses.items()}}
                            for c in diagnosis.PLANET_CAUSES}
    return out


def ablation_table(ctx: dict, full: np.ndarray, args) -> dict:
    """Each ablation against the full model with the same folds and seeds."""
    y, fold = ctx["label"], ctx["fold"]
    out = {}
    directory = RUN_DIR / "ablation"
    for name, ablation in diagnosis.ABLATIONS.items():
        if name == "full":
            preds, seeds = full[: args.seeds], list(range(args.seeds))
        else:
            preds = load_test_predictions(directory, name, len(y))
            seeds = list(range(preds.shape[0]))
        runs = int(np.isfinite(preds).all(axis=1).sum()) if len(preds) else 0
        done_folds = [f for f in range(h4.K_FOLDS) if len(preds) >= args.seeds and np.isfinite(preds[: args.seeds, fold == f]).all()]
        entry = {"label": ablation.label, "views_removed": list(ablation.view_names()), "scalars_removed": list(ablation.scalar_columns()),
                 "seeds": len(seeds), "complete": len(done_folds) == h4.K_FOLDS, "folds_done": len(done_folds), "complete_seeds": runs}
        if entry["complete"]:
            p = preds[: args.seeds]
            entry.update(measure(y, fold, p))
            entry["ensemble"] = measure(y, fold, p.mean(axis=0))
            ref = full[: args.seeds]
            for how in (EXACT, PAPER):
                d = [np.mean([recall_at(y[fold == f], s[fold == f], how=how) for s in p]) - np.mean([recall_at(y[fold == f], s[fold == f], how=how) for s in ref])
                     for f in range(h4.K_FOLDS)]
                entry[f"delta_recall_{how}"] = _stat(d)
            d = [np.mean([training.average_precision(y[fold == f], s[fold == f]) for s in p]) - np.mean([training.average_precision(y[fold == f], s[fold == f]) for s in ref])
                 for f in range(h4.K_FOLDS)]
            entry["delta_auc_pr"] = _stat(d)
        out[name] = entry
    return out


def occlusion_table(ctx: dict) -> dict | None:
    path = RUN_DIR / "occlusion.npz"
    if not path.exists():
        return None
    y, fold = ctx["label"], ctx["fold"]
    z = np.load(path)
    ref = z["full"]
    out = {}
    for name, ablation in diagnosis.ABLATIONS.items():
        p = z[name]
        entry = {"label": ablation.label, **measure(y, fold, p)}
        for how in (EXACT, PAPER):
            d = [np.mean([recall_at(y[fold == f], s[fold == f], how=how) for s in p]) - np.mean([recall_at(y[fold == f], s[fold == f], how=how) for s in ref])
                 for f in range(h4.K_FOLDS)]
            entry[f"delta_recall_{how}"] = _stat(d)
        out[name] = entry
    return out


def selection(ctx: dict) -> dict:
    """What the validation part says about each run of milestone 4: the epoch it chose and
    whether picking, in each fold, the seed with the best validation score helps."""
    y = ctx["label"]
    shape = (h4.K_FOLDS, 10)
    epoch, test_r, test_rp, test_ap, val_ap, val_r = (np.zeros(shape) for _ in range(6))
    for f in range(h4.K_FOLDS):
        for s in range(10):
            z = np.load(h4.pred_path(MODEL, f, s, False))
            yt, yv = y[z["test_index"]], y[z["val_index"]]
            epoch[f, s] = z["selected_epoch"]
            test_r[f, s], test_rp[f, s] = training.recall_at_precision(yt, z["test"]), diagnosis.recall_at_rounded_precision(yt, z["test"])
            test_ap[f, s] = training.average_precision(yt, z["test"])
            val_ap[f, s], val_r[f, s] = training.average_precision(yv, z["val"]), training.recall_at_precision(yv, z["val"])
    best_ap, best_r = val_ap.argmax(axis=1), val_r.argmax(axis=1)
    rows = np.arange(h4.K_FOLDS)
    early = epoch <= 8
    return {
        "selected_epoch_mean": float(epoch.mean()), "runs_with_epoch_8_or_less": int(early.sum()),
        "recall_exact_epoch_8_or_less": float(test_r[early].mean()), "recall_exact_epoch_9_or_more": float(test_r[~early].mean()),
        "all_seeds": {"recall_exact": _stat(test_r.mean(axis=1)), "recall_paper": _stat(test_rp.mean(axis=1)), "auc_pr": _stat(test_ap.mean(axis=1))},
        "best_val_auc_pr": {"recall_exact": _stat(test_r[rows, best_ap]), "recall_paper": _stat(test_rp[rows, best_ap]), "auc_pr": _stat(test_ap[rows, best_ap])},
        "best_val_recall": {"recall_exact": _stat(test_r[rows, best_r]), "recall_paper": _stat(test_rp[rows, best_r]), "auc_pr": _stat(test_ap[rows, best_r])},
        "within_fold_correlation_val_auc_pr_test_recall": float(np.corrcoef((val_ap - val_ap.mean(1, keepdims=True)).ravel(),
                                                                            (test_r - test_r.mean(1, keepdims=True)).ravel())[0, 1]),
    }


# --- Report: text ------------------------------------------------------------------

_es, _int = h4._es, h4._int


def _f(x: float, d: int = 3) -> str:
    return _es(f"{x:.{d}f}")


def _pm(s: dict, d: int = 3, level: str = "ci95") -> str:
    return _es(f"{s['mean']:.{d}f} ± {s[level]:.{d}f}")


def _signed(s: dict, d: int = 3) -> str:
    return _es(f"{s['mean']:+.{d}f} ± {s['ci95']:.{d}f}").replace("-", "−")


def text_direct(res: dict) -> str:
    d = res["direct"]
    lp = d["label_parity"]
    lines = [
        "**Qué lectura de «recall con precisión p» reproduce la tabla 5 de su artículo.** Con sus propias puntuaciones y etiquetas "
        "(tabla 3 del artículo, todas las señales a la vez):",
        "",
        "| Modelo de su tabla 3 | Precisión p | Publicado | Exacto (precisión ≥ p) | Precisión redondeada a dos decimales ≥ p |",
        "|---|---|---|---|---|",
    ]
    names = {"astronet": "Astronet (el suyo)", "exonet": "ExoNet", "exominer": "ExoMiner", "exominer_tce": "ExoMiner-TCE"}
    for e in d["definition"]:
        ok_e = abs(round(e["exact"], 3) - e["published"]) < 5e-4
        ok_r = abs(round(e["rounded"], 3) - e["published"]) < 5e-4
        lines.append(f"| {names[e['model']]} | {_es(str(e['precision']))} | {_f(e['published'])} | {_f(e['exact'])}{' ✓' if ok_e else ''} | "
                     f"{_f(e['rounded'])}{' ✓' if ok_r else ''} |")
    m = d["definition_matches"]
    fc = d["fold_check"]
    lines += [
        "",
        f"La lectura redondeada acierta {m['rounded']} de {m['of']} cifras al milésimo; la exacta, {m['exact']} de {m['of']} (las de p = 0,975, "
        "donde redondear a dos decimales no cambia nada, y la de Astronet con p = 0,99). Por particiones, con **sus** particiones: "
        f"redondeada {_pm(fc['0.99_paper'], level='ci99')} (publicado: 0,939 ± 0,030); exacta {_pm(fc['0.99_exact'], level='ci99')} "
        f"(media ± intervalo del 99 %). Con p = 0,975 y 0,95 pasa lo mismo: {_pm(fc['0.975_paper'], level='ci99')} y {_f(fc['0.95_paper']['mean'])} "
        "(publicado: 0,964 ± 0,009 y 0,988).",
        "",
        "**Las dos redes con las mismas señales, etiquetas, particiones y medida.** Media sobre las 10 particiones ± intervalo del **99 %** "
        "(el que usa ExoMiner), y entre paréntesis todas las señales a la vez:",
        "",
        "| Quién | Etiquetas | Particiones | Planetas | R@P0,99 exacto | R@P0,99 como lo mide ExoMiner (precisión ≥ 0,985) | AUC-PR |",
        "|---|---|---|---|---|---|---|",
        "| *ExoMiner, publicado (media de 10 modelos por partición)* | *suyas (2020)* | *suyas* | *2.291* | — | *0,939 ± 0,030 (0,936)* | *0,995 ± 0,002 (0,995)* |",
    ]
    for key in ("exominer_own", "exominer_own_ourfolds", "exominer_ours", "exominer_ours_noself", "pocket_ens_ours", "pocket_ens_noself",
                "pocket_ens_theirs", "pocket_single_ours", "pocket_single_theirs"):
        r = d["rows"][key]
        who = r["who"] + (", recalculado con sus puntuaciones" if key == "exominer_own" else "")
        lines.append(f"| {who} | {r['labels']} | {r['folds']} | {_int(r['n_pos'])} | {_pm(r['recall_exact'], level='ci99')} ({_f(r['recall_exact_pooled'])}) | "
                     f"{_pm(r['recall_paper'], level='ci99')} ({_f(r['recall_paper_pooled'])}) | {_pm(r['auc_pr'], level='ci99')} ({_f(r['auc_pr_pooled'])}) |")
    x = d["exominer_at_0.5"]
    refs = lp["our_new_planets_by_reference"]
    top = ", ".join(f"{k}: {v}" for k, v in list(refs.items())[:5])
    lines += [
        "",
        f"Comprobaciones con sus puntuaciones y etiquetas: AUC-PR {_f(x['auc_pr_exact'], 4)} (publicado 0,995; con el estimador de Keras de 1.000 umbrales "
        f"que usa su código, {_f(x['auc_pr_keras_1000_thresholds'], 4)}), precisión {_f(x['precision'])} y recall {_f(x['recall'])} con umbral 0,5 "
        f"(publicado 0,968 y 0,974), {x['false_positives']} falsas alarmas y {x['missed_planets']} planetas perdidos (su sección 6.2 dice 74 y 60).",
        "",
        f"**Etiquetas.** De sus {_int(lp['their_labelled'])} señales con etiqueta, {_int(lp['same_class_here'])} tienen aquí la misma clase; las "
        f"{lp['their_labelled'] - lp['same_class_here']} restantes son planetas suyos que aquí quedan sin etiqueta (confirmados y a la vez falsos positivos "
        f"certificados). Nosotros añadimos {_int(lp['our_new_planets'])} planetas confirmados después de su descarga (por referencia de descubrimiento: {top}) "
        f"y {lp['our_other_extra']} negativos. {_int(lp['self_validated_in_our_positives'])} de nuestros positivos los validó el propio ExoMiner "
        f"(Valizadegan y otros 2022 y 2023), con su puntuación como criterio: de los {_int(lp['our_new_planets'])} nuevos, ExoMiner da más de 0,99 a "
        f"{lp['new_planets_exominer_above_0.99']} y el bolsillo a {lp['new_planets_pocket_above_0.99']}.",
    ]
    return "\n".join(lines)


def _kic(n: dict) -> str:
    return f"{n['kepid']}-{n['tce_plnt_num']}"


def _note(n: dict) -> str:
    c = n["rv_comment"].replace("---", ", ")
    return c[:60] + ("…" if len(c) > 60 else "")


def text_errors(res: dict) -> str:
    e = res["errors"]
    op = e["operating_point"]
    t909, t939 = e["targets"]["0.909"], e["targets"]["0.939"]
    lines = [
        f"Media de las 10 semillas, todas las señales a la vez ({_int(op['n_pos'])} planetas). Hoy el mejor punto con precisión ≥ 0,99 está en la "
        f"puntuación {_f(op['threshold'])}: {_int(op['tp'])} planetas y {op['fp']} negativos por encima, recall {_f(op['recall'])}.",
        "",
        "| Objetivo | Planetas que hacen falta | Puntuación del último | Negativos por encima | Negativos que caben | Objetos que hay que arreglar (mínimo) |",
        "|---|---|---|---|---|---|",
    ]
    for target, t in (("0,909", t909), ("0,939", t939)):
        lines.append(f"| {target} | {_int(t['planets_needed'])} | {_f(t['cut_score'])} | {t['negatives_above_cut']} | {t['negatives_allowed']} | "
                     f"**{t['fixes_pooled_ensemble']}** |")
    s909, s939 = t909["fixes_pooled_single"], t939["fixes_pooled_single"]
    lines += [
        "",
        "«Arreglar» un objeto es darle la puntuación que le toca: un negativo deja de estar arriba o un planeta sube. El mínimo se calcula "
        "probando todos los umbrales (`diagnosis.best_recall_with_fixes`). Otras formas de contarlo:",
        "",
        "| Cómo se mide | Para 0,909 | Para 0,939 |",
        "|---|---|---|",
        f"| Media de 10, todas las señales, precisión exacta | {t909['fixes_pooled_ensemble']} | {t939['fixes_pooled_ensemble']} |",
        f"| Media de 10, todas las señales, precisión como la mide ExoMiner (≥ 0,985) | {max(t909['fixes_pooled_ensemble_paper_metric'], 0)} | "
        f"{t939['fixes_pooled_ensemble_paper_metric']} |",
        f"| Media de 10, media de las 10 particiones (reparto óptimo entre particiones) | {t909['fixes_fold_mean_ensemble']} | {t939['fixes_fold_mean_ensemble']} |",
        f"| Un solo modelo, todas las señales (media de las 10 semillas; mínimo–máximo) | {_f(float(np.mean(s909)), 1)} ({min(s909)}–{max(s909)}) | "
        f"{_f(float(np.mean(s939)), 1)} ({min(s939)}–{max(s939)}) |",
        f"| Un solo modelo, media de las 10 particiones (media de las 10 semillas) | {_f(float(np.mean(t909['fixes_fold_mean_single'])), 1)} | "
        f"{_f(float(np.mean(t939['fixes_fold_mean_single'])), 1)} |",
        "",
        f"**Los negativos que rompen la precisión.** Por encima de la puntuación {_f(t939['cut_score'])} (la del planeta n.º {_int(t939['planets_needed'])}) hay "
        f"{t939['negatives_above_cut']} negativos y solo caben {t939['negatives_allowed']}. Por causa probable (reglas sobre los catálogos públicos, "
        "no inspección visual; cada objeto va a la primera causa que le aplica):",
        "",
        "| Causa probable | Por encima del corte de 0,939 | De ellos, por encima del de 0,909 | AFP / NTP | ExoMiner también les da ≥ 0,5 | ExoMiner les da < 0,1 | "
        "R@P0,99 si se arreglan solo estos (todas las señales · media de particiones) |",
        "|---|---|---|---|---|---|---|",
    ]
    for c, v in sorted(e["negatives_by_cause"].items(), key=lambda kv: -kv[1]["above_cut"]):
        lines.append(f"| {diagnosis.NEGATIVE_CAUSES[c]} | {v['above_cut']} | {v['above_cut_909']} | {v['afp']} / {v['above_cut'] - v['afp']} | {v['exominer_also_high']} | "
                     f"{v['exominer_below_0.1']} | {_f(v['recall_if_fixed_pooled'])} · {_f(v['recall_if_fixed_fold_mean'])} |")
    above = [n for n in e["negatives"] if n["above_cut"]]
    lines += [
        "",
        f"Los {len(above)} negativos, de mayor a menor puntuación (la lista completa hasta la puntuación 0,5 está en [`hito5_errores.csv`](hito5_errores.csv)). "
        f"«Semillas» es en cuántas de las 10 queda por encima del umbral de hoy ({_f(op['threshold'])}).",
        "",
        "| N.º | Señal (KIC-n) | KOI | Clase | Bolsillo | Semillas | ExoMiner | Periodo (d) | MES | Causa probable | Banderas del Robovetter |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    short = {"dubious_label": "Etiqueta dudosa", "secondary_tce": "Secundario de binaria", "ephemeris_match": "Contaminación (efemérides)",
             "centroid_offset": "Binaria de fondo (centroide)", "eclipsing_binary": "Binaria eclipsante", "low_snr": "Poca señal",
             "other_false_alarm": "Otra falsa alarma"}
    for n in above:
        ex_s = _f(n["exominer_score"]) if np.isfinite(n["exominer_score"]) else "—"
        lines.append(f"| {n['rank']} | {_kic(n)} | {n['kepoi_name'] or '—'} | {n['label_class']} | {_f(n['score'])} | {n['seeds_above_threshold']} | {ex_s} | "
                     f"{_f(n['period'], 2)} | {_f(n['mes'], 1)} | {short[n['cause']]} | {_note(n) or '—'} |")
    sib = e["sibling_feature"]
    lines += [
        "",
        f"De esos {len(above)}, ExoMiner también da 0,5 o más a {e['shared_negatives_above_cut']}. Los «secundarios de binaria» los marca el Robovetter con "
        f"`IS_SEC_TCE` ({_int(sib['is_sec_tce'])} señales con etiqueta): {_int(sib['is_sec_tce_with_sibling'])} de ellas comparten estrella con otra señal del mismo "
        f"periodo (±1 %), algo que se calcula solo con la tabla de señales. Esa condición la cumplen {_int(sib['negatives_with_sibling'])} negativos y "
        f"{sib['planets_with_sibling']} planetas.",
        "",
        "**Una prueba sin entrenar nada.** Si a la señal más débil de cada pareja «misma estrella, mismo periodo» (la de menor estadístico de detección; "
        f"{_int(e['sibling_rule']['signals_zeroed'])} señales, {e['sibling_rule']['planets_zeroed']} de ellas planetas) se le pone puntuación 0, sin mirar ninguna etiqueta:",
        "",
        "| | R@P0,99 exacto | R@P0,99 como lo mide ExoMiner | AUC-PR |",
        "|---|---|---|---|",
        f"| Media de 10, hoy | {_pm(res['direct']['rows']['pocket_ens_ours']['recall_exact'])} | {_pm(res['direct']['rows']['pocket_ens_ours']['recall_paper'])} | "
        f"{_pm(res['direct']['rows']['pocket_ens_ours']['auc_pr'], 4)} |",
        f"| Media de 10, con esa regla | {_pm(e['sibling_rule']['ensemble']['recall_exact'])} | {_pm(e['sibling_rule']['ensemble']['recall_paper'])} | "
        f"{_pm(e['sibling_rule']['ensemble']['auc_pr'], 4)} |",
        f"| Un modelo, hoy | {_pm(res['direct']['rows']['pocket_single_ours']['recall_exact'])} | {_pm(res['direct']['rows']['pocket_single_ours']['recall_paper'])} | "
        f"{_pm(res['direct']['rows']['pocket_single_ours']['auc_pr'], 4)} |",
        f"| Un modelo, con esa regla | {_pm(e['sibling_rule']['single']['recall_exact'])} | {_pm(e['sibling_rule']['single']['recall_paper'])} | "
        f"{_pm(e['sibling_rule']['single']['auc_pr'], 4)} |",
        "",
        f"(Media sobre las 10 particiones ± intervalo del 95 %.) La regla quita {e['sibling_rule']['secondary_tce_negatives_above_cut_zeroed']} de los "
        f"{e['negatives_by_cause']['secondary_tce']['above_cut']} secundarios que estaban por encima del corte, y después de aplicarla harían falta "
        f"{e['sibling_rule']['fixes_to_939_pooled_ensemble']} arreglos más para 0,939 (media de 10, todas las señales). No es una propuesta de modelo: es la medida de "
        "cuánto hay que ganar dándole ese dato a la red.",
        "",
        f"**Los planetas que se pierden.** Con el umbral de hoy quedan fuera {_int(e['planets_lost_now'])} planetas. Si se arreglaran los negativos, "
        f"los {_int(e['planets_lost_now'] - e['planets_below_cut'])} que están entre el corte de 0,939 y el umbral de hoy entrarían solos; los "
        f"{_int(e['planets_below_cut'])} que quedan por debajo del corte seguirían perdidos (0,939 admite perder {_int(op['n_pos'] - t939['planets_needed'])}). "
        "Por causa probable:",
        "",
        "| Causa probable | Por debajo del corte de 0,939 | ExoMiner también les da < 0,5 | Confirmados después de 2020 | Entre el corte y el umbral de hoy | "
        "Planetas con esa condición en total |",
        "|---|---|---|---|---|---|",
    ]
    for c, v in sorted(e["planets_by_cause"].items(), key=lambda kv: -kv[1]["below_cut"]):
        lines.append(f"| {diagnosis.PLANET_CAUSES[c]} | {v['below_cut']} | {v['exominer_also_low']} | {v['confirmed_after_2020']} | "
                     f"{v['between_cut_and_threshold']} | {_int(v['all_planets'])} |")
    lines += [
        "",
        "Los 30 planetas con peor puntuación (la lista completa, en el CSV):",
        "",
        "| Señal (KIC-n) | Planeta | Bolsillo | ExoMiner | Periodo (d) | MES | Tránsitos | Radio (R⊕) | Causa probable |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    pshort = {"wrong_period": "Periodo mal", "controversial": "En disputa", "saturated": "Estrella saturada", "secondary": "Secundario visible",
              "centroid": "Centroide o reflejos", "few_transits": "Pocos tránsitos", "low_snr": "Poca señal", "not_transit_like": "No parece tránsito",
              "giant": "Tránsito profundo", "short_period": "Periodo muy corto", "unclear": "Sin causa clara"}
    for p in e["planets"][:30]:
        ex_s = _f(p["exominer_score"]) if np.isfinite(p["exominer_score"]) else "—"
        lines.append(f"| {_kic(p)} | {p['kepler_name'] or p['kepoi_name']} | {_f(p['score'])} | {ex_s} | {_f(p['period'], 2)} | {_f(p['mes'], 1)} | "
                     f"{_int(p['n_transits'])} | {_f(p['prad'], 1)} | {pshort[p['cause']]} |")
    return "\n".join(lines)


def text_spread(res: dict) -> str:
    s = res["spread"]
    lines = [
        "Descomposición de la dispersión de las 100 medidas (10 particiones × 10 semillas) del modelo de bolsillo completo "
        "(`diagnosis.variance_components`: efectos aleatorios de una vía):",
        "",
        "| Medida | Media | Dispersión entre semillas (misma partición) | Dispersión entre particiones (quitada la de semillas) | Parte de la varianza que es de la partición |",
        "|---|---|---|---|---|",
    ]
    names = {"recall_exact": "R@P0,99 exacto", "recall_paper": "R@P0,99 como lo mide ExoMiner", "auc_pr": "AUC-PR"}
    for k, label in names.items():
        v = s[k]
        d = 4 if k == "auc_pr" else 3
        lines.append(f"| {label} | {_f(v['mean'], d)} | {_f(v['seed_sd'], d)} | {_f(v['fold_sd'], d)} | {_es(format(100 * v['fold_share'], '.0f'))} % |")
    lines += [
        "",
        "R@P0,99 exacto por partición (media de las 10 semillas · media de 10 modelos), con los planetas de cada una:",
        "",
        "| Partición | " + " | ".join(str(f) for f in range(h4.K_FOLDS)) + " |",
        "|---|" + "---|" * h4.K_FOLDS,
        "| Planetas | " + " | ".join(str(n) for n in s["planets_per_fold"]) + " |",
        "| Un modelo (media de semillas) | " + " | ".join(_f(float(np.mean(r))) for r in s["recall_exact"]["table"]) + " |",
        "| Un modelo (mínimo–máximo de semillas) | " + " | ".join(f"{_f(min(r), 2)}–{_f(max(r), 2)}" for r in s["recall_exact"]["table"]) + " |",
        "| Media de 10 modelos | " + " | ".join(_f(v) for v in s["recall_exact"]["ensemble_folds"]) + " |",
        "",
        "Cuánto da promediar semillas (media sobre las 10 particiones; grupos disjuntos de semillas):",
        "",
        "| Modelos promediados | 1 | 2 | 3 | 5 | 10 |",
        "|---|---|---|---|---|---|",
        "| R@P0,99 exacto | " + " | ".join(_f(s["ensemble_size"][k]["recall_exact"]) for k in ("1", "2", "3", "5", "10")) + " |",
        "| R@P0,99 como lo mide ExoMiner | " + " | ".join(_f(s["ensemble_size"][k]["recall_paper"]) for k in ("1", "2", "3", "5", "10")) + " |",
    ]
    v = res["selection"]
    lines += [
        "",
        "Lo que ya dice la validación sobre cada modelo (sin mirar la prueba para elegir): un solo modelo por partición, elegido entre las 10 semillas.",
        "",
        "| Qué modelo se usa en cada partición | R@P0,99 exacto | R@P0,99 como lo mide ExoMiner | AUC-PR |",
        "|---|---|---|---|",
        f"| Una semilla cualquiera (media de las 10) | {_pm(v['all_seeds']['recall_exact'])} | {_pm(v['all_seeds']['recall_paper'])} | {_pm(v['all_seeds']['auc_pr'], 4)} |",
        f"| La semilla con mejor AUC-PR en validación | {_pm(v['best_val_auc_pr']['recall_exact'])} | {_pm(v['best_val_auc_pr']['recall_paper'])} | "
        f"{_pm(v['best_val_auc_pr']['auc_pr'], 4)} |",
        f"| La semilla con mejor R@P0,99 en validación | {_pm(v['best_val_recall']['recall_exact'])} | {_pm(v['best_val_recall']['recall_paper'])} | "
        f"{_pm(v['best_val_recall']['auc_pr'], 4)} |",
        "",
        f"La época elegida por la validación es de media la {_f(v['selected_epoch_mean'], 1)} de 20. Los {v['runs_with_epoch_8_or_less']} modelos que se quedaron con una "
        f"época ≤ 8 (cuando el paso de aprendizaje aún es alto) dan {_f(v['recall_exact_epoch_8_or_less'])} de R@P0,99; los demás, "
        f"{_f(v['recall_exact_epoch_9_or_more'])}.",
    ]
    return "\n".join(lines)


def text_paths(res: dict) -> str:
    c = res["paths"]
    r = c["rows"]
    params = 38_441
    lines = [
        "Con las predicciones guardadas del hito 4 (nada se entrena): elegir la semilla con la validación, promediar *k* semillas y la regla de la señal más "
        "débil de cada pareja «misma estrella, mismo periodo» (sección 3), solas y juntas. Media sobre las 10 particiones ± intervalo del 95 %.",
        "",
        "| Qué se mide | Parámetros al predecir | R@P0,99 exacto | R@P0,99 como lo mide ExoMiner | AUC-PR |",
        "|---|---|---|---|---|",
    ]
    for key, v in r.items():
        p = "≈ 2,5 millones" if key == "exominer" else _int(params * v["models"])
        label = f"*{v['label']}*" if key == "exominer" else v["label"]
        lines.append(f"| {label} | {p} | {_pm(v['recall_exact'])} | {_pm(v['recall_paper'])} | {_pm(v['auc_pr'], 4)} |")
    lt, lw = c["loss_total"], c["loss_worst"]
    lines += [
        "",
        "**Dónde está el AUC-PR que falta.** El área bajo la curva precisión–recall es la media, sobre los planetas, de la precisión en el umbral de cada "
        "uno; así que «1 − AUC-PR» se reparte exactamente entre los planetas (`diagnosis.ap_loss_per_positive`; todas las señales a la vez). "
        f"Lo que pierde cada uno: ExoMiner {_f(lt['exominer'], 4)}; bolsillo, media de 10 + regla, {_f(lt['mean10_rule'], 4)}; un modelo elegido "
        f"con la validación + regla, {_f(lt['picked_rule'], 4)}. Los 25 planetas peor puntuados se llevan {_f(lw['25']['exominer'], 4)} en ExoMiner y "
        f"{_f(lw['25']['mean10_rule'], 4)} en la media de 10 + regla (los 50 peores, {_f(lw['50']['exominer'], 4)} y {_f(lw['50']['mean10_rule'], 4)}).",
        "",
        "| Causa probable del planeta (sección 3) | Planetas | Pierde ExoMiner | Pierde la media de 10 + regla | Diferencia | Pierde un modelo elegido + regla |",
        "|---|---|---|---|---|---|",
    ]
    by = sorted(c["loss_by_cause"].items(), key=lambda kv: -(kv[1]["mean10_rule"] - kv[1]["exominer"]))
    for name, v in by:
        gap = _es(f"{v['mean10_rule'] - v['exominer']:+.4f}").replace("-", "−")
        lines.append(f"| {diagnosis.PLANET_CAUSES[name]} | {_int(v['planets'])} | {_f(v['exominer'], 4)} | {_f(v['mean10_rule'], 4)} | "
                     f"{gap} | {_f(v['picked_rule'], 4)} |")
    return "\n".join(lines)


def text_ablation(res: dict) -> str:
    a = res["ablation"]
    done = [k for k, v in a.items() if v["complete"]]
    lines = [
        f"Mismo modelo (38.441 parámetros), receta, particiones y semillas (0–{res['ablation_seeds'] - 1}) que el bolsillo completo del hito 4; las entradas quitadas "
        "se sustituyen por ceros al entrenar y al medir. Media sobre las 10 particiones ± intervalo del 95 %; «Δ» es la diferencia con el modelo completo, "
        "partición a partición (mismas semillas).",
        "",
        "| Entradas | Modelos | R@P0,99 exacto | Δ | R@P0,99 como lo mide ExoMiner | Δ | AUC-PR | Δ | R@P0,99 exacto, media de 3 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in done:
        v = a[name]
        n = f"{res['ablation_seeds']} × 10"
        if name == "full":
            lines.append(f"| **{v['label']}** | {n} | {_pm(v['recall_exact'])} | — | {_pm(v['recall_paper'])} | — | {_pm(v['auc_pr'], 4)} | — | {_pm(v['ensemble']['recall_exact'])} |")
        else:
            lines.append(f"| {v['label']} | {n} | {_pm(v['recall_exact'])} | {_signed(v['delta_recall_exact'])} | {_pm(v['recall_paper'])} | "
                         f"{_signed(v['delta_recall_paper'])} | {_pm(v['auc_pr'], 4)} | {_signed(v['delta_auc_pr'], 4)} | {_pm(v['ensemble']['recall_exact'])} |")
    pending = [f"{v['label']} ({v['folds_done']} de 10 particiones)" for k, v in a.items() if not v["complete"] and v["folds_done"] > 0]
    if pending:
        lines += ["", "Sin terminar (no entran en la tabla): " + "; ".join(pending) + "."]
    lines += ["", f"Configuraciones entrenadas: {len(done) - 1} ablaciones × 10 particiones × {res['ablation_seeds']} semillas = "
              f"{(len(done) - 1) * 10 * res['ablation_seeds']} modelos. Ninguna se ha elegido ni ajustado mirando estas cifras: la lista estaba fijada antes de entrenar."]
    o = res.get("occlusion")
    if o:
        lines += [
            "",
            "**Oclusión, sin reentrenar.** Los 100 modelos guardados del hito 4 (10 particiones × 10 semillas) puntúan su parte de prueba con un grupo de entradas "
            "a cero. No dice cuánto se perdería sin esa entrada (para eso está la tabla de arriba, que reentrena), sino cuánto se apoya en ella el modelo ya entrenado.",
            "",
            "| Entradas a cero al medir | R@P0,99 exacto | Δ | R@P0,99 como lo mide ExoMiner | Δ | AUC-PR |",
            "|---|---|---|---|---|---|",
        ]
        for name, v in o.items():
            if name == "full":
                lines.append(f"| **Ninguna (referencia)** | {_pm(v['recall_exact'])} | — | {_pm(v['recall_paper'])} | — | {_pm(v['auc_pr'], 4)} |")
            else:
                lines.append(f"| {v['label'].replace('Sin ', '').replace('Solo vistas (sin escalares)', 'todos los escalares').replace('Solo escalares (sin vistas)', 'todas las vistas')} | "
                             f"{_pm(v['recall_exact'])} | {_signed(v['delta_recall_exact'])} | {_pm(v['recall_paper'])} | {_signed(v['delta_recall_paper'])} | {_pm(v['auc_pr'], 4)} |")
    return "\n".join(lines)


def figure(res: dict, ctx: dict, pocket: np.ndarray, path) -> None:
    """Left: R@P0.99 of the pocket model and of ExoMiner with the two readings of the
    metric. Right: the top of the precision-recall curves, all signals at once."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.lines import Line2D

    ours, theirs = "#eb6834", "#2a78d6"
    ink, muted, grid, surface = "#1a1a19", "#52514e", "#e2e1dc", "#fcfcfb"
    rows = res["direct"]["rows"]
    rule = res["errors"]["sibling_rule"]["ensemble"]
    items = [
        ("ExoMiner, media de 10\n(sus etiquetas y particiones)", theirs, rows["exominer_own"]),
        ("Bolsillo, media de 10\n+ regla del secundario, sin entrenar", ours, rule),
        ("Bolsillo, media de 10", ours, rows["pocket_ens_ours"]),
        ("Bolsillo, un modelo", ours, rows["pocket_single_ours"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for target, label in ((0.909, "escalón 0,909"), (0.939, "ExoMiner publicado 0,939")):
        ax.axvline(target, color=grid, lw=1, zorder=1)
        ax.text(target, len(items) - 0.42, label, fontsize=8.5, color=muted, ha="right" if target < 0.92 else "left", va="bottom")
    for i, (label, color, r) in enumerate(items):
        a, b = r["recall_exact"]["mean"], r["recall_paper"]["mean"]
        ax.plot([a, b], [i, i], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.plot([a], [i], "o", ms=10, color=color, mec=surface, mew=2, zorder=3)
        ax.plot([b], [i], "D", ms=9, color=color, mec=surface, mew=2, zorder=3)
        ax.text(a - 0.006, i, _f(a), fontsize=9, color=ink, ha="right", va="center")
        ax.text(b + 0.006, i, _f(b), fontsize=9, color=ink, ha="left", va="center")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([it[0] for it in items], fontsize=9, color=ink)
    ax.set_ylim(-1.35, len(items) - 0.1)
    ax.set_xlim(0.78, 0.98)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95])
    ax.set_xlabel("R@P0,99 (media de las 10 particiones)")
    ax.set_title("La misma red, medida de dos formas", fontsize=10)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=9, color=muted, label="cuenta exacta: precisión ≥ 0,99"),
                       Line2D([], [], marker="D", ls="none", ms=8, color=muted, label="como la mide ExoMiner: precisión ≥ 0,985")],
              frameon=False, fontsize=8.5, loc="lower left")
    ax = axes[1]
    y = ctx["label"]
    have = np.isfinite(ctx["exominer_score"])
    for label, color, yy, score in (("Bolsillo, media de 10", ours, y, pocket.mean(axis=0)),
                                    ("ExoMiner, media de 10 (publicado)", theirs, y[have], ctx["exominer_score"][have])):
        order = np.argsort(-score, kind="mergesort")
        hit = yy[order].astype(bool)
        tp, fp = np.cumsum(hit), np.cumsum(~hit)
        ax.plot(tp / hit.sum(), tp / (tp + fp), color=color, lw=2, solid_joinstyle="round", label=label)
        for precision in (0.99, 0.985):
            ok = tp / (tp + fp) >= precision
            r = tp[ok].max() / hit.sum()
            ax.plot([r], [precision], "o" if precision == 0.99 else "D", ms=9, color=color, mec=surface, mew=2, zorder=3)
            ax.annotate(_f(r), (r, precision), xytext=(16, 6) if color == theirs else (-14, -24), textcoords="offset points", fontsize=8.5, color=ink, ha="center")
    for precision, label in ((0.99, "precisión 0,99"), (0.985, "precisión 0,985")):
        ax.axhline(precision, color=grid, lw=1, zorder=1)
        ax.text(0.803, precision - 0.0004, label, fontsize=8.5, color=muted, va="top", ha="left")
    ax.set_xlim(0.80, 1.0)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_ylim(0.96, 1.001)
    ax.set_xlabel("Recall (planetas encontrados)")
    ax.set_ylabel("Precisión")
    ax.set_title("Parte alta de la curva precisión–recall (todas las señales)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    comma2 = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
    comma3 = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.3f}".replace(".", ","))
    axes[0].xaxis.set_major_formatter(comma2)
    axes[1].xaxis.set_major_formatter(comma2)
    axes[1].yaxis.set_major_formatter(comma3)
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(grid)
        a.tick_params(colors=muted, length=0)
        a.set_axisbelow(True)
    axes[0].grid(axis="x", color=grid, lw=0.6)
    axes[1].grid(color=grid, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


def write_csv(res: dict, path) -> None:
    cols = ["kind", "rank", "kepid", "tce_plnt_num", "kepoi_name", "kepler_name", "label_class", "pocket_score_mean10", "seeds_above_threshold",
            "exominer_score", "relative_to_0.939_cut", "cause", "tce_period", "mes", "n_transits", "prad", "depth", "kepmag", "koi_disposition",
            "robovetter_flags", "robovetter_comment", "discovery_reference", "fold"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for kind, items, flag in (("negative", res["errors"]["negatives"], "above_cut"), ("planet", res["errors"]["planets"], "below_cut")):
            for n in items:
                rel = ("above" if n[flag] else "below") if kind == "negative" else ("below" if n[flag] else "above")
                w.writerow([kind, n["rank"], n["kepid"], n["tce_plnt_num"], n["kepoi_name"], n["kepler_name"], n["label_class"], f"{n['score']:.4f}",
                            n["seeds_above_threshold"], "" if not np.isfinite(n["exominer_score"]) else f"{n['exominer_score']:.4f}", rel, n["cause"],
                            f"{n['tce_period']:.5f}", f"{n['mes']:.1f}", int(n["n_transits"]), f"{n['prad']:.2f}", f"{n['depth']:.0f}",
                            "" if not np.isfinite(n["kepmag"]) else f"{n['kepmag']:.2f}", n["koi_disposition"], n["rv_flags"], n["rv_comment"],
                            n["disc_ref"], n["fold"]])


def update_doc(doc, blocks: dict[str, str]) -> None:
    content = doc.read_text()
    for name, text in blocks.items():
        start, end = f"<!-- {name}:inicio -->", f"<!-- {name}:fin -->"
        before, rest = content.split(start, 1)
        _, after = rest.split(end, 1)
        content = f"{before}{start}\n{text}\n{end}{after}"
    doc.write_text(content)


def report(args) -> None:
    ctx = load_context()
    n = len(ctx["label"])
    pocket = load_test_predictions(h4.RUN_DIR / "preds", MODEL, n)
    if not np.isfinite(pocket).all():
        raise SystemExit("milestone 4 predictions are incomplete: run scripts/hito4_dr25_cv.py run first")
    res = {
        "generated": time.strftime("%Y-%m-%d"),
        "command": "uv run python scripts/hito5_diagnostico.py ablate && uv run python scripts/hito5_diagnostico.py report",
        "n_signals": n, "n_planets": int(ctx["label"].sum()), "labels": ctx["counts"], "pocket_seeds": int(pocket.shape[0]),
        "ablation_seeds": args.seeds,
        "sources": {"exominer_scores": ex.ARXIV_SOURCE, "exominer_tables_sha256": ex.TABLES, "exominer_folds_commit": ex.REPO_COMMIT},
        "direct": direct_comparison(ctx, pocket),
        "errors": error_analysis(ctx, pocket),
        "spread": spread(ctx, pocket),
        "selection": selection(ctx),
        "ablation": ablation_table(ctx, pocket, args),
        "occlusion": occlusion_table(ctx),
        "paths": cheap_paths(ctx, pocket),
    }
    blocks = {"directa": text_direct(res), "errores": text_errors(res), "dispersion": text_spread(res), "ablacion": text_ablation(res),
              "camino": text_paths(res)}
    for text in blocks.values():
        print(text, "\n")
    if args.quick:
        return
    slim = json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    # The full lists of objects go to the CSV; the JSON keeps the ones shown in the document.
    slim["errors"]["negatives"] = [n for n in slim["errors"]["negatives"] if n["above_cut"]]
    slim["errors"]["planets"] = slim["errors"]["planets"][:30]
    (DOCS / "hito5_diagnostico.json").write_text(json.dumps(slim, indent=1, ensure_ascii=False) + "\n")
    write_csv(res, DOCS / "hito5_errores.csv")
    update_doc(DOCS / "hito5_diagnostico.md", blocks)
    figure(res, ctx, pocket, DOCS / "hito5_diagnostico.png")
    print("wrote docs/hito5_diagnostico.md, docs/hito5_diagnostico.json, docs/hito5_errores.csv, docs/hito5_diagnostico.png")




def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["ablate", "occlude", "report"])
    ap.add_argument("--configs", nargs="+", default=list(diagnosis.MAIN_ABLATIONS), choices=list(diagnosis.ABLATIONS),
                    help="ablations to train (default: the ten main ones)")
    ap.add_argument("--folds", type=int, default=h4.K_FOLDS)
    ap.add_argument("--seeds", type=int, default=ABLATION_SEEDS, help="seeds per ablation and fold")
    ap.add_argument("--occlusion-seeds", type=int, default=10, help="saved models of milestone 4 used per fold by occlude")
    ap.add_argument("--shard", default="0/1", help="i/n: do only the tasks whose index is i modulo n (to share the work among n processes)")
    ap.add_argument("--reverse", action="store_true", help="ablate: tasks from last to first (to share the work with processes that go forwards)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="ablate: 1 fold, 1 seed, 1 epoch; report: print the tables without writing files")
    args = ap.parse_args()
    if args.quick and args.command == "ablate":
        args.folds, args.seeds = 1, 1
    if args.command == "ablate":
        ablate(args)
    elif args.command == "occlude":
        occlude(args)
    else:
        report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
