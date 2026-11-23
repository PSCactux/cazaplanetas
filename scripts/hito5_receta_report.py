"""Report of milestone 5, step 2 (``uv run python scripts/hito5_receta.py report``).

Reads the runs of ``hito5_receta.py`` (and milestone 4's pocket model as the starting
point), measures every recipe on the test parts with the code of the diagnosis
(``hito5_diagnostico.measure``: exact R@P0.99, R@P0.99 as ExoMiner counts it, AUC-PR;
mean over the 10 folds and 95 % interval), and writes the tables of
docs/hito5_receta.md and docs/hito5_receta.json.

What is *chosen* (keep a recipe or not, which seed to keep) uses only the validation
part of each fold (:func:`selection_score`); the test parts only measure.
"""

from __future__ import annotations

import json
import time

import numpy as np

import hito4_dr25_cv as h4
import hito5_diagnostico as h5
import hito5_receta as R
from cazaplanetas import diagnosis, training

DOC = R.DOCS / "hito5_receta.md"
SCORE_WEIGHT = 0.1  # selection score: AUC-PR + 0.1 x mean of the two R@P0.99 (fixed before any run)


def selection_score(y: np.ndarray, p: np.ndarray) -> dict:
    """Validation score of one run: AUC-PR + 0.1 x (exact R@P0.99 + R@P0.99 as ExoMiner
    counts it) / 2. The weight puts 0.03 of recall on a par with 0.003 of AUC-PR."""
    ap = training.average_precision(y, p)
    r_exact = training.recall_at_precision(y, p, 0.99)
    r_paper = diagnosis.recall_at_rounded_precision(y, p, 0.99)
    return {"score": ap + SCORE_WEIGHT * 0.5 * (r_exact + r_paper), "auc_pr": ap, "recall_exact": r_exact, "recall_paper": r_paper}


def _file(name: str, fold: int, seed: int):
    if name == "base":  # milestone 4's pocket model: same folds, recipe and seeds
        return h4.pred_path("pocket_full", fold, seed, False)
    if name == "s_ref":
        name = R.SEARCH_PARENT
    return R.run_path(name, fold, seed)


def load_runs(name: str, n: int, folds, seeds) -> dict | None:
    """Test scores (seeds, n) on every signal of the listed folds, validation scores per
    (fold, seed) and costs; None if a run is missing."""
    test = np.full((len(seeds), n), np.nan)
    val, costs = {}, {"seconds": [], "epoch": [], "parameters": None}
    for f in folds:
        for si, s in enumerate(seeds):
            path = _file(name, f, s)
            if not path.exists():
                return None
            z = np.load(path)
            test[si, z["test_index"]] = z["test"]
            val[(f, s)] = (z["val_index"], z["val"])
            costs["seconds"].append(float(z["train_seconds"]))
            costs["epoch"].append(int(z["selected_epoch"]))
            costs["parameters"] = int(z["parameters"])
    return {"test": test, "val": val, "costs": costs, "folds": list(folds), "seeds": list(seeds)}


def complete_seeds(name: str, max_seeds: int = 10) -> int:
    k = 0
    while k < max_seeds and all(_file(name, f, k).exists() for f in range(h4.K_FOLDS)):
        k += 1
    return k


def validation_summary(runs: dict, y: np.ndarray) -> dict:
    per = {key: selection_score(y[idx], p) for key, (idx, p) in runs["val"].items()}
    return {"per_run": per, **{k: float(np.mean([v[k] for v in per.values()])) for k in ("score", "auc_pr", "recall_exact", "recall_paper")}}


def picked(runs: dict, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    out = np.full(len(y), np.nan)
    for f in runs["folds"]:
        m = fold == f
        best = max(runs["seeds"], key=lambda s: training.average_precision(y[runs["val"][(f, s)][0]], runs["val"][(f, s)][1]))
        out[m] = runs["test"][runs["seeds"].index(best)][m]
    return out


def delta(y, fold, a: np.ndarray, b: np.ndarray) -> dict:
    """Fold-by-fold difference of the single-model metrics (mean over seeds) of a minus b."""
    out = {}
    for how in (h5.EXACT, h5.PAPER):
        d = [np.mean([h5.recall_at(y[fold == f], s[fold == f], how=how) for s in a]) -
             np.mean([h5.recall_at(y[fold == f], s[fold == f], how=how) for s in b]) for f in range(h4.K_FOLDS)]
        out[f"recall_{how}"] = h5._stat(d)
    d = [np.mean([training.average_precision(y[fold == f], s[fold == f]) for s in a]) -
         np.mean([training.average_precision(y[fold == f], s[fold == f]) for s in b]) for f in range(h4.K_FOLDS)]
    out["auc_pr"] = h5._stat(d)
    return out


def ladder_entry(name: str, ctx: dict, seeds: int) -> dict | None:
    y, fold = ctx["label"], ctx["fold"]
    runs = load_runs(name, len(y), range(h4.K_FOLDS), range(seeds))
    if runs is None:
        return None
    t = runs["test"]
    return {
        "name": name, "label": R.RECIPES[name].label if name in R.RECIPES else name, "parent": R.RECIPES[name].parent if name in R.RECIPES else None,
        "seeds": seeds, "parameters": runs["costs"]["parameters"],
        "train_seconds": float(np.mean(runs["costs"]["seconds"])), "selected_epoch": float(np.mean(runs["costs"]["epoch"])),
        "single": h5.measure(y, fold, t), "picked": h5.measure(y, fold, picked(runs, y, fold)), "ensemble": h5.measure(y, fold, t.mean(axis=0)),
        "validation": {k: v for k, v in validation_summary(runs, y).items() if k != "per_run"},
        "_runs": runs,
    }


def compare_validation(a: dict, b: dict, y: np.ndarray) -> dict:
    """Paired (same fold and seed) validation difference of runs a minus runs b."""
    va, vb = validation_summary(a, y)["per_run"], validation_summary(b, y)["per_run"]
    keys = sorted(set(va) & set(vb))
    out = {"runs": len(keys)}
    for k in ("score", "auc_pr", "recall_exact", "recall_paper"):
        d = np.array([va[key][k] - vb[key][k] for key in keys])
        out[k] = float(d.mean())
        out[f"{k}_se"] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return out


def search_table(ctx: dict) -> dict:
    y = ctx["label"]
    ref = load_runs("s_ref", len(y), R.SEARCH_FOLDS, R.SEARCH_SEEDS)
    out = {"parent": R.SEARCH_PARENT, "folds": list(R.SEARCH_FOLDS), "seeds": list(R.SEARCH_SEEDS), "rows": {}}
    if ref is None:
        return out
    out["reference"] = {k: v for k, v in validation_summary(ref, y).items() if k != "per_run"}
    for name, rec in R.RECIPES.items():
        if not name.startswith("s_") or name == "s_ref":
            continue
        runs = load_runs(name, len(y), R.SEARCH_FOLDS, R.SEARCH_SEEDS)
        if runs is None:
            continue
        out["rows"][name] = {"label": rec.label, "parameters": runs["costs"]["parameters"],
                             "train_seconds": float(np.mean(runs["costs"]["seconds"])),
                             "validation": {k: v for k, v in validation_summary(runs, y).items() if k != "per_run"},
                             "delta": compare_validation(runs, ref, y)}
    return out


def with_rule(score: np.ndarray, rule: np.ndarray) -> np.ndarray:
    s = np.array(score, dtype=float, copy=True)
    s[..., rule] = 0.0
    return s


# --- Text ---------------------------------------------------------------------------------

_f, _pm, _signed, _es, _int = h5._f, h5._pm, h5._signed, h4._es, h4._int


def _m(m: dict, d_ap: int = 4) -> str:
    return f"{_pm(m['recall_exact'])} | {_pm(m['recall_paper'])} | {_pm(m['auc_pr'], d_ap)}"


def _sv(x: float, d: int = 4) -> str:
    return _es(f"{x:+.{d}f}").replace("-", "−")


def text_ladder(res: dict) -> str:
    lines = [
        f"Cada peldaño se entrena con 10 particiones × {res['ladder_seeds']} semillas (las mismas semillas 0–{res['ladder_seeds'] - 1} "
        "en todos) y se mide en las partes de prueba. Media sobre las 10 particiones ± intervalo del 95 %. «Un modelo» es la media "
        "de las medidas de las semillas; «elegido» es, en cada partición, la semilla con mejor AUC-PR en **validación**; «media de "
        f"{res['ladder_seeds']}» promedia las puntuaciones de las semillas. «Validación» es la puntuación con la que se decide "
        "(AUC-PR + 0,1 × la media de las dos cuentas de R@P0,99, en la parte de validación de cada partición) y su diferencia con "
        "el peldaño anterior, emparejada por partición y semilla.",
        "",
        "| Peldaño | Parámetros | Un modelo: R@P0,99 exacto | Como ExoMiner | AUC-PR | Elegido con la validación: exacto | Como ExoMiner | AUC-PR | "
        f"Media de {res['ladder_seeds']}: exacto | Como ExoMiner | AUC-PR | Validación | Δ validación | ¿Se queda? |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in res["ladder"]:
        dv = e.get("delta_validation")
        dtxt = "—" if dv is None else f"{_sv(dv['score'])} ± {_f(dv['score_se'], 4)}"
        keep = "—" if dv is None else ("sí" if e["kept"] else "no")
        lines.append(f"| {e['label']} | {_int(e['parameters'])} | {_m(e['single'])} | {_m(e['picked'])} | {_m(e['ensemble'])} | "
                     f"{_f(e['validation']['score'], 4)} | {dtxt} | {keep} |")
    lines += ["", "Lo que aporta cada peldaño en las partes de prueba (un modelo, partición a partición, frente al peldaño anterior):", "",
              "| Peldaño | Δ R@P0,99 exacto | Δ como ExoMiner | Δ AUC-PR | Δ validación: AUC-PR | Δ validación: exacto | Δ validación: como ExoMiner |",
              "|---|---|---|---|---|---|---|"]
    for e in res["ladder"]:
        if e.get("delta_test") is None:
            continue
        d, dv = e["delta_test"], e["delta_validation"]
        lines.append(f"| {e['label']} | {_signed(d['recall_exact'])} | {_signed(d['recall_paper'])} | {_signed(d['auc_pr'], 4)} | "
                     f"{_sv(dv['auc_pr'])} | {_sv(dv['recall_exact'], 3)} | {_sv(dv['recall_paper'], 3)} |")
    return "\n".join(lines)


def text_search(res: dict) -> str:
    s = res["search"]
    if "reference" not in s:
        return "*La búsqueda no ha empezado.*"
    lines = [
        f"Sobre el peldaño «{R.RECIPES[s['parent']].label}», un cambio cada vez, con {len(s['folds'])} particiones "
        f"({', '.join(str(f) for f in s['folds'])}) × {len(s['seeds'])} semillas y **solo la parte de validación** (la prueba de "
        "esas particiones no se lee). Δ: diferencia con la referencia, emparejada por partición y semilla, ± su error típico.",
        "",
        "| Cambio | Parámetros | Δ validación | Δ AUC-PR | Δ R@P0,99 exacto | Δ como ExoMiner | ¿Entra en la combinación? |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in sorted(s["rows"].items(), key=lambda kv: -kv[1]["delta"]["score"]):
        d = r["delta"]
        lines.append(f"| {r['label']} | {_int(r['parameters'])} | {_sv(d['score'])} ± {_f(d['score_se'], 4)} | {_sv(d['auc_pr'])} | "
                     f"{_sv(d['recall_exact'], 3)} | {_sv(d['recall_paper'], 3)} | {'sí' if name in res['search_winners'] else 'no'} |")
    ref = s["reference"]
    lines += ["", f"Referencia (validación, media de los {len(s['folds']) * len(s['seeds'])} modelos): puntuación {_f(ref['score'], 4)}, "
              f"AUC-PR {_f(ref['auc_pr'], 4)}, R@P0,99 exacto {_f(ref['recall_exact'])}, como ExoMiner {_f(ref['recall_paper'])}."]
    return "\n".join(lines)


def text_final(res: dict) -> str:
    rows = res["final_rows"]
    lines = [
        "Media sobre las 10 particiones ± intervalo del 95 % (entre paréntesis, ± intervalo del 99 %, el que usa ExoMiner), "
        "todas con nuestras etiquetas y particiones y el mismo código de medida.",
        "",
        "| Qué se mide | Parámetros al predecir | R@P0,99 exacto | R@P0,99 como lo mide ExoMiner | AUC-PR |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        m = r["m"]
        lines.append(f"| {r['label']} | {r['params']} | {_pm(m['recall_exact'])} ({_f(m['recall_exact']['ci99'])}) | "
                     f"{_pm(m['recall_paper'])} ({_f(m['recall_paper']['ci99'])}) | {_pm(m['auc_pr'], 4)} ({_f(m['auc_pr']['ci99'], 4)}) |")
    g = res["gaps"]
    lines += ["", "**Distancia a ExoMiner** (nosotros menos ExoMiner, partición a partición: media ± intervalo del 95 %, y entre "
              "paréntesis en cuántas de las 10 particiones quedamos igual o por encima; positivo es mejor):", "",
              "| Comparación | R@P0,99 exacto | Como ExoMiner | AUC-PR |", "|---|---|---|---|"]

    def cell(v, k, d):
        return f"{_sv(v[k], d)} ± {_f(v[k + '_ci95'], d)} ({v[k + '_folds_ahead']}/10)"

    for k, v in g.items():
        lines.append(f"| {v['label']} | {cell(v, 'recall_exact', 3)} | {cell(v, 'recall_paper', 3)} | {cell(v, 'auc_pr', 4)} |")
    return "\n".join(lines)


def text_choice(res: dict) -> str:
    c = res.get("single_choice")
    if not c:
        return "*Pendiente.*"
    lines = ["Puntuación de validación (10 particiones × 5 semillas, semillas 0–4), la misma regla de los peldaños:", "",
             "| Receta | Puntuación | AUC-PR | R@P0,99 exacto | Como ExoMiner |", "|---|---|---|---|---|"]
    for k, v in sorted(c["scores"].items(), key=lambda kv: -kv[1]["score"]):
        mark = " **(se queda)**" if k == c["chosen"] else ""
        lines.append(f"| {R.RECIPES[k].label}{mark} | {_f(v['score'], 4)} | {_f(v['auc_pr'], 4)} | {_f(v['recall_exact'])} | {_f(v['recall_paper'])} |")
    return "\n".join(lines)


def text_count(res: dict) -> str:
    c = res["count"]
    lines = ["| Qué | Configuraciones | Modelos entrenados |", "|---|---|---|"]
    for k, v in c["rows"].items():
        lines.append(f"| {k} | {v['configs']} | {_int(v['models'])} |")
    lines.append(f"| **Total** | **{c['configs']}** | **{_int(c['models'])}** |")
    return "\n".join(lines)


# --- Main ---------------------------------------------------------------------------------


def report(args) -> None:
    ctx = h5.load_context()
    y, fold = ctx["label"], ctx["fold"]
    n = len(y)
    seeds = R.SEEDS
    ladder_names = ["base", "sib", "sib_mag", "sib_mag_em", *[k for k in R.LADDER_EXTRA if k in R.RECIPES]]
    res = {"generated": time.strftime("%Y-%m-%d"), "ladder_seeds": seeds, "selection_weight": SCORE_WEIGHT, "ladder": []}
    entries = {}
    for name in ladder_names:
        e = ladder_entry(name, ctx, seeds)
        if e is None:
            continue
        entries[name] = e
        parent = e["parent"]
        while parent is not None and parent in entries and not entries[parent].get("kept", True):
            parent = entries[parent]["parent"]  # compare with the last step that was kept
        if parent in entries:
            e["delta_validation"] = compare_validation(e["_runs"], entries[parent]["_runs"], y)
            e["delta_test"] = delta(y, fold, e["_runs"]["test"], entries[parent]["_runs"]["test"])
            e["kept"] = e["delta_validation"]["score"] > 0
            e["compared_with"] = parent
        res["ladder"].append(e)
    res["search"] = search_table(ctx)
    res["search_winners"] = [k for k, r in res["search"]["rows"].items() if r["delta"]["score"] > 0 and r["delta"]["auc_pr"] > 0]

    # Before and after: milestone 4's model and the final recipe, with ExoMiner.
    exo = np.where(np.isfinite(ctx["exominer_score"]), ctx["exominer_score"], 0.0)
    base10 = load_runs("base", n, range(h4.K_FOLDS), range(10))
    final = R.FINAL
    k_final = complete_seeds(final)
    rows = [{"key": "exominer", "label": "*ExoMiner, media de 10 modelos (nuestras etiquetas y particiones)*", "params": "≈ 2,5 millones",
             "m": h5.measure(y, fold, exo)}]
    rule = ctx["weaker_of_pair"]
    rows += [
        {"key": "base_single", "label": "Antes: bolsillo del hito 4, un modelo (media de 10 semillas)", "params": "38.441", "m": h5.measure(y, fold, base10["test"])},
        {"key": "base_picked", "label": "Antes: un modelo elegido con la validación entre 10", "params": "38.441", "m": h5.measure(y, fold, picked(base10, y, fold))},
        {"key": "base_rule", "label": "Antes: un modelo + la regla del diagnóstico (sin entrenar)", "params": "38.441", "m": h5.measure(y, fold, with_rule(base10["test"], rule))},
        {"key": "base_ens10", "label": "Antes: media de 10", "params": "384.410", "m": h5.measure(y, fold, base10["test"].mean(0))},
        {"key": "base_ens10_rule", "label": "Antes: media de 10 + la regla", "params": "384.410", "m": h5.measure(y, fold, with_rule(base10["test"].mean(0), rule))},
    ]
    if k_final:
        fr = load_runs(final, n, range(h4.K_FOLDS), range(k_final))
        p = fr["costs"]["parameters"]
        rows += [
            {"key": "final_single", "label": f"Después: último peldaño, un modelo (media de {k_final} semillas)", "params": _int(p), "m": h5.measure(y, fold, fr["test"])},
            {"key": "final_picked", "label": f"Después: último peldaño, elegido con la validación entre {k_final}", "params": _int(p), "m": h5.measure(y, fold, picked(fr, y, fold))},
            {"key": "final_ens", "label": f"Después: último peldaño, media de {k_final}", "params": _int(p * k_final), "m": h5.measure(y, fold, fr["test"].mean(0))},
        ]
        res["final"] = {"name": final, "label": R.RECIPES[final].label, "seeds": k_final, "parameters": p}
    k_dist = complete_seeds(R.DISTILLED) if R.DISTILLED in R.RECIPES else 0
    if k_dist:
        dr = load_runs(R.DISTILLED, n, range(h4.K_FOLDS), range(k_dist))
        rows += [
            {"key": "distilled_single", "label": f"Después: destilado del conjunto de 10 (media de {k_dist} semillas)", "params": _int(dr["costs"]["parameters"]),
             "m": h5.measure(y, fold, dr["test"])},
            {"key": "distilled_picked", "label": f"Después: destilado, elegido con la validación entre {k_dist}", "params": _int(dr["costs"]["parameters"]),
             "m": h5.measure(y, fold, picked(dr, y, fold))},
        ]
    if k_final and "combo" in entries:  # the single model that stays: best validation score (seeds 0-4, all folds)
        cands = {final: entries[final]["_runs"], "combo": entries["combo"]["_runs"]}
        if k_dist >= R.SEEDS:
            cands[R.DISTILLED] = load_runs(R.DISTILLED, n, range(h4.K_FOLDS), range(R.SEEDS))
        scores = {k: {kk: v for kk, v in validation_summary(r, y).items() if kk != "per_run"} for k, r in cands.items()}
        chosen = max(scores, key=lambda k: scores[k]["score"])
        res["single_choice"] = {"scores": scores, "chosen": chosen, "label": R.RECIPES[chosen].label}
        if chosen == "combo":
            k_combo = complete_seeds("combo")
            cr = load_runs("combo", n, range(h4.K_FOLDS), range(k_combo))
            p = cr["costs"]["parameters"]
            rows += [{"key": "combo_single", "label": f"Después: la combinación, un modelo (media de {k_combo} semillas)",
                      "params": _int(p), "m": h5.measure(y, fold, cr["test"])},
                     {"key": "combo_picked", "label": f"Después: la combinación, elegido con la validación entre {k_combo}",
                      "params": _int(p), "m": h5.measure(y, fold, picked(cr, y, fold))},
                     {"key": "combo_ens", "label": f"Después: la combinación, media de {k_combo}", "params": _int(p * k_combo),
                      "m": h5.measure(y, fold, cr["test"].mean(0))}]
            res["combo_seeds"] = k_combo
    res["final_rows"] = [{k: v for k, v in r.items()} for r in rows]
    by = {r["key"]: r["m"] for r in rows}
    gaps = {}

    def gap(key, label, ours, theirs="exominer"):
        if ours in by:
            gaps[key] = {"label": label}
            for k in ("recall_exact", "recall_paper", "auc_pr"):  # paired by fold: mean and 95 % interval
                d = h5._stat(np.subtract(by[ours][k]["folds"], by[theirs][k]["folds"]))
                gaps[key][k], gaps[key][f"{k}_ci95"], gaps[key][f"{k}_folds_ahead"] = d["mean"], d["ci95"], int(np.sum(np.asarray(d["folds"]) >= 0))

    gap("before_single", "Antes, un modelo frente a su conjunto de 10", "base_single")
    gap("before_ens", "Antes, conjunto de 10 frente a su conjunto de 10", "base_ens10")
    for key, label in (("combo_single", "Después, la combinación (un modelo) frente a su conjunto de 10"),
                       ("combo_picked", "Después, la combinación elegida con la validación frente a su conjunto de 10"),
                       ("combo_ens", "Después, conjunto de la combinación frente a su conjunto de 10"), ("final_single", "Después, último peldaño, un modelo frente a su conjunto de 10"),
                       ("final_picked", "Después, último peldaño, elegido con la validación frente a su conjunto de 10"),
                       ("distilled_single", "Después, destilado frente a su conjunto de 10"),
                       ("distilled_picked", "Después, destilado elegido con la validación frente a su conjunto de 10"),
                       ("final_ens", "Después, conjunto del último peldaño frente a su conjunto de 10")):
        gap(key, label, key)
    res["gaps"] = gaps

    # How many configurations were tried.
    count_rows = {}
    ladder_trained = [e for e in res["ladder"] if e["name"] != "base"]
    count_rows["Peldaños de la escalera (10 particiones × 5 semillas)"] = {"configs": len(ladder_trained), "models": 50 * len(ladder_trained)}
    n_search = len(res["search"]["rows"])
    count_rows["Búsqueda, un cambio cada vez (3 particiones × 2 semillas, solo validación)"] = {"configs": n_search, "models": 6 * n_search}
    if k_final > R.SEEDS:
        count_rows["Semillas 5–9 de la receta final (conjunto de 10 y maestro de la destilación)"] = {"configs": 0, "models": 10 * (k_final - R.SEEDS)}
    if res.get("combo_seeds", 0) > R.SEEDS:
        count_rows["Semillas 5–9 de la combinación (su conjunto de 10)"] = {"configs": 0, "models": 10 * (res["combo_seeds"] - R.SEEDS)}
    if k_dist:
        count_rows["Destilación"] = {"configs": 1, "models": 10 * k_dist}
    res["count"] = {"rows": count_rows, "configs": sum(v["configs"] for v in count_rows.values()),
                    "models": sum(v["models"] for v in count_rows.values())}

    blocks = {"escalera": text_ladder(res), "busqueda": text_search(res), "final": text_final(res), "eleccion": text_choice(res), "cuenta": text_count(res)}
    for text in blocks.values():
        print(text, "\n")
    if args.quick:
        return
    for e in res["ladder"]:
        e.pop("_runs", None)
    slim = json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (R.DOCS / "hito5_receta.json").write_text(json.dumps(slim, indent=1, ensure_ascii=False) + "\n")
    if DOC.exists():
        h5.update_doc(DOC, blocks)
        print("wrote docs/hito5_receta.md and docs/hito5_receta.json")

