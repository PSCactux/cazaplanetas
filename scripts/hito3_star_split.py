"""Milestone 3: Astronet vs pocket model with a split by star and a threshold chosen on validation.

    uv run python scripts/hito3_star_split.py            # both experiments, ~1 h on an M3 Max
    uv run python scripts/hito3_star_split.py --quick    # smoke test: 1 seed, 1 epoch, writes nothing to docs/

Experiment A ("dr24"): the 15,737 labelled DR24 signals with Astronet's own views,
pooled and re-split 80/10/10 **by star** (no star in two parts). Same models and
recipes as milestone 2; 5 seeds each.

Experiment B ("propias"): the labelled DR24 signals that have a DR25 counterpart in our
own dataset (data/dr25/), split by star the same way. Each model is trained once on
Astronet's views and once on ours, for exactly the same signals and split, so the only
difference is who made the views.

In both, precision and recall are given with the 0.5 threshold and with the threshold
that maximises F1 on the validation part (chosen per model and seed, never on test).
Writes docs/hito3_results.json, the tables in docs/hito3.md and docs/hito3_*.png.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time

from cazaplanetas.env import DATA_DIR, REPO_ROOT, use_local_cache

use_local_cache()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cazaplanetas import astronet_data, dr25, models, splits, training  # noqa: E402
from cazaplanetas import views as V  # noqa: E402

DOCS = REPO_ROOT / "docs"
RUN_DIR = DATA_DIR / "hito3"
SPLIT_SEED = 0
FRACTIONS = (0.8, 0.1, 0.1)
PARTS = ("train", "val", "test")

# Same recipes as milestone 2 (scripts/hito2_astronet.py).
CONFIGS = {
    "astronet": training.TrainConfig(epochs=50, batch_size=64, lr=1e-5, optimizer="adam", select="last"),
    "pocket": training.TrainConfig(
        epochs=20, batch_size=64, lr=1e-3, weight_decay=1e-4, optimizer="adamw", schedule="cosine", select="best_val"
    ),
}
LABEL = {"astronet": "Astronet reentrenado", "pocket": "Astronet de bolsillo"}
METRICS = ("auc", "auc_pr", "recall_at_p099", "precision", "recall", "accuracy")


def _split(arrays: dict, idx: np.ndarray) -> astronet_data.Split:
    return astronet_data.Split(
        global_view=np.ascontiguousarray(arrays["global_view"][idx], dtype=np.float32),
        local_view=np.ascontiguousarray(arrays["local_view"][idx], dtype=np.float32),
        label=arrays["label"][idx],
        kepid=arrays["kepid"][idx],
        tce_plnt_num=arrays["tce_plnt_num"][idx],
        av_training_set=arrays["av_training_set"][idx],
    )


def by_star(arrays: dict) -> dict[str, astronet_data.Split]:
    part = splits.split_by_star(arrays["kepid"], FRACTIONS, seed=SPLIT_SEED)
    out = {name: _split(arrays, np.flatnonzero(part == i)) for i, name in enumerate(PARTS)}
    assert splits.shared_stars(out["train"].kepid, out["test"].kepid) == 0
    assert splits.shared_stars(out["train"].kepid, out["val"].kepid) == 0
    assert splits.shared_stars(out["val"].kepid, out["test"].kepid) == 0
    return out


def astronet_pool() -> dict:
    data = astronet_data.prepare()
    return {k: np.concatenate([getattr(data[s], k) for s in astronet_data.SPLITS])
            for k in ("global_view", "local_view", "label", "kepid", "tce_plnt_num", "av_training_set")}


def own_views_pool(pool: dict) -> tuple[dict, dict]:
    """The Astronet signals with a DR25 counterpart in our dataset: (astronet views, our views),
    aligned row by row, with the DR24 labels."""
    t25, t24 = dr25.read_csv(dr25.TCE_CSV), dr25.read_csv(dr25.DR24_CSV)
    match = dr25.match_dr24(t25, t24)
    built = dr25.load(("global_view", "local_view"))
    index = {(int(k), int(p)): i for i, (k, p) in enumerate(zip(built["kepid"], built["tce_plnt_num"]))}
    rows, ours = [], []
    for i, key in enumerate(zip(pool["kepid"].tolist(), pool["tce_plnt_num"].tolist())):
        j = index.get(match.get(key, (-1, -1)))
        if j is not None:
            rows.append(i)
            ours.append(j)
    rows, ours = np.array(rows), np.array(ours)
    theirs = {k: v[rows] for k, v in pool.items()}
    norm = V.normalize({"global_view": built["global_view"][ours], "local_view": built["local_view"][ours]})
    mine = {**theirs, "global_view": norm["global_view"].astype(np.float32), "local_view": norm["local_view"].astype(np.float32)}
    return theirs, mine


def summarize(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "values": a.tolist()}


def run_models(name: str, data: dict[str, astronet_data.Split], args, device, timing: bool) -> dict:
    tensors = {k: training.Tensors.from_split(v, device) for k, v in data.items()}
    y_val, y_test = data["val"].label, data["test"].label
    results = {}
    for m in args.models:
        cfg = CONFIGS[m]
        if args.quick:
            cfg = training.TrainConfig(**{**cfg.to_dict(), "epochs": 1})
        per_seed, val_probs, test_probs = [], [], []
        for seed in range(args.seeds):
            print(f"[{name}] {m}, seed {seed}", flush=True)
            training.seed_everything(seed)
            model = models.MODELS[m]()
            fit = training.train(model, tensors["train"], tensors["val"], cfg, seed=seed, log_every=0)
            pv, pt = training.predict(model, tensors["val"]), training.predict(model, tensors["test"])
            thr = training.choose_threshold(y_val, pv)
            r = {
                "seed": seed,
                "threshold": thr,
                "test_05": training.classification_metrics(y_test, pt, 0.5),
                "test_val_threshold": training.classification_metrics(y_test, pt, thr),
                "val_auc": training.roc_auc(y_val, pv),
                "train_seconds": fit["train_seconds"],
                "selected_epoch": fit["selected_epoch"],
            }
            print(f"   test AUC {r['test_05']['auc']:.4f}  thr {thr:.3f}  recall {r['test_05']['recall']:.3f} -> "
                  f"{r['test_val_threshold']['recall']:.3f}  precision {r['test_05']['precision']:.3f} -> "
                  f"{r['test_val_threshold']['precision']:.3f}  ({fit['train_seconds']:.0f} s)", flush=True)
            per_seed.append(r)
            val_probs.append(pv)
            test_probs.append(pt)
            if not args.quick:
                RUN_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), RUN_DIR / f"{name}_{m}_seed{seed}.pt")
        ens_v, ens_t = np.mean(val_probs, axis=0), np.mean(test_probs, axis=0)
        ens_thr = training.choose_threshold(y_val, ens_v)
        res = {
            "config": cfg.to_dict(),
            "parameters": models.count_parameters(model),
            "test_05": {k: summarize([r["test_05"][k] for r in per_seed]) for k in METRICS},
            "test_val_threshold": {k: summarize([r["test_val_threshold"][k] for r in per_seed]) for k in METRICS},
            "threshold": summarize([r["threshold"] for r in per_seed]),
            "missed_planets_05": summarize([r["test_05"]["fn"] for r in per_seed]),
            "missed_planets_val_threshold": summarize([r["test_val_threshold"]["fn"] for r in per_seed]),
            "false_alarms_05": summarize([r["test_05"]["fp"] for r in per_seed]),
            "false_alarms_val_threshold": summarize([r["test_val_threshold"]["fp"] for r in per_seed]),
            "train_seconds": summarize([r["train_seconds"] for r in per_seed]),
            "ensemble": {
                "threshold": ens_thr,
                "test_05": training.classification_metrics(y_test, ens_t, 0.5),
                "test_val_threshold": training.classification_metrics(y_test, ens_t, ens_thr),
            },
            "per_seed": per_seed,
            "test_prob_seed0": test_probs[0].tolist(),
        }
        if timing:
            model.to(device)
            res["inference_seconds_per_curve"] = {"device_batch1024": training.time_inference(model, tensors["test"], device, 1024)}
            threads = torch.get_num_threads()
            torch.set_num_threads(1)  # the small model is measured with one CPU thread (see docs/hito2.md)
            cpu = training.Tensors.from_split(data["test"], torch.device("cpu"))
            res["inference_seconds_per_curve"]["cpu_1thread_batch1"] = training.time_inference(
                model.to("cpu"), cpu, torch.device("cpu"), batch_size=1, n=200)
            torch.set_num_threads(threads)
        results[m] = res
    return results


def describe(data: dict[str, astronet_data.Split]) -> dict:
    return {k: {"tces": len(v), "stars": int(len(np.unique(v.kepid))), "planets": int(v.label.sum())} for k, v in data.items()}


def run(args) -> dict:
    device = training.pick_device(args.device)
    print(f"device: {device}  torch {torch.__version__}", flush=True)
    pool = astronet_pool()
    out = {
        "generated": time.strftime("%Y-%m-%d"),
        "command": "uv run python scripts/hito3_star_split.py" + (" --quick" if args.quick else ""),
        "environment": {"device": str(device), "torch": torch.__version__, "python": platform.python_version(),
                        "machine": platform.machine(), "system": platform.system()},
        "split": {"by": "star (kepid)", "fractions": FRACTIONS, "seed": SPLIT_SEED},
        "threshold_rule": "max F1 on the validation part, per model and seed",
        "seeds": list(range(args.seeds)),
        "experiments": {},
    }
    if "dr24" in args.experiments:
        data = by_star(pool)
        out["experiments"]["dr24"] = {"data": describe(data), "results": run_models("dr24", data, args, device, timing=True)}
    if "propias" in args.experiments:
        theirs, mine = own_views_pool(pool)
        d_theirs, d_mine = by_star(theirs), by_star(mine)
        assert all(np.array_equal(d_theirs[p].kepid, d_mine[p].kepid) for p in PARTS)
        out["experiments"]["propias"] = {
            "data": describe(d_theirs),
            "results": {
                "astronet_views": run_models("propias-astronet", d_theirs, args, device, timing=False),
                "own_views": run_models("propias-own", d_mine, args, device, timing=False),
            },
        }
    return out


# --- Markdown tables -------------------------------------------------------------


def _es(text: str) -> str:
    return re.sub(r"(\d)\.(\d)", r"\1,\2", text)


def _ms(s: dict, d: int = 3) -> str:
    return _es(f"{s['mean']:.{d}f} ± {s['std']:.{d}f}")


def _int(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def tables(out: dict, hito2: dict | None) -> str:
    lines = [f"Generado por `{out['command']}` el {out['generated']} ({out['environment']['device']}, "
             f"PyTorch {out['environment']['torch']}). Media ± desviación de {len(out['seeds'])} semillas. "
             "Umbral de validación: el que da el mejor F1 en la validación, elegido para cada modelo y semilla.", ""]
    if "dr24" in out["experiments"]:
        e = out["experiments"]["dr24"]
        d = e["data"]
        lines += [
            f"**A. Vistas de Astronet, partición por estrella.** Entrenamiento {_int(d['train']['tces'])} señales "
            f"({_int(d['train']['stars'])} estrellas), validación {_int(d['val']['tces'])}, prueba {_int(d['test']['tces'])} señales "
            f"de {_int(d['test']['stars'])} estrellas ({d['test']['planets']} planetas). Ninguna estrella está en dos partes.",
            "",
            "| Modelo | AUC | AUC-PR | Umbral | Precisión | Recall | Planetas perdidos | Falsas alarmas | Parámetros | Entrenamiento | Inferencia (MPS · CPU 1 hilo) |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for m, r in e["results"].items():
            inf = r.get("inference_seconds_per_curve", {})
            inf_txt = _es(f"{inf['device_batch1024'] * 1e6:.0f} µs · {inf['cpu_1thread_batch1'] * 1e3:.2f} ms") if inf else "—"
            for tag, key, thr in (("0,5", "test_05", None), ("validación", "test_val_threshold", r["threshold"])):
                t = r[key]
                thr_txt = "0,5" if thr is None else _ms(thr, 2)
                miss = r["missed_planets_05" if thr is None else "missed_planets_val_threshold"]
                fa = r["false_alarms_05" if thr is None else "false_alarms_val_threshold"]
                first = tag == "0,5"
                lines.append(
                    f"| {LABEL[m] if first else ''} | {_ms(t['auc'], 4) if first else ''} | {_ms(t['auc_pr']) if first else ''} | "
                    f"{thr_txt} | {_ms(t['precision'])} | {_ms(t['recall'])} | {_ms(miss, 1)} | {_ms(fa, 1)} | "
                    f"{_int(r['parameters']) if first else ''} | {_ms(r['train_seconds'], 0) + ' s' if first else ''} | {inf_txt if first else ''} |"
                )
        for m, r in e["results"].items():
            en = r["ensemble"]
            a, b = en["test_05"], en["test_val_threshold"]
            lines.append(
                f"| {LABEL[m]}, media de {len(out['seeds'])} | {_es(format(a['auc'], '.4f'))} | {_es(format(a['auc_pr'], '.3f'))} | "
                f"{_es(format(en['threshold'], '.2f'))} | {_es(format(b['precision'], '.3f'))} | {_es(format(b['recall'], '.3f'))} | "
                f"{b['fn']} | {b['fp']} | {_int(len(out['seeds']) * r['parameters'])} | — | — |"
            )
        lines.append("")
        if hito2:
            h = hito2["results"]
            lines += [
                "Las mismas recetas con la partición por señal de Astronet (hito 2), para comparar:",
                "",
                "| Modelo | Prueba de Astronet (1.574 señales) | Solo sus 690 señales de estrellas nuevas | Partición por estrella (esta) |",
                "|---|---|---|---|",
            ]
            for m in e["results"]:
                lines.append(f"| {LABEL[m]} | {_ms(h[m]['test']['auc'], 4)} | {_ms(h[m]['test_new_stars']['auc'], 4)} | "
                             f"{_ms(e['results'][m]['test_05']['auc'], 4)} |")
            lines.append("")
    if "propias" in out["experiments"]:
        e = out["experiments"]["propias"]
        d = e["data"]
        lines += [
            f"**B. Vistas propias frente a vistas de Astronet**, mismas {_int(sum(v['tces'] for v in d.values()))} señales "
            f"(las de DR24 con pareja en nuestro DR25), misma partición por estrella (prueba: {_int(d['test']['tces'])} señales, "
            f"{d['test']['planets']} planetas). Solo cambia quién hizo las vistas.",
            "",
            "| Modelo | Vistas | AUC | AUC-PR | Precisión (umbral de validación) | Recall (umbral de validación) |",
            "|---|---|---|---|---|---|",
        ]
        for m in e["results"]["astronet_views"]:
            for src, label in (("astronet_views", "de Astronet"), ("own_views", "propias (DR25)")):
                r = e["results"][src][m]
                lines.append(f"| {LABEL[m]} | {label} | {_ms(r['test_05']['auc'], 4)} | {_ms(r['test_05']['auc_pr'])} | "
                             f"{_ms(r['test_val_threshold']['precision'])} | {_ms(r['test_val_threshold']['recall'])} |")
        lines.append("")
    return "\n".join(lines)


def update_doc(doc, text: str) -> None:
    start, end = "<!-- tabla:inicio -->", "<!-- tabla:fin -->"
    content = doc.read_text()
    before, rest = content.split(start, 1)
    _, after = rest.split(end, 1)
    doc.write_text(f"{before}{start}\n{text}\n{end}{after}")


def figure(out: dict, hito2: dict | None, path) -> None:
    """Test AUC by evaluation condition, and recall/precision at 0.5 vs the validation threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    colors = {"astronet": "#2a78d6", "pocket": "#eb6834"}
    e = out["experiments"]["dr24"]["results"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    conds = ["Prueba de Astronet\n(por señal)", "Solo estrellas\nnuevas", "Partición\npor estrella"]
    for k, m in enumerate(e):
        h = hito2["results"][m] if hito2 else None
        vals = [h["test"]["auc"]["mean"], h["test_new_stars"]["auc"]["mean"]] if h else [np.nan, np.nan]
        errs = [h["test"]["auc"]["std"], h["test_new_stars"]["auc"]["std"]] if h else [0, 0]
        vals.append(e[m]["test_05"]["auc"]["mean"])
        errs.append(e[m]["test_05"]["auc"]["std"])
        x = np.arange(3) + (k - 0.5) * 0.18
        ax.errorbar(x, vals, yerr=errs, fmt="o", ms=8, color=colors[m], label=LABEL[m], capsize=3, lw=2)
    ax.set_xticks(range(3), conds)
    ax.set_ylabel("AUC en prueba")
    ax.set_title("AUC en prueba según cómo se parte (media ± desv. de 5 semillas)", fontsize=10)
    ax.grid(axis="y", color="#e2e1dc", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    for m in e:
        r = e[m]
        p0, r0 = r["test_05"]["precision"]["mean"], r["test_05"]["recall"]["mean"]
        p1, r1 = r["test_val_threshold"]["precision"]["mean"], r["test_val_threshold"]["recall"]["mean"]
        ax.annotate("", xy=(r1, p1), xytext=(r0, p0), arrowprops={"arrowstyle": "->", "color": colors[m], "lw": 2})
        ax.plot([r0], [p0], "o", ms=8, color=colors[m], mfc="white", mew=2)
        ax.plot([r1], [p1], "o", ms=8, color=colors[m], label=LABEL[m])
    ax.set_xlabel("Recall (planetas encontrados)")
    ax.set_ylabel("Precisión")
    ax.set_title("Umbral 0,5 (hueco) → umbral de validación (lleno)", fontsize=10)
    ax.grid(color="#e2e1dc", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    comma = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.3f}".replace(".", ","))
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.yaxis.set_major_formatter(comma)
    axes[1].xaxis.set_major_formatter(comma)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=["astronet", "pocket"], choices=list(CONFIGS))
    ap.add_argument("--experiments", nargs="+", default=["dr24", "propias"], choices=["dr24", "propias"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds = 1
    out = run(args)
    h2path = DOCS / "hito2_results.json"
    hito2 = json.loads(h2path.read_text()) if h2path.exists() else None
    text = tables(out, hito2)
    print(text)
    if args.quick:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        (RUN_DIR / "quick_results.json").write_text(json.dumps(out))
        return 0
    # Running one experiment keeps the other's results from a previous run.
    res_path = DOCS / "hito3_results.json"
    if res_path.exists() and set(args.experiments) != {"dr24", "propias"}:
        prev = json.loads(res_path.read_text())
        out["experiments"] = {**prev.get("experiments", {}), **out["experiments"]}
        text = tables(out, hito2)
    res_path.write_text(json.dumps(out, indent=1) + "\n")
    update_doc(DOCS / "hito3.md", text)
    if "dr24" in out["experiments"]:
        figure(out, hito2, DOCS / "hito3_particion.png")
    print("wrote docs/hito3_results.json, docs/hito3.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
