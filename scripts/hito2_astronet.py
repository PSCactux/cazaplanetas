"""Milestone 2: retrain Astronet and the pocket model on Astronet's own DR24 views.

One command, from scratch:

    uv run python scripts/hito2_astronet.py

1. downloads the 10 pre-computed TFRecord files (~145 MB, SHA-256 checked) into
   data/astronet/ and converts them to one .npz;
2. trains Astronet (paper recipe) and PocketAstronet with seeds 0..4 on the
   train split, using the validation split only for early stopping (pocket);
3. evaluates each model and the 5-seed average on the test split, measures
   training time and inference time per light curve;
4. writes docs/hito2_results.json and regenerates the tables in docs/hito2.md.

``--quick`` runs 1 seed x 1 epoch as a smoke test and writes nothing to docs/.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

from cazaplanetas.env import DATA_DIR, REPO_ROOT, use_local_cache

use_local_cache()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cazaplanetas import astronet_data, models, training  # noqa: E402

DOCS = REPO_ROOT / "docs"
RUN_DIR = DATA_DIR / "hito2"

# Astronet: Shallue & Vanderburg (2018), Sec. 5.2: batch 64, 50 epochs, Adam
# (lr 1e-5, betas 0.9/0.999, eps 1e-8), no dropout, random time reversal.
# The paper keeps the weights after the last epoch.
CONFIGS = {
    "astronet": training.TrainConfig(epochs=50, batch_size=64, lr=1e-5, optimizer="adam", select="last"),
    # Pocket: chosen on the validation split only (test never used for tuning).
    "pocket": training.TrainConfig(
        epochs=20, batch_size=64, lr=1e-3, weight_decay=1e-4, optimizer="adamw", schedule="cosine", select="best_val"
    ),
}

# Published reference numbers for Astronet on this same test split.
PUBLISHED = {
    "paper": {  # Shallue & Vanderburg 2018, Tables 1-2: average of 10 models
        "source": "Shallue & Vanderburg (2018), tablas 1 y 2 (media de 10 modelos)",
        "auc": 0.988,
        "accuracy": 0.960,
        "train": "90 min en una CPU (por modelo)",
    },
    "readme": {  # exoplanet-ml README, example output of `evaluate` (one model, 1574 test TCEs)
        "source": "README de exoplanet-ml (un modelo, 1.574 señales de prueba)",
        "auc": 0.988882,
        "accuracy": 0.9625159,
        "tp": 350,
        "fp": 49,
        "fn": 10,
        "tn": 1165,
    },
}
PUBLISHED["readme"]["precision"] = 350 / (350 + 49)
PUBLISHED["readme"]["recall"] = 350 / (350 + 10)

METRICS = ("auc", "auc_pr", "precision", "recall", "accuracy", "recall_at_p099")


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "values": a.tolist()}


def run(args) -> dict:
    device = training.pick_device(args.device)
    print(f"device: {device}  torch {torch.__version__}", flush=True)

    t = time.perf_counter()
    print("1. data", flush=True)
    data = astronet_data.prepare()
    for name, s in data.items():
        print(f"   {name}: {len(s)} TCEs, {int(s.label.sum())} planet candidates", flush=True)
    prep_seconds = time.perf_counter() - t

    tensors = {k: training.Tensors.from_split(v, device) for k, v in data.items()}
    test_cpu = training.Tensors.from_split(data["test"], torch.device("cpu"))
    y_test = data["test"].label
    # TCEs whose star has no TCE in the training split (Astronet splits by TCE, not by star).
    new_star = ~np.isin(data["test"].kepid, data["train"].kepid)

    seeds = list(range(args.seeds))
    results: dict = {}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        cfg = CONFIGS[name]
        if args.quick:
            cfg = training.TrainConfig(**{**cfg.to_dict(), "epochs": 1})
        per_seed, probs = [], []
        for seed in seeds:
            print(f"2. {name}, seed {seed}", flush=True)
            training.seed_everything(seed)
            model = models.MODELS[name]()
            fit = training.train(model, tensors["train"], tensors["val"], cfg, seed=seed)
            prob = training.predict(model, tensors["test"])
            m = training.classification_metrics(y_test, prob)
            m_new = training.classification_metrics(y_test[new_star], prob[new_star])
            print(
                f"   test AUC {m['auc']:.4f}  precision {m['precision']:.3f}  recall {m['recall']:.3f}"
                f"  ({fit['train_seconds']:.0f} s, epoch {fit['selected_epoch']})",
                flush=True,
            )
            per_seed.append({"seed": seed, "test": m, "test_new_stars": m_new, **fit})
            probs.append(prob)
            if not args.quick:
                torch.save(model.state_dict(), RUN_DIR / f"{name}_seed{seed}.pt")

        print(f"3. {name}: inference timing", flush=True)
        infer_mps = training.time_inference(model, tensors["test"], device, batch_size=1024)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        infer_cpu1 = training.time_inference(model.to("cpu"), test_cpu, torch.device("cpu"), batch_size=1, n=200)
        torch.set_num_threads(threads)

        ensemble = np.mean(probs, axis=0)
        results[name] = {
            "config": cfg.to_dict(),
            "parameters": models.count_parameters(model),
            "macs_per_curve": models.count_macs(model),
            "weights_kb": models.count_parameters(model) * 4 / 1024,
            "test": {k: summarize([r["test"][k] for r in per_seed]) for k in METRICS},
            "test_new_stars": {k: summarize([r["test_new_stars"][k] for r in per_seed]) for k in METRICS},
            "ensemble_test": training.classification_metrics(y_test, ensemble),
            "ensemble_test_new_stars": training.classification_metrics(y_test[new_star], ensemble[new_star]),
            "train_seconds": summarize([r["train_seconds"] for r in per_seed]),
            "selected_epoch": [r["selected_epoch"] for r in per_seed],
            "inference_seconds_per_curve": {"device_batch1024": infer_mps, "cpu_1thread_batch1": infer_cpu1},
            "per_seed": per_seed,
        }

    return {
        "generated": time.strftime("%Y-%m-%d"),
        "command": "uv run python scripts/hito2_astronet.py" + (" --quick" if args.quick else ""),
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "data": {
            "files": {k: v[2] for k, v in astronet_data.FILES.items()},
            "sizes": {k: len(v) for k, v in data.items()},
            "planet_candidates": {k: int(v.label.sum()) for k, v in data.items()},
            "test_new_stars": int(new_star.sum()),
            "prepare_seconds": prep_seconds,
        },
        "seeds": seeds,
        "published_astronet": PUBLISHED,
        "results": results,
    }


# --- Markdown tables -----------------------------------------------------------


def _ms(s: dict, digits: int = 3) -> str:
    return f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}"


def _num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _es(text: str) -> str:
    """Spanish decimal comma for the numbers in a cell."""
    return re.sub(r"(\d)\.(\d)", r"\1,\2", text)


def _int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def tables(out: dict) -> str:
    r, pub, env = out["results"], out["published_astronet"], out["environment"]
    n_seeds = len(out["seeds"])
    names = [n for n in ("astronet", "pocket") if n in r]
    label = {"astronet": "Astronet reentrenado", "pocket": "Astronet de bolsillo"}
    lines = [
        f"Generado por `{out['command']}` el {out['generated']} "
        f"({env['device']}, PyTorch {env['torch']}). Media ± desviación típica de {n_seeds} semillas; "
        f"«media de {n_seeds}» es el promedio de las probabilidades de los {n_seeds} modelos. "
        f"Prueba: {_int(out['data']['sizes']['test'])} señales, {out['data']['planet_candidates']['test']} planetas candidatos. "
        "Precisión y recall con umbral 0,5.",
        "",
        "| Modelo | AUC | Precisión | Recall | Exactitud | AUC-PR | Parámetros | Entrenamiento | Inferencia por curva (MPS, lotes de 1024) | Inferencia por curva (CPU, 1 hilo, de una en una) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in names:
        x = r[n]
        t = x["test"]
        inf = x["inference_seconds_per_curve"]
        lines.append(
            f"| {label[n]} | {_es(_ms(t['auc'], 4))} | {_es(_ms(t['precision']))} | {_es(_ms(t['recall']))} | "
            f"{_es(_ms(t['accuracy']))} | {_es(_ms(t['auc_pr']))} | {_int(x['parameters'])} | "
            f"{_es(_ms(x['train_seconds'], 0))} s | {_num(inf['device_batch1024'] * 1e6, 1)} µs | "
            f"{_num(inf['cpu_1thread_batch1'] * 1e3, 2)} ms |"
        )
    for n in names:
        e = r[n]["ensemble_test"]
        lines.append(
            f"| {label[n]}, media de {n_seeds} | {_num(e['auc'], 4)} | {_num(e['precision'])} | {_num(e['recall'])} | "
            f"{_num(e['accuracy'])} | {_num(e['auc_pr'])} | {_int(n_seeds * r[n]['parameters'])} | — | — | — |"
        )
    p, rd = pub["paper"], pub["readme"]
    lines += [
        f"| Astronet publicado: artículo, media de 10 | {_num(p['auc'])} | — | — | {_num(p['accuracy'])} | — | — | {p['train']} | — | — |",
        f"| Astronet publicado: README, un modelo | {_num(rd['auc'], 4)} | {_num(rd['precision'])} | {_num(rd['recall'])} | "
        f"{_num(rd['accuracy'])} | — | — | — | — | — |",
        "",
    ]
    if len(names) == 2:
        a, b = r["astronet"], r["pocket"]
        lines += [
            f"Coste relativo: el modelo de bolsillo tiene **{a['parameters'] / b['parameters']:.0f} veces menos parámetros** "
            f"({_num(b['weights_kb'], 0)} kB de pesos frente a {_num(a['weights_kb'] / 1024, 1)} MB), "
            f"{a['macs_per_curve'] / b['macs_per_curve']:.0f} veces menos operaciones por curva "
            f"({_num(b['macs_per_curve'] / 1e6, 1)} frente a {_num(a['macs_per_curve'] / 1e6, 1)} millones de multiplicaciones-sumas) "
            f"y entrena {_num(a['train_seconds']['mean'] / b['train_seconds']['mean'], 1)} veces más rápido.",
            "",
        ]
    lines += [
        f"Solo las {out['data']['test_new_stars']} señales de prueba cuya estrella no tiene ninguna señal en entrenamiento:",
        "",
        "| Modelo | AUC | Precisión | Recall | AUC-PR |",
        "|---|---|---|---|---|",
    ]
    for n in names:
        t = r[n]["test_new_stars"]
        lines.append(f"| {label[n]} | {_es(_ms(t['auc'], 4))} | {_es(_ms(t['precision']))} | {_es(_ms(t['recall']))} | {_es(_ms(t['auc_pr']))} |")
    return "\n".join(lines)


def update_doc(doc: Path, text: str) -> None:
    start, end = "<!-- tabla:inicio -->", "<!-- tabla:fin -->"
    content = doc.read_text()
    if start not in content or end not in content:
        raise SystemExit(f"{doc} has no {start} ... {end} markers")
    before, rest = content.split(start, 1)
    _, after = rest.split(end, 1)
    doc.write_text(f"{before}{start}\n{text}\n{end}{after}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=["astronet", "pocket"], choices=list(CONFIGS))
    ap.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    ap.add_argument("--quick", action="store_true", help="1 seed x 1 epoch smoke test; does not touch docs/")
    args = ap.parse_args()
    if args.quick:
        args.seeds = 1

    out = run(args)
    if args.quick:
        path = RUN_DIR / "quick_results.json"
        path.write_text(json.dumps(out, indent=1))
        print(tables(out))
        print(f"wrote {path.relative_to(REPO_ROOT)}")
        return 0
    (DOCS / "hito2_results.json").write_text(json.dumps(out, indent=1) + "\n")
    update_doc(DOCS / "hito2.md", tables(out))
    print(tables(out))
    print("wrote docs/hito2_results.json and docs/hito2.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
