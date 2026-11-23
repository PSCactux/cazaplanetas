"""Milestone 5, step 2: recipe improvements without new data (no light curve is read).

    uv run python scripts/hito5_receta.py run --configs sib          # train a recipe on 10 folds x 5 seeds (resumable)
    uv run python scripts/hito5_receta.py run --configs s_lr2 --search   # the 3 folds x 2 seeds of the search
    uv run python scripts/hito5_receta.py report                     # docs/hito5_receta.md and its JSON
    uv run python scripts/hito5_receta.py run --configs sib --quick  # smoke test: 1 fold, 1 seed, 1 epoch

Every recipe is an entry of :data:`RECIPES`: inputs (the same-period sibling signal, the
Kepler magnitude, the global view at P/2, ephemeris matches with other stars; see
``cazaplanetas.derived``), model size, loss, augmentations, normalisation and training
options. Recipes are tried in the order of the diagnosis (docs/hito5_diagnostico.md,
section 6) and each one is kept or dropped with the **validation** part of each fold,
never with the test part (:func:`selection_score`).

Same folds, labels and validation/test parts as milestone 4 (``hito4_dr25_cv``). Each
(recipe, fold, seed) writes its validation, test and all-signal predictions under
data/hito5/receta/, so ``run`` can be stopped and resumed and several processes can share
the work with ``--shard i/n``. The all-signal predictions are the teacher's targets for
distillation (only those of the training part are used).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

from cazaplanetas.env import DATA_DIR, REPO_ROOT, use_local_cache

use_local_cache()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hito4_dr25_cv as h4  # noqa: E402  (dataset, folds and recipe of milestone 4)
from cazaplanetas import derived, diagnosis, dr25, features, labels, models, training  # noqa: E402
from cazaplanetas import views as V  # noqa: E402

DOCS = REPO_ROOT / "docs"
RUN_DIR = DATA_DIR / "hito5" / "receta"
SEEDS = 5  # per fold for every recipe measured on the 10 folds
SEARCH_FOLDS, SEARCH_SEEDS = (1, 4, 7), (0, 1)  # the cheap search (validation only); fixed before any run
PARAMETER_CAP = 50_000


# --- Recipes ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    """What a run trains. ``scalars``: groups of derived scalars added to the 32 DR25
    columns ("sibling", "kepmag", "ephemeris"); ``views``: rebuilt views added as
    channels ("sibling_view" to the local branch, "half_global_view" to the global one);
    ``drop``: standard views left out; ``model`` and ``train``: overrides of
    ``models.PocketVet`` and of milestone 4's training recipe; ``scaler`` and ``clip``:
    standardisation of the scalars; ``centroid``: "self" (each centroid view divided by
    its own scatter, as in milestone 4) or "train" (by one scale from the training part,
    so the size of the shift is kept); ``teacher``: recipe whose 10-seed mean is the
    soft target (distillation)."""

    label: str
    parent: str | None = None
    scalars: tuple[str, ...] = ()
    views: tuple[str, ...] = ()
    drop: tuple[str, ...] = ()
    model: tuple[tuple[str, object], ...] = ()
    train: tuple[tuple[str, object], ...] = ()
    scaler: str = "robust"
    clip: float = 8.0
    centroid: str = "self"
    teacher: str | None = None
    arch: str = "vet"  # "vet" (models.PocketVet) or "branches" (models.PocketBranches)
    note: str = ""

    def train_config(self, quick: bool = False) -> training.TrainConfig:
        cfg = {**h4.POCKET_RECIPE, **dict(self.train)}
        if quick:
            cfg["epochs"] = 1
        return training.TrainConfig(**cfg)

    def scalar_columns(self) -> tuple[str, ...]:
        groups = {"sibling": derived.SIBLING_COLUMNS, "kepmag": ("kepmag",), "ephemeris": derived.EPHEMERIS_COLUMNS}
        return (*features.SCALAR_COLUMNS, *(c for g in self.scalars for c in groups[g]))

    def view_names(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        g = [n for n in models.PocketVet.GLOBAL_INPUTS if n not in self.drop]
        loc = [n for n in models.PocketVet.LOCAL_INPUTS if n not in self.drop]
        g += [n for n in self.views if n.endswith("global_view")]
        loc += [n for n in self.views if not n.endswith("global_view")]
        return tuple(g), tuple(loc)

    def make_model(self) -> torch.nn.Module:
        g, loc = self.view_names()
        if self.arch == "branches":
            have = set(g) | set(loc)
            branches = {k: tuple(v for v in vs if v in have) for k, vs in models.PocketBranches.BRANCHES.items()}
            return models.PocketBranches(len(self.scalar_columns()), branches={k: v for k, v in branches.items() if v}, **dict(self.model))
        return models.PocketVet(len(self.scalar_columns()), global_inputs=g, local_inputs=loc, **dict(self.model))


def _with(base: Recipe, label: str, parent: str, **changes) -> Recipe:
    d = {k: getattr(base, k) for k in Recipe.__dataclass_fields__}
    for k in ("train", "model"):
        if k in changes:
            changes[k] = tuple({**dict(d[k]), **dict(changes[k])}.items())
    d.update(label=label, parent=parent, teacher=None, note="", **changes)
    return Recipe(**d)


BASE = Recipe("Bolsillo del hito 4 (referencia)")
SIB = _with(BASE, "+ señal hermana (5 escalares y su vista)", "base", scalars=("sibling",), views=("sibling_view",))
SIB_MAG = _with(SIB, "+ magnitud Kepler y vista global a P/2", "sib", scalars=("sibling", "kepmag"), views=("sibling_view", "half_global_view"))
SIB_MAG_EM = _with(SIB_MAG, "+ coincidencia de efemérides con otras estrellas (5 escalares)", "sib_mag",
                   scalars=("sibling", "kepmag", "ephemeris"))

# The ladder of the diagnosis (each step on 10 folds x 5 seeds), then the search on top of
# the best ladder step (3 folds x 2 seeds, validation only). Search candidates are named
# "s_*" and change one thing each; the parent of the search is set in SEARCH_PARENT.
RECIPES: dict[str, Recipe] = {"base": BASE, "sib": SIB, "sib_mag": SIB_MAG, "sib_mag_em": SIB_MAG_EM}
SEARCH_PARENT = "sib_mag_em"


def _search(name: str, label: str, **changes) -> None:
    RECIPES[name] = _with(RECIPES[SEARCH_PARENT], label, SEARCH_PARENT, **changes)


_search("s_ref", "Referencia de la búsqueda (la del último peldaño)")
_search("s_zscore", "Normalización: media y desviación típica, recorte a 20 (como ExoMiner)", scaler="zscore", clip=20.0)
_search("s_centroid", "Normalización: centroide dividido por una escala del entrenamiento", centroid="train")
_search("s_posw", "Desequilibrio: peso 3 para los planetas", train={"pos_weight": 3.0})
_search("s_focal", "Pérdida focal (γ = 2)", train={"focal_gamma": 2.0})
_search("s_smooth", "Suavizado de etiquetas (0,05)", train={"label_smoothing": 0.05})
_search("s_shift", "Aumento: desplazamiento de fase (±3 tramos locales, ±10 globales)", train={"shift_local": 3, "shift_global": 10})
_search("s_noise", "Aumento: ruido gaussiano (0,05 de la profundidad) en las vistas de flujo", train={"noise": 0.05})
_search("s_ema", "Media móvil de los pesos (0,999 por paso)", train={"ema": 0.999})
_search("s_lr2", "Paso de aprendizaje 2·10⁻³", train={"lr": 2e-3})
_search("s_lr05", "Paso de aprendizaje 5·10⁻⁴", train={"lr": 5e-4})
_search("s_wd", "Decaimiento de pesos 10⁻³", train={"weight_decay": 1e-3})
_search("s_drop", "Abandono 0,3", model={"dropout": 0.3})
_search("s_ep30", "30 épocas", train={"epochs": 30})
_search("s_bs128", "Lote de 128", train={"batch_size": 128})
_search("s_wide", "Cabeza más ancha (escalares 64, capa final 48)", model={"scalar_hidden": 64, "hidden": 48})
_search("s_branches", "Una rama por prueba con cuello de 8 neuronas (como ExoMiner)", arch="branches")
_search("s_trim", "Sin vistas de impares/pares ni vista local (lo que la ablación dice que sobra)",
        drop=("odd_view", "even_view", "local_view"))


# After the search: the combination of the changes that won on validation (rule of
# hito5_receta_report.selection_score: validation score and validation AUC-PR both up),
# measured on 10 folds x 5 seeds like a ladder step.
SEARCH_WINNERS = ("s_branches", "s_trim", "s_smooth", "s_ep30", "s_noise", "s_centroid", "s_lr2", "s_wide")
RECIPES["combo"] = _with(
    RECIPES[SEARCH_PARENT], "+ los 8 cambios de la búsqueda que ganan en validación, juntos", SEARCH_PARENT,
    arch="branches", drop=("odd_view", "even_view", "local_view"), centroid="train",
    model={"scalar_hidden": 64, "hidden": 48},
    train={"label_smoothing": 0.05, "epochs": 30, "noise": 0.05, "lr": 2e-3})
LADDER_EXTRA: tuple[str, ...] = ("combo",)

# The recipe whose 10 seeds make the ensemble of 10 and the teacher (fixed when the search
# ended, before the combination was measured), and the distillation of that ensemble into
# one model of the same recipe: target = 0.5 x label + 0.5 x the teacher's mean score.
FINAL = SEARCH_PARENT
DISTILLED = "distilled"
RECIPES[DISTILLED] = Recipe(**{**{k: getattr(RECIPES[FINAL], k) for k in Recipe.__dataclass_fields__},
                               "label": "Destilado del conjunto de 10 (objetivo: ½ etiqueta + ½ media del conjunto)", "parent": FINAL,
                               "train": tuple({**dict(RECIPES[FINAL].train), "distill_alpha": 0.5}.items()), "teacher": FINAL})


# --- Data ---------------------------------------------------------------------------------


def derived_inputs(tce: dict, rows: np.ndarray, cache=RUN_DIR / "derived.npz") -> dict[str, np.ndarray]:
    """Derived scalars of every DR25 signal (cached), plus the sibling's offset."""
    if cache.exists():
        with np.load(cache) as z:
            full = {k: z[k] for k in z.files}
    else:
        dr25.fetch_stellar_kepmag()
        ks = dr25.read_csv(dr25.STELLAR_KEPMAG_CSV)
        mag = dict(zip(ks["kepid"].tolist(), ks["kepmag"].tolist()))
        kepmag = np.array([mag.get(int(k), np.nan) for k in tce["kepid"]])
        dur = tce["tce_duration"] / 24.0
        t = time.perf_counter()
        sib = derived.sibling_features(tce["kepid"], tce["tce_period"], tce["tce_time0bk"], dur, tce["tce_depth"], tce["tce_max_mult_ev"])
        em = derived.ephemeris_matches(tce["kepid"], tce["ra"], tce["dec"], tce["tce_period"], tce["tce_time0bk"], dur,
                                       tce["tce_depth"], tce["tce_max_mult_ev"], kepmag)
        print(f"derived inputs: {time.perf_counter() - t:.0f} s; siblings {int(sib['sib_has'].sum())}, "
              f"ephemeris matches {int(em['em_has'].sum())}", flush=True)
        full = {**sib, **em, "kepmag": kepmag, "kepid": tce["kepid"], "tce_plnt_num": tce["tce_plnt_num"]}
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **full)
    assert np.array_equal(full["kepid"], tce["kepid"]) and np.array_equal(full["tce_plnt_num"], tce["tce_plnt_num"])
    return {k: v[rows] for k, v in full.items() if k not in ("kepid", "tce_plnt_num")}


def load_dataset(need_centroid_raw: bool = False) -> dict:
    """Labelled DR25 signals: normalised views (the seven of milestone 4 and the two
    rebuilt ones), the DR25 scalars and the derived scalars (both transformed, not yet
    standardised)."""
    labels.fetch_all()
    tce = dr25.read_csv(dr25.TCE_CSV)
    lab = labels.build(tce)
    rows = np.flatnonzero(lab["label"] >= 0)
    built = dr25.load(V.VIEW_NAMES)
    index = {(int(k), int(p)): i for i, (k, p) in enumerate(zip(built["kepid"], built["tce_plnt_num"]))}
    vrow = np.array([index[(int(lab["kepid"][i]), int(lab["tce_plnt_num"][i]))] for i in rows])
    raw = {k: built[k][vrow] for k in features.VIEW_INPUTS}
    del built
    views = features.view_inputs(raw)
    der = derived_inputs(tce, rows)
    period, dur = tce["tce_period"][rows], tce["tce_duration"][rows] / 24.0
    has = der["sib_has"] > 0
    # Rebuilt views, scaled like the views they sit next to (see views.normalize).
    sib_raw = derived.sibling_view(raw["global_view"], period, dur, der["sib_offset"], has)
    local = raw["local_view"] - np.median(raw["local_view"], axis=-1, keepdims=True)
    scale = np.abs(local.min(axis=-1, keepdims=True))
    scale = np.where(scale > 0, scale, 1.0)
    sv = (sib_raw - np.median(sib_raw, axis=-1, keepdims=True)) / scale
    views["sibling_view"] = np.where(has[:, None], sv, 0.0).astype(np.float32)
    half = derived.half_period_view(raw["global_view"])
    half = half - np.median(half, axis=-1, keepdims=True)
    hs = np.abs(half.min(axis=-1, keepdims=True))
    views["half_global_view"] = (half / np.where(hs > 0, hs, 1.0)).astype(np.float32)
    centroid_raw = None
    if need_centroid_raw:
        norm = V.normalize({k: raw[k] for k in ("global_view", "local_view", "centroid_global_view", "centroid_local_view")})
        centroid_raw = {k: norm[k].astype(np.float32) for k in ("centroid_global_view", "centroid_local_view")}
    derived_scalars = {c: der[c] for c in derived.DERIVED_COLUMNS}
    features.assert_allowed(list(derived_scalars))
    return {
        "kepid": lab["kepid"][rows], "tce_plnt_num": lab["tce_plnt_num"][rows], "label": lab["label"][rows],
        "label_class": lab["label_class"][rows], "views": views, "centroid_raw": centroid_raw,
        "scalars": features.scalar_matrix(tce, rows), "derived": derived_scalars,
        "weaker_of_pair": diagnosis.same_period_siblings(tce["kepid"], tce["tce_period"], strength=tce["tce_max_mult_ev"])[rows],
    }


def scalar_inputs(data: dict, recipe: Recipe, train_rows: np.ndarray) -> np.ndarray:
    """DR25 and derived scalars of ``recipe``, standardised with the training part."""
    cols = recipe.scalar_columns()
    extra = [data["derived"][c] for c in cols[len(features.SCALAR_COLUMNS):]]
    x = np.concatenate([data["scalars"], np.stack(extra, axis=1)], axis=1) if extra else data["scalars"]
    return features.Standardizer(clip=recipe.clip, kind=recipe.scaler).fit(x[train_rows]).transform(x)


def centroid_inputs(data: dict, train_rows: np.ndarray, clip: float = 20.0) -> dict[str, np.ndarray]:
    """Centroid views (arcsec, median removed) divided by one robust scale per view taken
    from the training part, so a large shift stays larger than a small one."""
    out = {}
    for k, v in data["centroid_raw"].items():
        s = 1.4826 * np.median(np.abs(v[train_rows]))
        out[k] = np.clip(v / max(float(s), 1e-4), -clip, clip).astype(np.float32)
    return out


def device_tensors(data: dict, recipe: Recipe, parts: dict, device) -> training.Tensors:
    g, loc = recipe.view_names()
    names = (*g, *loc)
    features.assert_allowed(list(names))
    inputs = {n: data["views"][n] for n in names}
    if recipe.centroid == "train":
        inputs.update({k: v for k, v in centroid_inputs(data, parts["train"]).items() if k in inputs})
    inputs["scalars"] = scalar_inputs(data, recipe, parts["train"])
    return training.Tensors.from_arrays(inputs, data["label"], device)


def run_path(name: str, fold: int, seed: int, quick: bool = False):
    return RUN_DIR / ("quick" if quick else "runs") / f"{name}_fold{fold}_seed{seed}.npz"


def teacher_scores(name: str, fold: int, n: int) -> np.ndarray:
    """Mean over the 10 seeds of recipe ``name`` of its scores on every signal (fold
    ``fold``); only the training part is used as a target."""
    zs = [np.load(run_path(name, fold, s)) for s in range(10)]
    return np.mean([z["all"] for z in zs], axis=0)


# --- Training ------------------------------------------------------------------------------


def run(args) -> None:
    device = training.pick_device(args.device)
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    print(f"device: {device}  torch {torch.__version__}  load {os.getloadavg()[0]:.1f}  shard {shard}/{n_shards}  pid {os.getpid()}", flush=True)
    recipes = [RECIPES[n] for n in args.configs]
    data = load_dataset(need_centroid_raw=any(r.centroid == "train" for r in recipes))
    y = data["label"]
    folds = SEARCH_FOLDS if args.search else range(args.folds)
    seeds = SEARCH_SEEDS if args.search else range(args.seed_start, args.seed_start + args.seeds)
    if args.quick:
        folds, seeds = (0,), (0,)
    tasks = [(name, fold) for name in args.configs for fold in folds]
    tasks = [t for i, t in enumerate(tasks) if i % n_shards == shard]
    if args.reverse:  # from the last task to the first, to meet a process that goes forwards
        tasks = tasks[::-1]
    for name, fold in tasks:
        recipe = RECIPES[name]
        todo = [s for s in seeds if not run_path(name, fold, s, args.quick).exists()]
        if not todo:
            continue
        parts = h4.fold_parts(data["kepid"], fold)
        dev = device_tensors(data, recipe, parts, device)
        if recipe.teacher:
            while not all(run_path(recipe.teacher, fold, s).exists() for s in range(10)):  # teacher still training
                time.sleep(60)
            dev.soft = torch.from_numpy(teacher_scores(recipe.teacher, fold, len(y)).astype(np.float32)).to(device)
        sub = {k: dev.subset(v) for k, v in parts.items()}
        cfg = recipe.train_config(args.quick)
        for seed in (todo[::-1] if args.reverse else todo):
            path = run_path(name, fold, seed, args.quick)
            if path.exists():  # done meanwhile by another process
                continue
            t = time.perf_counter()
            training.seed_everything(seed)
            model = recipe.make_model()
            n_params = models.count_parameters(model)
            assert n_params <= PARAMETER_CAP, f"{name}: {n_params} parameters"
            fit = training.train(model, sub["train"], sub["val"], cfg, seed=seed, log_every=0)
            val, test, every = training.predict(model, sub["val"]), training.predict(model, sub["test"]), training.predict(model, dev)
            thr = training.choose_threshold(y[parts["val"]], val)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part.npz")
            np.savez(tmp, val=val, test=test, all=every.astype(np.float32), threshold=thr, val_index=parts["val"], test_index=parts["test"],
                     train_seconds=fit["train_seconds"], selected_epoch=fit["selected_epoch"], parameters=n_params,
                     val_auc_pr_history=np.array([h.get("val_auc_pr", np.nan) for h in fit["history"]]),
                     load=os.getloadavg()[0])
            tmp.replace(path)
            if not args.quick:
                torch.save(model.state_dict(), path.with_suffix(".pt"))
            print(f"[{name}] fold {fold} seed {seed}: val AUC-PR {training.average_precision(y[parts['val']], val):.4f}  "
                  f"({fit['train_seconds']:.0f} s, epoch {fit['selected_epoch']}, {n_params} params, load {os.getloadavg()[0]:.0f}, "
                  f"{time.perf_counter() - t:.0f} s total)", flush=True)
        del dev, sub
        if device.type == "mps":
            torch.mps.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--configs", nargs="+", default=["sib"], choices=list(RECIPES))
    ap.add_argument("--folds", type=int, default=h4.K_FOLDS)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--seed-start", type=int, default=0, help="first seed (5 to add seeds 5-9 to a recipe)")
    ap.add_argument("--search", action="store_true", help="only the folds and seeds of the search (validation only)")
    ap.add_argument("--shard", default="0/1", help="i/n: do only the tasks whose index is i modulo n")
    ap.add_argument("--reverse", action="store_true", help="tasks from last to first (to share a list with a process going forwards)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="1 fold, 1 seed, 1 epoch (files under data/hito5/receta/quick)")
    args = ap.parse_args()
    if args.command == "run":
        run(args)
    else:
        import hito5_receta_report as rep

        rep.report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
