"""The small files published in ``release/``: weights of the final models, their
predictions and a handful of prepared signals for the demo.

They let the verdict of milestone 5 (``scripts/hito5_veredicto.py report``) and the
demo (``scripts/demo.py``) run on a fresh clone without the DR25 views (0.65 GB built
from 118 GB of light curves) or any training. ``scripts/export_release.py`` writes
them from a full local run; ``release/SHA256SUMS`` lists their checksums.

* ``predictions_hito5_veredicto.npz``: for the final recipe (``combo``) and its distilled
  model (``distilled_combo``), the validation and test scores of every seed on every
  labelled DR25 signal (each signal is in the test part of one fold and in the
  validation part of another), with the signal keys and labels.
* ``weights/fold{f}.npz``: the state dicts of the 10 seeds of ``combo`` trained on fold
  ``f`` (25,745 parameters each, plus the running statistics of batch normalisation), one
  flat vector per seed; reading them needs no pickle.
* ``demo_signals.npz``: the model inputs of a few signals, already normalised with the
  training part of the fold whose models never saw them, with the scores recorded when
  those models were trained.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from cazaplanetas.env import REPO_ROOT

RELEASE_DIR = REPO_ROOT / "release"
PREDICTIONS = RELEASE_DIR / "predictions_hito5_veredicto.npz"
WEIGHTS_DIR = RELEASE_DIR / "weights"
DEMO_SIGNALS = RELEASE_DIR / "demo_signals.npz"
SHA256SUMS = RELEASE_DIR / "SHA256SUMS"
FOLDS = 10
SEEDS = 10
INPUT_PREFIX = "input/"


def weights_path(fold: int) -> Path:
    return WEIGHTS_DIR / f"fold{fold}.npz"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_checksums(path: Path = SHA256SUMS) -> dict[str, str]:
    """Relative path -> SHA-256, from a file in the format of ``sha256sum``."""
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.lstrip("*")] = digest
    return out


def verify_checksums(path: Path = SHA256SUMS) -> list[str]:
    """Names of the files whose content does not match ``SHA256SUMS`` (empty if all do)."""
    return [name for name, digest in read_checksums(path).items()
            if not (path.parent / name).exists() or sha256(path.parent / name) != digest]


def write_checksums(files: list[Path], path: Path = SHA256SUMS) -> None:
    lines = [f"{sha256(f)}  {f.relative_to(path.parent).as_posix()}" for f in sorted(files)]
    path.write_text("\n".join(lines) + "\n")


# --- Weights ------------------------------------------------------------------------------


def save_ensemble(state_dicts: dict[int, dict[str, torch.Tensor]], path: Path) -> None:
    """One flat float32 vector per seed (``seed{s}``) plus the names, shapes and dtypes of
    the tensors, shared by every seed (the state dicts must have the same layout)."""
    first = next(iter(state_dicts.values()))
    names = list(first)
    shapes = np.full((len(names), 4), -1, np.int64)
    for i, k in enumerate(names):
        shapes[i, : first[k].dim()] = first[k].shape
    arrays = {"names": np.array(names), "shapes": shapes, "dtypes": np.array([str(first[k].dtype).removeprefix("torch.") for k in names])}
    for s, sd in state_dicts.items():
        assert list(sd) == names
        arrays[f"seed{s}"] = np.concatenate([sd[k].detach().cpu().reshape(-1).to(torch.float32).numpy() for k in names])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_ensemble(fold: int, path: Path | None = None) -> dict[int, dict[str, torch.Tensor]]:
    """Seed -> state dict of the models of ``fold`` (to load into ``hito5_receta.RECIPES['combo'].make_model()``)."""
    with np.load(path or weights_path(fold), allow_pickle=False) as z:
        names, shapes, dtypes = z["names"].tolist(), z["shapes"], z["dtypes"].tolist()
        out = {}
        for key in sorted((k for k in z.files if k.startswith("seed")), key=lambda k: int(k.removeprefix("seed"))):
            flat, sd, at = z[key], {}, 0
            for name, shape, dtype in zip(names, shapes, dtypes):
                shape = tuple(int(d) for d in shape if d >= 0)
                size = int(np.prod(shape)) if shape else 1
                sd[name] = torch.from_numpy(flat[at:at + size].reshape(shape).copy()).to(getattr(torch, dtype))
                at += size
            assert at == len(flat)
            out[int(key.removeprefix("seed"))] = sd
    return out


# --- Predictions and demo signals ------------------------------------------------------------


def load_predictions(path: Path = PREDICTIONS) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_demo_signals(path: Path = DEMO_SIGNALS) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """(signal table, model inputs). The table has one row per signal (keys, names,
    label, fold, recorded scores); the inputs map each ``forward()`` argument to an
    array whose first axis is the signal."""
    with np.load(path, allow_pickle=False) as z:
        table = {k: z[k] for k in z.files if not k.startswith(INPUT_PREFIX)}
        inputs = {k.removeprefix(INPUT_PREFIX): z[k] for k in z.files if k.startswith(INPUT_PREFIX)}
    return table, inputs
