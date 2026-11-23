"""ExoMiner's published per-signal scores and cross-validation folds (data only).

Valizadegan et al. (2022, ApJ 926, 120) publish, as machine-readable tables, the score
of every model they trained for each Kepler DR25 signal:

* **Table 3** (``table_3.csv`` in the arXiv source of the paper, arXiv:2111.10009): the
  30,609 labelled signals with their label and the cross-validation scores of
  Astronet, ExoNet, ExoMiner and ExoMiner-TCE. Each score is the average of the ten
  models trained for the fold in which the signal was in the test part.
* **Table 13** (``table_13.csv``): the 1,921 KOIs they left unlabelled, scored with the
  average of the models of the ten folds (none of them was trained on these signals).

Their fold of each signal is not in the paper; it is in the fold tables that their
public repository carried between October 2021 and July 2022 (``data/cv/cv_fold_tables``
of github.com/nasa/ExoMiner at the commit below). Only these data files are downloaded
and read; no code of that repository (NASA Open Source Agreement) is used.
"""

from __future__ import annotations

import csv
import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
import requests

from cazaplanetas.env import DATA_DIR

EXOMINER_DIR = DATA_DIR / "exominer"
ARXIV_SOURCE = "https://arxiv.org/e-print/2111.10009v2"
TABLES = {  # file in the arXiv source -> SHA-256 of its content
    "table_3.csv": "7060e66b8da429d9295ed6a0c3519fef8ca340bc5d72d221fc961d468334622e",
    "table_13.csv": "4f25d4f656736539cad04fc943bdaa072b57f025fc98c72e252ab7a36e2c9e84",
}
REPO_COMMIT = "4baa57792bc238c2ad2c19f98d8b112491bfc774"  # 2021-11-10, nine days before the arXiv submission
FOLD_URL = (
    "https://raw.githubusercontent.com/nasa/ExoMiner/" + REPO_COMMIT + "/data/cv/cv_fold_tables/"
    "q1_q17_dr25_tce_2020.09.28_10.36.22_stellar_koi_cfp_norobovetterlabels_renamedcols_nomissingval_symsecphase_"
    "confirmedkoiperiod_sec_fold{fold}.csv"
)
FOLDS_CSV = EXOMINER_DIR / "cv_folds.csv"
MODELS = ("astronet", "exonet", "exominer", "exominer_tce")


def fetch_tables(out_dir: Path = EXOMINER_DIR) -> list[Path]:
    """Tables 3 and 13 from the arXiv source of the paper (about 27 MB, kept: 3 MB)."""
    paths = [out_dir / name for name in TABLES]
    if all(p.exists() for p in paths):
        return paths
    out_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(ARXIV_SOURCE, timeout=600, headers={"User-Agent": "cazaplanetas (research script)"})
    r.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:*") as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        for name, digest in TABLES.items():
            data = tar.extractfile(members[name]).read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError(f"{name}: unexpected content in the arXiv source")
            (out_dir / name).write_bytes(data)
    return paths


def fetch_folds(out: Path = FOLDS_CSV) -> Path:
    """Compact table (kepid, tce_plnt_num, fold) from ExoMiner's ten fold tables."""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(10):
        r = requests.get(FOLD_URL.format(fold=fold), timeout=600)
        r.raise_for_status()
        for row in csv.DictReader(io.StringIO(r.text)):
            rows.append((int(row["target_id"]), int(row["tce_plnt_num"]), fold))
    tmp = out.with_suffix(".part")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kepid", "tce_plnt_num", "fold"])
        w.writerows(sorted(rows))
    tmp.replace(out)
    return out


def read_labelled(path: Path | None = None) -> dict[str, np.ndarray]:
    """Table 3: kepid, tce_plnt_num, label_class ('PC', 'AFP', 'NTP'), label (1 = PC) and
    one array of scores per model in :data:`MODELS`."""
    with open(path or EXOMINER_DIR / "table_3.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {
        "kepid": np.array([int(r["target_id"]) for r in rows]),
        "tce_plnt_num": np.array([int(r["tce_plnt_num"]) for r in rows]),
        "label_class": np.array([r["original_label"] for r in rows]),
        "label": np.array([int(r["binary_label"]) for r in rows]),
    }
    for m in MODELS:
        out[m] = np.array([float(r[m]) for r in rows])
    return out


def read_unlabelled(path: Path | None = None) -> dict[str, np.ndarray]:
    """Table 13: kepid, tce_plnt_num and ExoMiner's score of the KOIs they left unlabelled."""
    with open(path or EXOMINER_DIR / "table_13.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "kepid": np.array([int(r["target_id"]) for r in rows]),
        "tce_plnt_num": np.array([int(r["tce_plnt_num"]) for r in rows]),
        "exominer": np.array([float(r["exominer"]) for r in rows]),
    }


def read_folds(path: Path = FOLDS_CSV) -> dict[tuple[int, int], int]:
    with open(path, newline="") as f:
        return {(int(r["kepid"]), int(r["tce_plnt_num"])): int(r["fold"]) for r in csv.DictReader(f)}


def align(kepid: np.ndarray, tce_plnt_num: np.ndarray, table: dict[str, np.ndarray], column: str, missing=np.nan) -> np.ndarray:
    """Values of ``table[column]`` for our signals (``missing`` where the table lacks them)."""
    index = {(int(k), int(p)): i for i, (k, p) in enumerate(zip(table["kepid"], table["tce_plnt_num"]))}
    rows = [index.get((int(k), int(p)), -1) for k, p in zip(kepid, tce_plnt_num)]
    values = table[column]
    return np.array([values[i] if i >= 0 else missing for i in rows])
