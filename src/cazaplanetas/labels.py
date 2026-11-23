"""Labels for the Kepler DR25 signals (TCEs), following the rules of ExoMiner.

Valizadegan et al. (2022, ApJ 926, 120, Sec. 4.1) label the Q1-Q17 DR25 TCEs as:

* **PC** (planet): the TCE is a KOI whose disposition in the *cumulative* KOI table
  is CONFIRMED. Unconfirmed candidates are left out.
* **AFP** (astrophysical false positive): the KOI is CERTIFIED FP in the Kepler
  Certified False Positive table (FPWG).
* **NTP** (non-transiting phenomenon): the TCE never became a KOI (the Robovetter
  called it "not transit-like"), or the KOI is CERTIFIED FA (false alarm) in the
  FPWG table.
* Rogue TCEs (``tce_rogue_flag = 1``: three-transit signals produced by a pipeline
  bug) are removed. Everything else (candidates, uncertified false positives) is
  ``UNK`` and is not used to train or to measure.

Here a KOI that is CONFIRMED *and* CERTIFIED FP (7 in 2026) is also ``UNK``: the two
tables disagree and we do not pick a side.

The Robovetter's own verdict (score and PC/FP disposition) for every DR25 TCE comes
from the public output file of the DR25 Robovetter (github.com/nasa/kepler-robovetter).
It is used **only** as a competitor to compare with, never as a model input.

Only the data files are downloaded; no code from the Robovetter or ExoMiner
repositories is used.
"""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path

import numpy as np
import requests

from cazaplanetas.dr25 import DR25_DIR, _tap_csv, read_csv

CUMULATIVE_CSV = DR25_DIR / "cumulative_koi.csv"
DR25_KOI_CSV = DR25_DIR / "q1_q17_dr25_koi.csv"
FPWG_CSV = DR25_DIR / "fpwg.csv"
ROBOVETTER_TXT = DR25_DIR / "kplr_dr25_obs_robovetter_output.txt"
CUMULATIVE_DETAILS_CSV = DR25_DIR / "cumulative_koi_details.csv"  # error analysis only (milestone 5)
KEPLER_PLANETS_CSV = DR25_DIR / "ps_kepler.csv"  # error analysis only (milestone 5)

ROBOVETTER_URL = "https://raw.githubusercontent.com/nasa/kepler-robovetter/master/kplr_dr25_obs_robovetter_output.txt"
ARCHIVE = "https://exoplanetarchive.ipac.caltech.edu"

CLASSES = ("PC", "AFP", "NTP")


# --- Downloads -----------------------------------------------------------------


def fetch_cumulative(out: Path = CUMULATIVE_CSV) -> Path:
    """Cumulative KOI table (dispositions after follow-up). Skipped if present."""
    if not out.exists():
        _tap_csv(
            "select kepid, kepoi_name, kepler_name, koi_disposition, koi_period, koi_tce_delivname "
            "from cumulative order by kepoi_name",
            out,
        )
    return out


def fetch_dr25_koi(out: Path = DR25_KOI_CSV) -> Path:
    """DR25 KOI table: links each KOI name to its DR25 TCE (kepid, koi_tce_plnt_num)."""
    if not out.exists():
        _tap_csv(
            "select kepid, kepoi_name, koi_tce_plnt_num, koi_tce_delivname, koi_pdisposition, koi_score "
            "from q1_q17_dr25_koi order by kepoi_name",
            out,
        )
    return out


def fetch_fpwg(out: Path = FPWG_CSV, retries: int = 3) -> Path:
    """Kepler Certified False Positive table. It is not served by the archive's TAP
    service, so it is read through the archive's table viewer: open the table (which
    creates a temporary workspace on their side) and ask for the CSV download."""
    if out.exists():
        return out
    for attempt in range(1, retries + 1):
        try:
            page = requests.get(f"{ARCHIVE}/cgi-bin/TblView/nph-tblView", params={"app": "ExoTbls", "config": "fpwg"}, timeout=120)
            page.raise_for_status()
            ws = re.search(r"'([0-9._]+_[0-9]+/TblView/[0-9._]+_[0-9]+)'", page.text).group(1)
            params = requests.get(f"{ARCHIVE}/cgi-bin/TblView/nph-prefDownload", params={"url": f"{ws}/fpwg_params.json"}, timeout=120)
            table = re.search(r'"fpwg"\s*:\s*"([^"$]+fpwg\.tbl)"', params.text).group(1)
            requests.get(
                f"{ARCHIVE}/cgi-bin/IceTable/nph-iceTbl",
                params={"log": "TblView.ExoplanetArchive", "workspace": ws, "table": table, "pltxaxis": "", "pltyaxis": "",
                        "checkbox": "1", "initialcheckedval": "0", "splitlabel": "0", "rowLabel": "rowlabel"},
                timeout=300,
            ).raise_for_status()
            r = requests.post(
                f"{ARCHIVE}/cgi-bin/IceTable/nph-iceTblDownload",
                data={"workspace": ws, "useTimestamp": "1", "table": table, "format": "CSV", "user": "", "label": "",
                      "columns": "all", "rows": "all", "mission": "ExoplanetArchive"},
                timeout=300,
            )
            r.raise_for_status()
            if "fpwg_disp_status" not in r.text[:20000]:
                raise RuntimeError("unexpected answer from the archive")
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".part")
            tmp.write_text(r.text)
            tmp.replace(out)
            return out
        except (requests.RequestException, AttributeError, RuntimeError) as e:
            print(f"  FPWG download failed (attempt {attempt}): {e}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError("could not download the Certified False Positive table")


def fetch_robovetter(out: Path = ROBOVETTER_TXT) -> Path:
    if not out.exists():
        r = requests.get(ROBOVETTER_URL, timeout=300)
        r.raise_for_status()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(out)
    return out


def fetch_cumulative_details(out: Path = CUMULATIVE_DETAILS_CSV) -> Path:
    """More columns of the cumulative KOI table, to describe errors (never model inputs)."""
    if not out.exists():
        _tap_csv(
            "select kepid, kepoi_name, kepler_name, koi_disposition, koi_pdisposition, koi_score, koi_fpflag_nt, koi_fpflag_ss, "
            "koi_fpflag_co, koi_fpflag_ec, koi_comment, koi_period, koi_kepmag from cumulative order by kepoi_name",
            out,
        )
    return out


def fetch_kepler_planets(out: Path = KEPLER_PLANETS_CSV) -> Path:
    """Discovery reference and date of the planets found by Kepler (Planetary Systems
    table, default parameter sets), to tell when and by whom each one was confirmed."""
    if not out.exists():
        _tap_csv(
            "select pl_name, disc_refname, disc_pubdate, disc_year, pl_controv_flag from ps "
            "where disc_facility like '%Kepler%' and default_flag=1 order by pl_name",
            out,
        )
    return out


def fetch_all() -> list[Path]:
    return [fetch_cumulative(), fetch_dr25_koi(), fetch_fpwg(), fetch_robovetter()]


# --- Readers -------------------------------------------------------------------


def read_fpwg(path: Path = FPWG_CSV) -> dict[str, str]:
    """kepoi_name -> fpwg_disp_status (comment lines start with '#')."""
    with open(path, newline="") as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        return {r["kepoi_name"]: r["fpwg_disp_status"].strip() for r in rows}


def read_robovetter(path: Path = ROBOVETTER_TXT) -> dict[tuple[int, int], tuple[float, str]]:
    """(kepid, tce_plnt_num) -> (Robovetter score, 'PC' or 'FP')."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            kepid, pl = parts[0].split("-")
            out[(int(kepid), int(pl))] = (float(parts[1]), parts[2])
    return out


def read_robovetter_flags(path: Path = ROBOVETTER_TXT) -> dict[tuple[int, int], tuple[str, str]]:
    """(kepid, tce_plnt_num) -> (four major flags as a string of 0/1: not transit-like,
    stellar eclipse, centroid offset, ephemeris match; minor descriptive flags)."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            kepid, pl = parts[0].split("-")
            out[(int(kepid), int(pl))] = ("".join(parts[3:7]), parts[7] if len(parts) > 7 else "")
    return out


def read_fpwg_details(path: Path = FPWG_CSV) -> dict[str, dict]:
    """kepoi_name -> the FPWG row (status, high-level flags, comment) as strings."""
    with open(path, newline="") as f:
        return {r["kepoi_name"]: r for r in csv.DictReader(line for line in f if not line.startswith("#"))}


# --- Labels ----------------------------------------------------------------------


def assign_labels(
    tce: dict[str, np.ndarray],
    koi_of_tce: dict[tuple[int, int], str],
    cumulative: dict[str, str],
    fpwg: dict[str, str],
) -> np.ndarray:
    """Class of every TCE row: 'PC', 'AFP', 'NTP', 'UNK' or 'ROGUE' (rules in the module doc).

    ``koi_of_tce`` maps (kepid, tce_plnt_num) to the KOI name of that DR25 TCE,
    ``cumulative`` maps KOI names to their cumulative disposition and ``fpwg`` to their
    FPWG status."""
    out = np.empty(len(tce["kepid"]), dtype="<U5")
    for i, (k, p) in enumerate(zip(tce["kepid"].tolist(), tce["tce_plnt_num"].tolist())):
        if int(tce["tce_rogue_flag"][i]) == 1:
            out[i] = "ROGUE"
            continue
        koi = koi_of_tce.get((int(k), int(p)))
        if koi is None:
            out[i] = "NTP"  # never became a KOI: the Robovetter found it not transit-like
            continue
        disp, status = cumulative.get(koi, ""), fpwg.get(koi, "")
        confirmed, cert_fp, cert_fa = disp == "CONFIRMED", status == "CERTIFIED FP", status == "CERTIFIED FA"
        if confirmed and (cert_fp or cert_fa):
            out[i] = "UNK"  # the two tables disagree
        elif confirmed:
            out[i] = "PC"
        elif cert_fp:
            out[i] = "AFP"
        elif cert_fa:
            out[i] = "NTP"
        else:
            out[i] = "UNK"
    return out


def build(tce: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Labels and Robovetter verdicts aligned with the rows of the DR25 TCE table.

    Returns kepid, tce_plnt_num, kepoi_name ('' if none), label_class, label (1 = PC,
    0 = AFP/NTP, -1 = not used), robovetter_score and robovetter_pc (bool)."""
    from cazaplanetas.dr25 import TCE_CSV

    tce = tce if tce is not None else read_csv(TCE_CSV)
    koi = read_csv(DR25_KOI_CSV)
    koi_of_tce = {(int(k), int(p)): str(n) for k, p, n in zip(koi["kepid"], koi["koi_tce_plnt_num"], koi["kepoi_name"])}
    cum = read_csv(CUMULATIVE_CSV)
    cumulative = {str(n): str(d) for n, d in zip(cum["kepoi_name"], cum["koi_disposition"])}
    cls = assign_labels(tce, koi_of_tce, cumulative, read_fpwg())
    rv = read_robovetter()
    keys = list(zip(tce["kepid"].tolist(), tce["tce_plnt_num"].tolist()))
    score = np.array([rv.get(key, (np.nan, ""))[0] for key in keys])
    pc = np.array([rv.get(key, (np.nan, ""))[1] == "PC" for key in keys])
    label = np.where(cls == "PC", 1, np.where(np.isin(cls, ("AFP", "NTP")), 0, -1))
    return {
        "kepid": np.asarray(tce["kepid"], dtype=np.int64),
        "tce_plnt_num": np.asarray(tce["tce_plnt_num"], dtype=np.int64),
        "kepoi_name": np.array([koi_of_tce.get(key, "") for key in keys]),
        "label_class": cls,
        "label": label.astype(np.int64),
        "robovetter_score": score,
        "robovetter_pc": pc,
    }


def counts(labels: dict[str, np.ndarray]) -> dict[str, int]:
    values, n = np.unique(labels["label_class"], return_counts=True)
    return {str(v): int(c) for v, c in zip(values, n)}
