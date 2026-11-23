"""Milestone 3: build our own Kepler DR25 dataset of views, resumable, in batches.

    uv run python scripts/hito3_dr25.py build     # download + views, ~3-6 h, can be stopped and resumed;
                                                  # at the end, stars that failed are processed once more
    uv run python scripts/hito3_dr25.py check     # compare 100 random global views with Astronet's
    uv run python scripts/hito3_dr25.py report    # failure list, manifest, docs/hito3_dataset.json

``build`` reads every long-cadence light curve of the 17,230 DR25 stars from the public
MAST copy on AWS, in memory (no FITS file is written to disk), and writes one shard
of views per batch of stars into data/dr25/views/. It stops if the disk has less than
30 GB free. Run it again to resume: finished batches are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

from cazaplanetas.env import DATA_DIR, REPO_ROOT, use_local_cache

use_local_cache()

import numpy as np  # noqa: E402

from cazaplanetas import astronet_data, dr25  # noqa: E402
from cazaplanetas import views as V  # noqa: E402

DOCS = REPO_ROOT / "docs"
LOG = dr25.DR25_DIR / "build.log"
STATS = dr25.DR25_DIR / "build_stats.json"
CHECK = dr25.DR25_DIR / "check_astronet.json"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def cmd_build(args) -> None:
    dr25.fetch_tce_table()
    table = dr25.read_csv(dr25.TCE_CSV)
    stars = dr25.tces_by_star(table)
    log(f"build: {len(table['kepid'])} TCEs on {len(stars)} stars, batches of {args.batch_size}, "
        f"{args.workers} processes x {args.threads} downloads, {dr25.free_gb():.1f} GB free")
    stats = dr25.build(stars, batch_size=args.batch_size, workers=args.workers, threads=args.threads,
                       min_free_gb=args.min_free_gb, limit_batches=args.limit_batches, log=log)
    old = json.loads(STATS.read_text()) if STATS.exists() else {"runs": []}
    old["runs"].append({"date": time.strftime("%Y-%m-%d %H:%M"), **stats})
    STATS.write_text(json.dumps(old, indent=1))
    log(f"build finished: {stats}")
    if not args.limit_batches:
        n = dr25.redo_failures(stars, batch_size=args.batch_size, workers=args.workers, threads=args.threads, log=log)
        log(f"failed stars processed again: {n}")


def _astronet_arrays():
    data = astronet_data.load()
    cat = {k: np.concatenate([getattr(data[s], k) for s in astronet_data.SPLITS])
           for k in ("kepid", "tce_plnt_num", "global_view", "local_view", "av_training_set")}
    return cat


def cmd_check(args) -> None:
    """Pearson correlation between our global view and Astronet's for random signals in both.

    Three variants for the same signals: (a) the dataset as built (DR25 ephemeris, transits
    masked while detrending); (b) DR24 ephemeris, as Astronet; (c) DR24 ephemeris and no
    masking, i.e. Astronet's recipe: this one tests that the pipeline reproduces theirs."""
    astronet_data.prepare()
    dr25.fetch_dr24_table()
    t25 = dr25.read_csv(dr25.TCE_CSV)
    t24 = dr25.read_csv(dr25.DR24_CSV)
    t24["tce_maxmesd"] = np.full(len(t24["kepid"]), np.nan)
    stars24 = dr25.tces_by_star(t24)
    match = dr25.match_dr24(t25, t24)
    a = _astronet_arrays()
    keys = list(zip(a["kepid"].tolist(), a["tce_plnt_num"].tolist()))
    matched = [i for i, k in enumerate(keys) if k in match]
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(matched, args.n, replace=False).tolist())
    log(f"check: {len(matched)} of {len(keys)} Astronet signals match a DR25 signal; checking {len(pick)}")

    # (a) from the built dataset
    built = dr25.load(("global_view", "local_view"))
    index = {(int(k), int(p)): i for i, (k, p) in enumerate(zip(built["kepid"], built["tce_plnt_num"]))}
    rows = []
    for n, i in enumerate(pick):
        k24 = keys[i]
        k25 = match[k24]
        row = {"kepid": k24[0], "dr24_plnt": k24[1], "dr25_plnt": k25[1], "label": str(a["av_training_set"][i])}
        j = index.get(k25)
        row["corr_dataset"] = dr25.pearson(built["global_view"][j], a["global_view"][i]) if j is not None else None
        for variant, mask in (("corr_dr24_ephemeris", True), ("corr_astronet_recipe", False)):
            r = dr25.process_star(k24[0], stars24[k24[0]], mask_transits=mask)
            e = [e for e in r["tces"] if e["tce_plnt_num"] == k24[1]]
            ok = not r["error"] and e and "views" in e[0]
            row[variant] = dr25.pearson(e[0]["views"]["global_view"], a["global_view"][i]) if ok else None
            if variant == "corr_astronet_recipe" and ok:
                row["corr_local_astronet_recipe"] = dr25.pearson(e[0]["views"]["local_view"], a["local_view"][i])
        rows.append(row)
        log(f"  {n + 1}/{len(pick)} {k24}: " + ", ".join(f"{k[5:]} {v:.3f}" for k, v in row.items() if k.startswith("corr") and v is not None))

    def summary(key):
        v = np.array([r[key] for r in rows if r.get(key) is not None])
        return {"n": int(len(v)), "median": float(np.median(v)), "mean": float(v.mean()),
                "frac_ge_095": float((v >= 0.95).mean()), "p10": float(np.percentile(v, 10))}

    out = {"seed": args.seed, "n_astronet": len(keys), "n_matched": len(matched), "rows": rows,
           "summary": {k: summary(k) for k in ("corr_dataset", "corr_dr24_ephemeris", "corr_astronet_recipe", "corr_local_astronet_recipe")}}
    CHECK.write_text(json.dumps(out, indent=1))
    log(f"check summary: {json.dumps(out['summary'])}")


def _reason(r: str) -> str:
    """Short category for a failure message."""
    if "no long-cadence" in r:
        return "sin curvas de cadencia larga en el archivo"
    if "no usable data" in r:
        return "sin puntos válidos"
    if "download failed" in r or "checksum" in r:
        return "descarga fallida"
    if "no transit falls" in r:
        return "ningún tránsito cae en datos válidos"
    if "no data in" in r:
        return "vista sin datos"
    return "otro: " + r[:80]


def cmd_report(args) -> None:
    table = dr25.read_csv(dr25.TCE_CSV)
    n_total = len(table["kepid"])
    stars = dr25.tces_by_star(table)
    d = dr25.load(("global_view",))
    n_ok = len(d["kepid"])
    fails = dr25.failures()
    # One row per failed TCE (a failed star fails all its TCEs).
    rows = []
    for f in fails:
        if f["tce_plnt_num"] is None:
            rows += [(f["kepid"], t.tce_plnt_num, f["reason"]) for t in stars[f["kepid"]]]
        else:
            rows.append((f["kepid"], f["tce_plnt_num"], f["reason"]))
    rows.sort()
    with open(DOCS / "hito3_fallos.csv", "w") as fh:
        fh.write("kepid,tce_plnt_num,motivo,mensaje\n")
        for k, p, r in rows:
            fh.write(f"{k},{p},{_reason(r)},\"{r.replace(chr(34), chr(39))[:200]}\"\n")
    # Processed, but worth a look: views with most bins empty (few transits on good data).
    full = dr25.load(())
    warn = []
    for j in range(len(full["kepid"])):
        notes = []
        for name in V.VIEW_NAMES:
            e, n = int(full[f"empty_{name}"][j]), V.VIEW_LENGTHS[name]
            if e == n:
                notes.append(f"{name} vacía (ceros)")
            elif e > n / 2 and not name.startswith("centroid"):
                notes.append(f"{name}: {e} de {n} tramos vacíos")
        if notes:
            warn.append((int(full["kepid"][j]), int(full["tce_plnt_num"][j]), int(full["n_transits"][j]), "; ".join(notes)))
    with open(DOCS / "hito3_avisos.csv", "w") as fh:
        fh.write("kepid,tce_plnt_num,transitos_con_datos,aviso\n")
        for k, p, nt, note in warn:
            fh.write(f"{k},{p},{nt},\"{note}\"\n")
    n_shards = len(dr25.shard_files())
    expected = (len(stars) + args.batch_size - 1) // args.batch_size
    stats = json.loads(STATS.read_text()) if STATS.exists() else {"runs": []}
    check = json.loads(CHECK.read_text()) if CHECK.exists() else None
    manifest = dr25.write_manifest({"tces_total": n_total, "tces_processed": n_ok, "tces_failed": len(rows)})
    shard_bytes = sum(f["bytes"] for f in manifest["files"] if f["path"].startswith("views/"))
    out = {
        "generated": time.strftime("%Y-%m-%d"),
        "tces_total": n_total,
        "stars_total": len(stars),
        "tces_processed": n_ok,
        "fraction_processed": n_ok / n_total,
        "tces_failed": len(rows),
        "failures_by_reason": dict(Counter(_reason(r) for _, _, r in rows).most_common()),
        "shards": n_shards,
        "shards_expected": expected,
        "dataset_bytes": manifest["total_bytes"],
        "views_bytes": shard_bytes,
        "manifest_sha256": dr25.sha256(dr25.MANIFEST),
        "tces_with_warnings": len(warn),
        "empty_bins": {name: {"tces_with_some": int((full[f"empty_{name}"] > 0).sum()),
                              "tces_with_all": int((full[f"empty_{name}"] == V.VIEW_LENGTHS[name]).sum())}
                       for name in V.VIEW_NAMES},
        "diagnostics": {
            "median_points_per_star": float(np.median(d["n_points"])),
            "median_transits_per_tce": float(np.median(d["n_transits"])),
            "tces_with_empty_global_bins": int((d["empty_global_view"] > 0).sum()),
        },
        "build_runs": stats["runs"],
        "check_astronet": None if check is None else {k: check[k] for k in ("seed", "n_astronet", "n_matched", "summary")},
        "data_dir_bytes": sum(p.stat().st_size for p in DATA_DIR.rglob("*") if p.is_file()),
    }
    (DOCS / "hito3_dataset.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "build_runs"}, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--batch-size", type=int, default=500)
    b.add_argument("--workers", type=int, default=10)
    b.add_argument("--threads", type=int, default=3)
    b.add_argument("--min-free-gb", type=float, default=30.0)
    b.add_argument("--limit-batches", type=int, default=None, help="stop after this many new batches")
    c = sub.add_parser("check")
    c.add_argument("--n", type=int, default=100)
    c.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("report")
    r.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()
    {"build": cmd_build, "check": cmd_check, "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
