"""Download a known planet's Kepler light curve, recover its period with BLS and plot it.

    uv run python scripts/fetch_and_plot.py                 # Kepler-10 b
    uv run python scripts/fetch_and_plot.py --target Kepler-6 --planet "Kepler-6 b"

Outputs (in docs/):
    <slug>_fold.png     light curve, BLS periodogram and folded transit
    <slug>_result.json  recovered vs published period, depth, timings, download size
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

from cazaplanetas.env import RAW_DIR, REPO_ROOT, use_local_cache

use_local_cache()  # must run before astropy / lightkurve are imported

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cazaplanetas import pipeline  # noqa: E402

# Fallback when the NASA Exoplanet Archive is unreachable (values from its
# pscomppars table; Kepler-10 b period from Dumusque et al. 2014).
KNOWN = {"Kepler-10 b": {"pl_orbper": 0.8374907, "pl_trandur": 1.81}}

# Colours: neutral ink for raw data, one accent per role (validated palette).
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#b9b8b2", "#e7e6e1"
BLUE, ORANGE, SURFACE = "#2a78d6", "#eb6834", "#fcfcfb"


def published_parameters(planet: str) -> dict:
    """Period (days) and duration (hours) from the NASA Exoplanet Archive."""
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

        table = NasaExoplanetArchive.query_criteria(
            table="pscomppars", select="pl_name,pl_orbper,pl_trandur", where=f"pl_name='{planet}'"
        )
        if len(table):
            row = table[0]
            return {
                "pl_orbper": float(np.asarray(row["pl_orbper"]).item()),
                "pl_trandur": float(np.asarray(row["pl_trandur"]).item()),
                "source": "NASA Exoplanet Archive pscomppars",
            }
    except Exception as exc:  # network down, schema change...
        print(f"warning: archive query failed ({exc}); using built-in values", file=sys.stderr)
    if planet in KNOWN:
        return {**KNOWN[planet], "source": "built-in fallback"}
    return {}


def directory_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*.fits")) / 1e6


def plot(lc, res: pipeline.BLSResult, periods, power, published: dict, title: str, out: Path) -> None:
    t = np.asarray(lc.time.value)
    ppm = (np.asarray(lc.flux.value) - 1.0) * 1e6

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })
    fig = plt.figure(figsize=(10, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.6])
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")

    # 1. Cleaned light curve over the whole mission.
    ax = fig.add_subplot(gs[0])
    ax.plot(t, ppm, ".", ms=1, color=MUTED, rasterized=True)
    lo, hi = np.percentile(ppm, [0.5, 99.5])
    ax.set_ylim(lo - 0.2 * (hi - lo), hi + 0.2 * (hi - lo))
    ax.set_xlabel("Time (BKJD = BJD - 2454833)")
    ax.set_ylabel("Flux (ppm)")
    ax.set_title(f"Detrended Kepler long-cadence light curve · {len(t):,} points", loc="left", color=INK_2, fontsize=10)

    # 2. BLS periodogram.
    ax = fig.add_subplot(gs[1])
    ax.plot(periods, power, color=BLUE, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Trial period (days)")
    ax.set_ylabel("BLS SNR")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.annotate(
        f"BLS peak {res.period:.6f} d (SDE {res.sde:.0f})",
        xy=(res.period, power.max()), xytext=(8, -4), textcoords="offset points", color=INK, va="top",
    )
    if published.get("pl_orbper"):
        ax.axvline(published["pl_orbper"], color=ORANGE, lw=1.2, ls="--", zorder=0)
        ax.text(published["pl_orbper"], 0.6, f"  published {published['pl_orbper']:.6f} d  ",
                transform=ax.get_xaxis_transform(), color=INK_2, va="center", ha="right", fontsize=9)
    ax.set_title(f"Box Least Squares search · {res.n_periods:,} trial periods in {res.seconds:.0f} s",
                 loc="left", color=INK_2, fontsize=10)

    # 3. Folded transit with the BLS box model.
    ax = fig.add_subplot(gs[2])
    window_h = max(6.0, 3 * res.duration * 24)
    x = pipeline.phase_hours(t, res.period, res.t0)
    near = np.abs(x) < window_h
    ax.plot(x[near], ppm[near], ".", ms=1.5, color=MUTED, rasterized=True, label="each 29.4-min measurement")
    bx, by, be = pipeline.binned(x[near], ppm[near], width=0.25, lo=-window_h, hi=window_h)
    ax.errorbar(bx, by, be, fmt="o", ms=4, color=BLUE, ecolor=BLUE, elinewidth=1, capsize=0,
                label="15-min bins (mean ± s.e.)")
    half = res.duration * 24 / 2
    ax.axvspan(-half, half, color=ORANGE, alpha=0.12, lw=0)
    box_x = [-window_h, -half, -half, half, half, window_h]
    box_y = [0, 0, -res.depth_ppm, -res.depth_ppm, 0, 0]
    ax.plot(box_x, box_y, color=ORANGE, lw=2, label="BLS box model")
    ax.annotate(f"transit · depth {res.depth_ppm:.0f} ppm · {res.duration * 24:.1f} h",
                xy=(half, -res.depth_ppm), xytext=(12, -14), textcoords="offset points", color=INK,
                arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8})
    spread = np.nanstd(by) if len(by) else 50
    ax.set_ylim(-res.depth_ppm - 6 * spread - 60, 6 * spread + 60)
    ax.set_xlim(-window_h, window_h)
    ax.set_xlabel("Hours from mid-transit")
    ax.set_ylabel("Flux (ppm)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK_2)
    ax.set_title(f"Folded at P = {res.period:.6f} d", loc="left", color=INK_2, fontsize=10)

    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="Kepler-10", help="star name or KIC id resolvable by MAST")
    ap.add_argument("--planet", default="Kepler-10 b", help="planet name in the NASA Exoplanet Archive")
    ap.add_argument("--pmin", type=float, default=0.5)
    ap.add_argument("--pmax", type=float, default=20.0)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", module="lightkurve")
    slug = args.planet.lower().replace(" ", "")
    t_start = time.perf_counter()

    collection = pipeline.download_kepler_long_cadence(args.target, RAW_DIR)
    t_download = time.perf_counter() - t_start
    lc = pipeline.clean(collection)
    res, periods, power = pipeline.bls_search(lc, args.pmin, args.pmax)
    published = published_parameters(args.planet)

    summary = {
        "target": args.target,
        "planet": args.planet,
        "quarters": len(collection),
        "points": len(lc),
        "download_mb_on_disk": round(directory_mb(RAW_DIR), 1),
        "download_seconds": round(t_download, 1),
        "bls": res.to_dict(),
        "published": published,
    }
    if published.get("pl_orbper"):
        summary["period_relative_error"] = abs(res.period - published["pl_orbper"]) / published["pl_orbper"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / f"{slug}_fold.png"
    plot(lc, res, periods, power, published, f"{args.planet}: transit recovered from public Kepler data", png)
    (args.out_dir / f"{slug}_result.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"saved {png.relative_to(REPO_ROOT) if png.is_relative_to(REPO_ROOT) else png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
