"""Keep every download and cache inside the repository's ``data/`` folder.

Astropy, astroquery and lightkurve write caches to the home directory by default.
Call :func:`use_local_cache` *before* importing them so a run never touches files
outside the project and ``data/`` can be deleted to start from scratch.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / ".cache"


def use_local_cache() -> None:
    for sub in ("astropy", "lightkurve"):
        (CACHE_DIR / sub).mkdir(parents=True, exist_ok=True)
    # Astropy (and libraries built on its config system) only honour these
    # variables when "<dir>/astropy" exists, which is created above.
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
    os.environ.setdefault("XDG_CONFIG_HOME", str(CACHE_DIR))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
