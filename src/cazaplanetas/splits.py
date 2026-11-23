"""Partitions by star: every signal of a star goes to the same part.

Astronet's published split is by signal (TCE), so 56 % of its test signals sit on
stars that also have signals in training (multi-planet systems, the primary and
secondary eclipses of a binary...). A model can then score well partly by
recognising the star. From milestone 3 on, every split in this project is by star.
"""

from __future__ import annotations

import numpy as np


def split_by_star(kepid: np.ndarray, fractions: tuple[float, ...] = (0.8, 0.1, 0.1), seed: int = 0) -> np.ndarray:
    """Assign each signal to a part (0, 1, ...) so that all signals of a star share it.

    Stars are shuffled with ``seed`` and dealt greedily to the part that is furthest
    below its target share of *signals*, so the parts match ``fractions`` closely even
    when some stars have many signals. Returns an int array aligned with ``kepid``."""
    kepid = np.asarray(kepid)
    frac = np.asarray(fractions, dtype=float)
    frac = frac / frac.sum()
    stars, inverse, counts = np.unique(kepid, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(stars))
    target = frac * len(kepid)
    filled = np.zeros(len(frac))
    part_of_star = np.empty(len(stars), dtype=np.int64)
    for s in order:
        p = int(np.argmax((target - filled) / target))
        part_of_star[s] = p
        filled[p] += counts[s]
    return part_of_star[inverse]


def star_folds(kepid: np.ndarray, k: int = 10, seed: int = 0) -> np.ndarray:
    """``k`` folds of (almost) equal size by star, for cross-validation."""
    return split_by_star(kepid, (1.0,) * k, seed)


def shared_stars(kepid_a: np.ndarray, kepid_b: np.ndarray) -> int:
    """Number of stars present in both sets (0 means no leakage)."""
    return len(np.intersect1d(np.unique(kepid_a), np.unique(kepid_b)))
