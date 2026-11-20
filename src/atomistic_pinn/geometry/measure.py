"""Bond-angle utilities shared by descriptors and the BOP layer (Sprint 1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cosine_matrix(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cosine of the angle between every ordered pair of vectors.

    Args:
        vectors: Array of shape (m, 3) — e.g. the distance vectors of one
            atom's neighbors from :class:`~atomistic_pinn.geometry.neighbor_list.NeighborList`.

    Returns:
        Array of shape (m, m) with ``C[j, k] = cos θ_jk`` where θ_jk is the
        angle between ``vectors[j]`` and ``vectors[k]``. Zero-norm vectors
        yield zeros (degenerate input; genuine neighbor vectors never
        vanish because the self pair is excluded). Values are clipped to
        [−1, 1] against round-off before use in Legendre polynomials.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    dots = vectors @ vectors.T
    denom = np.outer(norms, norms)
    safe = np.where(denom > 0.0, denom, 1.0)
    cos = np.where(denom > 0.0, dots / safe, 0.0)
    return np.clip(cos, -1.0, 1.0)
