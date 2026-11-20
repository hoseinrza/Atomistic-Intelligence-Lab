"""Truncated Gaussian radial function of Eq. (4) (Sprint 1).

    f(r) = (1/σ³) · exp[ −(r − r₀)² / σ² ] · f_c(r).

The σ⁻³ prefactor (with σ = 1.0 Å in the paper, hence prefactor 1.0) makes f
a Gaussian *density-like* weight in 3-D rather than a normalized probability.
The cutoff factor f_c of Eq. (5) forces the function (and, through it, the
descriptor sums) smoothly to zero at r_c.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from atomistic_pinn.constants import CUTOFF_D, CUTOFF_RADIUS, DESCRIPTOR_SIGMA
from atomistic_pinn.descriptors.cutoff import cutoff_function


def radial_function(
    r: NDArray[np.float64] | float,
    r0: float,
    sigma: float = DESCRIPTOR_SIGMA,
    r_c: float = CUTOFF_RADIUS,
    d: float = CUTOFF_D,
) -> NDArray[np.float64]:
    """Evaluate f(r) of Eq. (4).

    Args:
        r: Distance(s) in Å, scalar or array.
        r0: Gaussian center in Å (one of the 12 radial centers).
        sigma: Gaussian width in Å (paper: 1.0).
        r_c: Cutoff radius in Å (paper: 6.0).
        d: Truncation-range parameter in Å (paper: 1.5).

    Returns:
        Values of f with the broadcast shape of ``r`` (zero for r > r_c).
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    r_arr = np.asarray(r, dtype=np.float64)
    gaussian = np.exp(-((r_arr - r0) ** 2) / sigma**2) / sigma**3
    return gaussian * cutoff_function(r_arr, r_c=r_c, d=d)
