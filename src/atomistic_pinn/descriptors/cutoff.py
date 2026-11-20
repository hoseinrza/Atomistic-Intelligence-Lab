"""Cutoff function of Eq. (5) (Sprint 1).

    f_c(r) = (r − r_c)⁴ / (d⁴ + (r − r_c)⁴)   for r ≤ r_c, else 0.

The paper states that this function "and its derivatives up to the third go
to zero at a cutoff distance r_c". With x = r − r_c ≤ 0 the closed forms are

    f_c   = x⁴ / (d⁴ + x⁴)
    f_c′  = 4 x³ d⁴ / (d⁴ + x⁴)²
    f_c″  = 4 d⁴ x² (3 d⁴ − 5 x⁴) / (d⁴ + x⁴)³
    f_c‴  = 4 d⁴ x (6 d⁸ − 60 d⁴ x⁴ + 30 x⁸) / (d⁴ + x⁴)⁴

all of which vanish as x → 0⁻ (verified analytically and numerically in
tests). With the paper values r_c = 6 Å, d = 1.5 Å the function is already
≈ 0.996 at r = 0 and decays smoothly to 0 at r_c.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from atomistic_pinn.constants import CUTOFF_D, CUTOFF_RADIUS


def cutoff_function(
    r: NDArray[np.float64] | float,
    r_c: float = CUTOFF_RADIUS,
    d: float = CUTOFF_D,
) -> NDArray[np.float64]:
    """Evaluate f_c(r) of Eq. (5).

    Args:
        r: Distance(s) in Å, scalar or array.
        r_c: Cutoff radius in Å (paper: 6.0).
        d: Truncation-range parameter in Å (paper: 1.5).

    Returns:
        Values of f_c with the broadcast shape of ``r`` (zero for r > r_c).
    """
    r_arr = np.asarray(r, dtype=np.float64)
    x = r_arr - r_c
    inside = x**4 / (d**4 + x**4)
    return np.where(x <= 0.0, inside, 0.0)
