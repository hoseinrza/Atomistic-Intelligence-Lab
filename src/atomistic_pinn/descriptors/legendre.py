"""Legendre polynomials P_m of Eq. (3) (Sprint 1).

Evaluated with the Bonnet recurrence

    (m + 1) P_{m+1}(x) = (2m + 1) x P_m(x) − m P_{m−1}(x),

which is numerically stable on [−1, 1] for the orders used here (≤ 6). The
implementation is validated against ``scipy.special.eval_legendre`` (an
independent reference) in the unit tests.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def legendre_values(
    orders: Sequence[int],
    x: NDArray[np.float64],
) -> dict[int, NDArray[np.float64]]:
    """Evaluate P_m(x) for each requested order.

    Args:
        orders: Non-negative polynomial orders (any subset, in any order).
        x: Evaluation points, any shape; expected to lie in [−1, 1].

    Returns:
        Mapping ``order → array of P_m(x)`` with the shape of ``x``.

    Raises:
        ValueError: If ``orders`` is empty or contains negative orders.
    """
    if not orders:
        raise ValueError("at least one Legendre order is required")
    if any(m < 0 for m in orders):
        raise ValueError("Legendre orders must be non-negative")
    x_arr = np.asarray(x, dtype=np.float64)
    max_m = max(orders)
    values: dict[int, NDArray[np.float64]] = {}
    p_prev = np.ones_like(x_arr)  # P_0
    p_curr = x_arr.copy()  # P_1
    if 0 in orders:
        values[0] = np.ones_like(x_arr)
    if 1 in orders:
        values[1] = x_arr.copy()
    for m in range(2, max_m + 1):
        current = ((2 * m - 1) * x_arr * p_curr - (m - 1) * p_prev) / m
        p_prev, p_curr = p_curr, current
        if m in orders:
            values[m] = current
    return values
