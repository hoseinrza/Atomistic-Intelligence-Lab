"""Force evaluation (Sprint 2).

The paper computes forces by the **finite-difference method** (Methods);
that is the faithful default and the quantity used for the paper's force
comparisons. Autograd forces are provided as an engineering cross-check and
are verified against finite differences in the scientific tests.

Known edge case: for an atom with an *empty bond environment* (ρ_i = 0, e.g.
an isolated atom) the analytic gradient of the promotion term −σ√ρ is
singular while the true force is exactly zero. ``forces_autograd`` therefore
substitutes 0 for non-finite gradient entries (documented deviation, applies
only to those atoms); finite differences are unaffected because they never
form the derivative.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from atomistic_pinn.constants import CUTOFF_D, CUTOFF_RADIUS
from atomistic_pinn.physics.bop import total_energy


def _as_param_tensor(params: NDArray[np.float64] | torch.Tensor) -> torch.Tensor:
    if isinstance(params, torch.Tensor):
        return params.detach().to(torch.float64)
    return torch.tensor(np.asarray(params, dtype=np.float64))


def forces_finite_difference(
    positions: NDArray[np.float64],
    cell: NDArray[np.float64],
    pbc: list[bool] | tuple[bool, ...],
    params: NDArray[np.float64] | torch.Tensor,
    cutoff: float = CUTOFF_RADIUS,
    d: float = CUTOFF_D,
    step: float = 1e-4,
) -> NDArray[np.float64]:
    """Forces by central finite differences of the total energy — the
    paper's method (F_i = −∂E/∂r_i).

    Args:
        positions: ``(N, 3)`` positions in Å (numpy).
        cell: ``(3, 3)`` lattice rows in Å.
        pbc: Three periodic flags.
        params: ``(N, 8)`` per-atom BOP parameters.
        cutoff: Cutoff radius in Å.
        d: Truncation-range parameter in Å.
        step: Displacement in Å (central difference).

    Returns:
        ``(N, 3)`` forces in eV/Å. Neighbor lists are rebuilt for every
        displaced configuration, so the result is exact for the given
        potential form (up to the FD truncation error ~h²·E‴).
    """
    pos = np.array(positions, dtype=np.float64)
    cell_t = torch.tensor(np.asarray(cell, dtype=np.float64))
    params_t = _as_param_tensor(params)

    def energy(p: NDArray[np.float64]) -> float:
        return float(
            total_energy(
                torch.tensor(p, dtype=torch.float64), cell_t, pbc, params_t, cutoff, d
            ).detach()
        )

    n_atoms = pos.shape[0]
    forces = np.zeros((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        for c in range(3):
            plus = pos.copy()
            minus = pos.copy()
            plus[i, c] += step
            minus[i, c] -= step
            forces[i, c] = -(energy(plus) - energy(minus)) / (2.0 * step)
    return forces


def forces_autograd(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: list[bool] | tuple[bool, ...],
    params: torch.Tensor,
    cutoff: float = CUTOFF_RADIUS,
    d: float = CUTOFF_D,
) -> torch.Tensor:
    """Forces through reverse-mode autograd, ``(N, 3)`` in eV/Å.

    Engineering improvement over the paper's finite differences; agreement
    with :func:`forces_finite_difference` is enforced by a scientific test.
    Non-finite gradient entries (empty bond environments, see module
    docstring) are replaced by 0 — the physically correct force for those
    atoms. The leaf tensor is upcast to float64 so gradients carry full
    precision regardless of the caller's dtype.
    """
    pos = positions.detach().to(torch.float64).requires_grad_(True)
    energy = total_energy(pos, cell, pbc, params, cutoff, d)
    (grad,) = torch.autograd.grad(energy, pos)
    return torch.nan_to_num(-grad, nan=0.0, posinf=0.0, neginf=0.0)
