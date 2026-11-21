"""Fixed-BOP pre-fit (Sprint 3, ledger §7.13 option (a)).

Fits the eight shared parameters of the *fixed* BOP baseline (paper Fig. 8:
same functional form, parameters fitted once and fixed) by minimizing the
energy term of Eq. (13) over the database with DFP. One computation, two
uses: the fitted 8-vector is the γ-tether target ``p̄_η``, and the fit is the
fixed-BOP baseline for Sprint 3/4.

NOT the paper's own fit — its values are unpublished (ledger §7.15); any
result is reported as a *re-fit on the same functional form*.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from atomistic_pinn.constants import BOP_PARAMETER_NAMES
from atomistic_pinn.physics import atomic_energies
from atomistic_pinn.training.database import TrainingDatabase
from atomistic_pinn.training.dfp import DFPResult, dfp_minimize

PHYSICAL_INIT = np.array([1.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 1.0])
"""Physical initial guess in BOP_PARAMETER_NAMES order (synthetic fixture
convention; A·α < B·β so repulsion dominates at short range)."""


@dataclass
class FixedBOPFit:
    """Result of the fixed-BOP pre-fit / baseline fit.

    Attributes:
        parameters: ``(8,)`` fitted values in BOP_PARAMETER_NAMES order.
        objective: Final Σ_s (E_s − E_s^DFT)² (eV²).
        result: Full DFP record of the best restart.
    """

    parameters: np.ndarray
    objective: float
    result: DFPResult


def fixed_bop_objective(db: TrainingDatabase, x: np.ndarray) -> float:
    """Energy term of Eq. (13) with uniform per-atom parameters ``x`` (8,).

    No τ term (no network weights) and no γ term (the tether target *is*
    this fit's output).
    """
    values = np.asarray(x, dtype=np.float64)
    if values.size != len(BOP_PARAMETER_NAMES):
        raise ValueError(
            f"fixed-BOP parameter vector must have {len(BOP_PARAMETER_NAMES)} "
            f"entries, got {values.size}"
        )
    energies = _energies(db, values)
    targets = db.shifted_targets()
    return float(np.sum((energies - targets) ** 2))


def _energies(db: TrainingDatabase, values: np.ndarray) -> np.ndarray:
    energies = np.empty(db.n_structures)
    with torch.no_grad():
        for k, geometry in enumerate(db.geometries):
            per_atom = torch.tensor(
                np.broadcast_to(values, (geometry.n_atoms, len(values))).copy(),
                dtype=torch.float64,
            )
            per_atom_energies = atomic_energies(
                geometry.positions, geometry.cell, geometry.pbc, per_atom
            )
            energies[k] = float(per_atom_energies.sum())
    return energies


def _make_objective(
    db: TrainingDatabase, gradient: str
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """Build numpy-callable objective/gradient pair for the 8-parameter fit.

    The energy term is differentiable in torch (autograd route) or evaluated
    under ``no_grad`` for the numerical route (resolved §7.8).
    """
    targets = torch.tensor(db.shifted_targets(), dtype=torch.float64)

    def energies_tensor(x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for geometry in db.geometries:
            per_atom = x.unsqueeze(0).expand(geometry.n_atoms, -1)
            outputs.append(
                atomic_energies(geometry.positions, geometry.cell, geometry.pbc, per_atom).sum()
            )
        return torch.stack(outputs)

    def f(x: np.ndarray) -> float:
        with torch.no_grad():
            values = energies_tensor(torch.tensor(x, dtype=torch.float64))
        return float(((values - targets) ** 2).sum())

    def grad(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if gradient == "autograd":
            # Streaming accumulation: each structure's BOP graph is built
            # and immediately backward-ed (peak memory = one structure's
            # intermediates; the stacked variant retained every structure's
            # graph simultaneously and exhausted memory on real datasets).
            leaf = torch.tensor(x, dtype=torch.float64, requires_grad=True)
            for k, geometry in enumerate(db.geometries):
                per_atom = leaf.unsqueeze(0).expand(geometry.n_atoms, -1)
                energy = atomic_energies(
                    geometry.positions, geometry.cell, geometry.pbc, per_atom
                ).sum()
                ((energy - targets[k]) ** 2).backward()
            assert leaf.grad is not None  # at least one backward ran
            return leaf.grad.numpy()
        eps = np.sqrt(np.finfo(np.float64).eps)
        grad_values = np.empty_like(x)
        f0 = f(x)
        for i in range(x.size):
            h = eps * max(1.0, abs(float(x[i])))
            x_plus = x.copy()
            x_plus[i] += h
            grad_values[i] = (f(x_plus) - f0) / h
        return grad_values

    return f, grad


def fit_fixed_bop(
    db: TrainingDatabase,
    *,
    gradient: str = "numerical",
    n_restarts: int = 3,
    master_seed: int = 0,
    max_iterations: int = 200,
    gradient_tolerance: float = 1e-10,
) -> FixedBOPFit:
    """Fit the eight shared BOP parameters (best of seeded restarts).

    Args:
        db: Training database.
        gradient: ``"numerical"`` (resolved §7.8 route; 9 evaluations per
            gradient — cheap for 8 parameters) or ``"autograd"``.
        n_restarts: Seeded restarts (paper: "several times").
        master_seed: Master seed; restart seeds derive from it.
        max_iterations: DFP iteration cap per restart.
        gradient_tolerance: DFP gradient stop.

    Returns:
        :class:`FixedBOPFit` with the best-restart parameters.
    """
    f, grad = _make_objective(db, gradient)

    seed_sequence = np.random.SeedSequence(master_seed)
    best: FixedBOPFit | None = None
    for child in seed_sequence.spawn(n_restarts):
        rng = np.random.default_rng(child)
        # Small seeded jitter of the physical init keeps restarts distinct.
        x0 = PHYSICAL_INIT * (1.0 + 0.05 * rng.standard_normal(8))
        result = dfp_minimize(
            f,
            x0,
            grad,
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
        )
        if best is None or result.f < best.objective:
            best = FixedBOPFit(
                parameters=result.x.copy(),
                objective=result.f,
                result=result,
            )
    assert best is not None  # n_restarts ≥ 1
    return best
