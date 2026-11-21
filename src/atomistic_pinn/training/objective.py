"""Objective function of Eq. (13) and the two energy models (Sprint 3).

    ℰ = Σ_s ( E_s − E_s^DFT )²                      (unweighted, §7.11 literal)
      + τ Σ_{εκ} ( ‖w_εκ‖² + Σ_κ ‖b_κ‖² )            (all weights and biases)
      + γ Σ_η ( p_η − p̄_η )²                        (PINN only — tether)

Two energy models share this objective:

* :class:`PINNEnergyModel` — descriptors → PINN → per-atom BOP parameters →
  BOP total energy (the physics-informed route);
* :class:`BaselineEnergyModel` — descriptors → NN → per-atom energies,
  summed (the direct-regression baseline; has no BOP parameters, so the γ
  term is absent by construction).

The γ tether target ``p̄_η`` (ledger §7.13):

* ``"fixed_bop_prefit"`` — fixed 8-vector from the once-fitted fixed BOP
  (``training/fixed_bop.py``); also delivers that baseline;
* ``"self_mean"`` — the mean of the *current* predictions, treated as a
  constant within the evaluation (detached), so the term tethers each atom
  to the current population mean.

Optimization variables are one flat float64 vector of all network weights
and biases (DFP operates on flat vectors). Two gradient routes exist:
``numerical`` (resolved §7.8 decision; forward differences, one gradient =
``n_params + 1`` evaluations) and ``autograd`` (labeled fallback; exact and
much cheaper, via ``torch.func.functional_call``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from atomistic_pinn.models import PINN, BaselineNN
from atomistic_pinn.physics import total_energy
from atomistic_pinn.training.database import TrainingDatabase

_DIFFERENTIAL_STEP = math.sqrt(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class TetherConfig:
    """γ-tether settings (ledger §7.13)."""

    gamma: float
    mode: str  # "fixed_bop_prefit" | "self_mean"
    target: torch.Tensor | None = None  # (8,) for fixed_bop_prefit


class PINNEnergyModel:
    """Descriptors → PINN network → BOP parameters → BOP energies."""

    def __init__(self, network: PINN) -> None:
        self.network = network

    def structure_energies(self, db: TrainingDatabase) -> torch.Tensor:
        """Scalar total-energy tensor per structure, ``(S,)`` (eV)."""
        outputs = []
        for geometry, desc in zip(db.geometries, db.descriptors, strict=True):
            params = self.network(torch.as_tensor(desc, dtype=torch.float64))
            outputs.append(total_energy(geometry.positions, geometry.cell, geometry.pbc, params))
        return torch.stack(outputs)

    def structure_energies_functional(
        self, named_params: dict[str, torch.Tensor], db: TrainingDatabase
    ) -> torch.Tensor:
        """Same computation with parameters supplied functionally (autograd
        route through a flat vector)."""
        outputs = []
        for geometry, desc in zip(db.geometries, db.descriptors, strict=True):
            params = torch.func.functional_call(
                self.network, named_params, (torch.as_tensor(desc, dtype=torch.float64),)
            )
            outputs.append(total_energy(geometry.positions, geometry.cell, geometry.pbc, params))
        return torch.stack(outputs)

    def predicted_parameters(
        self, db: TrainingDatabase, named_params: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """All predicted per-atom parameters stacked, ``(Σ_s N_s, 8)``."""
        rows = []
        for desc in db.descriptors:
            desc_t = torch.as_tensor(desc, dtype=torch.float64)
            if named_params is None:
                rows.append(self.network(desc_t))
            else:
                rows.append(torch.func.functional_call(self.network, named_params, (desc_t,)))
        return torch.cat(rows, dim=0)

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        return self.network.named_parameters()


class BaselineEnergyModel:
    """Descriptors → NN → per-atom energies, summed per structure."""

    def __init__(self, network: BaselineNN) -> None:
        self.network = network

    def structure_energies(self, db: TrainingDatabase) -> torch.Tensor:
        outputs = []
        for desc in db.descriptors:
            atomic = self.network(torch.as_tensor(desc, dtype=torch.float64))
            outputs.append(atomic.sum())
        return torch.stack(outputs)

    def structure_energies_functional(
        self, named_params: dict[str, torch.Tensor], db: TrainingDatabase
    ) -> torch.Tensor:
        outputs = []
        for desc in db.descriptors:
            atomic = torch.func.functional_call(
                self.network, named_params, (torch.as_tensor(desc, dtype=torch.float64),)
            )
            outputs.append(atomic.sum())
        return torch.stack(outputs)

    def predicted_parameters(
        self, db: TrainingDatabase, named_params: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        """None — the baseline has no BOP parameters; γ does not apply."""
        return None

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        return self.network.named_parameters()


def _flat_to_named(
    model: PINNEnergyModel | BaselineEnergyModel, x: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Split a flat float64 vector into per-parameter tensors (functional
    forward), keeping the autograd graph connected to ``x``."""
    named: dict[str, torch.Tensor] = {}
    offset = 0
    for name, param in model.named_parameters():
        count = param.numel()
        named[name] = x[offset : offset + count].reshape(param.shape)
        offset += count
    if offset != x.numel():
        raise ValueError(f"flat vector has {x.numel()} entries, model needs {offset}")
    return named


def set_flat_params(model: PINNEnergyModel | BaselineEnergyModel, x: torch.Tensor) -> None:
    """Copy a flat vector into the module parameters (no autograd graph;
    used by the numerical route and checkpoint restore)."""
    named = _flat_to_named(model, x.to(torch.float64))
    current = dict(model.named_parameters())
    if set(current) != set(named):
        raise ValueError("flat vector does not match the model's parameters")
    with torch.no_grad():
        for name, param in current.items():
            param.copy_(named[name])


def get_flat_params(model: PINNEnergyModel | BaselineEnergyModel) -> torch.Tensor:
    """Flat float64 vector of all weights and biases (checkpoint format)."""
    return torch.nn.utils.parameters_to_vector(model.network.parameters()).detach().clone()


def _weight_regularization(model: PINNEnergyModel | BaselineEnergyModel) -> torch.Tensor:
    """τ-term numerator: Σ over all weights and biases of ‖·‖²."""
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.network.parameters():
        total = total + (parameter**2).sum()
    return total


def objective_value(
    model: PINNEnergyModel | BaselineEnergyModel,
    db: TrainingDatabase,
    x: torch.Tensor,
    *,
    tau: float,
    tether: TetherConfig | None,
) -> torch.Tensor:
    """Evaluate ℰ (Eq. 13) at flat parameter vector ``x``.

    Args:
        model: Energy model (PINN or baseline).
        db: Training database (targets shifted exactly once, at access time).
        x: Flat float64 parameter vector (may require grad for the autograd
            route).
        tau: Weight-regularization strength (§7.6; 0 disables).
        tether: γ-tether configuration (None disables; baseline uses None).

    Returns:
        Scalar float64 tensor ℰ in eV².
    """
    x = x.to(torch.float64)
    named = _flat_to_named(model, x)
    energies = model.structure_energies_functional(named, db)
    targets = torch.as_tensor(db.shifted_targets(), dtype=torch.float64)
    energy_term = ((energies - targets) ** 2).sum()
    total = energy_term + tau * _weight_regularization(model)
    if tether is not None and tether.gamma > 0.0:
        predictions = model.predicted_parameters(db, named_params=named)
        if predictions is None:
            raise ValueError(
                "the γ tether requires predicted BOP parameters; the "
                "baseline model has none (its γ is forced to 0)"
            )
        if tether.mode == "fixed_bop_prefit":
            target = tether.target
            if target is None:
                raise ValueError("fixed_bop_prefit tether requires the (8,) target")
            tether_term = ((predictions - target) ** 2).sum()
        elif tether.mode == "self_mean":
            mean = predictions.mean(dim=0).detach()  # constant within the evaluation
            tether_term = ((predictions - mean) ** 2).sum()
        else:
            raise ValueError(f"unknown tether mode {tether.mode!r}")
        total = total + tether.gamma * tether_term
    return total


def objective_function(
    model: PINNEnergyModel | BaselineEnergyModel,
    db: TrainingDatabase,
    *,
    tau: float,
    tether: TetherConfig | None,
    gradient: str,
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """Build numpy-callable objective and gradient functions for DFP.

    Args:
        model: Energy model.
        db: Training database.
        tau: Weight regularization.
        tether: γ-tether configuration (None disables).
        gradient: ``"numerical"`` (resolved §7.8 route) or ``"autograd"``
            (labeled fallback).

    Returns:
        ``(f, grad)`` — objective value and gradient over the flat vector.
    """
    n_params = sum(p.numel() for p in model.network.parameters())

    def f(x: np.ndarray) -> float:
        with torch.no_grad():
            value = objective_value(model, db, torch.tensor(x), tau=tau, tether=tether)
        return float(value)

    def grad_autograd(x: np.ndarray) -> np.ndarray:
        leaf = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        value = objective_value(model, db, leaf, tau=tau, tether=tether)
        (g,) = torch.autograd.grad(value, leaf)
        return g.numpy()

    def grad_numerical(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        grad = np.empty_like(x)
        f0 = f(x)
        for i in range(n_params):
            h = _DIFFERENTIAL_STEP * max(1.0, abs(float(x[i])))
            x_plus = x.copy()
            x_plus[i] += h
            grad[i] = (f(x_plus) - f0) / h
        return grad

    if gradient == "autograd":
        return f, grad_autograd
    if gradient == "numerical":
        return f, grad_numerical
    raise ValueError(f"unknown gradient route {gradient!r} (ledger §7.8)")
