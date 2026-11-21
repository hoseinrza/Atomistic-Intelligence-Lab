"""Prediction helpers: restore a trained model and evaluate it (Sprint 4).

Bridges the checkpoint world (``TrainingOutcome.best_vector``) to the
evaluation world: rebuild the network of the resolved architecture, load the
flat parameter vector, and produce per-structure energies / per-atom BOP
parameters / per-atom forces on any database. The learning↔physics contract
(Sprint 3 tests) guarantees these reproduce the BOP engine exactly for the
predicted parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from atomistic_pinn.models import PINN, BaselineNN
from atomistic_pinn.training.config import ResolvedTrainingConfig
from atomistic_pinn.training.database import TrainingDatabase
from atomistic_pinn.training.objective import (
    BaselineEnergyModel,
    PINNEnergyModel,
    set_flat_params,
)


def restore_model(
    resolved: ResolvedTrainingConfig, flat_vector: NDArray[np.float64] | torch.Tensor
) -> PINNEnergyModel | BaselineEnergyModel:
    """Rebuild the configured model and load a flat parameter vector.

    The seed is irrelevant once the parameters are overwritten; the
    architecture and activation come from the resolved config so the vector
    layout matches the checkpoint that produced it.
    """
    if resolved.model.type == "pinn":
        activation = resolved.model.activation
        assert activation is not None  # resolved (§7.4)
        model: PINNEnergyModel | BaselineEnergyModel = PINNEnergyModel(PINN(activation=activation))
    elif resolved.model.type == "baseline_nn":
        activation = resolved.model.activation
        assert activation is not None  # resolved (§7.4)
        model = BaselineEnergyModel(BaselineNN(activation=activation))
    else:
        raise ValueError(f"unknown model type {resolved.model.type!r}")
    set_flat_params(model, torch.as_tensor(np.asarray(flat_vector, dtype=np.float64)))
    return model


def predicted_structure_energies(
    model: PINNEnergyModel | BaselineEnergyModel, db: TrainingDatabase
) -> NDArray[np.float64]:
    """Predicted total energies per structure, ``(S,)`` in eV (no shift)."""
    return model.structure_energies(db).detach().numpy()


def predicted_bop_parameters(
    model: PINNEnergyModel, db: TrainingDatabase
) -> list[NDArray[np.float64]]:
    """Predicted per-atom BOP parameters, one ``(N_s, 8)`` array per structure.

    Raises:
        TypeError: If called with the baseline NN (no BOP parameters).
    """
    arrays: list[NDArray[np.float64]] = []
    with torch.no_grad():
        for desc in db.descriptors:
            arrays.append(model.network(torch.as_tensor(desc, dtype=torch.float64)).numpy())
    return arrays
