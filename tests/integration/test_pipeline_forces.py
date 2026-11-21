"""Integration tests: full-pipeline finite-difference forces (Sprint 4).

Independent checks:

1. **Constant-parameter contract** — a PINN whose output layer produces
   constant parameters must give pipeline FD forces identical to the
   physics-layer FD forces (`physics.forces`) computed with those same
   parameters: two independently implemented routes, same numbers.
2. **Constant-energy baseline** — the baseline NN with zeroed output weights
   has position-independent energy, so its pipeline FD forces vanish.
3. **Translational invariance** — ΣF ≈ 0 for a periodic configuration
   (FD truncation sets the tolerance).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from atomistic_pinn.evaluation.forces import (
    dataset_force_metrics,
    pipeline_forces_finite_difference,
)
from atomistic_pinn.models import PINN, BaselineNN
from atomistic_pinn.physics.forces import forces_finite_difference
from atomistic_pinn.training.objective import BaselineEnergyModel, PINNEnergyModel

CELL_8 = 8.0 * np.eye(3)
PBC = [True, True, True]
CLUSTER_PBC = [False, False, False]


def _pinn_constant_parameters() -> tuple[PINNEnergyModel, np.ndarray]:
    """PINN whose output layer ignores its input: params = final-layer bias."""
    network = PINN(seed=3)
    with torch.no_grad():
        network.network[-1].weight.zero_()
    bias = network.network[-1].bias.detach().numpy()  # (8,)
    return PINNEnergyModel(network), bias


def _baseline_constant_energy() -> BaselineEnergyModel:
    network = BaselineNN(seed=3)
    with torch.no_grad():
        network.network[-1].weight.zero_()
        network.network[-1].bias.fill_(-0.4)
    return BaselineEnergyModel(network)


class TestPipelineFiniteDifferenceForces:
    def test_constant_parameter_pinn_matches_physics_layer_fd(self) -> None:
        model, bias = _pinn_constant_parameters()
        positions = np.array([[0.0, 0.0, 0.0], [1.4, 0.1, 0.05], [0.2, 1.35, 0.0]])
        cell = np.zeros((3, 3))
        pipeline = pipeline_forces_finite_difference(model, positions, cell, CLUSTER_PBC, step=1e-4)
        params = np.broadcast_to(bias, (positions.shape[0], 8)).copy()
        physics = forces_finite_difference(positions, cell, CLUSTER_PBC, params, step=1e-4)
        np.testing.assert_allclose(pipeline, physics, rtol=1e-8, atol=1e-10)

    def test_constant_energy_baseline_gives_zero_forces(self) -> None:
        model = _baseline_constant_energy()
        positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.3, 0.4]])
        forces = pipeline_forces_finite_difference(model, positions, CELL_8, PBC)
        np.testing.assert_allclose(forces, 0.0, atol=1e-10)

    def test_sum_of_forces_vanishes_on_periodic_cell(self) -> None:
        model, _ = _pinn_constant_parameters()
        positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.3, 0.4], [4.1, 3.2, 1.1]])
        forces = pipeline_forces_finite_difference(model, positions, CELL_8, PBC)
        assert np.all(np.abs(forces.sum(axis=0)) < 1e-4)


class TestDatasetForceMetrics:
    def test_metrics_over_labeled_structures(self) -> None:
        # Use the shared conftest fixture indirectly: build two labeled
        # structures directly (positions + stored DFT forces).
        from atomistic_pinn.data.schema import AtomicStructure

        rng = np.random.default_rng(5)
        structures = []
        for k in range(2):
            positions = rng.uniform(0.0, 4.0, size=(3, 3))
            structures.append(
                AtomicStructure(
                    id=f"f{k}",
                    numbers=[13, 13, 13],
                    positions=positions.tolist(),
                    cell=CELL_8.tolist(),
                    pbc=PBC,
                    energy=-8.0,
                    forces=rng.normal(0.0, 0.1, size=(3, 3)).tolist(),
                )
            )
        model, _ = _pinn_constant_parameters()
        metrics = dataset_force_metrics(model, structures)
        assert metrics.n_structures == 2
        assert metrics.n_atoms == 6
        assert np.isfinite(metrics.rmse_ev_per_angstrom)
        assert metrics.mae_ev_per_angstrom > 0.0

    def test_requires_labeled_forces(self) -> None:
        from atomistic_pinn.data.schema import AtomicStructure

        structure = AtomicStructure(
            id="s",
            numbers=[13],
            positions=[[0.0, 0.0, 0.0]],
            cell=CELL_8.tolist(),
            pbc=PBC,
            energy=-3.0,
        )
        model, _ = _pinn_constant_parameters()
        with pytest.raises(ValueError, match="no structures carry force labels"):
            dataset_force_metrics(model, [structure])
