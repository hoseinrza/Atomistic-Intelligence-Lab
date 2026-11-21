"""Scientific validation: differentiable descriptor twin + pipeline forces.

Independent-reference checks (Sprint 4):

1. The torch descriptor path reproduces the numpy builder (Sprint 1) on
   random periodic and cluster structures — two independently implemented
   computations of Eq. (3) must agree to round-off (standard: rtol 1e-9,
   as for the Sprint 2 physics twin).
2. Full-pipeline autograd forces agree with full-pipeline finite-difference
   forces on a static configuration — the two routes compute the same
   derivative of the same energy by independent methods.
3. Translational invariance: ΣF = 0 (exact for the analytic gradient).
"""

from __future__ import annotations

import numpy as np
import torch

from atomistic_pinn.descriptors import compute_descriptors
from atomistic_pinn.descriptors.builder import DescriptorConfig
from atomistic_pinn.descriptors.torch_builder import (
    build_descriptor_neighbors,
    compute_descriptors_differentiable,
)
from atomistic_pinn.evaluation.forces import (
    pipeline_forces_autograd,
    pipeline_forces_finite_difference,
)
from atomistic_pinn.models import PINN
from atomistic_pinn.training.objective import PINNEnergyModel


def _perturbed_fcc_cell(n_cells: int = 2, jitter: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Small perturbed periodic FCC cell (n_cells³ conventional cells)."""
    a = 4.05
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) * a
    positions = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                for b in basis:
                    positions.append([ix * a + b[0], iy * a + b[1], iz * a + b[2]])
    positions = np.asarray(positions)
    rng = np.random.default_rng(17)
    positions = positions + rng.uniform(-jitter, jitter, positions.shape)
    cell = (n_cells * a) * np.eye(3)
    return positions, cell


class TestTorchDescriptorTwin:
    def test_matches_numpy_builder_on_periodic_structure(self) -> None:
        positions, cell = _perturbed_fcc_cell()
        cfg = DescriptorConfig()
        expected = compute_descriptors(positions, cell, [True, True, True], cfg)
        neighbors = build_descriptor_neighbors(
            positions, cell, [True, True, True], cfg.cutoff_radius
        )
        got = (
            compute_descriptors_differentiable(
                torch.tensor(positions), torch.tensor(cell), neighbors, cfg
            )
            .detach()
            .numpy()
        )
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)

    def test_matches_numpy_builder_on_cluster(self) -> None:
        rng = np.random.default_rng(23)
        positions = rng.uniform(0.0, 5.0, size=(5, 3))
        cell = np.zeros((3, 3))
        cfg = DescriptorConfig()
        expected = compute_descriptors(positions, cell, [False, False, False], cfg)
        neighbors = build_descriptor_neighbors(
            positions, cell, [False, False, False], cfg.cutoff_radius
        )
        got = (
            compute_descriptors_differentiable(
                torch.tensor(positions), torch.tensor(cell), neighbors, cfg
            )
            .detach()
            .numpy()
        )
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)

    def test_descriptors_are_differentiable_wrt_positions(self) -> None:
        positions, cell = _perturbed_fcc_cell()
        cfg = DescriptorConfig()
        neighbors = build_descriptor_neighbors(
            positions, cell, [True, True, True], cfg.cutoff_radius
        )
        pos = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
        descriptors = compute_descriptors_differentiable(pos, torch.tensor(cell), neighbors, cfg)
        assert descriptors.requires_grad
        loss = descriptors.sum()
        (grad,) = torch.autograd.grad(loss, pos)
        assert grad.shape == positions.shape
        assert np.all(np.isfinite(grad.detach().numpy()))


class TestPipelineForceRoutes:
    def test_autograd_forces_match_finite_difference(self) -> None:
        positions, cell = _perturbed_fcc_cell(n_cells=1)
        model = PINNEnergyModel(PINN(seed=11))
        autograd = (
            pipeline_forces_autograd(model, positions, cell, [True, True, True]).detach().numpy()
        )
        finite = pipeline_forces_finite_difference(
            model, positions, cell, [True, True, True], step=1e-4
        )
        np.testing.assert_allclose(autograd, finite, rtol=1e-4, atol=1e-5)

    def test_autograd_forces_sum_to_zero(self) -> None:
        positions, cell = _perturbed_fcc_cell(n_cells=1)
        model = PINNEnergyModel(PINN(seed=11))
        autograd = (
            pipeline_forces_autograd(model, positions, cell, [True, True, True]).detach().numpy()
        )
        np.testing.assert_allclose(autograd.sum(axis=0), 0.0, atol=1e-10)
