"""Full-pipeline finite-difference forces (Sprint 4, phase 4.1).

The paper computes forces by **finite differences** (Methods) — of the
*whole* potential. For the PINN that means the parameters themselves depend
on the local environment, so a faithful finite-difference force must
re-evaluate the complete pipeline at every displaced configuration:

    positions → descriptors (Eq. 3–5) → network → p_i (or E_i) → total energy

This is distinct from the physics-layer FD of ``physics.forces`` (fixed
parameters), which is exact for the fixed-BOP baseline and serves as the
independent cross-check for the constant-parameter contract test here.

Finite differences rebuild the neighbor list and descriptors at every
displaced configuration, so no stale-list risk exists on this route; the
C³-smooth cutoff keeps the energy function well-defined across the cutoff.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from atomistic_pinn.constants import CUTOFF_D, CUTOFF_RADIUS
from atomistic_pinn.data.schema import AtomicStructure
from atomistic_pinn.descriptors import DescriptorConfig, compute_descriptors
from atomistic_pinn.descriptors.torch_builder import (
    build_descriptor_neighbors,
    compute_descriptors_differentiable,
)
from atomistic_pinn.evaluation.metrics import ForceMetrics, force_metrics
from atomistic_pinn.physics.bop import total_energy
from atomistic_pinn.physics.neighbors import build_bop_neighbors
from atomistic_pinn.training.objective import BaselineEnergyModel, PINNEnergyModel


def _pipeline_total_energy(
    model: PINNEnergyModel | BaselineEnergyModel,
    positions: NDArray[np.float64],
    cell: NDArray[np.float64],
    pbc: list[bool] | tuple[bool, ...],
    descriptor_config: DescriptorConfig,
) -> float:
    """Total energy of the full pipeline at arbitrary positions (eV)."""
    descriptors = compute_descriptors(
        positions, cell, np.asarray(pbc, dtype=bool), descriptor_config
    )
    inputs = torch.as_tensor(descriptors, dtype=torch.float64)
    positions_t = torch.as_tensor(np.asarray(positions, dtype=np.float64))
    cell_t = torch.as_tensor(np.asarray(cell, dtype=np.float64))
    if isinstance(model, PINNEnergyModel):
        params = model.network(inputs)
        energy = total_energy(positions_t, cell_t, pbc, params, cutoff=CUTOFF_RADIUS, d=CUTOFF_D)
    else:
        atomic = model.network(inputs)
        energy = atomic.sum()
    return float(energy.detach())


def pipeline_forces_finite_difference(
    model: PINNEnergyModel | BaselineEnergyModel,
    positions: NDArray[np.float64],
    cell: NDArray[np.float64],
    pbc: list[bool] | tuple[bool, ...],
    descriptor_config: DescriptorConfig | None = None,
    step: float = 1e-4,
) -> NDArray[np.float64]:
    """Forces of the full pipeline by central finite differences — the
    paper's method, ``(N, 3)`` in eV/Å.

    One force set costs ``6·N`` pipeline evaluations (descriptors + network
    + energy each), so this route is reserved for *static* evaluation
    configurations, exactly as in the paper (`docs/md_plan.md` §2).
    """
    cfg = descriptor_config or DescriptorConfig()
    pos = np.array(positions, dtype=np.float64)
    n_atoms = pos.shape[0]
    forces = np.zeros((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        for c in range(3):
            plus = pos.copy()
            minus = pos.copy()
            plus[i, c] += step
            minus[i, c] -= step
            e_plus = _pipeline_total_energy(model, plus, cell, pbc, cfg)
            e_minus = _pipeline_total_energy(model, minus, cell, pbc, cfg)
            forces[i, c] = -(e_plus - e_minus) / (2.0 * step)
    return forces


def dataset_force_metrics(
    model: PINNEnergyModel | BaselineEnergyModel,
    structures: list[AtomicStructure],
    descriptor_config: DescriptorConfig | None = None,
    step: float = 1e-4,
) -> ForceMetrics:
    """Pipeline FD forces vs. the stored DFT forces over labeled structures.

    Forces are evaluated post-training only (paper methodology) and compared
    against the dataset's reference forces where present; structures without
    force labels are skipped. All pipeline forces use the paper-faithful
    finite-difference route.
    """
    labeled = [s for s in structures if s.forces is not None]
    if not labeled:
        raise ValueError(
            "no structures carry force labels; force evaluation requires a "
            "labeled dataset (contract: docs/data_format.md — forces are "
            "optional and never trained on)"
        )
    predicted_blocks: list[NDArray[np.float64]] = []
    reference_blocks: list[NDArray[np.float64]] = []
    for structure in labeled:
        forces_reference = structure.forces_array
        assert forces_reference is not None  # narrowed by the labeled filter
        predicted_blocks.append(
            pipeline_forces_finite_difference(
                model,
                structure.positions_array,
                structure.cell_array,
                structure.pbc,
                descriptor_config,
                step=step,
            )
        )
        reference_blocks.append(forces_reference)
    predicted = np.concatenate(predicted_blocks, axis=0)
    reference = np.concatenate(reference_blocks, axis=0)
    return force_metrics(predicted, reference, n_structures=len(labeled))


def pipeline_forces_autograd(
    model: PINNEnergyModel,
    positions: NDArray[np.float64] | torch.Tensor,
    cell: NDArray[np.float64] | torch.Tensor,
    pbc: list[bool] | tuple[bool, ...],
    descriptor_config: DescriptorConfig | None = None,
) -> torch.Tensor:
    """Full-pipeline forces by reverse-mode autograd, ``(N, 3)`` in eV/Å.

    The gradient of the total energy flows through the *whole* chain —
    descriptors (torch twin, fixed padded neighbor list), network, BOP —
    back to the positions. This is the engineering route used by the ASE
    calculator for dynamics (`docs/md_plan.md` §2); agreement with
    :func:`pipeline_forces_finite_difference` is a scientific test.
    """
    cfg = descriptor_config or DescriptorConfig()
    positions_np = np.asarray(
        positions.detach().numpy() if isinstance(positions, torch.Tensor) else positions,
        dtype=np.float64,
    )
    cell_np = np.asarray(cell.detach().numpy() if isinstance(cell, torch.Tensor) else cell)
    # Fixed lists: descriptor slots at r_c, BOP screening slots at 2·r_c.
    descriptor_neighbors = build_descriptor_neighbors(positions_np, cell_np, pbc, cfg.cutoff_radius)
    bop_neighbors = build_bop_neighbors(positions_np, cell_np, pbc, CUTOFF_RADIUS)

    pos = torch.as_tensor(positions_np, dtype=torch.float64).requires_grad_(True)
    cell_t = torch.as_tensor(cell_np, dtype=torch.float64)
    descriptors = compute_descriptors_differentiable(pos, cell_t, descriptor_neighbors, cfg)
    params = model.network(descriptors)
    energy = total_energy(
        pos, cell_t, pbc, params, cutoff=CUTOFF_RADIUS, d=CUTOFF_D, neighbors=bop_neighbors
    )
    (grad,) = torch.autograd.grad(energy, pos)
    return torch.nan_to_num(-grad, nan=0.0, posinf=0.0, neginf=0.0)
