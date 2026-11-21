"""ASE calculator for the PINN potential (Sprint 4, phase 4.3).

Wires energies and forces into the standard ASE workflow
(``atoms.get_potential_energy()`` / ``atoms.get_forces()``), exactly as
`docs/md_plan.md` §2–3 specifies:

* **Energy** — the full pipeline (descriptors → network → BOP) evaluated at
  the current positions.
* **Forces (autograd route — the MD force route)** — the gradient of the
  total energy through the whole chain, with the neighbor metadata built
  once and **held fixed** while the gradient flows; the C³-smooth cutoff is
  what makes this well-defined across ``r_c``.
* **Rebuild-on-demand** — the fixed lists are rebuilt when any atom has
  moved more than ``rebuild_threshold`` from the positions they were built
  at (the calculator tracks the maximum displacement between calls). The
  threshold is a **labeled engineering decision** (default 0.2 Å — small
  against the 1.5 Å truncation range; an MD-stability tuning knob).
* **Finite-difference route** — the paper-faithful method, available for
  static verification (``force_method="finite_difference"``); agreement of
  the two routes on static configurations is itself a scientific test
  (Sprint 4 gate 2), so the autograd deviation is validated, and training
  remains energy-only regardless (paper methodology unaffected).

Fidelity note: the calculator serves whatever trained checkpoint it is
given; results produced with it inherit the checkpoint dataset's fidelity
level (`docs/reproduction_protocol.md`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from atomistic_pinn.constants import CUTOFF_D, CUTOFF_RADIUS
from atomistic_pinn.descriptors.torch_builder import (
    DescriptorNeighbors,
    build_descriptor_neighbors,
    compute_descriptors_differentiable,
)
from atomistic_pinn.models import PINN
from atomistic_pinn.physics.bop import total_energy
from atomistic_pinn.physics.neighbors import BOPNeighbors, build_bop_neighbors
from atomistic_pinn.training.config import ResolvedTrainingConfig, load_training_config
from atomistic_pinn.training.objective import PINNEnergyModel, set_flat_params

# Labeled engineering decision (docs/md_plan.md §3): rebuild the fixed
# neighbor metadata when any atom has moved further than this from the
# positions the lists were built at.
DEFAULT_REBUILD_THRESHOLD_ANGSTROM = 0.2


class PINNCalculator(Calculator):
    """ASE calculator exposing the trained PINN potential.

    Args:
        resolved: Ledger-resolved training configuration (architecture,
            activation, descriptor settings).
        flat_params: Flat network parameter vector (checkpoint
            ``flat_params``).
        force_method: ``"autograd"`` (MD route, default) or
            ``"finite_difference"`` (paper's static-verification route).
        rebuild_threshold: Maximum atom displacement (Å) tolerated before
            the fixed neighbor metadata is rebuilt.
    """

    # ASE contract: mutable-looking class attribute is the framework pattern
    # (BaseCalculator declares it as an instance variable); keep it plain.
    implemented_properties = ["energy", "free_energy", "forces"]  # noqa: RUF012

    def __init__(
        self,
        resolved: ResolvedTrainingConfig,
        flat_params: np.ndarray | torch.Tensor,
        *,
        force_method: str = "autograd",
        rebuild_threshold: float = DEFAULT_REBUILD_THRESHOLD_ANGSTROM,
    ) -> None:
        super().__init__()
        if force_method not in ("autograd", "finite_difference"):
            raise ValueError(
                f"force_method must be 'autograd' or 'finite_difference', "
                f"got {force_method!r} (docs/md_plan.md §2)"
            )
        if rebuild_threshold <= 0.0:
            raise ValueError("rebuild_threshold must be positive (docs/md_plan.md §3)")
        self._resolved = resolved
        self._descriptor_config = resolved.descriptors.to_descriptor_config()
        self._model = PINNEnergyModel(PINN(activation=resolved.model.activation or "tanh"))
        set_flat_params(self._model, torch.as_tensor(np.asarray(flat_params, dtype=np.float64)))
        self._force_method = force_method
        self._rebuild_threshold = float(rebuild_threshold)
        self._cached_descriptor_neighbors: DescriptorNeighbors | None = None
        self._cached_bop_neighbors: BOPNeighbors | None = None
        self._positions_built_at: np.ndarray | None = None
        self._cell: np.ndarray | None = None
        self._pbc: tuple[bool, bool, bool] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config_path: str | Path,
        **kwargs: object,
    ) -> PINNCalculator:
        """Build the calculator from a training checkpoint + its config."""
        from atomistic_pinn.training.checkpoints import load_checkpoint

        payload = load_checkpoint(checkpoint_path)
        resolved = load_training_config(config_path)
        if payload.get("model_kind") != "pinn":
            raise ValueError(
                f"checkpoint model_kind {payload.get('model_kind')!r} is not "
                "'pinn'; the calculator exposes the PINN potential (the "
                "baseline NN has no physics layer)"
            )
        return cls(
            resolved,
            payload["flat_params"].numpy(),
            **kwargs,  # type: ignore[arg-type]
        )

    # -- list management ---------------------------------------------------

    def _ensure_neighbor_metadata(
        self, positions: np.ndarray, cell: np.ndarray, pbc: tuple[bool, bool, bool]
    ) -> None:
        """Rebuild the fixed lists if any atom moved past the threshold."""
        cached = (
            self._cached_descriptor_neighbors is not None
            and self._cached_bop_neighbors is not None
            and self._positions_built_at is not None
            and positions.shape == self._positions_built_at.shape
        )
        if cached:
            assert self._positions_built_at is not None
            displacement = float(np.linalg.norm(positions - self._positions_built_at, axis=1).max())
            if displacement <= self._rebuild_threshold:
                return
        self._cached_descriptor_neighbors = build_descriptor_neighbors(
            positions, cell, list(pbc), self._descriptor_config.cutoff_radius
        )
        self._cached_bop_neighbors = build_bop_neighbors(positions, cell, list(pbc), CUTOFF_RADIUS)
        self._positions_built_at = positions.copy()

    # -- ASE contract ------------------------------------------------------

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """ASE entry point: compute energy and forces for ``atoms``."""
        super().calculate(atoms, properties or ["energy"], system_changes)
        assert atoms is not None
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        cell = np.asarray(atoms.get_cell(), dtype=np.float64)
        pbc = (bool(atoms.get_pbc()[0]), bool(atoms.get_pbc()[1]), bool(atoms.get_pbc()[2]))
        self._cell = cell
        self._pbc = pbc

        energy = self._pipeline_energy(positions, cell, pbc)
        self.results["energy"] = float(energy)
        self.results["free_energy"] = float(energy)
        if self._force_method == "autograd":
            forces = self._pipeline_forces_autograd(positions, cell, pbc)
        else:
            forces = self._pipeline_forces_finite_difference(positions, cell, pbc)
        self.results["forces"] = forces

    # -- pipeline pieces ---------------------------------------------------

    def _pipeline_energy(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        pbc: tuple[bool, bool, bool],
    ) -> float:
        pos = torch.as_tensor(positions, dtype=torch.float64)
        cell_t = torch.as_tensor(cell, dtype=torch.float64)
        self._ensure_neighbor_metadata(positions, cell, pbc)
        assert self._cached_descriptor_neighbors is not None
        assert self._cached_bop_neighbors is not None
        descriptors = compute_descriptors_differentiable(
            pos.detach(), cell_t, self._cached_descriptor_neighbors, self._descriptor_config
        )
        params = self._model.network(descriptors)
        energy = total_energy(
            pos,
            cell_t,
            pbc,
            params.detach(),
            cutoff=CUTOFF_RADIUS,
            d=CUTOFF_D,
            neighbors=self._cached_bop_neighbors,
        )
        return float(energy.detach())

    def _pipeline_forces_autograd(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        pbc: tuple[bool, bool, bool],
    ) -> np.ndarray:
        self._ensure_neighbor_metadata(positions, cell, pbc)
        assert self._cached_descriptor_neighbors is not None
        assert self._cached_bop_neighbors is not None
        pos = torch.as_tensor(positions, dtype=torch.float64).requires_grad_(True)
        cell_t = torch.as_tensor(cell, dtype=torch.float64)
        descriptors = compute_descriptors_differentiable(
            pos, cell_t, self._cached_descriptor_neighbors, self._descriptor_config
        )
        params = self._model.network(descriptors)
        energy = total_energy(
            pos,
            cell_t,
            pbc,
            params,
            cutoff=CUTOFF_RADIUS,
            d=CUTOFF_D,
            neighbors=self._cached_bop_neighbors,
        )
        (grad,) = torch.autograd.grad(energy, pos)
        forces = torch.nan_to_num(-grad, nan=0.0, posinf=0.0, neginf=0.0)
        return forces.detach().numpy()

    def _pipeline_forces_finite_difference(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        pbc: tuple[bool, bool, bool],
        step: float = 1e-4,
    ) -> np.ndarray:
        from atomistic_pinn.evaluation.forces import pipeline_forces_finite_difference

        return pipeline_forces_finite_difference(
            self._model, positions, cell, list(pbc), self._descriptor_config, step=step
        )
