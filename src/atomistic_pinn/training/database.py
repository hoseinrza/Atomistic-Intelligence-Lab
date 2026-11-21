"""Training database: cached geometry + descriptors + shifted targets.

Holds everything the objective (Eq. 13) needs, computed once per training
run (descriptors depend only on the fixed structures; neighbor lists are
rebuilt per evaluation by the physics layer, per the Sprint 3 plan's
engineering note).

Energy-shift discipline (double-shift protection): raw energies are stored
as produced by the data source; :meth:`shifted_targets` applies the paper's
+0.38446 eV/atom shift *at call time*, only when the dataset declares
``energy_convention = "raw_dft"``. A dataset already marked ``"shifted"``
is passed through unchanged. Stored data is never mutated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch

from atomistic_pinn.constants import CUTOFF_RADIUS, ENERGY_SHIFT_EV_PER_ATOM
from atomistic_pinn.data.schema import Dataset
from atomistic_pinn.descriptors import DescriptorConfig, compute_structure_descriptors
from atomistic_pinn.exceptions import DataContractError
from atomistic_pinn.physics.neighbors import BOPNeighbors, build_bop_neighbors


@dataclass(frozen=True)
class StructureGeometry:
    """Per-structure torch geometry cache (float64)."""

    structure_id: str
    positions: torch.Tensor
    cell: torch.Tensor
    pbc: tuple[bool, bool, bool]
    n_atoms: int


class TrainingDatabase:
    """Precomputed training material for one fixed dataset.

    Attributes:
        geometries: per-structure cached torch geometry (descriptor order).
        descriptors: per-structure ``(N_s, 60)`` descriptor arrays.
        raw_energies: ``(S,)`` stored DFT energies (raw or shifted per the
            dataset's declared convention).
        energy_convention: convention declared by the dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        descriptor_config: DescriptorConfig | None = None,
    ) -> None:
        self._dataset = dataset
        self.descriptor_config = descriptor_config or DescriptorConfig()
        self.geometries: list[StructureGeometry] = []
        self.descriptors: list[np.ndarray] = []
        raw: list[float] = []
        for structure in dataset.structures:
            if structure.energy is None:
                raise DataContractError(
                    f"structure {structure.id!r} has no energy label; the "
                    "training database requires energies for every structure"
                )
            self.geometries.append(
                StructureGeometry(
                    structure_id=structure.id,
                    positions=torch.as_tensor(structure.positions_array, dtype=torch.float64),
                    cell=torch.as_tensor(structure.cell_array, dtype=torch.float64),
                    pbc=(bool(structure.pbc[0]), bool(structure.pbc[1]), bool(structure.pbc[2])),
                    n_atoms=structure.n_atoms,
                )
            )
            self.descriptors.append(
                compute_structure_descriptors(structure, self.descriptor_config)
            )
            raw.append(float(structure.energy))
        self.raw_energies = np.asarray(raw, dtype=np.float64)
        # Positions are fixed during training: the extended (2·r_c) neighbor
        # enumeration is computed once here and reused by every objective
        # evaluation (identical results, no rebuild cost). Positions-based
        # vectors/distances remain differentiable per evaluation.
        self.neighbors: list[BOPNeighbors] = [
            build_bop_neighbors(
                structure.positions_array,
                structure.cell_array,
                structure.pbc,
                CUTOFF_RADIUS,
            )
            for structure in dataset.structures
        ]
        self.energy_convention: str = dataset.metadata.energy_convention
        self.dataset_hash = dataset_content_hash(dataset)

    @property
    def n_structures(self) -> int:
        """Number of structures."""
        return len(self.geometries)

    @property
    def dataset(self) -> Dataset:
        """The underlying dataset (provenance and force labels)."""
        return self._dataset

    @property
    def is_synthetic(self) -> bool:
        """True for pipeline-validation databases (never reproduction results)."""
        return bool(self._dataset.metadata.extra.get("synthetic", False))

    def shifted_targets(self) -> np.ndarray:
        """DFT targets as used in training (eV), shift applied exactly here.

        ``raw_dft`` datasets get ``E + 0.38446 · N_atoms``; datasets declared
        ``shifted`` pass through unchanged. This is the only place the shift
        is applied — the double-shift corruption path is structurally closed.
        """
        if self.energy_convention == "raw_dft":
            n_atoms = np.asarray([g.n_atoms for g in self.geometries], dtype=np.float64)
            return self.raw_energies + ENERGY_SHIFT_EV_PER_ATOM * n_atoms
        if self.energy_convention == "shifted":
            return self.raw_energies.copy()
        raise DataContractError(
            f"unknown energy convention {self.energy_convention!r} (docs/data_format.md)"
        )

    def atoms_per_structure(self) -> np.ndarray:
        """``(S,)`` atom counts (for per-atom RMSE reporting)."""
        return np.asarray([g.n_atoms for g in self.geometries], dtype=np.float64)


def dataset_content_hash(dataset: Dataset) -> str:
    """Deterministic SHA-256 over the dataset content (provenance).

    Public so the evaluation layer (cross-validation) records the same hash
    as the training checkpoints.
    """
    payload = {
        "name": dataset.metadata.name,
        "energy_convention": dataset.metadata.energy_convention,
        "structures": [
            {
                "id": s.id,
                "numbers": s.numbers,
                "positions": s.positions,
                "cell": s.cell,
                "pbc": s.pbc,
                "energy": s.energy,
            }
            for s in dataset.structures
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
