"""Data contracts for atomic structures and datasets.

All quantities use the package units (Å, eV, eV/Å; see
:mod:`atomistic_pinn.constants`).

Energy convention
-----------------
Energies stored in a dataset must be labeled through
``DatasetMetadata.energy_convention``:

* ``"raw_dft"`` — energies are exactly as produced by the electronic
  structure code. This is the default and the convention used for every
  dataset produced by this project: the uniform +0.38446 eV per atom shift
  described in the paper is applied by the *training pipeline*, never baked
  into stored data (this avoids accidental double shifts).
* ``"shifted"`` — energies already include the paper's per-atom shift.

Structure identifiers
---------------------
Every structure carries a unique ``id``. Cross-validation fold assignment
(Sprint 4) is performed *per structure*, never per atom, so all atoms of one
supercell always stay in the same fold (leakage rule, docs/data_format.md).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from atomistic_pinn.constants import UNIT_ENERGY, UNIT_FORCE, UNIT_LENGTH

EnergyConvention = Literal["raw_dft", "shifted"]


def _require_finite(values: list[list[float]], name: str) -> None:
    """Raise if any entry of ``values`` is not a finite float."""
    for i, row in enumerate(values):
        for j, x in enumerate(row):
            if not math.isfinite(x):
                raise ValueError(f"{name}[{i}][{j}] is not finite: {x!r}")


class AtomicStructure(BaseModel):
    """A single atomic configuration with optional DFT labels.

    Attributes:
        id: Unique identifier within its dataset (used for fold assignment).
        numbers: Atomic numbers, length ``n_atoms``.
        positions: Cartesian positions in Å, shape ``(n_atoms, 3)``.
        cell: Lattice vectors as rows in Å, shape ``(3, 3)``. Rows may be
            zero for non-periodic (cluster) structures.
        pbc: Periodic boundary flags, one per lattice direction.
        energy: Total DFT energy in eV (optional; unlabeled structures are
            allowed, e.g. for trajectories generated later by this package).
        forces: DFT forces in eV/Å, shape ``(n_atoms, 3)`` (optional; the
            paper does not use forces for training, only for post-hoc
            evaluation).
        metadata: Free-form provenance (source id, structure group,
            temperature, ...). Recommended keys are listed in
            ``docs/data_format.md``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    numbers: list[int]
    positions: list[list[float]]
    cell: list[list[float]] = Field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    pbc: list[bool] = Field(default_factory=lambda: [False, False, False])
    energy: float | None = None
    forces: list[list[float]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        n = len(self.numbers)
        if n == 0:
            raise ValueError("a structure must contain at least one atom")
        if len(self.positions) != n:
            raise ValueError(f"positions has {len(self.positions)} rows, expected {n}")
        if any(len(row) != 3 for row in self.positions):
            raise ValueError("every position row must have length 3")
        if len(self.cell) != 3 or any(len(row) != 3 for row in self.cell):
            raise ValueError("cell must have shape (3, 3)")
        if len(self.pbc) != 3:
            raise ValueError("pbc must have exactly three flags")
        if self.forces is not None:
            if len(self.forces) != n:
                raise ValueError(f"forces has {len(self.forces)} rows, expected {n}")
            if any(len(row) != 3 for row in self.forces):
                raise ValueError("every force row must have length 3")
        _require_finite(self.positions, "positions")
        _require_finite(self.cell, "cell")
        if self.forces is not None:
            _require_finite(self.forces, "forces")
        return self

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the structure."""
        return len(self.numbers)

    @property
    def is_periodic(self) -> bool:
        """True if any lattice direction is periodic."""
        return any(self.pbc)

    @property
    def positions_array(self) -> NDArray[np.float64]:
        """Positions as an ``(n_atoms, 3)`` float64 array (Å)."""
        return np.asarray(self.positions, dtype=np.float64)

    @property
    def cell_array(self) -> NDArray[np.float64]:
        """Lattice vectors as a ``(3, 3)`` float64 array (rows, Å)."""
        return np.asarray(self.cell, dtype=np.float64)

    @property
    def forces_array(self) -> NDArray[np.float64] | None:
        """Forces as an ``(n_atoms, 3)`` float64 array (eV/Å), or None."""
        if self.forces is None:
            return None
        return np.asarray(self.forces, dtype=np.float64)


class DatasetMetadata(BaseModel):
    """Provenance and convention metadata for a dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str = ""
    source: str = ""
    """Citation or URL of the data producer (paper, database, or workflow)."""

    license: str = ""
    version: str = "0.1.0"
    energy_convention: EnergyConvention = "raw_dft"
    units: dict[str, str] = Field(
        default_factory=lambda: {
            "length": UNIT_LENGTH,
            "energy": UNIT_ENERGY,
            "force": UNIT_FORCE,
        }
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    """An immutable collection of structures with shared provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: DatasetMetadata
    structures: list[AtomicStructure]

    @model_validator(mode="after")
    def _check_ids(self) -> Self:
        if not self.structures:
            raise ValueError("a dataset must contain at least one structure")
        ids = [s.id for s in self.structures]
        duplicates = [i for i, count in Counter(ids).items() if count > 1]
        if duplicates:
            raise ValueError(f"structure ids must be unique; duplicates: {sorted(duplicates)}")
        return self

    @property
    def n_structures(self) -> int:
        """Number of structures in the dataset."""
        return len(self.structures)

    @property
    def total_atoms(self) -> int:
        """Total number of atoms across all structures."""
        return sum(s.n_atoms for s in self.structures)

    @property
    def n_labeled_energies(self) -> int:
        """Number of structures carrying a DFT energy label."""
        return sum(1 for s in self.structures if s.energy is not None)
