"""Converters between :class:`ase.Atoms` and the package data schema.

These helpers are the bridge between the data layer and the simulation layer
(ASE, Sprint 4). They perform no physics; they only copy data across
representations and preserve labels through a single-point calculator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from numpy.typing import NDArray

from atomistic_pinn.data.schema import AtomicStructure


def _calculator_results(atoms: Atoms) -> dict[str, Any]:
    """Return the results dict of the attached calculator, if any."""
    if atoms.calc is None:
        return {}
    results = getattr(atoms.calc, "results", None)
    return results if isinstance(results, dict) else {}


def structure_from_ase(
    atoms: Atoms,
    structure_id: str,
    *,
    energy: float | None = None,
    forces: NDArray[np.float64] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AtomicStructure:
    """Build an :class:`AtomicStructure` from an ASE ``Atoms`` object.

    Args:
        atoms: Source configuration.
        structure_id: Unique identifier for the resulting structure.
        energy: Total energy in eV. If omitted and ``atoms`` carries a
            calculator with stored results, the stored value is used.
        forces: Forces in eV/Å, shape ``(n_atoms, 3)``. Same fallback rule
            as ``energy``.
        metadata: Free-form provenance for the structure.

    Returns:
        A validated :class:`AtomicStructure`.
    """
    results = _calculator_results(atoms)
    if energy is None:
        stored_energy = results.get("energy")
        energy = float(stored_energy) if stored_energy is not None else None
    if forces is None:
        stored_forces = results.get("forces")
        if stored_forces is not None:
            forces = np.asarray(stored_forces, dtype=np.float64)

    return AtomicStructure(
        id=structure_id,
        numbers=[int(z) for z in atoms.numbers],
        positions=atoms.positions.tolist(),
        cell=atoms.cell.array.tolist(),
        pbc=[bool(p) for p in atoms.pbc],
        energy=energy,
        forces=None if forces is None else np.asarray(forces, dtype=np.float64).tolist(),
        metadata=dict(metadata) if metadata else {},
    )


def structure_to_ase(structure: AtomicStructure) -> Atoms:
    """Build an ASE ``Atoms`` object from an :class:`AtomicStructure`.

    DFT labels, when present, are attached through a
    :class:`~ase.calculators.singlepoint.SinglePointCalculator` so that
    ``atoms.get_potential_energy()`` / ``atoms.get_forces()`` reproduce the
    stored values without any computation.
    """
    atoms = Atoms(
        numbers=structure.numbers,
        positions=structure.positions_array,
        cell=structure.cell_array,
        pbc=structure.pbc,
    )
    if structure.energy is not None or structure.forces is not None:
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=structure.energy,
            forces=structure.forces_array,
        )
    return atoms
