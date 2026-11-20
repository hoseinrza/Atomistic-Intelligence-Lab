"""Integration tests for the ASE <-> schema converters."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator

from atomistic_pinn.data.converters import structure_from_ase, structure_to_ase

pytestmark = pytest.mark.integration


def test_bulk_al_round_trip_preserves_geometry_and_labels() -> None:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    rng = np.random.default_rng(42)
    forces = rng.normal(scale=0.1, size=(len(atoms), 3))
    atoms.calc = SinglePointCalculator(atoms, energy=-27.7, forces=forces)

    structure = structure_from_ase(atoms, structure_id="al_bulk")
    assert structure.n_atoms == len(atoms)
    assert structure.energy == pytest.approx(-27.7)
    assert structure.forces_array is not None
    np.testing.assert_allclose(structure.forces_array, forces, atol=1e-12)
    assert structure.is_periodic

    back = structure_to_ase(structure)
    np.testing.assert_allclose(back.positions, atoms.positions, atol=1e-12)
    np.testing.assert_allclose(back.cell.array, atoms.cell.array, atol=1e-12)
    assert back.pbc.tolist() == atoms.pbc.tolist()
    assert back.get_potential_energy() == pytest.approx(-27.7)
    np.testing.assert_allclose(back.get_forces(), forces, atol=1e-12)


def test_cluster_without_labels_round_trip() -> None:
    atoms = Atoms(
        "Al2",
        positions=[[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]],
        cell=[[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
        pbc=False,
    )
    structure = structure_from_ase(atoms, structure_id="dimer")
    assert structure.energy is None
    assert structure.forces is None

    back = structure_to_ase(structure)
    np.testing.assert_allclose(back.positions, atoms.positions, atol=1e-12)
    assert back.calc is None


def test_triclinic_cell_round_trip() -> None:
    cell = [[6.0, 0.2, 0.1], [0.3, 5.5, 0.0], [0.1, 0.0, 7.2]]
    atoms = Atoms("Al", positions=[[0.5, 0.5, 0.5]], cell=cell, pbc=True)

    structure = structure_from_ase(atoms, structure_id="tri")
    np.testing.assert_allclose(structure.cell_array, np.asarray(cell), atol=1e-12)

    back = structure_to_ase(structure)
    np.testing.assert_allclose(back.cell.array, np.asarray(cell), atol=1e-12)


def test_partial_labels_survive_round_trip() -> None:
    # Only an energy label, no forces: both must survive independently.
    atoms = Atoms("Al", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-3.36)

    structure = structure_from_ase(atoms, structure_id="single")
    assert structure.energy == pytest.approx(-3.36)
    assert structure.forces is None

    back = structure_to_ase(structure)
    assert back.get_potential_energy() == pytest.approx(-3.36)


def test_explicit_labels_override_calculator() -> None:
    atoms = Atoms("Al", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-3.36)

    structure = structure_from_ase(atoms, structure_id="override", energy=-1.0)
    assert structure.energy == pytest.approx(-1.0)


def test_metadata_is_carried_through() -> None:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    metadata = {"group": "fcc_bulk", "temperature_K": 900, "source_id": "db-0001"}
    structure = structure_from_ase(atoms, structure_id="hot", metadata=metadata)
    assert structure.metadata == metadata
