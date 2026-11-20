"""Neighbor-list validation against independent references.

References used here (never the implementation under test):

* ASE's ``neighbor_list`` (independent linked-cell implementation);
* direct enumeration of lattice translations for known cases (closed-form
  FCC geometry);
* brute-force enumeration over a generously large shift box.
"""

import itertools

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list

from atomistic_pinn.geometry.neighbor_list import NeighborList

pytestmark = pytest.mark.scientific_validation


def _our_pairs(
    nl: NeighborList, *, include_self_images: bool
) -> dict[tuple[int, int, tuple[int, int, int]], float]:
    pairs: dict[tuple[int, int, tuple[int, int, int]], float] = {}
    for i in range(nl.n_atoms):
        for j, shift, dist in zip(nl.indices[i], nl.shifts[i], nl.distances[i], strict=True):
            if i == j and not include_self_images:
                continue
            key = (int(i), int(j), tuple(int(s) for s in shift))
            pairs[key] = float(dist)
    return pairs


def _ase_pairs(atoms: Atoms, cutoff: float) -> dict[tuple[int, int, tuple[int, int, int]], float]:
    i, j, d, dvec = neighbor_list("ijdD", atoms, cutoff)
    inv_cell = np.linalg.inv(atoms.cell.array)
    pairs: dict[tuple[int, int, tuple[int, int, int]], float] = {}
    for a, b, dist, vec in zip(i, j, d, dvec, strict=True):
        if a == b:
            continue  # self-image entries are validated separately
        shift = np.rint((vec - atoms.positions[b] + atoms.positions[a]) @ inv_cell)
        pairs[(int(a), int(b), tuple(int(s) for s in shift))] = float(dist)
    return pairs


def test_matches_ase_on_fcc_supercell() -> None:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2))
    cutoff = 6.0
    ours = _our_pairs(
        NeighborList.build(atoms.positions, atoms.cell.array, atoms.pbc, cutoff),
        include_self_images=False,
    )
    reference = _ase_pairs(atoms, cutoff)
    assert set(ours) == set(reference)
    for key, dist in ours.items():
        assert reference[key] == pytest.approx(dist, abs=1e-10)


def test_matches_ase_on_triclinic_cell() -> None:
    rng = np.random.default_rng(21)
    lengths = rng.uniform(7.0, 9.0, size=3)
    cell = np.diag(lengths) + rng.uniform(-0.8, 0.8, size=(3, 3))
    cell[0, 0] = lengths[0]  # keep positive volume comfortably
    atoms = Atoms("Al7", positions=rng.uniform(0.0, 7.0, size=(7, 3)), cell=cell, pbc=True)
    cutoff = 5.0
    ours = _our_pairs(
        NeighborList.build(atoms.positions, atoms.cell.array, atoms.pbc, cutoff),
        include_self_images=False,
    )
    reference = _ase_pairs(atoms, cutoff)
    assert set(ours) == set(reference)
    for key, dist in ours.items():
        assert reference[key] == pytest.approx(dist, abs=1e-10)


def test_matches_ase_with_partial_pbc() -> None:
    rng = np.random.default_rng(33)
    cell = np.diag([9.0, 9.0, 30.0])
    positions = rng.uniform(0.0, 9.0, size=(10, 3))
    positions[:, 2] += 12.0
    atoms = Atoms("Al10", positions=positions, cell=cell, pbc=[True, True, False])
    cutoff = 6.0
    ours = _our_pairs(
        NeighborList.build(atoms.positions, atoms.cell.array, atoms.pbc, cutoff),
        include_self_images=False,
    )
    reference = _ase_pairs(atoms, cutoff)
    assert set(ours) == set(reference)
    for key, dist in ours.items():
        assert reference[key] == pytest.approx(dist, abs=1e-10)


def test_fcc_primitive_self_images_match_closed_form_lattice_translations() -> None:
    # In the FCC primitive cell the only atom interacts with its own periodic
    # images; the self-image distances are exactly |S @ cell| for lattice
    # translations S. Enumerate translations directly (independent of the
    # implementation) and compare with the stored self-image entries.
    atoms = bulk("Al", "fcc", a=4.05)
    cutoff = 6.0
    cell = atoms.cell.array

    expected: dict[tuple[int, int, int], float] = {}
    for s in itertools.product(range(-3, 4), repeat=3):
        if s == (0, 0, 0):
            continue
        dist = float(np.linalg.norm(np.asarray(s, dtype=float) @ cell))
        if dist < cutoff:
            expected[s] = dist
    assert expected, "FCC primitive cell must have self images within 6 Å"
    assert min(expected.values()) == pytest.approx(4.05 / np.sqrt(2.0), rel=1e-10)

    nl = NeighborList.build(atoms.positions, cell, atoms.pbc, cutoff)
    stored = {
        tuple(int(x) for x in shift): float(dist)
        for j, shift, dist in zip(nl.indices[0], nl.shifts[0], nl.distances[0], strict=True)
        if j == 0
    }
    assert set(stored) == set(expected)
    for key, dist in expected.items():
        assert stored[key] == pytest.approx(dist, abs=1e-12)


def test_matches_brute_force_on_random_triclinic_structure() -> None:
    # Independent brute force over a generously large shift box; catches any
    # error in the analytic shift-range derivation.
    rng = np.random.default_rng(42)
    lengths = rng.uniform(6.0, 9.0, size=3)
    cell = np.diag(lengths) + rng.uniform(-0.5, 0.5, size=(3, 3))
    positions = rng.uniform(0.0, 7.0, size=(8, 3))
    cutoff = 4.5

    brute: dict[tuple[int, int, tuple[int, int, int]], float] = {}
    for s in itertools.product(range(-4, 5), repeat=3):
        image = positions + np.asarray(s, dtype=float) @ cell
        dvec = image[None, :, :] - positions[:, None, :]
        dist = np.linalg.norm(dvec, axis=2)
        row, col = np.nonzero(dist < cutoff)
        for a, b in zip(row, col, strict=True):
            if a == b and s == (0, 0, 0):
                continue
            brute[(int(a), int(b), s)] = float(dist[a, b])

    nl = NeighborList.build(positions, cell, [True] * 3, cutoff)
    ours = _our_pairs(nl, include_self_images=True)
    assert set(ours) == set(brute)
    for key, dist in ours.items():
        assert brute[key] == pytest.approx(dist, abs=1e-12)
