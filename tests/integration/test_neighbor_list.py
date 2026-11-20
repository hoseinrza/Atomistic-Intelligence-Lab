"""Integration tests for neighbor-list behavior (self-consistency checks)."""

import numpy as np
import pytest

from atomistic_pinn.exceptions import DataContractError
from atomistic_pinn.geometry.neighbor_list import NeighborList

pytestmark = pytest.mark.integration


def _random_triclinic_cell(rng: np.random.Generator) -> np.ndarray:
    """Random invertible cell with edge lengths 6-9 Å and skewed angles."""
    lengths = rng.uniform(6.0, 9.0, size=3)
    angles = np.deg2rad(rng.uniform(60.0, 120.0, size=3))
    # Lattice vectors via the standard triclinic parametrization.
    a, b, c = lengths
    cos_ab, cos_ac, cos_bc = np.cos(angles)
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * cos_ac, b * np.sqrt(1.0 - cos_ac**2), 0.0])
    cx = c * cos_bc
    cy = c * (cos_ab - cos_bc * cos_ac) / np.sqrt(1.0 - cos_ac**2)
    cz = np.sqrt(max(c**2 - cx**2 - cy**2, 1e-6))
    return np.array([va, vb, np.array([cx, cy, cz])])


def test_reciprocal_pair_symmetry() -> None:
    # d(i, j, S) = d(j, i, −S) must hold for every stored pair.
    rng = np.random.default_rng(11)
    cell = _random_triclinic_cell(rng)
    positions = rng.uniform(0.0, 6.0, size=(10, 3))
    nl = NeighborList.build(positions, cell, [True, True, True], cutoff=5.0)
    pairs = {
        (int(i), int(j), tuple(int(s) for s in shift)): float(dist)
        for i in range(nl.n_atoms)
        for j, shift, dist in zip(nl.indices[i], nl.shifts[i], nl.distances[i], strict=True)
    }
    assert pairs, "test structure produced no pairs"
    for (i, j, shift), dist in pairs.items():
        mirror = (j, i, tuple(-s for s in shift))
        assert mirror in pairs
        assert pairs[mirror] == pytest.approx(dist, abs=1e-12)


def test_deterministic_ordering() -> None:
    rng = np.random.default_rng(3)
    cell = np.eye(3) * 9.0
    positions = rng.uniform(0.0, 9.0, size=(12, 3))
    nl1 = NeighborList.build(positions, cell, [True] * 3, cutoff=6.0)
    nl2 = NeighborList.build(positions, cell, [True] * 3, cutoff=6.0)
    for a, b in zip(nl1.indices, nl2.indices, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(nl1.shifts, nl2.shifts, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(nl1.distances, nl2.distances, strict=True):
        np.testing.assert_array_equal(a, b)


def test_partial_pbc_has_no_offplane_shifts() -> None:
    # Slab setup: periodic in x and y only; no neighbor may carry a
    # z-component in its shift, and in-plane geometry must still be correct.
    rng = np.random.default_rng(5)
    cell = np.diag([8.0, 8.0, 30.0])
    positions = rng.uniform(0.0, 8.0, size=(14, 3))
    positions[:, 2] = rng.uniform(10.0, 20.0, size=14)  # keep planes apart
    nl = NeighborList.build(positions, cell, [True, True, False], cutoff=6.0)
    for shifts in nl.shifts:
        np.testing.assert_array_equal(shifts[:, 2], 0)
    # Every stored distance must be reproducible from the stored shift.
    for i, (idx, shifts, vecs, dists) in enumerate(
        zip(nl.indices, nl.shifts, nl.vectors, nl.distances, strict=True)
    ):
        expected = positions[idx] + shifts.astype(np.float64) @ cell - positions[i]
        np.testing.assert_allclose(vecs, expected, atol=1e-12)
        np.testing.assert_allclose(dists, np.linalg.norm(expected, axis=1), atol=1e-12)


def test_distances_reproduced_from_stored_shifts_periodic() -> None:
    rng = np.random.default_rng(9)
    cell = _random_triclinic_cell(rng)
    positions = rng.uniform(-0.2, 1.2 * 7.0, size=(9, 3))  # mildly unwrapped
    nl = NeighborList.build(positions, cell, [True] * 3, cutoff=5.0)
    for i, (idx, shifts, vecs, dists) in enumerate(
        zip(nl.indices, nl.shifts, nl.vectors, nl.distances, strict=True)
    ):
        expected = positions[idx] + shifts.astype(np.float64) @ cell - positions[i]
        np.testing.assert_allclose(vecs, expected, atol=1e-12)
        np.testing.assert_allclose(dists, np.linalg.norm(expected, axis=1), atol=1e-12)


def test_invalid_inputs_raise_data_contract_error() -> None:
    positions = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(DataContractError, match="singular"):
        NeighborList.build(positions, np.zeros((3, 3)), [True] * 3, cutoff=4.0)
    with pytest.raises(DataContractError, match="cutoff"):
        NeighborList.build(positions, np.eye(3) * 9.0, [True] * 3, cutoff=0.0)
    with pytest.raises(DataContractError, match="non-finite"):
        NeighborList.build(np.array([[np.nan, 0.0, 0.0]]), np.eye(3) * 9.0, [True] * 3, cutoff=4.0)
    with pytest.raises(DataContractError, match="positions"):
        NeighborList.build(np.zeros((2, 2)), np.eye(3) * 9.0, [False] * 3, cutoff=4.0)
