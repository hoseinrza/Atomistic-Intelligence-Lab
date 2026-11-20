"""Scientific validation of the Eq. (3) local structural descriptors.

Every reference here is independent of the implementation under test:

* closed-form values for a two-neighbor cluster (hand-derived sums of
  Legendre polynomials);
* closed-form vector identities g^(0) = (Σ f)² and
  g^(1) = |Σ f r̂|² that the double-sum form must satisfy;
* exact invariances (translation, rotation, permutation);
* periodic-image completeness (supercell replication must reproduce the
  infinite-crystal environment);
* ASE-built structures as physically known inputs.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from atomistic_pinn.data.converters import structure_from_ase
from atomistic_pinn.descriptors import (
    DescriptorConfig,
    compute_descriptors,
    compute_structure_descriptors,
)
from atomistic_pinn.descriptors.radial import radial_function
from atomistic_pinn.geometry.neighbor_list import NeighborList

pytestmark = pytest.mark.scientific_validation

R_C = 6.0
D = 1.5


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Random proper rotation matrix (det = +1) via QR decomposition."""
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_descriptor_dimensions_and_layout() -> None:
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    structure = structure_from_ase(atoms, structure_id="al4")
    desc = compute_structure_descriptors(structure)
    assert desc.shape == (4, 60)
    # Order-major layout: a single-order run must equal the corresponding
    # block of the full run (m = 0 → slots 0-11, m = 2 → slots 24-35).
    desc0 = compute_structure_descriptors(structure, DescriptorConfig(orders=(0,)))
    np.testing.assert_allclose(desc0, desc[:, 0:12], rtol=0.0, atol=1e-15)
    desc2 = compute_structure_descriptors(structure, DescriptorConfig(orders=(2,)))
    np.testing.assert_allclose(desc2, desc[:, 24:36], rtol=0.0, atol=1e-15)


def test_translation_invariance() -> None:
    rng = np.random.default_rng(1)
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2))
    atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
    structure = structure_from_ase(atoms, structure_id="perturbed")
    desc = compute_structure_descriptors(structure)
    shifted = structure.model_copy(
        update={"positions": (structure.positions_array + np.array([0.7, -1.2, 3.3])).tolist()}
    )
    desc_shifted = compute_structure_descriptors(shifted)
    np.testing.assert_allclose(desc_shifted, desc, rtol=0.0, atol=1e-12)


def test_rotation_invariance_periodic_and_cluster() -> None:
    rng = np.random.default_rng(2)
    rotation = _random_rotation(rng)

    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2))
    atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
    structure = structure_from_ase(atoms, structure_id="periodic")
    rotated = structure.model_copy(
        update={
            "positions": (structure.positions_array @ rotation.T).tolist(),
            "cell": (structure.cell_array @ rotation.T).tolist(),
        }
    )
    np.testing.assert_allclose(
        compute_structure_descriptors(rotated),
        compute_structure_descriptors(structure),
        rtol=0.0,
        atol=1e-10,
    )

    cluster = structure_from_ase(
        Atoms("Al9", positions=rng.uniform(0.0, 10.0, size=(9, 3))),
        structure_id="cluster",
    )
    cluster_rotated = cluster.model_copy(
        update={"positions": (cluster.positions_array @ rotation.T).tolist()}
    )
    np.testing.assert_allclose(
        compute_structure_descriptors(cluster_rotated),
        compute_structure_descriptors(cluster),
        rtol=0.0,
        atol=1e-10,
    )


def test_permutation_invariance() -> None:
    rng = np.random.default_rng(4)
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2))
    atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
    structure = structure_from_ase(atoms, structure_id="perm")
    desc = compute_structure_descriptors(structure)
    perm = rng.permutation(structure.n_atoms)
    permuted = structure.model_copy(
        update={
            "positions": structure.positions_array[perm].tolist(),
            "numbers": [structure.numbers[k] for k in perm],
        }
    )
    desc_permuted = compute_structure_descriptors(permuted)
    np.testing.assert_allclose(desc_permuted, desc[perm], rtol=0.0, atol=1e-12)


def test_fcc_site_symmetry_gives_identical_rows() -> None:
    # All four atoms of the conventional FCC cell are equivalent FCC sites;
    # their descriptor rows must coincide.
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    desc = compute_structure_descriptors(structure_from_ase(atoms, structure_id="al4"))
    assert float(np.abs(desc - desc[0]).max()) < 1e-12


def test_supercell_replication_reproduces_crystal_environment() -> None:
    # Every atom of the 4x4x4 FCC supercell (box 16.2 Å) has its full
    # infinite-crystal environment within the 6 Å cutoff, exactly as every
    # atom of the 1x1x1 cell has through its periodic images. The descriptor
    # rows must therefore match — a strong test of periodic-image
    # completeness across different cell sizes.
    small = bulk("Al", "fcc", a=4.05, cubic=True)
    desc_small = compute_structure_descriptors(structure_from_ase(small, structure_id="small"))
    big = small.repeat((4, 4, 4))
    desc_big = compute_structure_descriptors(structure_from_ase(big, structure_id="big"))
    assert float(np.abs(desc_big - desc_small[0]).max()) < 1e-10


def test_periodic_images_actually_contribute() -> None:
    # One atom in a small periodic box has many self images within 6 Å and
    # must get a nonzero descriptor; the same isolated atom gets zeros.
    periodic = structure_from_ase(
        Atoms("Al", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 3.0, pbc=True),
        structure_id="periodic_single",
    )
    isolated = structure_from_ase(
        Atoms("Al", positions=[[0.0, 0.0, 0.0]]),
        structure_id="isolated",
    )
    desc_periodic = compute_structure_descriptors(periodic)
    desc_isolated = compute_structure_descriptors(isolated)
    assert np.abs(desc_periodic).max() > 1e-6
    np.testing.assert_array_equal(desc_isolated, 0.0)


def test_closed_form_two_neighbor_cluster() -> None:
    # Atom 0 with two equidistant neighbors at 90 degrees. With a single
    # radial center at r0 = 2.5 (Gaussian peak, sigma = 1) all radial
    # weights equal f = f_c(2.5), and the double sum collapses to
    # hand-derived values:
    #   g0 = (2f)²,  g1 = 2f²,  g2 = f²,  g4 = (2 + 2·3/8) f²,  g6 = (2 − 2·5/16) f²
    config = DescriptorConfig(
        orders=(0, 1, 2, 4, 6),
        radial_centers=(2.5,),
        include_same_pair=True,
    )
    positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 2.5, 0.0]])
    desc = compute_descriptors(positions, np.zeros((3, 3)), np.array([False] * 3), config)

    x = 2.5 - R_C
    f = x**4 / (D**4 + x**4)
    f2 = f * f
    expected = np.array([4.0 * f2, 2.0 * f2, 1.0 * f2, 2.75 * f2, 1.375 * f2])
    np.testing.assert_allclose(desc[0], expected, rtol=1e-12, atol=1e-14)


def test_closed_form_two_neighbor_cluster_excluding_same_pair() -> None:
    # Same cluster with j = k terms excluded:
    #   g0 = 2f²,  g1 = 0,  g2 = −f²,  g4 = 3/8·2f²,  g6 = −5/16·2f²
    config = DescriptorConfig(
        orders=(0, 1, 2, 4, 6),
        radial_centers=(2.5,),
        include_same_pair=False,
    )
    positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 2.5, 0.0]])
    desc = compute_descriptors(positions, np.zeros((3, 3)), np.array([False] * 3), config)

    x = 2.5 - R_C
    f = x**4 / (D**4 + x**4)
    f2 = f * f
    expected = np.array([2.0 * f2, 0.0, -1.0 * f2, 0.75 * f2, -0.625 * f2])
    np.testing.assert_allclose(desc[0], expected, rtol=1e-12, atol=1e-14)


def test_double_sum_vector_identities() -> None:
    # g^(0)[c] = (Σ_j f_c(r_ij))² and g^(1)[c] = |Σ_j f_c(r_ij) r̂_ij|²
    # must hold per radial center c for arbitrary configurations.
    rng = np.random.default_rng(6)
    positions = rng.uniform(0.0, 12.0, size=(15, 3))
    config = DescriptorConfig(orders=(0, 1))
    desc = compute_descriptors(positions, np.eye(3) * 13.0, np.array([True] * 3), config)

    neighbor_list = NeighborList.build(
        positions, np.eye(3) * 13.0, np.array([True] * 3), config.cutoff_radius
    )
    centers = np.asarray(config.radial_centers)
    for i in range(positions.shape[0]):
        vectors = neighbor_list.vectors[i]
        if vectors.shape[0] == 0:
            continue
        unit = vectors / neighbor_list.distances[i][:, None]
        for c, r0 in enumerate(centers):
            weights = radial_function(neighbor_list.distances[i], r0=float(r0))
            expected0 = float(weights.sum()) ** 2
            expected1 = float(
                np.sum(weights[:, None] * unit, axis=0) @ np.sum(weights[:, None] * unit, axis=0)
            )
            assert desc[i, c] == pytest.approx(expected0, rel=1e-10, abs=1e-14)
            assert desc[i, 1 * len(centers) + c] == pytest.approx(expected1, rel=1e-10, abs=1e-14)


def test_numerical_stability_under_extreme_volumes() -> None:
    rng = np.random.default_rng(8)
    for a in (2.0, 4.05, 8.0):
        atoms = bulk("Al", "fcc", a=a, cubic=True).repeat((2, 2, 2))
        desc = compute_structure_descriptors(structure_from_ase(atoms, structure_id=f"a{a}"))
        assert np.isfinite(desc).all()
    liquid = structure_from_ase(
        Atoms(
            "Al20", positions=rng.uniform(0.0, 10.0, size=(20, 3)), cell=np.eye(3) * 10.0, pbc=True
        ),
        structure_id="liquid",
    )
    assert np.isfinite(compute_structure_descriptors(liquid)).all()


def test_descriptor_config_validation() -> None:
    with pytest.raises(ValueError, match="orders"):
        DescriptorConfig(orders=())
    with pytest.raises(ValueError, match="sigma"):
        DescriptorConfig(sigma=0.0)
    with pytest.raises(ValueError, match="radial_centers"):
        DescriptorConfig(radial_centers=())
    with pytest.raises(ValueError, match="cutoff_radius"):
        DescriptorConfig(cutoff_radius=-1.0)
