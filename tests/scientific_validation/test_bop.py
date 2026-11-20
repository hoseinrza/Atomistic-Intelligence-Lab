"""Scientific validation of the BOP engine (Eqs. 6-12).

The centerpiece is an **independent reference implementation** — plain
Python loops transcribed literally from the paper's equations, sharing no
code with the vectorized torch implementation except the (already
independently validated) numpy neighbor list. The two must agree to
float64 round-off on arbitrary structures.

Physical parameter sets below are *synthetic test fixtures* chosen to be
physically sensible (A·α < B·β so that pair repulsion dominates at short
range; positive σ, a, λ). The paper's fitted BOP parameter values are not
published in its main text (see docs/equations.md §7.12), so no "true"
parameter set exists to test against.
"""

import math

import numpy as np
import pytest
import torch
from ase.build import bulk

from atomistic_pinn.constants import BOP_PARAMETER_NAMES, CUTOFF_D, CUTOFF_RADIUS
from atomistic_pinn.exceptions import DataContractError
from atomistic_pinn.geometry.neighbor_list import NeighborList
from atomistic_pinn.physics import (
    atomic_energies,
    forces_autograd,
    forces_finite_difference,
    total_energy,
)

pytestmark = pytest.mark.scientific_validation

PARAM_SET_1 = {
    "A": 1.2,
    "B": 1.0,
    "alpha": 1.5,
    "beta": 4.0,
    "a": 1.5,
    "h": 0.0,
    "sigma": 2.0,
    "lambda": 1.5,
}
PARAM_SET_2 = {
    "A": 0.8,
    "B": 1.4,
    "alpha": 2.0,
    "beta": 3.5,
    "a": 1.1,
    "h": -0.25,
    "sigma": 1.2,
    "lambda": 2.2,
}


def make_params(n: int, pset: dict[str, float], requires_grad: bool = False) -> torch.Tensor:
    values = np.stack(
        [np.full(n, pset[name], dtype=np.float64) for name in BOP_PARAMETER_NAMES], axis=1
    )
    tensor = torch.tensor(values, requires_grad=requires_grad)
    return tensor


def _fc(r: np.ndarray | float) -> np.ndarray:
    x = np.asarray(r, dtype=np.float64) - CUTOFF_RADIUS
    return np.where(x <= 0, x**4 / (CUTOFF_D**4 + x**4), 0.0)


def reference_atomic_energies(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: list[bool],
    pset: dict[str, float],
) -> np.ndarray:
    """Independent loop-based transcription of Eqs. (6)-(12).

    Deliberately slow and literal: per-atom python loops over the extended
    (2 r_c) neighbor list, no vectorization tricks, no padding.
    """
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    n_atoms = len(positions)
    neighbor_list = NeighborList.build(positions, cell, pbc, 2.0 * CUTOFF_RADIUS)
    energies = np.zeros(n_atoms)
    for i in range(n_atoms):
        vec = neighbor_list.vectors[i]
        r = neighbor_list.distances[i]
        m = len(r)
        if m == 0:
            continue  # no neighbors: E_i = 0
        lam = pset["lambda"][i]
        diff = vec[:, None, :] - vec[None, :, :]
        r_kl = np.linalg.norm(diff, axis=2)
        arg = r[None, :] + r_kl - r[:, None]  # Eq. (10) argument
        w = np.where(
            np.eye(m, dtype=bool),
            1.0,
            1.0 - _fc(arg) * np.exp(-(lam**2) * arg),
        )
        s = np.prod(w, axis=1)  # Eq. (9)
        bond = r < CUTOFF_RADIUS
        fc_r = _fc(r)
        cosines = (vec @ vec.T) / np.outer(r, r)
        z = np.zeros(m)
        for a in np.nonzero(bond)[0]:
            acc = 0.0
            for b in np.nonzero(bond)[0]:
                if b == a:
                    continue
                acc += s[b] * (cosines[a, b] + pset["h"][i]) ** 2 * fc_r[b]
            z[a] = pset["a"][i] ** 2 * acc  # Eq. (8)
        bfac = 1.0 / np.sqrt(1.0 + z)  # Eq. (7)
        energy = 0.0
        for j in np.nonzero(bond)[0]:
            energy += (
                0.5
                * (
                    math.exp(-pset["A"][i] * pset["alpha"][i] * r[j])
                    - s[j] * bfac[j] * math.exp(-pset["B"][i] * pset["beta"][i] * r[j])
                )
                * fc_r[j]
            )
        rho = float(np.sum(s[bond] * bfac[bond] * fc_r[bond]))
        energy += -pset["sigma"][i] * math.sqrt(rho)  # Eq. (11)
        energies[i] = energy
    return energies


def _random_triclinic_cell(rng: np.random.Generator) -> np.ndarray:
    lengths = rng.uniform(6.0, 8.0, size=3)
    cell = np.diag(lengths) + rng.uniform(-0.5, 0.5, size=(3, 3))
    cell[0, 0] = lengths[0]
    return cell


def test_dimer_matches_closed_form() -> None:
    r = 3.0
    positions = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64)
    cell = torch.zeros((3, 3), dtype=torch.float64)
    params = make_params(2, PARAM_SET_1)

    energies = atomic_energies(positions, cell, [False, False, False], params)
    total = total_energy(positions, cell, [False, False, False], params)

    fc = float(_fc(r))
    rep = math.exp(-PARAM_SET_1["A"] * PARAM_SET_1["alpha"] * r)
    att = math.exp(-PARAM_SET_1["B"] * PARAM_SET_1["beta"] * r)
    # Single bond: empty screening product → S = 1; empty z sum → b = 1.
    expected_atomic = 0.5 * (rep - att) * fc - PARAM_SET_1["sigma"] * math.sqrt(fc)
    expected_total = (rep - att) * fc - 2.0 * PARAM_SET_1["sigma"] * math.sqrt(fc)
    np.testing.assert_allclose(energies.numpy(), [expected_atomic] * 2, rtol=1e-12)
    assert float(total) == pytest.approx(expected_total, rel=1e-12)


def test_screening_changes_energy_and_reference_agrees() -> None:
    # Three collinear atoms: the middle atom almost completely screens the
    # outer bond (arg = 0). The reference implementation must reproduce the
    # same effect.
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    cell = np.zeros((3, 3))
    pbc = [False, False, False]
    params = make_params(3, PARAM_SET_1)
    got = atomic_energies(torch.tensor(positions), torch.tensor(cell), pbc, params).numpy()
    expected = reference_atomic_energies(
        positions, cell, pbc, {k: np.full(3, v) for k, v in PARAM_SET_1.items()}
    )
    np.testing.assert_allclose(got, expected, rtol=1e-10)
    # And the screened configuration differs from the unscreened pair alone.
    pair_only = reference_atomic_energies(
        positions[:2], cell, pbc, {k: np.full(2, v) for k, v in PARAM_SET_1.items()}
    )
    assert not np.isclose(got[0], pair_only[0], rtol=1e-3)


def test_matches_independent_reference_on_various_structures() -> None:
    rng = np.random.default_rng(17)
    structures: list[tuple[str, np.ndarray, np.ndarray, list[bool]]] = []

    cell = _random_triclinic_cell(rng)
    structures.append(
        ("triclinic_pbc", rng.uniform(0.0, 6.0, size=(5, 3)), cell, [True, True, True])
    )
    structures.append(
        ("cluster", rng.uniform(0.0, 9.0, size=(8, 3)), np.zeros((3, 3)), [False] * 3)
    )
    fcc = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2))
    fcc.positions += rng.normal(scale=0.1, size=fcc.positions.shape)
    structures.append(("perturbed_fcc", fcc.positions, fcc.cell.array, [True, True, True]))
    structures.append(
        ("tiny_pbc_selfimages", np.array([[0.1, 0.2, 0.3]]), np.eye(3) * 3.0, [True] * 3)
    )
    slab_cell = np.diag([8.0, 8.0, 30.0])
    slab = rng.uniform(0.0, 8.0, size=(6, 3))
    slab[:, 2] += 12.0
    structures.append(("slab_partial_pbc", slab, slab_cell, [True, True, False]))

    for name, positions, cell, pbc in structures:
        for pset in (PARAM_SET_1, PARAM_SET_2):
            n = len(positions)
            params = make_params(n, pset)
            got = atomic_energies(torch.tensor(positions), torch.tensor(cell), pbc, params).numpy()
            expected = reference_atomic_energies(
                positions, cell, pbc, {k: np.full(n, v) for k, v in pset.items()}
            )
            np.testing.assert_allclose(
                got, expected, rtol=1e-9, atol=1e-12, err_msg=f"structure {name}"
            )


def test_total_energy_equals_sum_of_atomic() -> None:
    rng = np.random.default_rng(23)
    positions = rng.uniform(0.0, 8.0, size=(6, 3))
    cell = np.eye(3) * 9.0
    params = make_params(6, PARAM_SET_1)
    atomic = atomic_energies(torch.tensor(positions), torch.tensor(cell), [True] * 3, params)
    total = total_energy(torch.tensor(positions), torch.tensor(cell), [True] * 3, params)
    assert float(total) == pytest.approx(float(atomic.sum()), rel=1e-14)


def test_permutation_covariance() -> None:
    rng = np.random.default_rng(29)
    positions = rng.uniform(0.0, 8.0, size=(7, 3))
    cell = np.eye(3) * 10.0
    params = make_params(7, PARAM_SET_2)
    perm = rng.permutation(7)
    energies = atomic_energies(torch.tensor(positions), torch.tensor(cell), [True] * 3, params)
    energies_permuted = atomic_energies(
        torch.tensor(positions[perm]),
        torch.tensor(cell),
        [True] * 3,
        params[perm],
    )
    np.testing.assert_allclose(energies_permuted.numpy(), energies.numpy()[perm], rtol=1e-12)


def test_translation_and_rotation_invariance() -> None:
    rng = np.random.default_rng(31)
    positions = rng.uniform(0.0, 7.0, size=(6, 3))
    cell = np.eye(3) * 8.0
    params = make_params(6, PARAM_SET_1)
    base = float(total_energy(torch.tensor(positions), torch.tensor(cell), [True] * 3, params))

    shifted = positions + np.array([1.3, -2.1, 0.7])
    assert float(
        total_energy(torch.tensor(shifted), torch.tensor(cell), [True] * 3, params)
    ) == pytest.approx(base, rel=1e-12)

    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    rotated_positions = positions @ q.T
    rotated_cell = cell @ q.T
    assert float(
        total_energy(
            torch.tensor(rotated_positions), torch.tensor(rotated_cell), [True] * 3, params
        )
    ) == pytest.approx(base, rel=1e-12)


def test_periodic_images_interact_isolated_atom_does_not() -> None:
    params = make_params(1, PARAM_SET_1)
    periodic_energy = float(
        total_energy(
            torch.tensor([[0.05, 0.1, 0.15]]),
            torch.eye(3) * 3.0,
            [True] * 3,
            params,
        )
    )
    isolated_energy = float(
        total_energy(
            torch.tensor([[0.05, 0.1, 0.15]]),
            torch.zeros((3, 3), dtype=torch.float64),
            [False] * 3,
            params,
        )
    )
    assert isolated_energy == pytest.approx(0.0, abs=1e-14)
    assert abs(periodic_energy) > 1e-3  # self images are real neighbors


def test_energy_rises_under_strong_compression() -> None:
    # Paper Fig. 6 physics: the compressed dimer must be higher in energy
    # than the near-equilibrium dimer (repulsion wins at short range).
    params = make_params(2, PARAM_SET_1)
    cell = torch.zeros((3, 3), dtype=torch.float64)
    pbc = [False, False, False]
    compressed = float(
        total_energy(torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]), cell, pbc, params)
    )
    relaxed = float(
        total_energy(torch.tensor([[0.0, 0.0, 0.0], [2.86, 0.0, 0.0]]), cell, pbc, params)
    )
    assert compressed > relaxed


def test_fd_forces_match_autograd_forces() -> None:
    rng = np.random.default_rng(37)
    fcc = bulk("Al", "fcc", a=4.05, cubic=True)
    fcc.positions += rng.normal(scale=0.15, size=fcc.positions.shape)
    params = make_params(4, PARAM_SET_1)

    fd = forces_finite_difference(fcc.positions, fcc.cell.array, [True] * 3, params)
    auto = forces_autograd(
        torch.tensor(fcc.positions), torch.tensor(fcc.cell.array), [True] * 3, params
    )
    np.testing.assert_allclose(fd, auto.numpy(), rtol=1e-5, atol=1e-6)


def test_autograd_forces_match_closed_form_dimer_derivative() -> None:
    # For an isolated pair the closed form E_total(r) is known exactly;
    # differentiate it numerically (pure python) and compare with autograd.
    params = make_params(2, PARAM_SET_1)
    r = 2.5
    positions = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64)
    forces = forces_autograd(
        positions, torch.zeros((3, 3), dtype=torch.float64), [False] * 3, params
    )

    def closed_form(rr: float) -> float:
        fc = float(_fc(rr))
        rep = math.exp(-PARAM_SET_1["A"] * PARAM_SET_1["alpha"] * rr)
        att = math.exp(-PARAM_SET_1["B"] * PARAM_SET_1["beta"] * rr)
        return (rep - att) * fc - 2.0 * PARAM_SET_1["sigma"] * math.sqrt(fc)

    h = 1e-6
    derivative = (closed_form(r + h) - closed_form(r - h)) / (2.0 * h)
    # Atom 1 sits at +r: F_x(1) = −dE/dr. Atom 0: F_x(0) = +dE/dr.
    assert float(forces[1, 0]) == pytest.approx(-derivative, rel=1e-6)
    assert float(forces[0, 0]) == pytest.approx(+derivative, rel=1e-6)
    # Transverse components vanish by symmetry.
    np.testing.assert_allclose(forces[:, 1:].numpy(), 0.0, atol=1e-12)


def test_isolated_atom_energy_and_forces_are_zero() -> None:
    params = make_params(1, PARAM_SET_1)
    positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    energy = atomic_energies(
        positions, torch.zeros((3, 3), dtype=torch.float64), [False] * 3, params
    )
    assert float(energy[0]) == pytest.approx(0.0, abs=1e-14)
    forces = forces_autograd(
        positions, torch.zeros((3, 3), dtype=torch.float64), [False] * 3, params
    )
    np.testing.assert_allclose(forces.detach().numpy(), 0.0, atol=1e-14)
    fd = forces_finite_difference(positions.numpy(), np.zeros((3, 3)), [False] * 3, params)
    np.testing.assert_allclose(fd, 0.0, atol=1e-14)


def test_bond_order_and_screening_stay_in_physical_ranges() -> None:
    # z ≥ 0 → b ∈ (0, 1]; partial screening factors ∈ (0, 1]; row products
    # S ∈ (0, 1] — on random structures this must hold without any clamps.
    rng = np.random.default_rng(41)
    from atomistic_pinn.physics import build_bop_neighbors
    from atomistic_pinn.physics.bop import _param, bond_order_factors, effective_bond_counts
    from atomistic_pinn.physics.neighbors import neighbor_geometry
    from atomistic_pinn.physics.screening import (
        partial_screening_matrix,
        screening_argument_matrix,
        screening_factors,
    )

    positions = torch.tensor(rng.uniform(0.0, 8.0, size=(6, 3)), dtype=torch.float64)
    cell = torch.tensor(np.eye(3) * 9.0)
    params = make_params(6, PARAM_SET_2)
    nbrs = build_bop_neighbors(positions.numpy(), cell.numpy(), [True] * 3, CUTOFF_RADIUS)
    vec, dist, bond = neighbor_geometry(positions, cell, nbrs, CUTOFF_RADIUS)
    arg = screening_argument_matrix(vec, dist)
    partial = partial_screening_matrix(arg, _param(params, "lambda") ** 2)
    screening = screening_factors(partial)
    z = effective_bond_counts(
        vec,
        dist,
        bond,
        screening,
        _param(params, "a"),
        _param(params, "h"),
        CUTOFF_RADIUS,
        CUTOFF_D,
    )
    b = bond_order_factors(z)
    assert bool((z >= 0.0).all())
    assert bool(((b > 0.0) & (b <= 1.0)).all())
    assert bool(((partial > 0.0) & (partial <= 1.0)).all())
    real = screening[nbrs.valid]
    assert bool(((real > 0.0) & (real <= 1.0)).all())


def test_invalid_parameter_shape_rejected() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    cell = torch.zeros((3, 3), dtype=torch.float64)
    bad_params = torch.ones((2, 7), dtype=torch.float64)
    with pytest.raises(DataContractError, match="params"):
        atomic_energies(positions, cell, [False] * 3, bad_params)
    with pytest.raises(DataContractError, match="params"):
        atomic_energies(positions, cell, [False] * 3, torch.ones((3, 8)))
