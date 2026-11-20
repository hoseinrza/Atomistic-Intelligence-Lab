"""Unit tests for the screening module (Eqs. 9-10, torch implementation)."""

import numpy as np
import pytest
import torch

from atomistic_pinn.descriptors.cutoff import cutoff_function as cutoff_numpy
from atomistic_pinn.physics.screening import (
    cutoff_function,
    partial_screening_matrix,
    screening_argument_matrix,
    screening_factors,
)

pytestmark = pytest.mark.unit

R_C = 6.0
D = 1.5


def test_torch_cutoff_matches_numpy_twin() -> None:
    grid = torch.linspace(0.0, 8.0, 81, dtype=torch.float64)
    np_grid = grid.numpy()
    np.testing.assert_allclose(
        cutoff_function(grid, R_C, D).numpy(),
        cutoff_numpy(np_grid, R_C, D),
        rtol=1e-14,
        atol=1e-18,
    )


def test_argument_matrix_geometry() -> None:
    # Two neighbors of one center: a at (3, 0, 0), b at (0, 4, 0).
    vectors = torch.tensor([[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]], dtype=torch.float64)
    distances = torch.linalg.norm(vectors, dim=2)
    arg = screening_argument_matrix(vectors, distances)
    # arg[a, b] = r_ib + r_ab − r_ia; for b = a the diagonal is r_a + 0 − r_a = 0.
    assert float(arg[0, 0, 0]) == pytest.approx(0.0, abs=1e-14)
    assert float(arg[0, 1, 1]) == pytest.approx(0.0, abs=1e-14)
    # arg[a=0, b=1] = r_ib + r_ab − r_ia = 4 + 5 − 3 = 6
    assert float(arg[0, 0, 1]) == pytest.approx(6.0, abs=1e-12)
    # arg[a=1, b=0] = r_ia + r_ab − r_ib = 3 + 5 − 4 = 4
    assert float(arg[0, 1, 0]) == pytest.approx(4.0, abs=1e-12)


def test_partial_screening_closed_forms() -> None:
    # Bond a = (3,0,0); screeners on the bond axis, on the cutoff spheroid,
    # and far outside it.
    vectors = torch.tensor(
        [[[3.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 30.0]]],
        dtype=torch.float64,
    )
    distances = torch.linalg.norm(vectors, dim=2)
    arg = screening_argument_matrix(vectors, distances)
    lam2 = torch.tensor([2.25])  # λ = 1.5
    w = partial_screening_matrix(arg, lam2, R_C, D)

    # Screener on the bond axis between i and a: arg = 1.5 + 1.5 − 3 = 0
    # → W = 1 − f_c(0) — almost complete screening.
    expected_axis = 1.0 - R_C**4 / (D**4 + R_C**4)
    assert float(w[0, 0, 1]) == pytest.approx(expected_axis, rel=1e-12)
    # Screener on the cutoff spheroid: arg = 4 + 5 − 3 = 6 = r_c → W = 1.
    assert float(w[0, 0, 2]) == pytest.approx(1.0, abs=1e-14)
    # Screener far outside: arg ≈ 30 + 30.15 − 3 ≈ 57 → W = 1.
    assert float(w[0, 0, 3]) == pytest.approx(1.0, abs=1e-14)
    # Diagonal excluded.
    assert float(w[0, 2, 2]) == pytest.approx(1.0)
    # All partial factors lie in (0, 1].
    assert bool(((w > 0.0) & (w <= 1.0)).all())


def test_screening_factors_are_row_products() -> None:
    vectors = torch.tensor(
        [[[3.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 30.0]]],
        dtype=torch.float64,
    )
    distances = torch.linalg.norm(vectors, dim=2)
    arg = screening_argument_matrix(vectors, distances)
    lam2 = torch.tensor([2.25])
    w = partial_screening_matrix(arg, lam2, R_C, D)
    s = screening_factors(w)
    expected_row0 = float(w[0, 0, 1] * w[0, 0, 2] * w[0, 0, 3])
    assert float(s[0, 0]) == pytest.approx(expected_row0, rel=1e-14)


def test_argument_nonnegative() -> None:
    # Triangle inequality guarantees arg ≥ 0 for genuine neighbor sets.
    rng = np.random.default_rng(3)
    vectors = torch.tensor(rng.uniform(-6.0, 6.0, size=(4, 9, 3)), dtype=torch.float64)
    distances = torch.linalg.norm(vectors, dim=2)
    arg = screening_argument_matrix(vectors, distances)
    assert bool((arg >= -1e-12).all())
