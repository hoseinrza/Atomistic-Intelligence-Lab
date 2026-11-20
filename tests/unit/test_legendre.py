"""Unit tests for the Legendre polynomial evaluation (Bonnet recurrence)."""

import numpy as np
import pytest
import scipy.special as sp

from atomistic_pinn.descriptors.legendre import legendre_values

pytestmark = pytest.mark.unit

ORDERS = (0, 1, 2, 4, 6)


def test_matches_scipy_eval_legendre() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, size=200)
    values = legendre_values(ORDERS, x)
    for m in ORDERS:
        np.testing.assert_allclose(values[m], sp.eval_legendre(m, x), rtol=1e-12, atol=1e-14)


def test_endpoint_values() -> None:
    x = np.array([-1.0, 1.0])
    values = legendre_values(ORDERS, x)
    for m in ORDERS:
        assert float(values[m][1]) == pytest.approx(1.0)
        assert float(values[m][0]) == pytest.approx((-1.0) ** m)


def test_known_central_values() -> None:
    # P4(0) = 3/8 and P6(0) = -5/16: standard values, also used by the
    # closed-form descriptor test.
    values = legendre_values((4, 6), np.array([0.0]))
    assert float(values[4][0]) == pytest.approx(3.0 / 8.0)
    assert float(values[6][0]) == pytest.approx(-5.0 / 16.0)


def test_subset_of_orders_is_consistent() -> None:
    x = np.linspace(-1.0, 1.0, 11)
    full = legendre_values(ORDERS, x)
    partial = legendre_values((2,), x)
    np.testing.assert_array_equal(partial[2], full[2])


def test_matrix_input_preserves_shape() -> None:
    x = np.linspace(-1.0, 1.0, 9).reshape(3, 3)
    values = legendre_values((4,), x)
    assert values[4].shape == (3, 3)


def test_invalid_orders_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        legendre_values((), np.array([0.0]))
    with pytest.raises(ValueError, match="non-negative"):
        legendre_values((-1,), np.array([0.0]))
