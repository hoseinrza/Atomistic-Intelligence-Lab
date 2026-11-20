"""Unit tests for the Eq. (4) truncated Gaussian radial function."""

import numpy as np
import pytest

from atomistic_pinn.descriptors.radial import radial_function

pytestmark = pytest.mark.unit

R0 = 2.5
R_C = 6.0
D = 1.5


def _fc(r: float) -> float:
    x = r - R_C
    return x**4 / (D**4 + x**4) if r <= R_C else 0.0


def test_peak_value_matches_closed_form() -> None:
    # At r = r0 the Gaussian factor is 1, so f(r0) = sigma^-3 * f_c(r0);
    # with the default sigma = 1.0 the prefactor is 1.
    expected = _fc(R0)
    assert float(radial_function(R0, R0)) == pytest.approx(expected, rel=1e-12)


def test_is_a_local_maximum_at_the_center() -> None:
    peak = float(radial_function(R0, R0))
    assert peak > float(radial_function(R0 - 0.5, R0))
    assert peak > float(radial_function(R0 + 0.5, R0))


def test_zero_beyond_cutoff() -> None:
    np.testing.assert_array_equal(radial_function(np.array([7.0, 100.0]), R0), 0.0)


def test_negligible_near_cutoff() -> None:
    # The C^3 cutoff makes the function vanish extremely fast: the value
    # one micrometre inside the cutoff is below 1e-20.
    assert abs(float(radial_function(R_C - 1e-6, R0))) < 1e-20


def test_sigma_prefactor_scales_as_inverse_cube() -> None:
    # f(r0) with sigma = 2 must equal f_c(r0) / 2^3.
    expected = _fc(R0) / 8.0
    assert float(radial_function(R0, R0, sigma=2.0)) == pytest.approx(expected, rel=1e-12)


def test_non_positive_sigma_rejected() -> None:
    with pytest.raises(ValueError, match="sigma"):
        radial_function(R0, R0, sigma=0.0)
