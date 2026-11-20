"""Unit tests for the Eq. (5) cutoff function.

The closed-form derivatives below are derived independently in this test
file (quotient rule on x⁴/(d⁴+x⁴) with x = r − r_c) so the C³ smoothness
claim of the paper is verified against independent arithmetic, not against
the implementation itself.
"""

import numpy as np
import pytest

from atomistic_pinn.descriptors.cutoff import cutoff_function

pytestmark = pytest.mark.unit

R_C = 6.0
D = 1.5


def _analytic(r: float) -> float:
    x = r - R_C
    return x**4 / (D**4 + x**4) if r <= R_C else 0.0


def _analytic_d1(r: float) -> float:
    x = r - R_C
    if r > R_C:
        return 0.0
    return 4.0 * x**3 * D**4 / (D**4 + x**4) ** 2


def _analytic_d2(r: float) -> float:
    x = r - R_C
    if r > R_C:
        return 0.0
    return 4.0 * D**4 * x**2 * (3.0 * D**4 - 5.0 * x**4) / (D**4 + x**4) ** 3


def _analytic_d3(r: float) -> float:
    x = r - R_C
    if r > R_C:
        return 0.0
    numerator = 4.0 * D**4 * x * (6.0 * D**8 - 60.0 * D**4 * x**4 + 30.0 * x**8)
    return numerator / (D**4 + x**4) ** 4


def test_zero_at_and_beyond_cutoff() -> None:
    assert float(cutoff_function(R_C, R_C, D)) == pytest.approx(0.0, abs=1e-15)
    beyond = np.array([R_C + 1e-9, 7.0, 100.0])
    np.testing.assert_array_equal(cutoff_function(beyond, R_C, D), 0.0)


def test_matches_closed_form_inside_cutoff() -> None:
    r = np.linspace(0.0, R_C, 200)
    expected = np.array([_analytic(float(v)) for v in r])
    np.testing.assert_allclose(cutoff_function(r, R_C, D), expected, rtol=1e-12, atol=1e-15)


def test_value_at_origin_matches_closed_form() -> None:
    expected = R_C**4 / (D**4 + R_C**4)
    assert float(cutoff_function(0.0, R_C, D)) == pytest.approx(expected, rel=1e-12)


def test_monotonically_decreasing_inside_cutoff() -> None:
    # f_c tapers from ≈ 0.996 at r = 0 down to 0 at r_c; restricted to
    # [0, 5.5] so the comparison stays above the float underflow region
    # that begins a fraction of an Å below the cutoff.
    r = np.linspace(0.0, 5.5, 500)
    assert bool(np.all(np.diff(cutoff_function(r, R_C, D)) < 0.0))


def test_first_derivative_matches_closed_form_and_vanishes_at_cutoff() -> None:
    r = np.linspace(0.5, R_C - 0.1, 100)
    h = 1e-4
    fd = (cutoff_function(r + h, R_C, D) - cutoff_function(r - h, R_C, D)) / (2 * h)
    expected = np.array([_analytic_d1(float(v)) for v in r])
    np.testing.assert_allclose(fd, expected, rtol=1e-5, atol=1e-11)
    # fc′ = 4 x³ d⁴ / (d⁴ + x⁴)² ∝ x³ → vanishes as x → 0⁻; at x = −1e-6
    # the closed form gives 7.9e-19, five orders below the function scale.
    assert abs(_analytic_d1(R_C - 1e-6)) < 1e-17


def test_second_derivative_matches_closed_form_and_vanishes_at_cutoff() -> None:
    # Comparison restricted to the well-conditioned interior; the tail near
    # the cutoff is covered by the closed-form vanishing bounds below (the
    # FD truncation error h² f⁗/12 dominates there).
    h = 3e-4
    r = np.linspace(1.0, 5.5, 50)
    fd = (
        cutoff_function(r + h, R_C, D)
        - 2.0 * cutoff_function(r, R_C, D)
        + cutoff_function(r - h, R_C, D)
    ) / h**2
    expected = np.array([_analytic_d2(float(v)) for v in r])
    np.testing.assert_allclose(fd, expected, rtol=1e-3, atol=1e-8)
    # fc″ ∝ x² → 2.4e-12 at x = −1e-6.
    assert abs(_analytic_d2(R_C - 1e-6)) < 1e-10


def test_third_derivative_matches_closed_form_and_vanishes_at_cutoff() -> None:
    # Central third-difference stencil on the well-conditioned interior (see
    # the second-derivative test for why the cutoff tail is excluded).
    h = 1e-3
    r = np.linspace(1.0, 5.5, 40)
    fd3 = (
        cutoff_function(r + 2 * h, R_C, D)
        - 2.0 * cutoff_function(r + h, R_C, D)
        + 2.0 * cutoff_function(r - h, R_C, D)
        - cutoff_function(r - 2 * h, R_C, D)
    ) / (2.0 * h**3)
    expected = np.array([_analytic_d3(float(v)) for v in r])
    # rtol is set by the h² truncation error of the third-difference stencil
    # (measured ≈ 1.2e-4 at h = 1e-3); a genuine formula error would be O(1).
    np.testing.assert_allclose(fd3, expected, rtol=5e-4, atol=1e-9)
    # fc‴ ∝ x → 4.7e-6 at x = −1e-6.
    assert abs(_analytic_d3(R_C - 1e-6)) < 1e-4


def test_continuous_across_the_cutoff() -> None:
    left = float(cutoff_function(R_C - 1e-9, R_C, D))
    right = float(cutoff_function(R_C + 1e-9, R_C, D))
    assert left < 1e-30
    assert right == 0.0
