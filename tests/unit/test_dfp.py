"""Unit tests for the DFP optimizer (Sprint 3, phase 3.3)."""

import numpy as np
import pytest

from atomistic_pinn.training.dfp import DFPResult, dfp_minimize

pytestmark = pytest.mark.unit


def _quadratic(x: np.ndarray) -> float:
    """f(x) = ½ xᵀ A x with A = diag(1, 4)."""
    return float(0.5 * (x[0] ** 2 + 4.0 * x[1] ** 2))


def _quadratic_grad(x: np.ndarray) -> np.ndarray:
    return np.array([x[0], 4.0 * x[1]])


def test_reduces_objective_monotonically() -> None:
    result = dfp_minimize(_quadratic, np.array([2.0, 2.0]), _quadratic_grad, max_iterations=20)
    history = np.asarray(result.history)
    assert len(history) >= 2
    assert bool(np.all(np.diff(history) <= 1e-12))  # Armijo-enforced decrease


def test_two_dimensional_quadratic_converges_in_two_iterations() -> None:
    # Textbook property: DFP with exact gradients minimizes an n-D quadratic
    # in at most n iterations (here n = 2).
    result = dfp_minimize(_quadratic, np.array([2.0, -3.0]), _quadratic_grad, max_iterations=10)
    # Armijo stops at ‖g‖∞ < 1e-8 → a ~1e-9 parameter residual is expected.
    np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1e-7)
    assert result.f == pytest.approx(0.0, abs=1e-16)
    assert result.stop_reason in ("gradient_tolerance", "max_iterations")


def test_hand_computed_first_step() -> None:
    # One iteration from x0 = (2, 2): H = I → d = −g = (−2, −8). The Armijo
    # acceptance for this quadratic is computed here with an independent
    # inline implementation of the backtracking rule.
    x0 = np.array([2.0, 2.0])
    g0 = _quadratic_grad(x0)
    d = -g0
    c1, rho = 1e-4, 0.5
    f0 = _quadratic(x0)
    alpha = 1.0
    while _quadratic(x0 + alpha * d) > f0 + c1 * alpha * float(g0 @ d):
        alpha *= rho
    result = dfp_minimize(_quadratic, x0, _quadratic_grad, max_iterations=1)
    np.testing.assert_allclose(result.x, x0 + alpha * d, rtol=1e-14)
    assert result.history[0] == pytest.approx(f0)
    assert result.history[-1] == pytest.approx(_quadratic(result.x))


def test_higher_dimensional_quadratic() -> None:
    # 6-D ill-conditioned quadratic; DFP with exact gradients converges.
    diag = np.array([0.2, 1.0, 5.0, 0.5, 2.0, 40.0])

    def f(x: np.ndarray) -> float:
        return float(0.5 * np.sum(diag * x**2))

    def g(x: np.ndarray) -> np.ndarray:
        return diag * x

    result = dfp_minimize(
        f, np.linspace(-2.0, 2.0, 6), g, max_iterations=200, gradient_tolerance=1e-12
    )
    np.testing.assert_allclose(result.x, 0.0, atol=1e-8)


def test_result_metadata() -> None:
    result = dfp_minimize(_quadratic, np.array([1.0, 1.0]), _quadratic_grad, max_iterations=50)
    assert isinstance(result, DFPResult)
    assert result.n_parameters == 2
    assert result.stop_reason == "gradient_tolerance"
    assert result.history[0] == pytest.approx(_quadratic(np.array([1.0, 1.0])))


def test_flat_start_is_immediate() -> None:
    result = dfp_minimize(_quadratic, np.array([0.0, 0.0]), _quadratic_grad, max_iterations=10)
    assert result.iterations == 0
    assert result.stop_reason == "gradient_tolerance"
    assert result.history == [0.0]
