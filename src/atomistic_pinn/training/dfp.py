"""Davidson–Fletcher–Powell quasi-Newton optimizer (Sprint 3, ledger §7.8).

Textbook DFP on a flat float64 vector with a backtracking-Armijo line
search:

* inverse-Hessian update ``H⁺ = H − (H y)(H y)ᵀ/(yᵀ H y) + s sᵀ/(yᵀ s)``
  with ``s = x⁺ − x``, ``y = g⁺ − g``; warm-started with ``H = I``;
* the curvature condition ``yᵀ s > 0`` must hold for the update; on failure
  ``H`` is reset to ``I`` (standard textbook safeguard, documented);
* Armijo condition ``f(x + αd) ≤ f(x) + c₁ α gᵀ d`` with backtracking
  (α ← ρ α); a failed line search resets ``H`` once, then stops;
* monotone decrease of ``f`` is guaranteed by the Armijo acceptance.

The paper states DFP but not its variant details (line search, gradient
computation) — ledger §7.8: this is the labeled textbook implementation with
the resolved numerical-gradient route; autograd gradients are the labeled
fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DFPResult:
    """Outcome of one DFP run.

    Attributes:
        x: Final parameter vector.
        f: Final objective value.
        iterations: Iterations executed.
        history: Objective value per accepted iterate (monotone by Armijo).
        gradient_norm: Final infinity-norm of the gradient.
        stop_reason: ``"gradient_tolerance"`` | ``"max_iterations"`` |
            ``"line_search_failed"``.
    """

    x: np.ndarray
    f: float
    iterations: int
    history: list[float] = field(default_factory=list)
    gradient_norm: float = 0.0
    stop_reason: str = ""

    @property
    def n_parameters(self) -> int:
        """Length of the parameter vector."""
        return int(self.x.size)


def dfp_minimize(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    grad: Callable[[np.ndarray], np.ndarray],
    *,
    max_iterations: int = 100,
    gradient_tolerance: float = 1e-8,
    armijo_c1: float = 1e-4,
    backtracking_rho: float = 0.5,
    max_line_search_steps: int = 30,
) -> DFPResult:
    """Minimize ``f`` with textbook DFP.

    Args:
        f: Objective (numpy scalar).
        x0: Initial flat parameter vector.
        grad: Gradient function (numerical or autograd route).
        max_iterations: Iteration cap.
        gradient_tolerance: Stop when ‖g‖∞ falls below this.
        armijo_c1: Armijo sufficient-decrease constant.
        backtracking_rho: Backtracking factor (0 < ρ < 1).
        max_line_search_steps: Backtracking step cap.

    Returns:
        :class:`DFPResult` with the full accepted-iterate history.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    f_x = float(f(x))
    g = np.asarray(grad(x), dtype=np.float64)
    identity = np.eye(x.size)
    hessian = identity.copy()

    result = DFPResult(x=x, f=f_x, iterations=0, history=[f_x])
    for _ in range(max_iterations):
        result.gradient_norm = float(np.max(np.abs(g))) if g.size else 0.0
        if result.gradient_norm < gradient_tolerance:
            result.stop_reason = "gradient_tolerance"
            return result

        direction = -hessian @ g
        g_dot_d = float(g @ direction)
        if g_dot_d >= 0:  # not a descent direction (numerical breakdown)
            hessian = identity.copy()
            direction = -g
            g_dot_d = float(g @ direction)

        alpha = _backtracking_armijo(
            f,
            x,
            f_x,
            g,
            direction,
            g_dot_d,
            armijo_c1,
            backtracking_rho,
            max_line_search_steps,
        )
        if alpha is None:
            if not np.allclose(hessian, identity):
                # One reset-and-retry before giving up (textbook safeguard).
                hessian = identity.copy()
                direction = -g
                g_dot_d = float(g @ direction)
                alpha = _backtracking_armijo(
                    f,
                    x,
                    f_x,
                    g,
                    direction,
                    g_dot_d,
                    armijo_c1,
                    backtracking_rho,
                    max_line_search_steps,
                )
            if alpha is None:
                result.stop_reason = "line_search_failed"
                return result

        step = alpha * direction
        x_plus = x + step
        f_plus = float(f(x_plus))
        g_plus = np.asarray(grad(x_plus), dtype=np.float64)

        y = g_plus - g
        h_y = hessian @ y
        y_h_y = float(y @ h_y)
        y_s = float(y @ step)
        if y_s > 1e-14 and y_h_y > 1e-14:
            hessian = hessian - np.outer(h_y, h_y) / y_h_y + np.outer(step, step) / y_s
        else:
            hessian = identity.copy()  # curvature condition failed (documented)

        x, f_x, g = x_plus, f_plus, g_plus
        result.x = x.copy()
        result.f = f_x
        result.iterations += 1
        result.history.append(f_x)

    result.gradient_norm = float(np.max(np.abs(g))) if g.size else 0.0
    result.stop_reason = "max_iterations"
    return result


def _backtracking_armijo(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    f_x: float,
    g: np.ndarray,
    direction: np.ndarray,
    g_dot_d: float,
    c1: float,
    rho: float,
    max_steps: int,
) -> float | None:
    """Backtracking line search satisfying the Armijo condition."""
    alpha = 1.0
    for _ in range(max_steps):
        candidate = x + alpha * direction
        if float(f(candidate)) <= f_x + c1 * alpha * g_dot_d:
            return alpha
        alpha *= rho
    return None
