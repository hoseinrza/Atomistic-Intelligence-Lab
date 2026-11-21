"""Screening factors of Eqs. (9)-(10), torch/differentiable (Sprint 2).

Eq. (10) gives the partial screening of bond ``(i, a)`` by extended neighbor
``b`` of atom ``i``:

    S_{i,a,b} = 1 − f_c( r_ib + r_ab − r_ia ) · exp( −λ_i² ( r_ib + r_ab − r_ia ) )

The argument is constant on spheroids with poles at the bond atoms ``i`` and
``a`` (sum of distances to two foci), which is the paper's physical
justification: atoms outside the cutoff spheroid (argument ≥ r_c) do not
screen (factor 1); an atom on the bond axis between i and a (argument → 0)
screens almost completely.

For each atom ``i`` the partial factors form a matrix ``W[a, b]`` over its
extended neighbors (diagonal ≡ 1, a ≠ b excluded from products), and the
full screening factors of Eq. (9) are the row products ``S_ia = Π_b W[a,b]``.
Far and padded entries contribute exactly 1 (see ``physics/neighbors.py``),
so a single ``2·r_c`` list serves every bond of the atom.
"""

from __future__ import annotations

import torch

from atomistic_pinn.constants import CUTOFF_D, CUTOFF_RADIUS


def cutoff_function(
    r: torch.Tensor,
    r_c: float = CUTOFF_RADIUS,
    d: float = CUTOFF_D,
) -> torch.Tensor:
    """Eq. (5) on torch tensors — differentiable twin of the numpy version
    in ``atomistic_pinn.descriptors.cutoff`` (same closed form)."""
    x = r - r_c
    x2 = x * x  # x*x is ~10x faster than x**4 on common CPUs
    x4 = x2 * x2
    inside = x4 / (d**4 + x4)
    return inside * (x <= 0).to(inside.dtype)


def screening_argument_matrix(
    vectors: torch.Tensor,
    distances: torch.Tensor,
) -> torch.Tensor:
    """Argument matrix ``arg[a, b] = r_ib + r_ab − r_ia`` per atom.

    Args:
        vectors: ``(N, m, 3)`` neighbor vectors relative to their center.
        distances: ``(N, m)`` neighbor distances (‖vectors‖, consistent
            including sentinel slots).

    Returns:
        ``(N, m, m)`` argument matrix. ``r_ab`` is computed from the vectors
        themselves (Gram identity, clipped at 0 against round-off), so the
        sentinel slots remain exactly neutral.
    """
    r_squared = distances**2
    gram = vectors @ vectors.transpose(1, 2)
    r_ab_squared = r_squared[:, :, None] + r_squared[:, None, :] - 2.0 * gram
    r_ab = torch.sqrt(torch.clamp(r_ab_squared, min=0.0))
    # arg[a, b] = distances[b] + r_ab − distances[a]
    return distances[:, None, :] + r_ab - distances[:, :, None]


def partial_screening_matrix(
    arg: torch.Tensor,
    lam_squared: torch.Tensor,
    r_c: float = CUTOFF_RADIUS,
    d: float = CUTOFF_D,
) -> torch.Tensor:
    """Eq. (10) matrix: ``W[a, b]`` for every atom; diagonal (a == b) set to 1.

    Args:
        arg: ``(N, m, m)`` argument matrix from :func:`screening_argument_matrix`.
        lam_squared: ``(N,)`` squared per-atom λ parameters.
        r_c: Cutoff radius in Å (paper: 6.0).
        d: Truncation-range parameter in Å (paper: 1.5).

    Returns:
        ``(N, m, m)`` partial screening factors in (0, 1].
    """
    w = 1.0 - cutoff_function(arg, r_c, d) * torch.exp(-lam_squared[:, None, None] * arg)
    m = arg.shape[1]
    idx = torch.arange(m, device=arg.device)
    w[:, idx, idx] = 1.0  # a == b excluded from products (in-place, O(m))
    return w


def screening_factors(partial: torch.Tensor) -> torch.Tensor:
    """Eq. (9): full screening factors as row products, ``(N, m)``.

    The diagonal is already 1, so the plain product over the second axis
    equals the product over ``b ≠ a``.
    """
    return partial.prod(dim=2)
