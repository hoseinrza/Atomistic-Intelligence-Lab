"""The analytical bond-order potential, Eqs. (6)-(12) (Sprint 2).

    E_i = ½ Σ_{j≠i} [ e^{−A_iα_i r_ij} − S_ij b_ij e^{−B_iβ_i r_ij} ] f_c(r_ij) + E_i^(p)
    b_ij = (1 + z_ij)^(−1/2)
    z_ij = a_i² Σ_{k≠i,j} S_ik (cos θ_jik + h_i)² f_c(r_ik)
    S_ij = Π_{k≠i,j} S_ijk                              (screening.py)
    E_i^(p) = −σ_i ( Σ_{j≠i} S_ij b_ij f_c(r_ij) )^(1/2)

All eight parameters (A, B, α, β, a, h, σ, λ) are *per-atom* quantities —
in the PINN model they are predicted by the network; here they arrive as an
``(N, 8)`` tensor in the order of ``constants.BOP_PARAMETER_NAMES``. Sums
carry the center atom's parameters, and the ½ prefactors restore pair
symmetry once ``E_i`` is summed over atoms.

Numerical safety (proven properties, no clamps):

* ``z_ij ≥ 0`` (squares times non-negative f_c, screening in (0, 1]) and
  ``ρ_i ≥ 0`` by construction, so both square roots (Eqs. 7, 11) never see
  negative arguments;
* exponentials have non-positive arguments for physical (positive) A·α,
  B·β and λ² — bounded by 1, no overflow; unphysical negative parameters
  could overflow and are the caller's responsibility (the paper constrains
  nothing — see docs/equations.md §7.5);

Implementation: one extended ``2·r_c`` neighbor list per call
(``physics/neighbors.py``), everything vectorized over atoms with padded
slot count m. Transient tensors are ``(N, m, m)`` — for typical training
supercells (~35 atoms, ~450 images within 12 Å) this is tens of MB; very
large structures should be chunked per atom block (documented engineering
note for Sprint 3/4).
"""

from __future__ import annotations

import torch

from atomistic_pinn.constants import BOP_PARAMETER_NAMES, CUTOFF_D, CUTOFF_RADIUS
from atomistic_pinn.exceptions import DataContractError
from atomistic_pinn.physics.neighbors import (
    BOPNeighbors,
    build_bop_neighbors,
    neighbor_geometry,
)
from atomistic_pinn.physics.screening import (
    cutoff_function,
    partial_screening_matrix,
    screening_argument_matrix,
    screening_factors,
)

_PARAM_INDEX = {name: k for k, name in enumerate(BOP_PARAMETER_NAMES)}


def _param(params: torch.Tensor, name: str) -> torch.Tensor:
    """Column ``name`` of the (N, 8) parameter tensor."""
    return params[:, _PARAM_INDEX[name]]


def bond_order_factors(z: torch.Tensor) -> torch.Tensor:
    """Eq. (7): ``b = (1 + z)^(−1/2)``. ``z ≥ 0`` by construction."""
    return 1.0 / torch.sqrt(1.0 + z)


def effective_bond_counts(
    vectors: torch.Tensor,
    distances: torch.Tensor,
    bond_mask: torch.Tensor,
    screening: torch.Tensor,
    a_param: torch.Tensor,
    h_param: torch.Tensor,
    r_c: float,
    d: float,
) -> torch.Tensor:
    """Eq. (8): ``z_ij`` for every slot of every atom, ``(N, m)``.

    The sum runs over bond-tier slots ``k ≠ j`` (``f_c(r_ik)`` makes the
    restriction to ``r_ik < r_c`` exact). Non-bond slots come out as 0.
    """
    _n_atoms, m = distances.shape
    gram = vectors @ vectors.transpose(1, 2)
    denom = distances[:, :, None] * distances[:, None, :]
    cosines = gram / denom
    eye = torch.eye(m, dtype=torch.bool, device=distances.device)
    angular = (cosines + h_param[:, None, None]) ** 2 * torch.where(
        eye, torch.zeros_like(cosines), torch.ones_like(cosines)
    )
    # k-terms: S_ik · f_c(r_ik), restricted to the bond tier.
    terms = screening * cutoff_function(distances, r_c, d) * bond_mask.to(distances.dtype)
    z = a_param[:, None] ** 2 * torch.einsum("nb,njb->nj", terms, angular)  # Eq. (8): a_i²
    return z


def atomic_energies(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: list[bool] | tuple[bool, ...],
    params: torch.Tensor,
    cutoff: float = CUTOFF_RADIUS,
    d: float = CUTOFF_D,
    neighbors: BOPNeighbors | None = None,
) -> torch.Tensor:
    """Atomic energies ``E_i`` of Eq. (6) for every atom, ``(N,)``.

    Args:
        positions: ``(N, 3)`` float64 tensor (may require grad — gradients
            w.r.t. positions support the autograd force cross-check).
        cell: ``(3, 3)`` float64 tensor of lattice rows.
        pbc: Three periodic flags.
        params: ``(N, 8)`` float64 tensor of per-atom BOP parameters in the
            order ``A, B, alpha, beta, a, h, sigma, lambda``.
        cutoff: Cutoff radius r_c in Å (paper: 6.0).
        d: Truncation-range parameter in Å (paper: 1.5).

    Returns:
        ``(N,)`` float64 tensor of atomic energies in eV. Inputs are
        upcast to float64 — all physics runs in float64 regardless of the
        caller's dtype (torch's default float32 is silently upcast).
    """
    positions = positions.to(torch.float64)
    cell = cell.to(torch.float64)
    params = params.to(torch.float64)
    if params.ndim != 2 or params.shape[1] != len(BOP_PARAMETER_NAMES):
        raise DataContractError(
            f"params must have shape (N, {len(BOP_PARAMETER_NAMES)}), got {tuple(params.shape)}"
        )
    n_atoms = positions.shape[0]
    if params.shape[0] != n_atoms:
        raise DataContractError(
            f"params has {params.shape[0]} rows but positions has {n_atoms} atoms"
        )

    if neighbors is None:
        neighbors = build_bop_neighbors(
            positions.detach().cpu().numpy(),
            cell.detach().cpu().numpy(),
            pbc,
            cutoff,
        )
    vectors, distances, bond_mask = neighbor_geometry(positions, cell, neighbors, cutoff)

    lam_squared = _param(params, "lambda") ** 2
    arg = screening_argument_matrix(vectors, distances)
    partial = partial_screening_matrix(arg, lam_squared, cutoff, d)
    screening = screening_factors(partial)  # (N, m): S_ia for every slot

    z = effective_bond_counts(
        vectors,
        distances,
        bond_mask,
        screening,
        _param(params, "a"),
        _param(params, "h"),
        cutoff,
        d,
    )
    b_factors = bond_order_factors(z)  # (N, m)
    fc_r = cutoff_function(distances, cutoff, d)
    bond = bond_mask.to(distances.dtype)

    # Eq. (6): pair terms over bond-tier slots j.
    repulsive = torch.exp(-(_param(params, "A") * _param(params, "alpha"))[:, None] * distances)
    attractive = torch.exp(-(_param(params, "B") * _param(params, "beta"))[:, None] * distances)
    pair = 0.5 * (repulsive - screening * b_factors * attractive) * fc_r
    pair_energy = (pair * bond).sum(dim=1)

    # Eq. (11): promotion / embedding energy.
    rho = (screening * b_factors * fc_r * bond).sum(dim=1)
    promotion = -_param(params, "sigma") * torch.sqrt(rho)

    return pair_energy + promotion


def total_energy(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: list[bool] | tuple[bool, ...],
    params: torch.Tensor,
    cutoff: float = CUTOFF_RADIUS,
    d: float = CUTOFF_D,
    neighbors: BOPNeighbors | None = None,
) -> torch.Tensor:
    """Total energy ``E = Σ_i E_i`` (scalar tensor, eV)."""
    return atomic_energies(positions, cell, pbc, params, cutoff, d, neighbors).sum()
