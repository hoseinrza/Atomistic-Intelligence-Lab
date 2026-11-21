"""Differentiable (torch) descriptor path, Eq. (3) (Sprint 4).

Torch twin of :mod:`atomistic_pinn.descriptors.builder` with the *same*
mathematical semantics, expressed on padded neighbor metadata so that
gradients flow from the total energy back to the atomic positions:

    positions → vectors/distances (torch, fixed list) → f(r) → P_m(cos θ)
              → g_i^(m)[c] sums → network → p_i → BOP → E_s

The neighbor *list* is built once (numpy, deterministic, `geometry` layer)
and held fixed; the C³-smooth cutoff of Eq. (5) is what keeps the
descriptor computation differentiable across the cutoff as atoms move
(``docs/md_plan.md`` §3). Agreement with the numpy builder is enforced by
the scientific tests to the same standard as the Sprint 2 physics twin
(rtol 1e-9 on random structures).

Padding trick (same as ``physics.neighbors``): invalid slots carry a large
constant offset vector whose distance lies far beyond the cutoff, so their
radial factor is exactly zero and they contribute nothing to any sum while
adding no position dependence (hence no spurious gradients).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from atomistic_pinn.descriptors.builder import DescriptorConfig
from atomistic_pinn.geometry.neighbor_list import NeighborList

_PADDING_FACTOR = 1.5


@dataclass(frozen=True)
class DescriptorNeighbors:
    """Padded neighbor metadata for the differentiable descriptor path.

    Attributes:
        indices: ``(N, m)`` int64 neighbor indices (invalid slots 0).
        shifts: ``(N, m, 3)`` int64 lattice-image shifts.
        valid: ``(N, m)`` bool mask of real neighbor slots.
    """

    indices: torch.Tensor
    shifts: torch.Tensor
    valid: torch.Tensor

    @property
    def max_neighbors(self) -> int:
        """Padded slot count ``m``."""
        return int(self.indices.shape[1])


def build_descriptor_neighbors(
    positions: NDArray[np.float64],
    cell: NDArray[np.float64],
    pbc: list[bool] | tuple[bool, ...],
    cutoff: float,
) -> DescriptorNeighbors:
    """Build padded descriptor neighbor metadata (numpy list, torch layout).

    The underlying enumeration is the deterministic full-PBC
    ``NeighborList.build`` of Sprint 1 at the descriptor cutoff; entries are
    padded to a fixed ``(N, m)`` layout so the torch recomputation is
    batched and position-independent per slot.
    """
    neighbor_list = NeighborList.build(
        np.asarray(positions, dtype=np.float64),
        np.asarray(cell, dtype=np.float64),
        np.asarray(pbc, dtype=bool),
        cutoff,
    )
    n_atoms = neighbor_list.n_atoms
    max_neighbors = max((arr.size for arr in neighbor_list.indices), default=0)
    indices = np.zeros((n_atoms, max_neighbors), dtype=np.int64)
    shifts = np.zeros((n_atoms, max_neighbors, 3), dtype=np.int64)
    valid = np.zeros((n_atoms, max_neighbors), dtype=bool)
    for i in range(n_atoms):
        k = len(neighbor_list.indices[i])
        if k:
            indices[i, :k] = neighbor_list.indices[i]
            shifts[i, :k] = neighbor_list.shifts[i]
            valid[i, :k] = True
    return DescriptorNeighbors(
        indices=torch.from_numpy(indices),
        shifts=torch.from_numpy(shifts),
        valid=torch.from_numpy(valid),
    )


def _torch_cutoff(r: torch.Tensor, r_c: float, d: float) -> torch.Tensor:
    """Eq. (5) cutoff factor (torch twin of ``descriptors.cutoff``)."""
    x = r - r_c
    x4 = (x * x) * (x * x)
    inside = x4 / (d**4 + x4)
    return torch.where(x <= 0.0, inside, torch.zeros_like(inside))


def _torch_radial(r: torch.Tensor, r0: float, sigma: float, r_c: float, d: float) -> torch.Tensor:
    """Eq. (4) truncated Gaussian (torch twin of ``descriptors.radial``)."""
    gaussian = torch.exp(-((r - r0) ** 2) / sigma**2) / sigma**3
    return gaussian * _torch_cutoff(r, r_c, d)


def _torch_legendre(orders: tuple[int, ...], x: torch.Tensor) -> dict[int, torch.Tensor]:
    """Bonnet recurrence (torch twin of ``descriptors.legendre``)."""
    values: dict[int, torch.Tensor] = {}
    p_prev = torch.ones_like(x)  # P_0
    p_curr = x.clone()  # P_1
    if 0 in orders:
        values[0] = torch.ones_like(x)
    if 1 in orders:
        values[1] = x.clone()
    for m in range(2, max(orders) + 1):
        current = ((2 * m - 1) * x * p_curr - (m - 1) * p_prev) / m
        p_prev, p_curr = p_curr, current
        if m in orders:
            values[m] = current
    return values


def _cosine_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """cos θ_jk between all neighbor-vector pairs (torch twin of
    ``geometry.measure.cosine_matrix``), shape ``(N, m, m)``.

    Mirrors the numpy semantics: zero-norm vectors yield zero cosines and
    values are clipped to [−1, 1] against round-off before the Legendre
    evaluation.
    """
    norms = torch.linalg.norm(vectors, dim=2)  # (N, m)
    dots = torch.einsum("njc,nkc->njk", vectors, vectors)  # (N, m, m)
    denom = norms[:, :, None] * norms[:, None, :]
    safe = torch.where(denom > 0.0, denom, torch.ones_like(denom))
    cos = torch.where(denom > 0.0, dots / safe, torch.zeros_like(dots))
    return torch.clamp(cos, -1.0, 1.0)


def compute_descriptors_differentiable(
    positions: torch.Tensor,
    cell: torch.Tensor,
    neighbors: DescriptorNeighbors,
    config: DescriptorConfig | None = None,
) -> torch.Tensor:
    """Eq. (3) descriptors as a differentiable function of positions.

    Args:
        positions: ``(N, 3)`` tensor (may require grad); upcast to float64.
        cell: ``(3, 3)`` lattice rows.
        neighbors: Padded metadata from :func:`build_descriptor_neighbors`.
        config: Descriptor configuration (defaults to the paper values).

    Returns:
        ``(N, k)`` float64 tensor, order-major layout identical to the
        numpy builder.
    """
    cfg = config or DescriptorConfig()
    positions = positions.to(torch.float64)
    cell = cell.to(torch.float64)
    offset_value = _PADDING_FACTOR * cfg.cutoff_radius
    offset = torch.full((3,), offset_value, dtype=positions.dtype)
    vectors = (
        positions[neighbors.indices]
        + neighbors.shifts.to(positions.dtype) @ cell
        - positions[:, None, :]
    )
    vectors = torch.where(neighbors.valid[:, :, None], vectors, offset)
    distances = torch.linalg.norm(vectors, dim=2)  # (N, m)
    centers = torch.tensor(cfg.radial_centers, dtype=torch.float64)
    radial = torch.stack(
        [
            _torch_radial(distances, float(r0), cfg.sigma, cfg.cutoff_radius, cfg.cutoff_d)
            for r0 in centers
        ],
        dim=2,
    )  # (N, m, C)
    polynomials = _torch_legendre(cfg.orders, _cosine_matrix(vectors))  # order → (N, m, m)
    columns: list[torch.Tensor] = []
    for order in cfg.orders:
        p = polynomials[order]  # (N, m, m)
        if not cfg.include_same_pair:
            p = p - torch.diag_embed(torch.diagonal(p, dim1=1, dim2=2))
        # g[n, c] = Σ_{a,b} P[n, a, b] · rad[n, a, c] · rad[n, b, c]
        g = torch.einsum("nab,nac,nbc->nc", p, radial, radial)
        columns.append(g)
    return torch.cat(columns, dim=1)  # order-major: (N, k)
