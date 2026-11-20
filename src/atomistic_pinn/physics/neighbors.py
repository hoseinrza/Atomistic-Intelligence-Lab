"""Extended neighbor data for the BOP physics layer (Sprint 2).

The screening argument of Eq. (10) for a bond (i, a) screened by neighbor b
is ``r_ib + r_ab − r_ia``, which is smaller than r_c only for atoms within
``r_c + r_ia ≤ 2·r_c`` of the center. The physics layer therefore builds ONE
extended neighbor list at ``2·r_c`` and derives the bond tier
(``r < r_c``, needed by Eqs. 6, 8 and 11) by masking.

Padding: atoms are padded to a common slot count ``m``. Padded slots carry a
far-away sentinel *vector* so that every padded contribution is exactly
neutral:

* ``‖v_sentinel‖ ≫ r_c``  → the slot is never in the bond tier;
* the screening argument ``arg[a, b] = r_ib + r_ab − r_ia`` for a padded b
  is ≥ ``2‖v_sentinel‖ − 2·r_c ≫ r_c`` → ``f_c = 0`` → the partial screening
  factor W = 1 → products (Eq. 9) are unchanged.

Neighbor lists are rebuilt from (detached) positions on every call; forces
via finite differences (the paper's method) rely on this.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from atomistic_pinn.geometry.neighbor_list import NeighborList

_PADDING_FACTOR = 10.0
"""Sentinel distance factor: padded vectors sit at 10·cutoff per component."""


@dataclass(frozen=True)
class BOPNeighbors:
    """Padded extended-neighbor metadata for one configuration.

    Attributes:
        indices: ``(N, m)`` int64 neighbor indices (0 in padded slots).
        shifts: ``(N, m, 3)`` int64 lattice shifts (row-vector convention).
        valid: ``(N, m)`` bool, True for genuine neighbor slots.
    """

    indices: torch.Tensor
    shifts: torch.Tensor
    valid: torch.Tensor

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return int(self.indices.shape[0])

    @property
    def max_neighbors(self) -> int:
        """Padded slot count m."""
        return int(self.indices.shape[1])


def build_bop_neighbors(
    positions: NDArray[np.float64],
    cell: NDArray[np.float64],
    pbc: NDArray[np.bool_] | Sequence[bool],
    cutoff: float,
) -> BOPNeighbors:
    """Build the extended (``2·cutoff``) neighbor metadata from numpy inputs.

    Args:
        positions: Cartesian positions (N, 3) in Å (numpy; detached — the
            differentiable vectors are recomputed in torch from these
            indices/shifts by :func:`neighbor_geometry`).
        cell: Lattice rows (3, 3) in Å.
        pbc: Three periodic flags.
        cutoff: BOP cutoff r_c in Å; the list is built at ``2·cutoff``.

    Returns:
        Padded :class:`BOPNeighbors` on the CPU.
    """
    neighbor_list = NeighborList.build(
        np.asarray(positions, dtype=np.float64),
        np.asarray(cell, dtype=np.float64),
        pbc,
        2.0 * cutoff,
    )
    n_atoms = neighbor_list.n_atoms
    max_neighbors = max((len(a) for a in neighbor_list.indices), default=0)
    max_neighbors = max(max_neighbors, 1)

    indices = np.zeros((n_atoms, max_neighbors), dtype=np.int64)
    shifts = np.zeros((n_atoms, max_neighbors, 3), dtype=np.int64)
    valid = np.zeros((n_atoms, max_neighbors), dtype=bool)
    for i in range(n_atoms):
        k = len(neighbor_list.indices[i])
        if k:
            indices[i, :k] = neighbor_list.indices[i]
            shifts[i, :k] = neighbor_list.shifts[i]
            valid[i, :k] = True
    return BOPNeighbors(
        indices=torch.from_numpy(indices),
        shifts=torch.from_numpy(shifts),
        valid=torch.from_numpy(valid),
    )


def neighbor_geometry(
    positions: torch.Tensor,
    cell: torch.Tensor,
    neighbors: BOPNeighbors,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable neighbor geometry from padded metadata.

    Args:
        positions: ``(N, 3)`` float64 tensor (may require grad).
        cell: ``(3, 3)`` float64 tensor.
        neighbors: Padded metadata from :func:`build_bop_neighbors`.
        cutoff: BOP cutoff r_c in Å (used for the bond tier and the
            sentinel offset).

    Returns:
        ``(vectors, distances, bond_mask)`` with shapes ``(N, m, 3)``,
        ``(N, m)`` and ``(N, m)``. Distances are ``‖vectors‖`` so that the
        sentinel slots are mutually consistent; gradients flow through the
        valid slots. Inputs are upcast to float64 — all physics runs in
        float64 regardless of the caller's dtype.
    """
    positions = positions.to(torch.float64)
    cell = cell.to(torch.float64)
    offset_value = _PADDING_FACTOR * cutoff
    offset = torch.full((3,), offset_value, dtype=positions.dtype, device=positions.device)
    vectors = (
        positions[neighbors.indices]
        + neighbors.shifts.to(positions.dtype) @ cell
        - positions[:, None, :]
    )
    vectors = torch.where(neighbors.valid[:, :, None], vectors, offset)
    distances = torch.linalg.norm(vectors, dim=2)
    bond_mask = distances < cutoff
    return vectors, distances, bond_mask
