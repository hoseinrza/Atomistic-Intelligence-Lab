"""Neighbor lists with full periodic-image enumeration (Sprint 1).

The paper's descriptor sum (Eq. 3) and BOP sums (Eqs. 6-12) run over *all*
neighbors within the cutoff, including periodic images: in a small DFT
supercell the same atom can appear several times within 6 Å of another atom,
and each image is a distinct physical neighbor of the periodic system. The
neighbor list therefore enumerates every (j, lattice shift) pair with
|r_ij| < cutoff, including images of atom i itself (shift ≠ 0); only the
exact self pair (i, i, shift = 0) is excluded.

Implementation: brute-force enumeration over the lattice shift box derived
from the cutoff (``_shift_ranges``), vectorized over atom pairs, with a
one-shell safety margin so that positions displaced by up to one cell width
from the home cell (unwrapped coordinates) cannot lose neighbors. For the
database sizes of this project (tens to hundreds of atoms per supercell)
this is fast enough; a cell-list acceleration can replace the internals
later without changing the interface (engineering note,
docs/architecture.md).

Conventions:

* distance vectors point from atom i to the neighbor image:
  ``r_ij = positions[j] + shift @ cell − positions[i]``;
* shifts are integer lattice translations in row-vector convention
  (``shift @ cell``);
* each atom's entries are sorted deterministically by
  (distance, j, shift);
* the cutoff comparison is strict (``< cutoff``) — the cutoff function of
  Eq. (5) vanishes at r_c, so the boundary never contributes physically.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from atomistic_pinn.exceptions import DataContractError


def _shift_ranges(
    cell: NDArray[np.float64],
    pbc: NDArray[np.bool_],
    cutoff: float,
) -> list[range]:
    """Lattice shift ranges that certainly cover all images within ``cutoff``.

    For lattice direction k the perpendicular plane spacing is
    ``w_k = 1/|b_k|`` with b_k the reciprocal rows (``b_k · a_m = δ_km``).
    Any image with ``|S_k| ≥ cutoff / w_k + 1`` is at least one plane spacing
    beyond the cutoff from every point of the home cell and cannot
    contribute; one extra shell is kept as a safety margin for unwrapped
    positions. Non-periodic directions use only shift 0.
    """
    if not bool(np.any(pbc)):
        return [range(0, 1)] * 3
    recip = np.linalg.inv(cell).T
    ranges: list[range] = []
    for k in range(3):
        if not pbc[k]:
            ranges.append(range(0, 1))
        else:
            n_k = int(np.ceil(cutoff * float(np.linalg.norm(recip[k])))) + 1
            ranges.append(range(-n_k, n_k + 1))
    return ranges


@dataclass(frozen=True)
class NeighborList:
    """All periodic-image neighbor pairs within ``cutoff`` of every atom.

    Attributes:
        cutoff: Cutoff radius in Å used to build the list (all stored
            distances are strictly smaller).
        indices: Per atom i, integer array of neighbor indices j.
        shifts: Per atom i, integer shift vectors of shape (m_i, 3).
        vectors: Per atom i, distance vectors of shape (m_i, 3), in Å.
        distances: Per atom i, distances of shape (m_i,), in Å.
    """

    cutoff: float
    indices: tuple[NDArray[np.int64], ...]
    shifts: tuple[NDArray[np.int64], ...]
    vectors: tuple[NDArray[np.float64], ...]
    distances: tuple[NDArray[np.float64], ...]

    @property
    def n_atoms(self) -> int:
        """Number of atoms the list was built for."""
        return len(self.indices)

    @property
    def n_entries(self) -> int:
        """Total number of directed neighbor entries (i → j images)."""
        return int(sum(arr.size for arr in self.indices))

    @classmethod
    def build(
        cls,
        positions: NDArray[np.float64],
        cell: NDArray[np.float64],
        pbc: Sequence[bool] | NDArray[np.bool_],
        cutoff: float,
    ) -> NeighborList:
        """Build the neighbor list for a configuration.

        Args:
            positions: Cartesian positions, shape (n_atoms, 3), in Å.
            cell: Lattice vectors as rows, shape (3, 3), in Å. Must be
                invertible when any direction is periodic.
            pbc: One periodic flag per lattice direction.
            cutoff: Cutoff radius in Å (positive).

        Returns:
            A deterministic :class:`NeighborList`.

        Raises:
            DataContractError: If shapes are invalid, values are non-finite,
                the cutoff is not positive, or a singular cell is combined
                with periodic boundaries.
        """
        positions = np.asarray(positions, dtype=np.float64)
        cell = np.asarray(cell, dtype=np.float64)
        pbc_arr = np.asarray(pbc, dtype=bool)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise DataContractError("positions must have shape (n_atoms, 3)")
        if positions.shape[0] == 0:
            raise DataContractError("neighbor lists require at least one atom")
        if not bool(np.isfinite(positions).all()):
            raise DataContractError("positions contain non-finite values")
        if cutoff <= 0.0:
            raise DataContractError(f"cutoff must be positive, got {cutoff}")
        n_atoms = positions.shape[0]
        if bool(np.any(pbc_arr)):
            if cell.shape != (3, 3) or not bool(np.isfinite(cell).all()):
                raise DataContractError("periodic structures require a finite (3, 3) cell")
            if abs(float(np.linalg.det(cell))) < 1e-12:
                raise DataContractError("cell is singular but periodic boundaries are requested")

        i_chunks: list[NDArray[np.int64]] = []
        j_chunks: list[NDArray[np.int64]] = []
        s_chunks: list[NDArray[np.int64]] = []
        v_chunks: list[NDArray[np.float64]] = []
        d_chunks: list[NDArray[np.float64]] = []

        for shift in itertools.product(*_shift_ranges(cell, pbc_arr, cutoff)):
            s_vec = np.asarray(shift, dtype=np.int64)
            image = positions + s_vec.astype(np.float64) @ cell
            dvec = image[None, :, :] - positions[:, None, :]
            dist = np.linalg.norm(dvec, axis=2)
            mask = dist < cutoff
            if shift == (0, 0, 0):
                np.fill_diagonal(mask, False)
            row, col = np.nonzero(mask)
            if row.size == 0:
                continue
            i_chunks.append(row.astype(np.int64))
            j_chunks.append(col.astype(np.int64))
            s_chunks.append(np.broadcast_to(s_vec, (row.size, 3)).copy())
            v_chunks.append(dvec[row, col])
            d_chunks.append(dist[row, col])

        empty_i: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        empty_s: NDArray[np.int64] = np.empty((0, 3), dtype=np.int64)
        empty_v: NDArray[np.float64] = np.empty((0, 3), dtype=np.float64)
        empty_d: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        if not i_chunks:
            return cls(
                cutoff=float(cutoff),
                indices=tuple(empty_i.copy() for _ in range(n_atoms)),
                shifts=tuple(empty_s.copy() for _ in range(n_atoms)),
                vectors=tuple(empty_v.copy() for _ in range(n_atoms)),
                distances=tuple(empty_d.copy() for _ in range(n_atoms)),
            )

        all_i = np.concatenate(i_chunks)
        all_j = np.concatenate(j_chunks)
        all_s = np.concatenate(s_chunks)
        all_v = np.concatenate(v_chunks)
        all_d = np.concatenate(d_chunks)
        order = np.lexsort((all_s[:, 0], all_s[:, 1], all_s[:, 2], all_j, all_d, all_i))
        all_i, all_j = all_i[order], all_j[order]
        all_s, all_v, all_d = all_s[order], all_v[order], all_d[order]

        starts = np.searchsorted(all_i, np.arange(n_atoms), side="left")
        stops = np.searchsorted(all_i, np.arange(n_atoms), side="right")
        indices = tuple(all_j[s:e] for s, e in zip(starts, stops, strict=True))
        shifts = tuple(all_s[s:e] for s, e in zip(starts, stops, strict=True))
        vectors = tuple(all_v[s:e] for s, e in zip(starts, stops, strict=True))
        distances = tuple(all_d[s:e] for s, e in zip(starts, stops, strict=True))
        return cls(
            cutoff=float(cutoff),
            indices=indices,
            shifts=shifts,
            vectors=vectors,
            distances=distances,
        )
