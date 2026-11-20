"""Descriptor construction, Eq. (3) (Sprint 1).

    g_i^(m) = Σ_{j,k} P_m(cos θ_jik) · f(r_ij) · f(r_ik)

with the double sum running over ordered neighbor pairs (j, k) within the
cutoff, *including* j = k (literal reading of Σ_{j,k}; see
docs/equations.md §7.2). With this reading the descriptors are angular
power-spectrum-like quantities and the identities

    g_i^(0)[c] = ( Σ_j f_c(r_ij) )²                (squared local density)
    g_i^(1)[c] = | Σ_j f_c(r_ij) r̂_ij |²           (squared "dipole")

hold per radial center c — exactly the paper's interpretation that g^(0)
measures local density and g^(1) measures deviation from spherical symmetry
(vanishing for spherically symmetric environments). Both identities are
verified independently in the scientific tests.

Output layout (per atom): order-major — entry index is
``order_slot * n_centers + center_slot`` with ``orders`` in config order
(paper: 0, 1, 2, 4, 6) and ``radial_centers`` ascending. With the defaults
this gives slots 0-11 for m = 0, 12-23 for m = 1, 24-35 for m = 2,
36-47 for m = 4, 48-59 for m = 6.

Numerical notes:

* the neighbor list restricts the sum to images within the cutoff; the
  cutoff factor makes the truncated tail exactly zero, so this is exact;
* atoms with no neighbors inside the cutoff get an all-zero descriptor row;
* accumulation in float64 with deterministic ordering (sorted neighbor
  entries) keeps results reproducible to round-off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from atomistic_pinn.constants import (
    CUTOFF_D,
    CUTOFF_RADIUS,
    DEFAULT_RADIAL_CENTERS,
    DESCRIPTOR_LEGENDRE_ORDERS,
    DESCRIPTOR_SIGMA,
)
from atomistic_pinn.data.schema import AtomicStructure
from atomistic_pinn.descriptors.legendre import legendre_values
from atomistic_pinn.descriptors.radial import radial_function
from atomistic_pinn.geometry.measure import cosine_matrix
from atomistic_pinn.geometry.neighbor_list import NeighborList


@dataclass(frozen=True)
class DescriptorConfig:
    """Configuration of the local structural descriptors, Eq. (3).

    Attributes:
        orders: Legendre polynomial orders (paper: (0, 1, 2, 4, 6)).
        radial_centers: Gaussian centers r₀ in Å. The paper uses 12 values
            without giving them; the default grid is a labeled
            implementation decision (see ``constants.DEFAULT_RADIAL_CENTERS``).
        sigma: Gaussian width in Å (paper: 1.0).
        cutoff_radius: Cutoff radius in Å (paper: 6.0).
        cutoff_d: Truncation-range parameter in Å (paper: 1.5).
        include_same_pair: Include the j = k terms of the double sum
            (literal reading of the paper; set False only for labeled
            comparison experiments).
    """

    orders: tuple[int, ...] = DESCRIPTOR_LEGENDRE_ORDERS
    radial_centers: tuple[float, ...] = DEFAULT_RADIAL_CENTERS
    sigma: float = DESCRIPTOR_SIGMA
    cutoff_radius: float = CUTOFF_RADIUS
    cutoff_d: float = CUTOFF_D
    include_same_pair: bool = True

    def __post_init__(self) -> None:
        if not self.orders or any(m < 0 for m in self.orders):
            raise ValueError("orders must be a non-empty tuple of non-negative integers")
        if not self.radial_centers or any(c < 0.0 for c in self.radial_centers):
            raise ValueError("radial_centers must be non-empty and non-negative")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.cutoff_radius <= 0.0:
            raise ValueError(f"cutoff_radius must be positive, got {self.cutoff_radius}")
        if self.cutoff_d <= 0.0:
            raise ValueError(f"cutoff_d must be positive, got {self.cutoff_d}")

    @property
    def n_descriptors(self) -> int:
        """Descriptor count k = n_orders × n_centers."""
        return len(self.orders) * len(self.radial_centers)


def compute_descriptors(
    positions: NDArray[np.float64],
    cell: NDArray[np.float64],
    pbc: NDArray[np.bool_],
    config: DescriptorConfig | None = None,
) -> NDArray[np.float64]:
    """Compute the Eq. (3) descriptors for every atom.

    Args:
        positions: Cartesian positions (n_atoms, 3), in Å.
        cell: Lattice vectors as rows (3, 3), in Å.
        pbc: Three periodic flags.
        config: Descriptor configuration (defaults to the paper values).

    Returns:
        Array of shape (n_atoms, k) with k = ``config.n_descriptors``.
    """
    cfg = config or DescriptorConfig()
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    pbc = np.asarray(pbc, dtype=bool)
    neighbor_list = NeighborList.build(positions, cell, pbc, cfg.cutoff_radius)
    centers = np.asarray(cfg.radial_centers, dtype=np.float64)
    out = np.zeros((positions.shape[0], cfg.n_descriptors), dtype=np.float64)
    for i in range(positions.shape[0]):
        vectors = neighbor_list.vectors[i]
        if vectors.shape[0] == 0:
            continue  # empty environment: row stays zero
        distances = neighbor_list.distances[i]
        cosines = cosine_matrix(vectors)
        radial = np.stack(
            [
                radial_function(
                    distances,
                    r0=float(r0),
                    sigma=cfg.sigma,
                    r_c=cfg.cutoff_radius,
                    d=cfg.cutoff_d,
                )
                for r0 in centers
            ]
        )
        pair_weight = radial[:, :, None] * radial[:, None, :]  # (n_c, m, m)
        polynomials = np.stack(
            [legendre_values(cfg.orders, cosines)[m] for m in cfg.orders]
        )  # (n_o, m, m)
        g_oc = np.einsum("oab,cab->oc", polynomials, pair_weight)
        if not cfg.include_same_pair:
            diag_poly = np.diagonal(polynomials, axis1=1, axis2=2)  # (n_o, m)
            diag_weight = np.diagonal(pair_weight, axis1=1, axis2=2)  # (n_c, m)
            g_oc -= np.einsum("oa,ca->oc", diag_poly, diag_weight)
        out[i] = g_oc.reshape(-1)  # order-major layout
    return out


def compute_structure_descriptors(
    structure: AtomicStructure,
    config: DescriptorConfig | None = None,
) -> NDArray[np.float64]:
    """Convenience wrapper computing descriptors for an :class:`AtomicStructure`."""
    return compute_descriptors(
        structure.positions_array,
        structure.cell_array,
        np.asarray(structure.pbc, dtype=bool),
        config,
    )
