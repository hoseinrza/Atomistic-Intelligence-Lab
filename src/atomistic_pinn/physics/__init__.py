"""Physics layer: the analytical bond-order potential, Eqs. (6)-(12) (Sprint 2).

Implemented on torch tensors (CPU-compatible) so that training can
differentiate the energy with respect to the network-predicted BOP
parameters, while forces for simulation follow the paper's finite-difference
method (``physics/forces.py``).
"""

from atomistic_pinn.physics.bop import atomic_energies, bond_order_factors, total_energy
from atomistic_pinn.physics.forces import forces_autograd, forces_finite_difference
from atomistic_pinn.physics.neighbors import BOPNeighbors, build_bop_neighbors
from atomistic_pinn.physics.screening import (
    cutoff_function,
    partial_screening_matrix,
    screening_argument_matrix,
    screening_factors,
)

__all__ = [
    "BOPNeighbors",
    "atomic_energies",
    "bond_order_factors",
    "build_bop_neighbors",
    "cutoff_function",
    "forces_autograd",
    "forces_finite_difference",
    "partial_screening_matrix",
    "screening_argument_matrix",
    "screening_factors",
    "total_energy",
]
