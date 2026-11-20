"""Atomistic PINN Lab.

A research-grade reproduction of the physically informed neural network (PINN)
bond-order potential for aluminum introduced in:

    G. P. Purja Pun, R. Batra, R. Ramprasad, and Y. Mishin,
    "Physically informed artificial neural networks for atomistic modeling
    of materials", Nature Communications 10, 2339 (2019).
    https://doi.org/10.1038/s41467-019-10343-5

The package is currently at Sprint 0 (project foundation): unit conventions,
physical and paper-derived constants, and the atomic-structure data contracts
are defined. The geometry, descriptor, physics, learning, evaluation, and
simulation layers follow in later sprints (see docs/architecture.md).
"""

from atomistic_pinn import constants

__version__ = "0.1.0"

__all__ = ["__version__", "constants"]
