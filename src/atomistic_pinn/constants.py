"""Unit conventions, physical constants, and paper-derived constants.

Units
-----
Every public interface of ``atomistic_pinn`` uses one consistent unit system:

=============  ========  =======================
Quantity       Unit      Symbol used in the docs
=============  ========  =======================
distance       angstrom  Å
energy         eV        eV
force          eV/Å      eV Å⁻¹
temperature    kelvin    K
=============  ========  =======================

Paper-derived constants
-----------------------
Values transcribed from G. P. Purja Pun, R. Batra, R. Ramprasad, and
Y. Mishin, "Physically informed artificial neural networks for atomistic
modeling of materials", Nature Communications 10, 2339 (2019),
DOI 10.1038/s41467-019-10343-5. Each group records its source (Methods
equations, Table 1, Table 2, or Results text). Quantities the paper does
*not* specify are deliberately absent from this module; they are tracked in
``docs/equations.md`` (§ "Known uncertainties") and must not be silently
invented.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Unit conventions
# ---------------------------------------------------------------------------
UNIT_LENGTH: str = "angstrom"
UNIT_ENERGY: str = "eV"
UNIT_FORCE: str = "eV/angstrom"
UNIT_TEMPERATURE: str = "kelvin"

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018, expressed in the package units)
# ---------------------------------------------------------------------------
BOLTZMANN_EV_PER_K: float = 8.617333262e-5
"""Boltzmann constant k_B in eV/K."""

AVOGADRO: float = 6.02214076e23
"""Avogadro constant in 1/mol."""

ELEMENTARY_CHARGE: float = 1.602176634e-19
"""Elementary charge in coulomb."""

# ---------------------------------------------------------------------------
# Aluminum (the single-component system studied in the paper)
# ---------------------------------------------------------------------------
ALUMINUM_SYMBOL: str = "Al"
ALUMINUM_ATOMIC_NUMBER: int = 13
ALUMINUM_MASS_AMU: float = 26.9815385

# ---------------------------------------------------------------------------
# Local structural descriptors — Methods, Eqs. (3)-(5)
# ---------------------------------------------------------------------------
DESCRIPTOR_LEGENDRE_ORDERS: tuple[int, ...] = (0, 1, 2, 4, 6)
"""Legendre polynomial orders m used in Eq. (3)."""

N_RADIAL_CENTERS: int = 12
"""Number of Gaussian centers r₀ ("12 r0 values", Methods)."""

DEFAULT_RADIAL_CENTERS: tuple[float, ...] = tuple(0.5 * i for i in range(N_RADIAL_CENTERS))
"""Default 12 radial centers r₀ in Å: 0.0, 0.5, 1.0, ..., 5.5.

IMPLEMENTATION DECISION, not a paper fact: the paper states that 12 r₀ values
are used but does not give their coordinates (docs/equations.md §7.1). This
evenly spaced 0.5 Å grid spans the range of Al interatomic distances
(nearest-neighbor shell ≈ 2.4-3.0 Å, second shell ≈ 4.0 Å, compressed bonds
below 2.4 Å). To be revised if the paper's Supplementary Information becomes
available; the choice is exposed through ``DescriptorConfig.radial_centers``
so alternative grids can be compared without code changes.
"""

N_DESCRIPTORS_PER_ATOM: int = 60
"""Descriptor count k = 5 orders × 12 radial centers."""

DESCRIPTOR_SIGMA: float = 1.0
"""Gaussian width σ in Å in Eq. (4); the paper states "we chose σ = 1.0"."""

CUTOFF_RADIUS: float = 6.0
"""Cutoff radius r_c in Å of Eq. (5); "fixed at r_c = 6 Å" (Methods)."""

CUTOFF_D: float = 1.5
"""Truncation-range parameter d in Å of Eq. (5); "fixed at ... d = 1.5 Å"."""

# ---------------------------------------------------------------------------
# Bond-order potential — Methods, Eqs. (6)-(12)
# ---------------------------------------------------------------------------
BOP_PARAMETER_NAMES: tuple[str, ...] = ("A", "B", "alpha", "beta", "a", "h", "sigma", "lambda")
"""The eight per-atom BOP parameters (p₁...p₈) predicted by the PINN network."""

N_BOP_PARAMETERS: int = 8
"""Number of adjustable BOP parameters per atom; m = 8 (Methods)."""

# ---------------------------------------------------------------------------
# Network architectures — Table 1
# ---------------------------------------------------------------------------
PINN_ARCHITECTURE: tuple[int, ...] = (60, 15, 15, 8)
"""PINN: 60 descriptors → two hidden layers of 15 → 8 BOP parameters."""

NN_ARCHITECTURE: tuple[int, ...] = (60, 16, 16, 1)
"""Baseline NN: 60 descriptors → two hidden layers of 16 → atomic energy."""

NN_PRIME_ARCHITECTURE: tuple[int, ...] = (47, 18, 18, 1)
"""NN' baseline (PROPhet package, Behler-Parrinello features).

Note: NN' is not planned for exact reproduction; see docs/equations.md.
"""

# ---------------------------------------------------------------------------
# Training protocol — Methods
# ---------------------------------------------------------------------------
N_CV_FOLDS: int = 10
"""Number of cross-validation folds."""

ENERGY_SHIFT_EV_PER_ATOM: float = 0.38446
"""Uniform shift added to all DFT energies before training (per atom).

Paper: "prior to the training all DFT energies were uniformly shifted by
0.38446 eV per atom to match the experimental cohesive energy of Al, 3.36 eV
per atom". The shift is applied by the training pipeline, never baked into
stored datasets (see docs/data_format.md).
"""

EXPERIMENTAL_COHESIVE_ENERGY_EV_PER_ATOM: float = 3.36
"""Experimental cohesive energy of Al that the energy shift is calibrated to."""

DFT_COHESIVE_ENERGY_EV_PER_ATOM: float = 3.7480
"""DFT equilibrium cohesive energy E₀ from Table 2 (magnitude, eV per atom)."""

# ---------------------------------------------------------------------------
# Dataset sizes — Results text
# ---------------------------------------------------------------------------
TRAINING_SUPERCELLS: int = 3649
"""Number of training/validation supercells in the paper's DFT database."""

TRAINING_ATOMS: int = 127592
"""Number of atoms in the training/validation database."""

TEST_SUPERCELLS: int = 26425
"""Number of testing supercells (held-out DFT data, Supplementary Table 3)."""

TEST_ATOMS: int = 2376388
"""Number of atoms in the testing database."""

# ---------------------------------------------------------------------------
# Reference accuracy — Table 1 (energies, meV/atom) and Results text (forces)
# ---------------------------------------------------------------------------
TABLE1_REFERENCE_RMSE: dict[str, dict[str, float]] = {
    "NN": {"params": 1265.0, "train": 3.36, "val": 3.85},
    "NN_prime": {"params": 1225.0, "train": 3.62, "val": 3.54},
    "PINN": {"params": 1283.0, "train": 3.46, "val": 3.59},
}
"""Training/validation RMSE from Table 1, in meV per atom."""

TARGET_RMSE_MEV_PER_ATOM: tuple[float, float] = (3.0, 4.0)
"""Paper goal: "RMSE of training and validation close to 3-4 meV per atom"."""

FORCE_RMSE_REFERENCE_EV_PER_ANGSTROM: dict[str, float] = {
    "pinn_validation": 0.1,
    "nn_validation": 0.2,
    "pinn_testing": 0.1,
    "nn_testing_dislocation": 0.4,
    "nn_testing_hcp": 0.6,
}
"""Approximate (≈) force RMSE values quoted in the Results text, in eV/Å."""
