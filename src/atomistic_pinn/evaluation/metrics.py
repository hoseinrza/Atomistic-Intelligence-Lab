"""Evaluation metrics (Sprint 4, phase 4.1).

Energy errors are computed **per atom** (meV/atom) against the *shifted*
DFT targets — exactly the convention of the training objective (Eq. 13,
``TrainingDatabase.shifted_targets``): the paper's +0.38446 eV/atom shift is
applied once, pipeline-side. Force errors are computed in eV/Å against the
stored DFT forces of the dataset (forces are never a training target — paper
methodology — they only enter this post-training evaluation).

Fidelity rule (`docs/reproduction_protocol.md`): every metric produced from
the alternative dataset carries the fidelity level
``"methodological reproduction (alternative data)"``; the reference bands of
paper Table 1 (3–4 meV/atom) are comparison bands, never pass/fail gates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FIDELITY_ALTERNATIVE_DATA = "methodological reproduction (alternative data)"


@dataclass(frozen=True)
class EnergyMetrics:
    """Energy error summary over a set of structures.

    Attributes:
        rmse_mev_per_atom: Root-mean-square per-atom error (meV/atom).
        mae_mev_per_atom: Mean absolute per-atom error (meV/atom).
        max_abs_error_mev_per_atom: Largest absolute per-atom error.
        n_structures: Number of structures evaluated.
        n_atoms: Total number of atoms over the structures.
    """

    rmse_mev_per_atom: float
    mae_mev_per_atom: float
    max_abs_error_mev_per_atom: float
    n_structures: int
    n_atoms: int

    def with_fidelity(self, alternative_data: bool) -> dict[str, object]:
        """Metric dict plus the fidelity level of the producing dataset."""
        return {
            "rmse_mev_per_atom": self.rmse_mev_per_atom,
            "mae_mev_per_atom": self.mae_mev_per_atom,
            "max_abs_error_mev_per_atom": self.max_abs_error_mev_per_atom,
            "n_structures": self.n_structures,
            "n_atoms": self.n_atoms,
            "fidelity": FIDELITY_ALTERNATIVE_DATA if alternative_data else "exact reproduction",
        }


@dataclass(frozen=True)
class ForceMetrics:
    """Force error summary over a set of structures (eV/Å).

    Attributes:
        rmse_ev_per_angstrom: Root-mean-square force-component error.
        mae_ev_per_angstrom: Mean absolute force-component error.
        max_abs_error_ev_per_angstrom: Largest absolute component error.
        n_structures: Number of structures evaluated.
        n_atoms: Total number of atoms over the structures.
    """

    rmse_ev_per_angstrom: float
    mae_ev_per_angstrom: float
    max_abs_error_ev_per_angstrom: float
    n_structures: int
    n_atoms: int


def per_atom_energy_errors(
    predicted_ev: NDArray[np.float64],
    targets_ev: NDArray[np.float64],
    n_atoms: NDArray[np.float64] | NDArray[np.int_],
) -> NDArray[np.float64]:
    """Per-atom energy errors ``(E_pred − E_target)/n_atoms`` in eV/atom.

    Args:
        predicted_ev: ``(S,)`` predicted total energies (eV).
        targets_ev: ``(S,)`` reference total energies (eV, shifted convention).
        n_atoms: ``(S,)`` atom counts.

    Raises:
        ValueError: On shape mismatches or non-positive atom counts.
    """
    predicted = np.asarray(predicted_ev, dtype=np.float64)
    targets = np.asarray(targets_ev, dtype=np.float64)
    counts = np.asarray(n_atoms, dtype=np.float64)
    if predicted.shape != targets.shape or predicted.shape != counts.shape:
        raise ValueError(
            f"shape mismatch: predicted {predicted.shape}, targets {targets.shape}, "
            f"n_atoms {counts.shape}"
        )
    if np.any(counts <= 0):
        raise ValueError("n_atoms must be positive for every structure")
    return (predicted - targets) / counts


def energy_metrics(
    predicted_ev: NDArray[np.float64],
    targets_ev: NDArray[np.float64],
    n_atoms: NDArray[np.float64] | NDArray[np.int_],
) -> EnergyMetrics:
    """Energy RMSE/MAE in meV/atom over a set of structures."""
    errors = per_atom_energy_errors(predicted_ev, targets_ev, n_atoms) * 1000.0
    return EnergyMetrics(
        rmse_mev_per_atom=float(np.sqrt(np.mean(errors**2))),
        mae_mev_per_atom=float(np.mean(np.abs(errors))),
        max_abs_error_mev_per_atom=float(np.max(np.abs(errors))),
        n_structures=int(errors.shape[0]),
        n_atoms=int(np.asarray(n_atoms).sum()),
    )


def force_metrics(
    predicted: NDArray[np.float64],
    reference: NDArray[np.float64],
    n_structures: int = 0,
) -> ForceMetrics:
    """Force MAE/RMSE in eV/Å over stacked ``(N_total, 3)`` force arrays.

    Args:
        predicted: Concatenated predicted forces, shape ``(N_total, 3)``.
        reference: Concatenated reference (DFT) forces, same shape.
        n_structures: Number of contributing structures (provenance only).

    Raises:
        ValueError: On shape mismatch or empty input.
    """
    predicted_arr = np.asarray(predicted, dtype=np.float64)
    reference_arr = np.asarray(reference, dtype=np.float64)
    if predicted_arr.shape != reference_arr.shape:
        raise ValueError(
            f"force shape mismatch: predicted {predicted_arr.shape}, "
            f"reference {reference_arr.shape}"
        )
    if predicted_arr.size == 0:
        raise ValueError("force metrics require at least one atom")
    errors = (predicted_arr - reference_arr).ravel()
    return ForceMetrics(
        rmse_ev_per_angstrom=float(np.sqrt(np.mean(errors**2))),
        mae_ev_per_angstrom=float(np.mean(np.abs(errors))),
        max_abs_error_ev_per_angstrom=float(np.max(np.abs(errors))),
        n_structures=n_structures,
        n_atoms=int(predicted_arr.shape[0]),
    )
