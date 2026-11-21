"""Evaluation layer (Sprint 4): metrics, predictions, forces, cross-validation.

Dependency rule (`docs/architecture.md`): this layer may import models,
physics, and data — never the reverse.
"""

from atomistic_pinn.evaluation.forces import (
    dataset_force_metrics,
    pipeline_forces_autograd,
    pipeline_forces_finite_difference,
)
from atomistic_pinn.evaluation.metrics import (
    FIDELITY_ALTERNATIVE_DATA,
    EnergyMetrics,
    ForceMetrics,
    energy_metrics,
    force_metrics,
    per_atom_energy_errors,
)
from atomistic_pinn.evaluation.predict import (
    predicted_bop_parameters,
    predicted_structure_energies,
    restore_model,
)

__all__ = [
    "FIDELITY_ALTERNATIVE_DATA",
    "EnergyMetrics",
    "ForceMetrics",
    "dataset_force_metrics",
    "energy_metrics",
    "force_metrics",
    "per_atom_energy_errors",
    "pipeline_forces_autograd",
    "pipeline_forces_finite_difference",
    "predicted_bop_parameters",
    "predicted_structure_energies",
    "restore_model",
]
