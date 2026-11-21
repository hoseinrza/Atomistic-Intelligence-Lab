"""K-fold cross-validation driver (Sprint 4, phase 4.1).

Wraps the Sprint 3 training driver exactly as planned (`docs/sprints.md`,
Sprint 4 deliverables): per-structure fold assignment, per-fold and averaged
energy metrics, derived seeds per fold, and full provenance (config hash,
dataset content hash, fold assignment, fold seeds).

Per fold:
1. training subset → ``TrainingDatabase`` → ``run_training`` (the Sprint 3
   driver, including the fixed-BOP pre-fit when the γ tether uses it);
2. training metrics on the training subset (the driver's own RMSE plus the
   MAE from this layer);
3. validation metrics on the held-out fold via the restored winner;
4. the fold's fixed-BOP baseline evaluated on the held-out fold (paper
   Fig. 8 re-fit per fold — never the paper's own parameter values, §7.15).

Reference bands (paper Table 1: 3–4 meV/atom) are comparison bands, not
gates, for alternative data (`docs/reproduction_protocol.md`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from atomistic_pinn.data.schema import Dataset
from atomistic_pinn.evaluation.folds import FoldAssignment, assign_folds
from atomistic_pinn.evaluation.metrics import (
    FIDELITY_ALTERNATIVE_DATA,
    EnergyMetrics,
    energy_metrics,
)
from atomistic_pinn.evaluation.predict import (
    predicted_structure_energies,
    restore_model,
)
from atomistic_pinn.physics.bop import atomic_energies
from atomistic_pinn.training.config import ExperimentCfg, ResolvedTrainingConfig
from atomistic_pinn.training.database import TrainingDatabase, dataset_content_hash
from atomistic_pinn.training.trainer import TrainingOutcome, run_training


@dataclass(frozen=True)
class FoldResult:
    """One fold's outcome.

    Attributes:
        fold_index: Held-out fold index (0-based).
        seed: Derived seed used for this fold's training.
        n_train: Training structures.
        n_validation: Held-out structures.
        train_metrics: Energy metrics on the training subset.
        validation_metrics: Energy metrics on the held-out subset.
        fixed_bop_validation_metrics: Fixed-BOP baseline on the held-out
            subset (None when no pre-fit was configured).
        objective: Best final objective of the fold (eV²).
        stop_reason: Why the fold's winning restart stopped.
    """

    fold_index: int
    seed: int
    n_train: int
    n_validation: int
    train_metrics: EnergyMetrics
    validation_metrics: EnergyMetrics
    fixed_bop_validation_metrics: EnergyMetrics | None
    objective: float
    stop_reason: str


@dataclass(frozen=True)
class CVResult:
    """Aggregated cross-validation record (full provenance payload)."""

    model_kind: str
    n_folds: int
    fold_results: list[FoldResult]
    mean_validation_rmse_mev_per_atom: float
    std_validation_rmse_mev_per_atom: float
    mean_validation_mae_mev_per_atom: float
    fixed_bop_mean_validation_rmse_mev_per_atom: float | None
    config_hash: str
    dataset_hash: str
    fold_assignment: dict[str, int]
    fold_partition_seed: int
    fidelity: str = FIDELITY_ALTERNATIVE_DATA

    def summary(self) -> dict[str, object]:
        """JSON-ready summary dict (per-fold + averaged, provenance included)."""
        return {
            "model_kind": self.model_kind,
            "n_folds": self.n_folds,
            "mean_validation_rmse_mev_per_atom": self.mean_validation_rmse_mev_per_atom,
            "std_validation_rmse_mev_per_atom": self.std_validation_rmse_mev_per_atom,
            "mean_validation_mae_mev_per_atom": self.mean_validation_mae_mev_per_atom,
            "fixed_bop_mean_validation_rmse_mev_per_atom": (
                self.fixed_bop_mean_validation_rmse_mev_per_atom
            ),
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "fold_partition_seed": self.fold_partition_seed,
            "fold_assignment": self.fold_assignment,
            "fidelity": self.fidelity,
            "folds": [
                {
                    "fold_index": f.fold_index,
                    "seed": f.seed,
                    "n_train": f.n_train,
                    "n_validation": f.n_validation,
                    "train_rmse_mev_per_atom": f.train_metrics.rmse_mev_per_atom,
                    "validation_rmse_mev_per_atom": f.validation_metrics.rmse_mev_per_atom,
                    "validation_mae_mev_per_atom": f.validation_metrics.mae_mev_per_atom,
                    "fixed_bop_validation_rmse_mev_per_atom": (
                        None
                        if f.fixed_bop_validation_metrics is None
                        else f.fixed_bop_validation_metrics.rmse_mev_per_atom
                    ),
                    "objective_ev2": f.objective,
                    "stop_reason": f.stop_reason,
                }
                for f in self.fold_results
            ],
        }


def run_cross_validation(
    resolved: ResolvedTrainingConfig,
    dataset: Dataset,
    *,
    n_folds: int,
    fold_seed: int,
    shuffle: bool = True,
    max_folds: int | None = None,
    alternative_data: bool = True,
) -> CVResult:
    """Run the paper's K-fold protocol around the Sprint 3 driver.

    Args:
        resolved: Ledger-resolved training configuration (one model).
        dataset: The full dataset to partition (ANI-Al for the
            methodological reproduction).
        n_folds: Number of folds K (paper: 10).
        fold_seed: Master seed of the partition and of the derived fold
            seeds.
        shuffle: Random partition (paper protocol) vs. contiguous smoke
            split.
        max_folds: Run only the first ``max_folds`` folds (development
            budget; None runs all K).
        alternative_data: Fidelity label of the producing dataset.

    Returns:
        The aggregated :class:`CVResult`.
    """
    descriptor_config = resolved.descriptors.to_descriptor_config()
    ids = [s.id for s in dataset.structures]
    assignment: FoldAssignment = assign_folds(ids, n_folds, fold_seed, shuffle=shuffle)
    folds_to_run = range(n_folds) if max_folds is None else range(min(max_folds, n_folds))

    fold_results: list[FoldResult] = []
    structures_by_id = {s.id: s for s in dataset.structures}
    for fold_index in folds_to_run:
        training_ids, validation_ids = assignment.train_validation_split(ids, fold_index)
        train_dataset = Dataset(
            metadata=dataset.metadata,
            structures=[structures_by_id[i] for i in training_ids],
        )
        validation_dataset = Dataset(
            metadata=dataset.metadata,
            structures=[structures_by_id[i] for i in validation_ids],
        )
        train_db = TrainingDatabase(train_dataset, descriptor_config)
        validation_db = TrainingDatabase(validation_dataset, descriptor_config)
        fold_resolved = resolved.model_copy(
            update={
                "experiment": ExperimentCfg(
                    name=f"{resolved.experiment.name}_fold{fold_index}",
                    seed=assignment.fold_seeds[fold_index],
                    output_dir=resolved.experiment.output_dir,
                )
            }
        )
        outcome = run_training(fold_resolved, train_db)

        model = restore_model(fold_resolved, outcome.best_vector)
        train_metrics = energy_metrics(
            predicted_structure_energies(model, train_db),
            train_db.shifted_targets(),
            train_db.atoms_per_structure(),
        )
        validation_metrics = energy_metrics(
            predicted_structure_energies(model, validation_db),
            validation_db.shifted_targets(),
            validation_db.atoms_per_structure(),
        )
        fixed_bop_metrics = (
            _fixed_bop_validation_metrics(outcome, validation_db)
            if outcome.fixed_bop is not None
            else None
        )
        fold_results.append(
            FoldResult(
                fold_index=fold_index,
                seed=assignment.fold_seeds[fold_index],
                n_train=len(training_ids),
                n_validation=len(validation_ids),
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                fixed_bop_validation_metrics=fixed_bop_metrics,
                objective=float(outcome.best_objective),
                stop_reason=outcome.stop_reason,
            )
        )

    rmses = np.asarray([f.validation_metrics.rmse_mev_per_atom for f in fold_results])
    maes = np.asarray([f.validation_metrics.mae_mev_per_atom for f in fold_results])
    fixed_rmses = [
        f.fixed_bop_validation_metrics.rmse_mev_per_atom
        for f in fold_results
        if f.fixed_bop_validation_metrics is not None
    ]
    return CVResult(
        model_kind=resolved.model.type,
        n_folds=n_folds,
        fold_results=fold_results,
        mean_validation_rmse_mev_per_atom=float(np.mean(rmses)),
        std_validation_rmse_mev_per_atom=float(np.std(rmses)),
        mean_validation_mae_mev_per_atom=float(np.mean(maes)),
        fixed_bop_mean_validation_rmse_mev_per_atom=(
            float(np.mean(fixed_rmses)) if fixed_rmses else None
        ),
        config_hash=resolved.config_hash,
        dataset_hash=dataset_content_hash(dataset),
        fold_assignment=assignment.fold_of,
        fold_partition_seed=fold_seed,
        fidelity=FIDELITY_ALTERNATIVE_DATA if alternative_data else "exact reproduction",
    )


def _fixed_bop_validation_metrics(
    outcome: TrainingOutcome,
    validation_db: TrainingDatabase,
) -> EnergyMetrics:
    """Fixed-BOP baseline energies on the held-out fold (Fig. 8 re-fit).

    The fold's pre-fitted parameters are broadcast to every atom of every
    validation structure and evaluated through the physics layer — no
    learning-stack involvement.
    """
    assert outcome.fixed_bop is not None  # guarded by the caller
    parameters = outcome.fixed_bop.parameters
    predicted: list[float] = []
    for geometry in validation_db.geometries:
        params = torch.tensor(
            np.broadcast_to(np.asarray(parameters, dtype=np.float64), (geometry.n_atoms, 8)).copy(),
            dtype=torch.float64,
        )
        energy = atomic_energies(geometry.positions, geometry.cell, geometry.pbc, params).sum()
        predicted.append(float(energy.detach()))
    return energy_metrics(
        np.asarray(predicted),
        validation_db.shifted_targets(),
        validation_db.atoms_per_structure(),
    )
