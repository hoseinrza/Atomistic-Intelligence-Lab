"""Reporting layer: training reports (Sprint 3 subset).

Writes a JSON metrics summary and matplotlib figures into ``results/``
(generated artifacts, never committed). Figure generation degrades
gracefully when matplotlib is unavailable — the JSON summary is always
written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from atomistic_pinn.models import PINN, BaselineNN
from atomistic_pinn.training.database import TrainingDatabase
from atomistic_pinn.training.objective import BaselineEnergyModel, PINNEnergyModel, set_flat_params
from atomistic_pinn.training.trainer import TrainingOutcome


def write_training_report(
    out_dir: str | Path,
    outcome: TrainingOutcome,
    db: TrainingDatabase,
    resolved_config: dict,
    activation: str,
) -> Path:
    """Write the training report (metrics.json + figures).

    Args:
        out_dir: Report directory (created; typically ``results/<name>``).
        outcome: Training outcome.
        db: The training database (for parity data and RMSE recomputation).
        resolved_config: Canonical resolved-config dict (provenance).
        activation: Resolved activation name (model reconstruction).

    Returns:
        The path of the written ``metrics.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct the winning model to regenerate parity/parameter data.
    if outcome.model_kind == "pinn":
        pinn_network = PINN(seed=outcome.restart_seeds[outcome.best_restart], activation=activation)
        model: PINNEnergyModel | BaselineEnergyModel = PINNEnergyModel(pinn_network)
    else:
        baseline_network = BaselineNN(
            seed=outcome.restart_seeds[outcome.best_restart], activation=activation
        )
        model = BaselineEnergyModel(baseline_network)
    set_flat_params(model, torch.tensor(outcome.best_vector, dtype=torch.float64))
    predicted = model.structure_energies(db).detach().numpy()
    targets = db.shifted_targets()
    n_atoms = db.atoms_per_structure()
    per_atom_errors = (predicted - targets) / n_atoms
    rmse_mev = float(1000.0 * np.sqrt(np.mean(per_atom_errors**2)))
    mae_mev = float(1000.0 * np.mean(np.abs(per_atom_errors)))

    metrics = {
        "model_kind": outcome.model_kind,
        "objective_ev2": outcome.best_objective,
        "rmse_mev_per_atom": rmse_mev,
        "mae_mev_per_atom": mae_mev,
        "best_restart": outcome.best_restart,
        "stop_reason": outcome.stop_reason,
        "iterations": outcome.best_dfp.iterations,
        "n_parameters": outcome.n_parameters,
        "config_hash": outcome.config_hash,
        "dataset_hash": outcome.dataset_hash,
        "git_commit": outcome.git_commit,
        "restart_seeds": outcome.restart_seeds,
        "fidelity_note": (
            "pipeline validation on the synthetic development database — not a reproduction result"
            if db.is_synthetic
            else "methodological reproduction (alternative data)"
        ),
        "fixed_bop": None
        if outcome.fixed_bop is None
        else {
            "parameters": outcome.fixed_bop.parameters.tolist(),
            "objective_ev2": outcome.fixed_bop.objective,
        },
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    _write_figures(out_dir, outcome, predicted, targets, model, db, activation)
    return metrics_path


def _write_figures(
    out_dir: Path,
    outcome: TrainingOutcome,
    predicted: np.ndarray,
    targets: np.ndarray,
    model: PINNEnergyModel | BaselineEnergyModel,
    db: TrainingDatabase,
    activation: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # JSON summary already written; figures are optional

    fig, ax = plt.subplots(figsize=(5, 4))
    for k, history in enumerate(outcome.histories):
        ax.plot(history, label=f"restart {k}" if len(outcome.histories) > 1 else None)
    ax.set_xlabel("DFP iteration")
    ax.set_ylabel("ℰ (eV²)")
    ax.set_yscale("log")
    if len(outcome.histories) > 1:
        ax.legend()
    ax.set_title("Objective per restart")
    fig.tight_layout()
    fig.savefig(out_dir / "objective_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(targets, predicted, s=14, alpha=0.7)
    lims = [min(targets.min(), predicted.min()), max(targets.max(), predicted.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("shifted DFT target (eV)")
    ax.set_ylabel("predicted (eV)")
    ax.set_title("Energy parity")
    fig.tight_layout()
    fig.savefig(out_dir / "energy_parity.png", dpi=150)
    plt.close(fig)

    if outcome.model_kind == "pinn":
        predicted_params = model.predicted_parameters(db)
        assert predicted_params is not None
        params = predicted_params.detach().numpy()
        names = ["A", "B", "alpha", "beta", "a", "h", "sigma", "lambda"]
        fig, axes = plt.subplots(2, 4, figsize=(12, 5))
        for k, (axis, name) in enumerate(zip(axes.flat, names, strict=True)):
            axis.hist(params[:, k], bins=24)
            axis.set_title(name)
        fig.suptitle("Predicted parameter distributions (§7.5 drift check)")
        fig.tight_layout()
        fig.savefig(out_dir / "parameter_distributions.png", dpi=150)
        plt.close(fig)
