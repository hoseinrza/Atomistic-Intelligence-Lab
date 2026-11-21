"""Checkpoint save/load with full provenance (Sprint 3).

Payload: best model state + flat vector, the winning DFP state *including
the inverse-Hessian approximation*, resolved config (+ hash), dataset hash,
git commit, master/derived seeds, restart index, fold id (None until Sprint
4), metrics, and the fixed-BOP pre-fit record. Float64 tensors round-trip
bit-exactly through ``torch.save``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from atomistic_pinn.training.trainer import TrainingOutcome

CHECKPOINT_FORMAT = 1


def save_checkpoint(
    path: str | Path,
    outcome: TrainingOutcome,
    resolved_config: dict,
) -> Path:
    """Persist the training outcome with full provenance.

    Args:
        path: Target ``.pt`` path (parent directories are created).
        outcome: Training outcome (trainer.py).
        resolved_config: Canonical dict of the resolved configuration.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_kind": outcome.model_kind,
        # The flat vector is the authoritative model state: the architecture
        # is fixed by model_kind, and set_flat_params restores it exactly.
        "flat_params": torch.tensor(outcome.best_vector, dtype=torch.float64),
        "dfp": {
            "x": torch.tensor(outcome.best_dfp.x, dtype=torch.float64),
            "iterations": outcome.best_dfp.iterations,
            "history": outcome.best_dfp.history,
            "gradient_norm": outcome.best_dfp.gradient_norm,
            "stop_reason": outcome.best_dfp.stop_reason,
        },
        "resolved_config": resolved_config,
        "config_hash": outcome.config_hash,
        "dataset_hash": outcome.dataset_hash,
        "git_commit": outcome.git_commit,
        "master_seed_note": "derived restart seeds recorded below",
        "restart_seeds": outcome.restart_seeds,
        "best_restart": outcome.best_restart,
        "fold_id": None,  # Sprint 4 wraps cross-validation around the driver
        "metrics": {
            "objective_ev2": outcome.best_objective,
            "rmse_mev_per_atom": outcome.rmse_mev_per_atom,
            "n_parameters": outcome.n_parameters,
        },
        "fixed_bop": (
            None
            if outcome.fixed_bop is None
            else {
                "parameters": outcome.fixed_bop.parameters.tolist(),
                "objective": outcome.fixed_bop.objective,
            }
        ),
        "p_bar": None if outcome.p_bar is None else outcome.p_bar.tolist(),
        "histories": outcome.histories,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path) -> dict:
    """Load a checkpoint payload verbatim (bit-exact float64 tensors)."""
    return torch.load(Path(path), weights_only=False)
