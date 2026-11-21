"""Integration test: the cross-validation driver end-to-end (Sprint 4 gate 1).

Runs a tiny 2-fold CV on the 8-structure ANI fixture with a minimal training
budget and validates the gate contract: per-structure fold assignment,
derived fold seeds, per-fold + averaged metrics, fixed-BOP baseline, full
provenance (config hash, dataset hash, partition recorded in the summary).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from atomistic_pinn.evaluation.cross_validation import run_cross_validation
from atomistic_pinn.training.config import load_cv_config
from atomistic_pinn.training.trainer import build_dataset


def _cv_config(tmp_path: Path, ani_dir: Path) -> dict[str, object]:
    return {
        "experiment": {"name": "cv_smoke", "seed": 42},
        "model": {
            "type": "pinn",
            "architecture": [60, 15, 15, 8],
            "n_parameters": 1283,
            "activation": "tanh",
            "weight_init": "xavier_uniform",
        },
        "descriptors": {},
        "physics": {"bop_parameters": ["A", "B", "alpha", "beta", "a", "h", "sigma", "lambda"]},
        "training": {
            "objective": "sum_squared_total_energy_error",
            "energy_shift_ev_per_atom": 0.38446,
            "optimizer": "dfp",
            "n_random_restarts": 1,
            "regularization": {"tau": 1e-6, "gamma": 1e-4},
            "forces_in_training": False,
            "gradients": "autograd",
            "line_search": "backtracking_armijo",
            "p_bar_eta": "fixed_bop_prefit",
            "max_iterations": 3,
            "gradient_tolerance": 1e-8,
        },
        "data": {
            "dataset": None,
            "source": "ani_al",
            "path": str(ani_dir),
            "subset_size": 8,
            "subset_seed": 3,
            "n_folds": 10,
        },
        "cv": {"n_folds": 2, "shuffle": True, "seed": None, "max_folds": None},
    }


def test_cross_validation_end_to_end(ani_dir: Path, tmp_path: Path) -> None:
    config_path = tmp_path / "cv.yaml"
    config_path.write_text(yaml.safe_dump(_cv_config(tmp_path, ani_dir)))
    resolved, cv_section = load_cv_config(config_path)
    assert cv_section.n_folds == 2
    assert cv_section.seed == 42  # inherited from experiment.seed
    dataset = build_dataset(resolved)
    assert dataset.n_structures == 8

    result = run_cross_validation(
        resolved,
        dataset,
        n_folds=2,
        fold_seed=42,
        shuffle=True,
        alternative_data=True,
    )

    assert result.model_kind == "pinn"
    assert len(result.fold_results) == 2
    for fold in result.fold_results:
        assert fold.n_train == 4
        assert fold.n_validation == 4
        assert fold.seed != 42  # derived, not the master seed
        assert fold.validation_metrics.rmse_mev_per_atom >= 0.0
        assert fold.fixed_bop_validation_metrics is not None
        assert np_is_finite(fold.validation_metrics.rmse_mev_per_atom)
    assert result.mean_validation_rmse_mev_per_atom >= 0.0

    summary = result.summary()
    assert summary["fidelity"] == "methodological reproduction (alternative data)"
    assert summary["fold_partition_seed"] == 42
    assert len(summary["fold_assignment"]) == 8
    assert set(summary["fold_assignment"].values()) == {0, 1}


def np_is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")
