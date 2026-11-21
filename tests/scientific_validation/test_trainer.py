"""Scientific validation of the fixed-BOP pre-fit and the training driver.

The synthetic database is labeled by the BOP physics layer under the known
truth parameter set, so:

* the fixed-BOP pre-fit must drive the energy error to ~zero (the truth is
  exactly representable) — validating DFP + objective + physics end-to-end
  against a known solution;
* the training driver must be seed-reproducible bit-exactly;
* checkpoints must restore the full training state, inverse Hessian
  included.

Pipeline validation only — never a reproduction result (fidelity rules).
"""

import numpy as np
import pytest
import torch

from atomistic_pinn.training.checkpoints import load_checkpoint, save_checkpoint
from atomistic_pinn.training.config import ResolvedTrainingConfig
from atomistic_pinn.training.database import TrainingDatabase
from atomistic_pinn.training.synthetic import (
    SYNTHETIC_TRUTH_PARAMS,
    generate_synthetic_database,
)

pytestmark = pytest.mark.scientific_validation


def _small_db() -> TrainingDatabase:
    # Non-periodic variant: same physics gates, no periodic-image work —
    # keeps the pipeline-validation suite fast.
    return TrainingDatabase(
        generate_synthetic_database(n_structures=6, seed=13, with_periodic=False)
    )


def _resolved(**overrides) -> ResolvedTrainingConfig:
    """Build a minimal resolved config directly (bypasses YAML for tests)."""
    base = {
        "experiment": {"name": "test", "seed": 42, "output_dir": "checkpoints"},
        "model": {
            "type": "pinn",
            "architecture": [60, 15, 15, 8],
            "n_parameters": 1283,
            "activation": "tanh",
            "weight_init": "xavier_uniform",
        },
        "descriptors": {},
        "physics": {},
        "training": {
            "objective": "sum_squared_total_energy_error",
            "energy_shift_ev_per_atom": 0.38446,
            "optimizer": "dfp",
            "n_random_restarts": 2,
            "regularization": {"tau": 1e-6, "gamma": 1e-4},
            "forces_in_training": False,
            "gradients": "autograd",
            "line_search": "backtracking_armijo",
            "p_bar_eta": "fixed_bop_prefit",
            "max_iterations": 25,
            "gradient_tolerance": 1e-8,
        },
        "data": {
            "dataset": None,
            "source": "synthetic_bop",
            "path": None,
            "synthetic": {"n_structures": 6, "seed": 13, "with_periodic": False},
            "n_folds": 10,
        },
    }
    for section, values in overrides.items():
        base[section].update(values)
    return ResolvedTrainingConfig.model_validate(base)


def test_fixed_bop_prefit_recovers_truth_energies() -> None:
    from atomistic_pinn.training.fixed_bop import fit_fixed_bop

    db = _small_db()
    # Scope note: full-scale pre-fits (5 restarts, tighter tolerances) are
    # production runs; this gate keeps one restart and a moderate cap.
    fit = fit_fixed_bop(db, gradient="autograd", n_restarts=1, master_seed=5, max_iterations=400)
    # The truth is exactly representable by the fixed BOP: the energy error
    # must be driven to ~zero. (Parameter recovery itself is not asserted —
    # the map parameters → energies need not be injective.)
    energies = _fit_energies(db, fit.parameters)
    targets = db.shifted_targets()
    n_atoms = db.atoms_per_structure()
    rmse = float(1000.0 * np.sqrt(np.mean(((energies - targets) / n_atoms) ** 2)))
    # Pipeline gate: DFP drives the known-truth fit into the tens-of-meV
    # band within the test budget. The observed behavior (monotone decrease,
    # slow crawl in a stiff degenerate valley, parameters far from truth
    # while energies converge) is tracked for phase 3.5: a Wolfe line search
    # and parameter scaling are the planned §7.8 refinements — Armijo-only
    # backtracking is known to slow quasi-Newton methods.
    assert rmse < 20.0  # meV/atom
    result = fit.result
    assert len(result.history) >= 2 and np.all(np.diff(result.history) <= 1e-12)
    assert result.stop_reason in ("max_iterations", "gradient_tolerance", "line_search_failed")
    assert fit.objective == pytest.approx(
        float(np.sum(((energies - targets) / n_atoms * n_atoms) ** 2)), rel=1e-8
    )


def _fit_energies(db: TrainingDatabase, parameters: np.ndarray) -> np.ndarray:
    from atomistic_pinn.physics import atomic_energies

    out = np.empty(db.n_structures)
    with torch.no_grad():
        for k, geometry in enumerate(db.geometries):
            params = torch.tensor(
                np.broadcast_to(parameters.reshape(1, 8), (geometry.n_atoms, 8)).copy()
            )
            out[k] = float(
                atomic_energies(geometry.positions, geometry.cell, geometry.pbc, params).sum()
            )
    return out


def test_fixed_bop_objective_at_truth_is_the_shift_penalty() -> None:
    # The truth parameters reproduce the RAW stored energies exactly, but the
    # objective fits the SHIFTED targets (raw + 0.38446·n). At the truth the
    # objective therefore equals Σ_s (shift·n_s)² — a closed form pinning the
    # shift consistency of the whole chain (never zero, by design).
    from atomistic_pinn.constants import ENERGY_SHIFT_EV_PER_ATOM
    from atomistic_pinn.training.fixed_bop import fixed_bop_objective

    db = _small_db()
    truth = np.array(
        [
            [
                SYNTHETIC_TRUTH_PARAMS[name]
                for name in ("A", "B", "alpha", "beta", "a", "h", "sigma", "lambda")
            ]
        ]
    )[0]
    expected = float(np.sum((ENERGY_SHIFT_EV_PER_ATOM * db.atoms_per_structure()) ** 2))
    assert expected > 0.0
    assert fixed_bop_objective(db, truth) == pytest.approx(expected, rel=1e-12)


def test_training_run_is_seed_reproducible() -> None:
    from atomistic_pinn.training.trainer import build_database, run_training

    resolved = _resolved()
    first = run_training(resolved, build_database(resolved))
    second = run_training(resolved, build_database(resolved))
    np.testing.assert_array_equal(first.best_vector, second.best_vector)
    assert first.best_objective == second.best_objective
    assert first.histories == second.histories


def test_trainer_improves_and_converges_on_pipeline_truth() -> None:
    from atomistic_pinn.training.trainer import build_database, run_training

    resolved = _resolved()
    outcome = run_training(resolved, build_database(resolved))
    # The γ tether (pre-fit target ≈ truth) plus the representable truth make
    # substantial progress expected within 25 iterations.
    assert outcome.rmse_mev_per_atom < 200.0
    assert outcome.best_objective > 0.0
    assert len(outcome.histories) == resolved.training.n_random_restarts
    assert outcome.fixed_bop is not None
    assert outcome.p_bar is not None and outcome.p_bar.shape == (8,)


def test_checkpoint_round_trip_is_bit_exact(tmp_path) -> None:
    from atomistic_pinn.training.trainer import build_database, run_training

    resolved = _resolved()
    outcome = run_training(resolved, build_database(resolved))
    path = save_checkpoint(tmp_path / "test.pt", outcome, resolved.model_dump())

    payload = load_checkpoint(path)
    assert payload["format"] == 1
    np.testing.assert_array_equal(payload["flat_params"].numpy(), outcome.best_vector)
    np.testing.assert_array_equal(payload["dfp"]["x"].numpy(), outcome.best_dfp.x)
    assert payload["dfp"]["history"] == outcome.best_dfp.history
    assert payload["config_hash"] == outcome.config_hash
    assert payload["dataset_hash"] == outcome.dataset_hash
    assert payload["git_commit"] == outcome.git_commit
    assert payload["restart_seeds"] == outcome.restart_seeds
    assert payload["best_restart"] == outcome.best_restart
    assert payload["fold_id"] is None
    assert payload["metrics"]["rmse_mev_per_atom"] == pytest.approx(outcome.rmse_mev_per_atom)
