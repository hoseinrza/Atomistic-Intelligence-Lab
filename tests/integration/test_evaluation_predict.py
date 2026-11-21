"""Integration tests: restoring a trained model and predicting on a database."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from atomistic_pinn.evaluation.predict import (
    predicted_bop_parameters,
    predicted_structure_energies,
    restore_model,
)
from atomistic_pinn.training.config import load_training_config
from atomistic_pinn.training.objective import get_flat_params
from atomistic_pinn.training.trainer import build_database

pytestmark = pytest.mark.integration


def _ani_config(tmp_path: Path, ani_dir: Path) -> dict[str, object]:
    return {
        "experiment": {"name": "eval_predict", "seed": 42},
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
            "n_random_restarts": 2,
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
    }


@pytest.fixture
def resolved_db(ani_dir: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(_ani_config(tmp_path, ani_dir)))
    resolved = load_training_config(config_path)
    return resolved, build_database(resolved)


class TestRestoreModel:
    def test_predictions_deterministic_under_fixed_vector(
        self, resolved_db: tuple[object, object]
    ) -> None:
        resolved, db = resolved_db
        model = restore_model(resolved, np.linspace(-0.01, 0.01, 1283))
        energies = predicted_structure_energies(model, db)
        assert energies.shape == (db.n_structures,)
        model_again = restore_model(resolved, np.linspace(-0.01, 0.01, 1283))
        np.testing.assert_allclose(
            predicted_structure_energies(model_again, db), energies, rtol=0, atol=0
        )

    def test_default_seed_vector_round_trip(self, resolved_db: tuple[object, object]) -> None:
        resolved, db = resolved_db
        probe = restore_model(resolved, np.zeros(1283))
        vector = get_flat_params(probe).numpy()
        model = restore_model(resolved, vector)
        expected = probe.structure_energies(db).detach().numpy()
        np.testing.assert_allclose(predicted_structure_energies(model, db), expected)

    def test_bop_parameter_shapes(self, resolved_db: tuple[object, object]) -> None:
        resolved, db = resolved_db
        assert resolved.model.type == "pinn"
        model = restore_model(resolved, np.zeros(1283))
        params = predicted_bop_parameters(model, db)
        assert len(params) == db.n_structures
        for array, geometry in zip(params, db.geometries, strict=True):
            assert array.shape == (geometry.n_atoms, 8)
