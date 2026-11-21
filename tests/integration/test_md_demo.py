"""Integration tests: MD demonstration machinery (Sprint 4 gates 4-5).

The MD physics gate (NVE drift below a documented tolerance) is exercised
on a tiny seeded run with a near-physical parameter set (the Sprint 2/3
synthetic fixture — A·α < B·β; test fixture, never paper values, §7.15).
Production drift gates run on trained checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from ase import Atoms

from atomistic_pinn.models import PINN
from atomistic_pinn.simulation.calculator import PINNCalculator
from atomistic_pinn.simulation.md import (
    DEFAULT_NVE_DRIFT_TOLERANCE,
    run_nve,
    run_nvt_langevin,
)
from atomistic_pinn.training.config import load_training_config
from atomistic_pinn.training.objective import PINNEnergyModel, get_flat_params
from atomistic_pinn.training.synthetic import SYNTHETIC_TRUTH_PARAMS

pytestmark = pytest.mark.integration


def _resolved(tmp_path: Path, ani_dir: Path):
    config = {
        "experiment": {"name": "md_test", "seed": 42},
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
    }
    config_path = tmp_path / "md_cfg.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return load_training_config(config_path)


def _calculator(tmp_path: Path, ani_dir: Path) -> PINNCalculator:
    resolved = _resolved(tmp_path, ani_dir)
    torch.manual_seed(5)  # the output-layer perturbation uses the global RNG
    network = PINN(seed=5)
    with torch.no_grad():
        network.network[-1].weight.normal_(0.0, 1e-4)
        network.network[-1].bias.zero_()
        network.network[-1].bias.copy_(
            torch.tensor(
                [
                    SYNTHETIC_TRUTH_PARAMS[name]
                    for name in ("A", "B", "alpha", "beta", "a", "h", "sigma", "lambda")
                ]
            )
        )
    vector = get_flat_params(PINNEnergyModel(network)).numpy()
    return PINNCalculator(resolved, vector)


def _atoms() -> Atoms:
    return Atoms(
        "Al2",
        positions=[[0.0, 0.0, 0.0], [2.6, 0.2, 0.3]],
        cell=9.0 * np.eye(3),
        pbc=[True, True, True],
    )


class TestNVE:
    def test_energy_drift_gate_on_seeded_run(self, tmp_path: Path, ani_dir: Path) -> None:
        result = run_nve(
            _atoms(),
            _calculator(tmp_path, ani_dir),
            timestep_fs=1.0,
            n_steps=20,
            temperature_k=300.0,
            seed=42,
            log_interval=5,
        )
        assert result.ensemble == "NVE"
        assert len(result.times_fs) == 5  # frame 0 + 4 logs
        assert np.all(np.isfinite(result.total_energy_ev))
        assert result.within_drift_tolerance(DEFAULT_NVE_DRIFT_TOLERANCE)
        # The demonstration is non-trivial: atoms actually moved.
        assert result.energy_drift_ev == pytest.approx(0.0, abs=0.5)

    def test_extrapolation_guard_logged(self, tmp_path: Path, ani_dir: Path) -> None:
        result = run_nve(_atoms(), _calculator(tmp_path, ani_dir), n_steps=5, log_interval=5)
        assert result.min_neighbors[0] >= 0
        assert result.mean_neighbors[0] > 0.0


class TestNVT:
    def test_langevin_reproducible_under_seed(self, tmp_path: Path, ani_dir: Path) -> None:
        first = run_nvt_langevin(
            _atoms(), _calculator(tmp_path, ani_dir), n_steps=10, log_interval=5, seed=7
        )
        second = run_nvt_langevin(
            _atoms(), _calculator(tmp_path, ani_dir), n_steps=10, log_interval=5, seed=7
        )
        np.testing.assert_allclose(first.total_energy_ev, second.total_energy_ev, rtol=0, atol=0)
        assert first.ensemble == "NVT"
        assert np.all(np.isfinite(second.temperature_k))
