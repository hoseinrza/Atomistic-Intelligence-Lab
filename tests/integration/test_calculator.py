"""Integration tests: ASE calculator contract (Sprint 4 gates 2-3).

Covers the calculator contract (shapes/types, energy equals a direct
pipeline call), the FD-vs-autograd agreement on static configurations at
the calculator level, translational invariance (ΣF = 0), and the
rebuild-on-demand threshold mechanics (docs/md_plan.md §3).
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
from atomistic_pinn.training.config import load_training_config
from atomistic_pinn.training.objective import PINNEnergyModel, get_flat_params

pytestmark = pytest.mark.integration


def _resolved(tmp_path: Path, ani_dir: Path):
    config = {
        "experiment": {"name": "calc_test", "seed": 42},
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
    config_path = tmp_path / "calc_cfg.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return load_training_config(config_path)


def _mild_network() -> PINN:
    """Network outputting near-fixture physical parameters (small forces)."""
    from atomistic_pinn.training.synthetic import SYNTHETIC_TRUTH_PARAMS

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
    return network


@pytest.fixture
def calculator(tmp_path: Path, ani_dir: Path) -> PINNCalculator:
    resolved = _resolved(tmp_path, ani_dir)
    vector = get_flat_params(PINNEnergyModel(_mild_network())).numpy()
    return PINNCalculator(resolved, vector)


def _two_atom_atoms() -> Atoms:
    atoms = Atoms(
        "Al2",
        positions=[[0.0, 0.0, 0.0], [2.5, 0.3, 0.4]],
        cell=8.0 * np.eye(3),
        pbc=[True, True, True],
    )
    return atoms


class TestCalculatorContract:
    def test_energy_and_force_shapes(self, calculator: PINNCalculator) -> None:
        atoms = _two_atom_atoms()
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        assert np.isscalar(energy) or np.asarray(energy).ndim == 0
        assert forces.shape == (2, 3)
        assert np.all(np.isfinite(forces))

    def test_energy_matches_direct_pipeline_call(self, calculator: PINNCalculator) -> None:
        atoms = _two_atom_atoms()
        atoms.calc = calculator
        energy = float(atoms.get_potential_energy())
        from atomistic_pinn.evaluation.forces import _pipeline_total_energy

        direct = _pipeline_total_energy(
            calculator._model,
            atoms.get_positions(),
            atoms.get_cell(),
            [True, True, True],
            calculator._descriptor_config,
        )
        assert energy == pytest.approx(direct, rel=1e-12)

    def test_autograd_forces_match_finite_difference(self, tmp_path: Path, ani_dir: Path) -> None:
        resolved = _resolved(tmp_path, ani_dir)
        vector = get_flat_params(PINNEnergyModel(_mild_network())).numpy()
        atoms = _two_atom_atoms()
        atoms.calc = PINNCalculator(resolved, vector, force_method="autograd")
        autograd = atoms.get_forces().copy()
        atoms.calc = PINNCalculator(resolved, vector, force_method="finite_difference")
        finite = atoms.get_forces()
        np.testing.assert_allclose(autograd, finite, rtol=1e-4, atol=1e-5)

    def test_sum_of_forces_vanishes(self, calculator: PINNCalculator) -> None:
        atoms = _two_atom_atoms()
        atoms.calc = calculator
        forces = atoms.get_forces()
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-10)


class TestRebuildOnDemand:
    def test_small_displacement_reuses_lists(self, calculator: PINNCalculator) -> None:
        atoms = _two_atom_atoms()
        atoms.calc = calculator
        atoms.get_potential_energy()
        built_at = calculator._positions_built_at.copy()
        atoms.positions = atoms.positions + 0.05  # below the 0.2 Å threshold
        atoms.get_potential_energy()
        np.testing.assert_allclose(calculator._positions_built_at, built_at)

    def test_large_displacement_rebuilds_lists(self, calculator: PINNCalculator) -> None:
        atoms = _two_atom_atoms()
        atoms.calc = calculator
        atoms.get_potential_energy()
        built_at = calculator._positions_built_at.copy()
        atoms.positions = atoms.positions + 0.5  # above the threshold
        atoms.get_potential_energy()
        assert not np.allclose(calculator._positions_built_at, built_at)

    def test_rebuilt_lists_keep_energy_correct(self, calculator: PINNCalculator) -> None:
        atoms = _two_atom_atoms()
        atoms.calc = calculator
        atoms.get_potential_energy()
        atoms.positions = atoms.positions + 0.5
        energy = float(atoms.get_potential_energy())
        from atomistic_pinn.evaluation.forces import _pipeline_total_energy

        direct = _pipeline_total_energy(
            calculator._model,
            atoms.get_positions(),
            atoms.get_cell(),
            [True, True, True],
            calculator._descriptor_config,
        )
        assert energy == pytest.approx(direct, rel=1e-12)


class TestArgumentContracts:
    def test_unknown_force_method_rejected(self, tmp_path: Path, ani_dir: Path) -> None:
        resolved = _resolved(tmp_path, ani_dir)
        with pytest.raises(ValueError, match="force_method"):
            PINNCalculator(resolved, np.zeros(1283), force_method="magic")

    def test_nonpositive_rebuild_threshold_rejected(self, tmp_path: Path, ani_dir: Path) -> None:
        resolved = _resolved(tmp_path, ani_dir)
        with pytest.raises(ValueError, match="rebuild_threshold"):
            PINNCalculator(resolved, np.zeros(1283), rebuild_threshold=0.0)
