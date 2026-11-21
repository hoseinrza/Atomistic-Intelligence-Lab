"""Integration tests for the ``atomistic-pinn train`` command (Sprint 3)."""

from pathlib import Path

import numpy as np
import yaml
from typer.testing import CliRunner

from atomistic_pinn.cli import app
from atomistic_pinn.training.checkpoints import load_checkpoint

runner = CliRunner()

BASE_CONFIG = {
    "experiment": {"name": "cli_test", "seed": 42, "output_dir": None},
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
        "source": "synthetic_bop",
        "path": None,
        "synthetic": {"n_structures": 4, "seed": 7},
        "n_folds": 10,
    },
}


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    import copy

    cfg = copy.deepcopy(BASE_CONFIG)
    for section, values in (overrides or {}).items():
        cfg[section].update(values)
    cfg["experiment"]["output_dir"] = str(tmp_path / "checkpoints")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def test_train_command_runs_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(tmp_path)
    result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert result.exit_code == 0, result.output

    checkpoint = tmp_path / "checkpoints" / "cli_test.pt"
    report = tmp_path / "results" / "cli_test" / "metrics.json"
    assert checkpoint.exists()
    assert report.exists()

    payload = load_checkpoint(checkpoint)
    assert payload["model_kind"] == "pinn"
    assert payload["flat_params"].numel() == 1283
    assert payload["fold_id"] is None
    assert payload["git_commit"] != ""

    import json

    metrics = json.loads(report.read_text())
    assert metrics["n_parameters"] == 1283
    assert "rmse_mev_per_atom" in metrics
    assert "synthetic" in metrics["fidelity_note"]


def test_train_rejects_null_activation(tmp_path) -> None:
    config_path = _write_config(tmp_path, overrides={"model": {"activation": None}})
    result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert result.exit_code == 1
    assert "§7.4" in result.output


def test_train_rejects_forces_in_training(tmp_path) -> None:
    config_path = _write_config(tmp_path, overrides={"training": {"forces_in_training": True}})
    result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert result.exit_code == 1
    assert "forces" in result.output


def test_train_rejects_null_p_bar_eta(tmp_path) -> None:
    config_path = _write_config(tmp_path, overrides={"training": {"p_bar_eta": None}})
    result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert result.exit_code == 1
    assert "§7.13" in result.output


def test_shipped_configs_are_valid(tmp_path) -> None:
    # The shipped YAML configs must parse and resolve (with the small
    # synthetic database; the numerical-gradient default keeps real runs
    # faithful and is swapped for autograd only in this fast check).
    from atomistic_pinn.training.config import load_training_config

    repo_configs = Path(__file__).resolve().parents[2] / "configs"
    for name in ("pinn_al.yaml", "baseline_nn.yaml"):
        text = (repo_configs / name).read_text()
        cfg = yaml.safe_load(text)
        cfg["training"]["gradients"] = "autograd"
        cfg["training"]["max_iterations"] = 2
        cfg["data"]["synthetic"] = {"n_structures": 4, "seed": 7}
        cfg["experiment"]["name"] = f"smoke_{name}"
        cfg["experiment"]["output_dir"] = str(tmp_path)
        path = tmp_path / name
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        resolved = load_training_config(path)
        assert resolved.model.activation == "tanh"


def test_checkpoint_flat_params_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = _write_config(
        tmp_path,
        overrides={
            "model": {
                "type": "baseline_nn",
                "architecture": [60, 16, 16, 1],
                "n_parameters": 1265,
            },
            "training": {
                "p_bar_eta": None,
                "max_iterations": 2,
                "regularization": {"tau": 1e-6, "gamma": 0.0},
            },
        },
    )
    result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    payload = load_checkpoint(tmp_path / "checkpoints" / "cli_test.pt")
    assert payload["model_kind"] == "baseline_nn"
    assert payload["flat_params"].numel() == 1265
    metrics = (tmp_path / "results" / "cli_test" / "metrics.json").read_text()
    assert "baseline_nn" in metrics
    np.testing.assert_allclose(payload["metrics"]["objective_ev2"] >= 0.0, True)
