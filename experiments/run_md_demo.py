"""Run the MD demonstration protocol (Sprint 4; docs/md_plan.md).

Usage:
    # production demo on a trained checkpoint (stage B/C spec of configs/md_demo.yaml):
    python experiments/run_md_demo.py --checkpoint checkpoints/pinn_al_ani_dev.pt

    # quick machinery demonstration on the synthetic fixture parameter set
    # (labeled: physics-layer demonstration, NOT a trained potential):
    python experiments/run_md_demo.py --fixture \
        --conventional-cells 1 --nve-steps 500 --nvt-steps 200

Writes to results/md_demo/ (generated, never committed):
    nve_trajectory.extxyz / nvt_trajectory.extxyz, energy_temperature.csv,
    md_demo.png, summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from ase import Atoms
from ase.build import bulk

from atomistic_pinn.models import PINN
from atomistic_pinn.simulation.calculator import PINNCalculator
from atomistic_pinn.simulation.md import run_nve, run_nvt_langevin
from atomistic_pinn.training.objective import PINNEnergyModel, get_flat_params
from atomistic_pinn.training.synthetic import SYNTHETIC_TRUTH_PARAMS


def build_atoms(cells: int, a: float) -> Atoms:
    """Ideal FCC Al supercell: ``cells``^3 conventional cells (4 atoms each)."""
    atoms = bulk("Al", "fcc", a=a, cubic=True)
    if cells > 1:
        atoms = atoms.repeat((cells, cells, cells))
    atoms.pbc = [True, True, True]
    return atoms


def build_calculator(resolved, checkpoint: str | None, fixture: bool) -> tuple[PINNCalculator, str]:
    """Calculator from a trained checkpoint, or the labeled fixture fallback."""
    if checkpoint is not None:
        calc = PINNCalculator.from_checkpoint(checkpoint, resolved)
        return calc, "trained checkpoint"
    if not fixture:
        raise SystemExit(
            "provide --checkpoint PATH for the trained potential, or pass "
            "--fixture for the synthetic-parameter physics demonstration"
        )
    torch.manual_seed(5)  # reproducible mild output-layer perturbation
    network = PINN(seed=5)
    with torch.no_grad():
        network.network[-1].weight.normal_(0.0, 1e-4)
        network.network[-1].bias.zero_()
        network.network[-1].bias.copy_(
            torch.tensor(
                [
                    SYNTHETIC_TRUTH_PARAMS[n]
                    for n in ("A", "B", "alpha", "beta", "a", "h", "sigma", "lambda")
                ]
            )
        )
    vector = get_flat_params(PINNEnergyModel(network)).numpy()
    return PINNCalculator(
        resolved, vector
    ), "fixture parameters (physics demonstration, not trained)"


def write_csv(path: Path, result) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_fs",
                "total_energy_ev",
                "potential_energy_ev",
                "temperature_k",
                "min_neighbors",
                "mean_neighbors",
            ]
        )
        for i in range(len(result.times_fs)):
            writer.writerow(
                [
                    f"{result.times_fs[i]:.2f}",
                    f"{result.total_energy_ev[i]:.8f}",
                    f"{result.potential_energy_ev[i]:.8f}",
                    f"{result.temperature_k[i]:.4f}",
                    f"{result.min_neighbors[i]:.0f}",
                    f"{result.mean_neighbors[i]:.4f}",
                ]
            )


def plot_stages(nve, nvt, path: Path, n_atoms: int, source: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4))
    axes[0, 0].plot(nve.times_fs, nve.total_energy_ev, color="#0B1C2C", lw=1.4)
    axes[0, 0].set_title(
        f"(a) NVE total energy — drift {nve.energy_drift_fraction * 100:.4f}%", fontsize=10
    )
    axes[0, 1].plot(nve.times_fs, nve.temperature_k, color="#529286", lw=1.2)
    axes[0, 1].axhline(300.0, color="#5B6B7D", ls="--", lw=1)
    axes[0, 1].set_title("(b) NVE temperature", fontsize=10)
    axes[1, 0].plot(nvt.times_fs, nvt.total_energy_ev, color="#D4875A", lw=1.4)
    axes[1, 0].set_title("(c) NVT (Langevin) total energy", fontsize=10)
    axes[1, 1].plot(nvt.times_fs, nvt.temperature_k, color="#529286", lw=1.2)
    axes[1, 1].axhline(300.0, color="#5B6B7D", ls="--", lw=1)
    axes[1, 1].set_title("(d) NVT temperature", fontsize=10)
    for ax in axes.ravel():
        ax.set_xlabel("time (fs)")
        ax.grid(True, ls="--", lw=0.5, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel("total energy (eV)")
    for ax in (axes[0, 1], axes[1, 1]):
        ax.set_ylabel("temperature (K)")
    fig.suptitle(f"MD demonstration — {n_atoms} atoms, forces: {source}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md-config", default="configs/md_demo.yaml")
    parser.add_argument("--model-config", default="configs/pinn_al.yaml")
    parser.add_argument("--checkpoint", default=None, help="trained .pt checkpoint (PINN)")
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="synthetic fixture parameters instead of a checkpoint",
    )
    parser.add_argument("--conventional-cells", type=int, default=None)
    parser.add_argument("--nve-steps", type=int, default=None)
    parser.add_argument("--nvt-steps", type=int, default=None)
    parser.add_argument("--out", default="results/md_demo")
    args = parser.parse_args()

    md_cfg = yaml.safe_load(Path(args.md_config).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = args.conventional_cells or md_cfg["system"]["conventional_cells"]
    a_lattice = md_cfg["system"]["lattice_parameter_a"]
    nve_steps = args.nve_steps or md_cfg["nve"]["steps"]
    nvt_steps = args.nvt_steps or md_cfg["nvt"]["steps"]
    log_interval = md_cfg["logging"]["interval_steps"]

    # model config provides the architecture/activation/descriptors only
    from atomistic_pinn.training.config import load_training_config

    resolved = load_training_config(args.model_config)
    calc, source = build_calculator(resolved, args.checkpoint, args.fixture)

    atoms = build_atoms(cells, a_lattice)
    n_atoms = len(atoms)
    print(f"[md-demo] {n_atoms} atoms ({cells}x{cells}x{cells} cells), forces: {source}")

    print("[md-demo] stage B: NVE ...")
    nve = run_nve(
        atoms,
        calc,
        timestep_fs=md_cfg["nve"]["timestep_fs"],
        n_steps=nve_steps,
        temperature_k=md_cfg["nve"]["temperature_k"],
        seed=md_cfg["nve"]["seed"],
        log_interval=log_interval,
        trajectory_path=str(out_dir / "nve_trajectory.extxyz"),
    )
    print(f"[md-demo] NVE drift fraction: {nve.energy_drift_fraction:.3e}")

    print("[md-demo] stage C: NVT (Langevin) ...")
    nvt = run_nvt_langevin(
        atoms,
        calc,
        temperature_k=md_cfg["nvt"]["temperature_k"],
        timestep_fs=md_cfg["nvt"]["timestep_fs"],
        n_steps=nvt_steps,
        friction_fs_inv=md_cfg["nvt"]["friction_fs_inv"],
        seed=md_cfg["nvt"]["seed"],
        log_interval=log_interval,
        trajectory_path=str(out_dir / "nvt_trajectory.extxyz"),
    )

    write_csv(out_dir / "energy_temperature.csv", nve)  # NVE record
    plot_stages(nve, nvt, out_dir / "md_demo.png", n_atoms, source)

    summary = {
        "n_atoms": n_atoms,
        "conventional_cells": cells,
        "force_source": source,
        "nve": {**nve.summary(), "log_interval": log_interval},
        "nvt": {**nvt.summary(), "log_interval": log_interval, "thermostat": "langevin"},
        "md_config": args.md_config,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[md-demo] artifacts written to {out_dir}/")


if __name__ == "__main__":
    main()
