"""Molecular-dynamics demonstration (Sprint 4, phase 4.3; docs/md_plan.md).

MD is a *demonstration*: proof that the trained PINN potential is usable in
a standard atomistic workflow and stays physically stable over a short
trajectory. No paper metric is claimed from it.

Protocol (`docs/md_plan.md` §4, all values labeled engineering decisions):

* forces through the autograd route (``PINNCalculator``, FD-validated on
  static configurations — Sprint 4 gate 2);
* stage B — NVE: the strongest correctness gate is **total-energy drift**
  below a documented tolerance over a fixed, seeded step count;
* stage C — NVT (Langevin, seeded): temperature stability and a usable
  trajectory;
* initial velocities from Maxwell–Boltzmann at the target temperature with
  zero total momentum;
* extrapolation guard: per frame, the minimum/mean neighbor count within
  the cutoff is logged — frames whose environments leave the training
  domain are *flagged in the log*, never silently used
  (`docs/reproduction_protocol.md` fidelity rules);
* reproducibility: identical seeds produce identical trajectories; the
  caller records config/checkpoint/dataset hashes with the run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms, units
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

from atomistic_pinn.simulation.calculator import PINNCalculator

# Labeled engineering decisions (docs/md_plan.md §4 — proposed defaults):
DEFAULT_TIMESTEP_FS = 1.0
DEFAULT_LOG_INTERVAL_STEPS = 10
# Primary MD physics gate tolerance (documented): relative total-energy
# drift over an NVE run. The machinery tests use a generous gate on tiny
# runs; the production gate value is set per demonstration run and logged.
DEFAULT_NVE_DRIFT_TOLERANCE = 5e-2


@dataclass(frozen=True)
class MDResult:
    """Record of one MD demonstration run.

    Attributes:
        ensemble: ``"NVE"`` or ``"NVT"``.
        timestep_fs: Integration timestep.
        n_steps: Steps taken.
        times_fs: Log times (fs), one entry per logged frame.
        total_energy_ev: Total (potential + kinetic) energy per frame (eV).
        potential_energy_ev: Potential energy per frame (eV).
        temperature_k: Instantaneous temperature per frame (K).
        min_neighbors: Minimum neighbor count within the cutoff per frame
            (extrapolation guard).
        mean_neighbors: Mean neighbor count within the cutoff per frame.
        energy_drift_ev: ``E_total(last) − E_total(first)`` (eV).
        energy_drift_fraction: Drift relative to ``|E_total(first)|``.
        seed: The run's seed (Langevin noise / velocity init).
    """

    ensemble: str
    timestep_fs: float
    n_steps: int
    times_fs: np.ndarray
    total_energy_ev: np.ndarray
    potential_energy_ev: np.ndarray
    temperature_k: np.ndarray
    min_neighbors: np.ndarray
    mean_neighbors: np.ndarray
    energy_drift_ev: float
    energy_drift_fraction: float
    seed: int

    def within_drift_tolerance(self, tolerance: float) -> bool:
        """The primary MD physics gate (docs/md_plan.md §5)."""
        return bool(abs(self.energy_drift_fraction) <= tolerance)

    def summary(self) -> dict[str, object]:
        """JSON-ready summary (generated artifacts are never committed)."""
        return {
            "ensemble": self.ensemble,
            "timestep_fs": self.timestep_fs,
            "n_steps": self.n_steps,
            "seed": self.seed,
            "initial_total_energy_ev": float(self.total_energy_ev[0]),
            "final_total_energy_ev": float(self.total_energy_ev[-1]),
            "energy_drift_ev": self.energy_drift_ev,
            "energy_drift_fraction": self.energy_drift_fraction,
            "final_temperature_k": float(self.temperature_k[-1]),
            "min_neighbors_first_frame": float(self.min_neighbors[0]),
            "min_neighbors_last_frame": float(self.min_neighbors[-1]),
        }


def _neighbor_statistics(calculator: PINNCalculator, atoms: Atoms) -> tuple[float, float]:
    """Min/mean neighbor count within the BOP cutoff per atom (guard logging).

    The counts come from the calculator's fixed neighbor metadata — the
    same lists the potential evaluated this frame — with the bond tier
    recomputed at the current positions (exact; see
    ``PINNCalculator.neighbor_counts``)."""
    counts = calculator.neighbor_counts(
        np.asarray(atoms.get_positions(), dtype=np.float64),
        np.asarray(atoms.get_cell(), dtype=np.float64),
        (bool(atoms.get_pbc()[0]), bool(atoms.get_pbc()[1]), bool(atoms.get_pbc()[2])),
    )
    return float(np.min(counts)), float(np.mean(counts))


def _seed_velocities(atoms: Atoms, temperature_k: float, seed: int) -> None:
    """Maxwell–Boltzmann velocities with zero total momentum (seeded)."""
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_k, rng=np.random.default_rng(seed)
    )
    Stationary(atoms)


def _observe(calculator: PINNCalculator, atoms: Atoms) -> tuple[float, float, float, float, float]:
    potential = float(atoms.get_potential_energy())
    kinetic = float(atoms.get_kinetic_energy())
    min_neighbors, mean_neighbors = _neighbor_statistics(calculator, atoms)
    return (
        potential + kinetic,
        potential,
        float(atoms.get_temperature()),
        min_neighbors,
        mean_neighbors,
    )


def run_nve(
    atoms: Atoms,
    calculator: PINNCalculator,
    *,
    timestep_fs: float = DEFAULT_TIMESTEP_FS,
    n_steps: int = 1000,
    temperature_k: float = 300.0,
    seed: int = 0,
    log_interval: int = DEFAULT_LOG_INTERVAL_STEPS,
    trajectory_path: str | None = None,
) -> MDResult:
    """Stage B — microcanonical demonstration (the drift gate)."""
    atoms.calc = calculator
    _seed_velocities(atoms, temperature_k, seed)
    times, totals, potentials, temperatures, min_n, mean_n = _collect_frames(
        VelocityVerlet(atoms, timestep_fs * units.fs),
        calculator,
        n_steps,
        log_interval,
        trajectory_path,
    )
    drift = float(totals[-1] - totals[0])
    reference = float(abs(totals[0])) or 1.0
    return MDResult(
        ensemble="NVE",
        timestep_fs=timestep_fs,
        n_steps=n_steps,
        times_fs=times,
        total_energy_ev=totals,
        potential_energy_ev=potentials,
        temperature_k=temperatures,
        min_neighbors=min_n,
        mean_neighbors=mean_n,
        energy_drift_ev=drift,
        energy_drift_fraction=drift / reference,
        seed=seed,
    )


def run_nvt_langevin(
    atoms: Atoms,
    calculator: PINNCalculator,
    *,
    temperature_k: float = 300.0,
    timestep_fs: float = DEFAULT_TIMESTEP_FS,
    n_steps: int = 1000,
    friction_fs_inv: float = 0.02,
    seed: int = 0,
    log_interval: int = DEFAULT_LOG_INTERVAL_STEPS,
    trajectory_path: str | None = None,
) -> MDResult:
    """Stage C — canonical demonstration (seeded Langevin thermostat)."""
    atoms.calc = calculator
    _seed_velocities(atoms, temperature_k, seed)
    dynamics = Langevin(
        atoms,
        timestep_fs * units.fs,
        temperature_K=temperature_k,
        friction=friction_fs_inv / units.fs,
        rng=np.random.default_rng(seed),
    )
    times, totals, potentials, temperatures, min_n, mean_n = _collect_frames(
        dynamics, calculator, n_steps, log_interval, trajectory_path
    )
    drift = float(totals[-1] - totals[0])
    reference = float(abs(totals[0])) or 1.0
    return MDResult(
        ensemble="NVT",
        timestep_fs=timestep_fs,
        n_steps=n_steps,
        times_fs=times,
        total_energy_ev=totals,
        potential_energy_ev=potentials,
        temperature_k=temperatures,
        min_neighbors=min_n,
        mean_neighbors=mean_n,
        energy_drift_ev=drift,
        energy_drift_fraction=drift / reference,
        seed=seed,
    )


def _collect_frames(
    dynamics: MolecularDynamics,
    calculator: PINNCalculator,
    n_steps: int,
    log_interval: int,
    trajectory_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run ``n_steps`` and log frames every ``log_interval`` steps.

    When ``trajectory_path`` is given, every logged frame is appended to an
    ASE extxyz trajectory alongside the scalar record (generated artifact —
    never committed).
    """
    atoms = dynamics.atoms
    times, totals, potentials, temperatures, min_n, mean_n = [], [], [], [], [], []

    def log() -> None:
        t, e_pot, temp, n_min, n_mean = _observe(calculator, atoms)
        times.append(dynamics.get_time() / units.fs)
        totals.append(t)
        potentials.append(e_pot)
        temperatures.append(temp)
        min_n.append(n_min)
        mean_n.append(n_mean)
        if trajectory_path is not None:
            write(trajectory_path, atoms, format="extxyz", append=True)

    log()
    steps_done = 0
    while steps_done < n_steps:
        batch = min(log_interval, n_steps - steps_done)
        dynamics.run(batch)
        steps_done += batch
        log()
    return (
        np.asarray(times),
        np.asarray(totals),
        np.asarray(potentials),
        np.asarray(temperatures),
        np.asarray(min_n),
        np.asarray(mean_n),
    )
