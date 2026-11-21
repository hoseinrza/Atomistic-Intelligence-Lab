"""Training driver: ledger-resolved config → pre-fit → restarts × DFP (Sprint 3).

Cross-validation is NOT here — Sprint 4 wraps it around this driver. Sprint 3
trains on a single documented development database to prove the machinery.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import numpy as np
import torch

from atomistic_pinn.data.ani_al import dataset_from_ani_al
from atomistic_pinn.exceptions import ConfigurationError
from atomistic_pinn.models import PINN, BaselineNN
from atomistic_pinn.training.config import ResolvedTrainingConfig
from atomistic_pinn.training.database import TrainingDatabase
from atomistic_pinn.training.dfp import DFPResult, dfp_minimize
from atomistic_pinn.training.fixed_bop import FixedBOPFit, fit_fixed_bop
from atomistic_pinn.training.objective import (
    BaselineEnergyModel,
    PINNEnergyModel,
    TetherConfig,
    get_flat_params,
    objective_function,
    set_flat_params,
)
from atomistic_pinn.training.synthetic import generate_synthetic_database


@dataclass
class TrainingOutcome:
    """Complete record of one training run (checkpoint payload base).

    Attributes:
        model_kind: ``"pinn"`` | ``"baseline_nn"``.
        best_objective: Lowest final ℰ over restarts (eV²).
        best_restart: Index of the winning restart.
        best_vector: Flat parameter vector of the winner.
        best_dfp: DFP record of the winner (inverse Hessian included).
        histories: Objective history per restart.
        rmse_mev_per_atom: Energy RMSE on the training database (meV/atom).
        n_parameters: Flat parameter count.
        p_bar: γ-tether target (PINN, fixed_bop_prefit) or None.
        fixed_bop: The pre-fit record (also the fixed-BOP baseline) or None.
        restart_seeds: Derived seed per restart (provenance).
        stop_reason: Stop reason of the winning restart.
        config_hash: Hash of the resolved configuration.
        dataset_hash: Hash of the training database.
        git_commit: Short commit hash at run time (or "unknown").
    """

    model_kind: str
    best_objective: float
    best_restart: int
    best_vector: np.ndarray
    best_dfp: DFPResult
    histories: list[list[float]] = field(default_factory=list)
    rmse_mev_per_atom: float = 0.0
    n_parameters: int = 0
    p_bar: np.ndarray | None = None
    fixed_bop: FixedBOPFit | None = None
    restart_seeds: list[int] = field(default_factory=list)
    stop_reason: str = ""
    config_hash: str = ""
    dataset_hash: str = ""
    git_commit: str = "unknown"


def git_commit() -> str:
    """Short commit hash of the working tree (best effort)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def training_rmse_mev_per_atom(
    model: PINNEnergyModel | BaselineEnergyModel, db: TrainingDatabase
) -> float:
    """Energy RMSE in meV/atom over the database (per-atom errors)."""
    energies = model.structure_energies(db).detach().numpy()
    targets = db.shifted_targets()
    n_atoms = db.atoms_per_structure()
    per_atom_errors = (energies - targets) / n_atoms
    return float(1000.0 * np.sqrt(np.mean(per_atom_errors**2)))


def run_training(resolved: ResolvedTrainingConfig, db: TrainingDatabase) -> TrainingOutcome:
    """Execute the resolved training plan (pre-fit → restarts × DFP → best).

    Args:
        resolved: Ledger-resolved configuration.
        db: Training database (descriptors precomputed).

    Returns:
        :class:`TrainingOutcome` — the checkpoint/report payload.
    """
    training = resolved.training
    is_pinn = resolved.model.type == "pinn"
    activation_raw = resolved.model.activation
    assert activation_raw is not None  # resolved (§7.4)
    activation_value: str = activation_raw
    assert training.n_random_restarts is not None  # resolved (§7.7)
    n_restarts: int = training.n_random_restarts
    tau_raw = training.regularization.tau
    assert tau_raw is not None  # resolved (§7.6)
    tau_value: float = tau_raw

    fixed_fit: FixedBOPFit | None = None
    tether: TetherConfig | None = None
    if is_pinn:
        gamma_raw = training.regularization.gamma
        assert gamma_raw is not None  # resolved (§7.6)
        gamma_value: float = gamma_raw
        if training.p_bar_eta == "fixed_bop_prefit":
            # One computation, two uses: γ target + fixed-BOP baseline (§7.13).
            fixed_fit = fit_fixed_bop(
                db,
                gradient=training.gradients,
                n_restarts=3,
                master_seed=resolved.experiment.seed,
            )
            tether = TetherConfig(
                gamma=gamma_value,
                mode="fixed_bop_prefit",
                target=torch.tensor(fixed_fit.parameters, dtype=torch.float64),
            )
        else:  # self_mean (§7.13 option b)
            tether = TetherConfig(gamma=gamma_value, mode="self_mean")

    def make_model(seed: int) -> PINNEnergyModel | BaselineEnergyModel:
        if is_pinn:
            return PINNEnergyModel(PINN(seed=seed, activation=activation_value))
        return BaselineEnergyModel(BaselineNN(seed=seed, activation=activation_value))

    seed_sequence = np.random.SeedSequence(resolved.experiment.seed)
    children = seed_sequence.spawn(n_restarts)
    restart_seeds = [int(child.generate_state(1)[0]) for child in children]

    best_vector: np.ndarray | None = None
    best_result: DFPResult | None = None
    best_restart = -1
    histories: list[list[float]] = []
    n_params = 0

    for restart, seed in enumerate(restart_seeds):
        model = make_model(seed)
        n_params = sum(p.numel() for p in model.network.parameters())
        x0 = get_flat_params(model).numpy()
        f, grad = objective_function(
            model,
            db,
            tau=tau_value,
            tether=tether,
            gradient=training.gradients,
        )
        result = dfp_minimize(
            f,
            x0,
            grad,
            max_iterations=training.max_iterations,
            gradient_tolerance=training.gradient_tolerance,
        )
        histories.append(list(result.history))
        if best_result is None or result.f < best_result.f:
            best_result = result
            best_restart = restart
            best_vector = result.x.copy()

    assert best_result is not None and best_vector is not None  # ≥ 1 restart

    # Evaluate the winner through the module-state path (same numbers as the
    # flat-vector path — the contract tests pin this equivalence).
    best_model = make_model(restart_seeds[best_restart])
    set_flat_params(best_model, torch.tensor(best_vector, dtype=torch.float64))
    rmse = training_rmse_mev_per_atom(best_model, db)

    return TrainingOutcome(
        model_kind=resolved.model.type,
        best_objective=float(best_result.f),
        best_restart=best_restart,
        best_vector=best_vector,
        best_dfp=best_result,
        histories=histories,
        rmse_mev_per_atom=rmse,
        n_parameters=n_params,
        p_bar=None if fixed_fit is None else fixed_fit.parameters.copy(),
        fixed_bop=fixed_fit,
        restart_seeds=restart_seeds,
        stop_reason=best_result.stop_reason,
        config_hash=resolved.config_hash,
        dataset_hash=db.dataset_hash,
        git_commit=git_commit(),
    )


def build_database(resolved: ResolvedTrainingConfig) -> TrainingDatabase:
    """Construct the training database from the resolved data section."""
    if resolved.data.source == "synthetic_bop":
        assert resolved.data.synthetic is not None
        dataset = generate_synthetic_database(
            n_structures=resolved.data.synthetic.n_structures,
            seed=resolved.data.synthetic.seed,
            with_periodic=resolved.data.synthetic.with_periodic,
        )
        return TrainingDatabase(dataset, resolved.descriptors.to_descriptor_config())
    if resolved.data.source == "ani_al":
        # config resolution enforces both (ConfigurationError at load time);
        # the guards keep the types honest if this is called with a raw cfg.
        if resolved.data.path is None or resolved.data.subset_size is None:
            raise ConfigurationError(
                "data.source='ani_al' requires data.path and data.subset_size "
                "(data/README.md; docs/sprint3.md §6)"
            )
        dataset = dataset_from_ani_al(
            resolved.data.path,
            max_structures=resolved.data.subset_size,
            subset_seed=resolved.data.subset_seed,
        )
        return TrainingDatabase(dataset, resolved.descriptors.to_descriptor_config())
    raise ConfigurationError(
        "loading a dataset from source='path' (extxyz) lands with the Sprint "
        "4 data work; use data.source='synthetic_bop' for pipeline validation "
        "or data.source='ani_al' for the alternative dataset (data/README.md)"
        + (f" (requested: {resolved.data.path})" if resolved.data.path else "")
    )
