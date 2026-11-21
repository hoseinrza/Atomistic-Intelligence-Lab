"""Training layer: database, objective, DFP, trainer, configs (Sprint 3)."""

from atomistic_pinn.training.checkpoints import load_checkpoint, save_checkpoint
from atomistic_pinn.training.config import (
    ResolvedTrainingConfig,
    TrainingConfig,
    load_training_config,
)
from atomistic_pinn.training.database import StructureGeometry, TrainingDatabase
from atomistic_pinn.training.dfp import DFPResult, dfp_minimize
from atomistic_pinn.training.fixed_bop import fit_fixed_bop, fixed_bop_objective
from atomistic_pinn.training.objective import (
    BaselineEnergyModel,
    PINNEnergyModel,
    TetherConfig,
    get_flat_params,
    objective_function,
    objective_value,
    set_flat_params,
)
from atomistic_pinn.training.trainer import (
    TrainingOutcome,
    build_database,
    run_training,
)

__all__ = [
    "BaselineEnergyModel",
    "DFPResult",
    "PINNEnergyModel",
    "ResolvedTrainingConfig",
    "StructureGeometry",
    "TetherConfig",
    "TrainingConfig",
    "TrainingDatabase",
    "TrainingOutcome",
    "build_database",
    "dfp_minimize",
    "fit_fixed_bop",
    "fixed_bop_objective",
    "get_flat_params",
    "load_checkpoint",
    "load_training_config",
    "objective_function",
    "objective_value",
    "run_training",
    "save_checkpoint",
    "set_flat_params",
]
