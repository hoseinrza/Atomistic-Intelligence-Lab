"""Training configuration: Pydantic validation + ledger resolution (Sprint 3).

The YAML configs pin paper facts; slots the paper does not specify are
``null``. :func:`load_training_config` parses the YAML, validates the
structure with Pydantic, then *resolves* the ledger items — any ``null``
slot Sprint 3 consumes raises :class:`ConfigurationError` naming the ledger
item, so nothing is ever filled silently. Paper facts embedded in the config
(architecture, parameter counts, energy shift, forces-in-training=false) are
re-validated against ``constants`` rather than trusted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from atomistic_pinn.constants import (
    ENERGY_SHIFT_EV_PER_ATOM,
    NN_ARCHITECTURE,
    NN_PRIME_ARCHITECTURE,
    PINN_ARCHITECTURE,
)
from atomistic_pinn.descriptors import DescriptorConfig
from atomistic_pinn.exceptions import ConfigurationError

SUPPORTED_TETHER_MODES = ("fixed_bop_prefit", "self_mean")


def _fc_param_count(architecture: tuple[int, ...]) -> int:
    """Weights + biases of a fully connected stack (paper Table 1 check)."""
    return sum(
        architecture[i] * architecture[i + 1] + architecture[i + 1]
        for i in range(len(architecture) - 1)
    )


class DescriptorCfg(BaseModel):
    """Descriptor section (paper values; §7.1-§7.3 decisions recorded)."""

    model_config = ConfigDict(extra="forbid")

    legendre_orders: tuple[int, ...] = (0, 1, 2, 4, 6)
    n_radial_centers: int = 12
    sigma: float = 1.0
    radial_centers: tuple[float, ...] = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
    )
    cutoff_radius: float = 6.0
    cutoff_d: float = 1.5
    include_same_pair: bool = True

    def to_descriptor_config(self) -> DescriptorConfig:
        """Build the immutable descriptor configuration."""
        if len(self.radial_centers) != self.n_radial_centers:
            raise ConfigurationError(
                f"descriptors.radial_centers has {len(self.radial_centers)} "
                f"entries but n_radial_centers = {self.n_radial_centers}"
            )
        n_descriptors = len(self.legendre_orders) * self.n_radial_centers
        if n_descriptors != 60:
            raise ConfigurationError(
                f"the networks consume 60 descriptors; "
                f"len(orders) × n_radial_centers = {n_descriptors}"
            )
        return DescriptorConfig(
            orders=tuple(self.legendre_orders),
            radial_centers=tuple(self.radial_centers),
            sigma=self.sigma,
            cutoff_radius=self.cutoff_radius,
            cutoff_d=self.cutoff_d,
            include_same_pair=self.include_same_pair,
        )


class ExperimentCfg(BaseModel):
    """Experiment identity and seeding."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    seed: int
    output_dir: str = "checkpoints"


class ModelCfg(BaseModel):
    """Model section; the pinned facts are re-validated against constants."""

    model_config = ConfigDict(extra="forbid")

    type: str  # "pinn" | "baseline_nn" (checked in resolve)
    architecture: list[int]
    n_parameters: int
    activation: str | None = None
    weight_init: str = "xavier_uniform"


class RegularizationCfg(BaseModel):
    """τ (weights/biases) and γ (parameter tether) of Eq. (13); §7.6."""

    model_config = ConfigDict(extra="forbid")

    tau: float | None = None
    gamma: float | None = None


class TrainingCfg(BaseModel):
    """Training protocol; unresolved ledger slots are ``None`` until resolve."""

    model_config = ConfigDict(extra="forbid")

    objective: str = "sum_squared_total_energy_error"
    energy_shift_ev_per_atom: float = ENERGY_SHIFT_EV_PER_ATOM
    optimizer: str = "dfp"
    n_random_restarts: int | None = None
    regularization: RegularizationCfg = RegularizationCfg()
    forces_in_training: bool = False
    # Sprint 3 resolutions (docs/sprint3.md §1):
    gradients: str = "numerical"
    line_search: str = "backtracking_armijo"
    p_bar_eta: str | None = None
    max_iterations: int = 100
    gradient_tolerance: float = 1e-8


class SyntheticCfg(BaseModel):
    """Synthetic development database (pipeline validation only, not DFT)."""

    model_config = ConfigDict(extra="forbid")

    n_structures: int = 24
    seed: int = 7


class DataCfg(BaseModel):
    """Dataset section: original (null), synthetic stand-in, or real path."""

    model_config = ConfigDict(extra="forbid")

    dataset: str | None = None  # label of the original paper database
    source: str | None = None
    path: str | None = None
    synthetic: SyntheticCfg | None = None
    n_folds: int = 10


class TrainingConfig(BaseModel):
    """Parsed but *unresolved* configuration (ledger slots may be None)."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentCfg
    model: ModelCfg
    descriptors: DescriptorCfg = DescriptorCfg()
    physics: dict[str, object] = Field(default_factory=dict)
    training: TrainingCfg = TrainingCfg()
    data: DataCfg = DataCfg()


class ResolvedTrainingConfig(BaseModel):
    """Configuration with every Sprint 3 slot resolved (no nulls in use)."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentCfg
    model: ModelCfg
    descriptors: DescriptorCfg
    physics: dict[str, object]
    training: TrainingCfg
    data: DataCfg

    @property
    def config_hash(self) -> str:
        """SHA-256 of the canonical resolved config (provenance)."""
        blob = json.dumps(
            self.model_dump(), sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return hashlib.sha256(blob).hexdigest()


def load_training_config(path: str | Path) -> ResolvedTrainingConfig:
    """Load, validate, and ledger-resolve a training YAML config.

    Raises:
        ConfigurationError: On unknown fields, invalid values, paper-fact
            mismatches, or a ``null`` slot Sprint 3 consumes (the error names
            the ledger item).
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigurationError(f"config {path} must be a YAML mapping")
    try:
        parsed = TrainingConfig.model_validate(raw)
    except Exception as error:  # pydantic ValidationError or similar
        raise ConfigurationError(f"invalid configuration {path}: {error}") from error
    return _resolve(parsed)


def _resolve(cfg: TrainingConfig) -> ResolvedTrainingConfig:
    """Resolve ledger items; nothing may be filled silently (hard rule)."""
    model = cfg.model
    if model.type not in ("pinn", "baseline_nn"):
        raise ConfigurationError(f"model.type must be 'pinn' or 'baseline_nn', got {model.type!r}")
    expected_architecture = {
        "pinn": PINN_ARCHITECTURE,
        "baseline_nn": NN_ARCHITECTURE,
    }[model.type]
    if tuple(model.architecture) != expected_architecture:
        raise ConfigurationError(
            f"model.architecture {model.architecture} does not match the "
            f"paper Table 1 value {list(expected_architecture)} for "
            f"{model.type!r}"
        )
    if model.n_parameters != _fc_param_count(tuple(model.architecture)):
        raise ConfigurationError(
            f"model.n_parameters {model.n_parameters} does not match the "
            f"recomputed count {_fc_param_count(tuple(model.architecture))} "
            f"(Table 1 consistency check)"
        )
    if model.activation is None:
        raise ConfigurationError(
            "model.activation is null — ledger §7.4 (activation functions) "
            "unresolved; the labeled proposal is 'tanh' (docs/sprint3.md §1)"
        )
    if model.activation not in ("tanh", "sigmoid"):
        raise ConfigurationError(
            f"model.activation must be 'tanh' or 'sigmoid' (ledger §7.4), got {model.activation!r}"
        )
    if model.weight_init != "xavier_uniform":
        raise ConfigurationError(
            f"model.weight_init must be 'xavier_uniform' (Sprint 3 decision), "
            f"got {model.weight_init!r}"
        )
    if model.type == "pinn" and tuple(model.architecture) == NN_PRIME_ARCHITECTURE:
        raise ConfigurationError(
            "NN' is the third-party PROPhet baseline and is not reproduced "
            "(ledger §7.9); use model.type='baseline_nn'"
        )

    training = cfg.training
    if training.objective != "sum_squared_total_energy_error":
        raise ConfigurationError(
            f"training.objective must be 'sum_squared_total_energy_error' "
            f"(Eq. 13 literal, §7.11), got {training.objective!r}"
        )
    if training.energy_shift_ev_per_atom != ENERGY_SHIFT_EV_PER_ATOM:
        raise ConfigurationError(
            f"training.energy_shift_ev_per_atom must equal the paper value "
            f"{ENERGY_SHIFT_EV_PER_ATOM} (applied pipeline-side only); got "
            f"{training.energy_shift_ev_per_atom}"
        )
    if training.optimizer != "dfp":
        raise ConfigurationError(
            f"training.optimizer must be 'dfp' (paper Methods), got {training.optimizer!r}"
        )
    if training.gradients not in ("numerical", "autograd"):
        raise ConfigurationError(
            f"training.gradients must be 'numerical' (resolved §7.8) or "
            f"'autograd' (labeled fallback), got {training.gradients!r}"
        )
    if training.line_search != "backtracking_armijo":
        raise ConfigurationError(
            f"training.line_search must be 'backtracking_armijo' (resolved "
            f"§7.8), got {training.line_search!r}"
        )
    if training.n_random_restarts is None:
        raise ConfigurationError(
            "training.n_random_restarts is null — ledger §7.7 (paper says "
            "'several times'); the labeled proposal is 5 (docs/sprint3.md §1)"
        )
    if training.forces_in_training:
        raise ConfigurationError(
            "training.forces_in_training=true contradicts the paper: forces "
            "are never trained or validated (Methods); any change would be a "
            "labeled deviation requiring a ledger update"
        )
    if training.regularization.tau is None:
        raise ConfigurationError(
            "training.regularization.tau is null — ledger §7.6 "
            "(regularization strength unpublished); tune by documented sweep "
            "(docs/sprint3.md §6)"
        )

    training = training.model_copy()
    if model.type == "baseline_nn":
        if training.regularization.gamma not in (None, 0.0):
            raise ConfigurationError(
                "baseline_nn has no BOP parameters — "
                "training.regularization.gamma must be 0.0 or null"
            )
        training.regularization.gamma = 0.0
        if training.p_bar_eta not in (None, *SUPPORTED_TETHER_MODES):
            raise ConfigurationError(f"unknown training.p_bar_eta {training.p_bar_eta!r} (§7.13)")
    else:
        if training.regularization.gamma is None:
            raise ConfigurationError(
                "training.regularization.gamma is null — ledger §7.6 (tether "
                "strength unpublished); tune by documented sweep"
            )
        if training.p_bar_eta is None:
            raise ConfigurationError(
                "training.p_bar_eta is null — ledger §7.13 (p̄_η is undefined "
                "without a convention); labeled default 'fixed_bop_prefit' "
                "(docs/sprint3.md §3.2a)"
            )
        if training.p_bar_eta not in SUPPORTED_TETHER_MODES:
            raise ConfigurationError(
                f"training.p_bar_eta must be one of {SUPPORTED_TETHER_MODES} "
                f"(§7.13), got {training.p_bar_eta!r}"
            )

    data = cfg.data
    if data.source is None:
        raise ConfigurationError(
            "data.source is null — a training dataset is required. The "
            "original database is not public (data/README.md); use "
            "'synthetic_bop' for pipeline validation or supply the "
            "alternative dataset decision (docs/reproduction_protocol.md)"
        )
    if data.source == "path":
        if data.path is None:
            raise ConfigurationError("data.path is required when source='path'")
        if not Path(data.path).exists():
            raise ConfigurationError(f"data.path does not exist: {data.path}")
    elif data.source == "synthetic_bop":
        if data.synthetic is None:
            raise ConfigurationError("data.synthetic is required when source='synthetic_bop'")
    else:
        raise ConfigurationError(
            f"unknown data.source {data.source!r}; supported: 'synthetic_bop', "
            "'path' (extxyz loader lands with the real dataset)"
        )

    return ResolvedTrainingConfig(
        experiment=cfg.experiment,
        model=model,
        descriptors=cfg.descriptors,
        physics=cfg.physics,
        training=training,
        data=data,
    )
