"""Scientific validation of the Eq. (13) objective (Sprint 3, phase 3.2).

The independent arithmetic here uses the physics layer with *hand-set*
network parameters (final-layer weights zeroed, final bias = chosen
parameters), so the network provably outputs constant known parameters —
making every term hand-checkable. The learning↔physics contract
(model energies == BOP-engine energies for the same parameters) is pinned
bit-level here too.
"""

import numpy as np
import pytest
import torch

from atomistic_pinn.constants import BOP_PARAMETER_NAMES, ENERGY_SHIFT_EV_PER_ATOM
from atomistic_pinn.models import PINN, BaselineNN
from atomistic_pinn.physics import atomic_energies
from atomistic_pinn.training.database import TrainingDatabase
from atomistic_pinn.training.objective import (
    BaselineEnergyModel,
    PINNEnergyModel,
    TetherConfig,
    _flat_to_named,
    get_flat_params,
    objective_value,
    set_flat_params,
)
from atomistic_pinn.training.synthetic import (
    SYNTHETIC_TRUTH_PARAMS,
    generate_synthetic_database,
)

pytestmark = pytest.mark.scientific_validation

N_STRUCTURES = 6


@pytest.fixture(scope="module")
def db() -> TrainingDatabase:
    dataset = generate_synthetic_database(n_structures=N_STRUCTURES, seed=11)
    return TrainingDatabase(dataset)


def _pinn_outputting(constant_params: np.ndarray) -> PINNEnergyModel:
    """PINN forced to output ``constant_params`` for every atom."""
    network = PINN(seed=0)
    last = network.network[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(torch.tensor(constant_params, dtype=torch.float64))
    return PINNEnergyModel(network)


def _truth_array() -> np.ndarray:
    return np.array([[SYNTHETIC_TRUTH_PARAMS[name] for name in BOP_PARAMETER_NAMES]])


def test_energy_term_matches_independent_physics_computation(db) -> None:
    truth = np.broadcast_to(_truth_array(), (1, 8)).copy()
    model = _pinn_outputting(truth[0])
    x = get_flat_params(model)
    tau = 0.0
    value = float(objective_value(model, db, x, tau=tau, tether=None))
    # Independent computation: physics layer directly, raw target shift.
    expected = 0.0
    for geometry in db.geometries:
        params = torch.tensor(np.broadcast_to(truth, (geometry.n_atoms, 8)).copy())
        energy = float(
            atomic_energies(geometry.positions, geometry.cell, geometry.pbc, params).sum()
        )
        n_atoms = geometry.n_atoms
        raw = db.raw_energies[db.geometries.index(geometry)]
        target = raw + ENERGY_SHIFT_EV_PER_ATOM * n_atoms
        expected += (energy - target) ** 2
    assert value == pytest.approx(expected, rel=1e-12)


def test_tau_term_is_sum_of_squared_weights(db) -> None:
    model = _pinn_outputting(_truth_array()[0])
    x = get_flat_params(model)
    tau = 0.05
    full = float(objective_value(model, db, x, tau=tau, tether=None))
    energy_only = float(objective_value(model, db, x, tau=0.0, tether=None))
    reg = sum(float((p**2).sum()) for p in model.network.parameters())
    assert full == pytest.approx(energy_only + tau * reg, rel=1e-12)


def test_gamma_fixed_target_term_is_hand_computable(db) -> None:
    # Bias set to truth + δ: every atom predicts exactly (truth + δ), so
    # γ Σ (p − p̄)² = γ · n_total_atoms · 8 · δ² — a closed form.
    delta = 0.1
    params = _truth_array()[0] + delta
    model = _pinn_outputting(params)
    x = get_flat_params(model)
    gamma = 0.01
    p_bar = torch.tensor(_truth_array()[0], dtype=torch.float64)
    value = float(
        objective_value(
            model,
            db,
            x,
            tau=0.0,
            tether=TetherConfig(gamma=gamma, mode="fixed_bop_prefit", target=p_bar),
        )
    )
    energy_only = float(objective_value(model, db, x, tau=0.0, tether=None))
    n_total = int(db.atoms_per_structure().sum())
    expected_tether = gamma * n_total * 8 * delta**2
    assert value == pytest.approx(energy_only + expected_tether, rel=1e-12)


def test_energy_shift_applied_exactly_once(db) -> None:
    targets = db.shifted_targets()
    raw = db.raw_energies
    n_atoms = db.atoms_per_structure()
    np.testing.assert_allclose(targets, raw + ENERGY_SHIFT_EV_PER_ATOM * n_atoms, rtol=1e-14)
    # Idempotent and non-mutating.
    again = db.shifted_targets()
    np.testing.assert_array_equal(again, targets)
    np.testing.assert_array_equal(db.raw_energies, raw)


def test_contract_model_energies_equal_bop_engine(db) -> None:
    # Learning↔physics contract: the full PINN pipeline with fixed predicted
    # parameters reproduces the BOP engine exactly (Sprint 3 gate).
    truth = np.broadcast_to(_truth_array(), (1, 8)).copy()
    model = _pinn_outputting(truth[0])
    model_energies = model.structure_energies(db).detach().numpy()
    for k, geometry in enumerate(db.geometries):
        params = torch.tensor(np.broadcast_to(truth, (geometry.n_atoms, 8)).copy())
        reference = float(
            atomic_energies(geometry.positions, geometry.cell, geometry.pbc, params).sum()
        )
        assert model_energies[k] == pytest.approx(reference, rel=1e-14, abs=1e-14)


def test_functional_and_module_paths_agree(db) -> None:
    # Numerical route (module state) and autograd route (functional call)
    # must produce identical energies for identical parameters.
    model = _pinn_outputting(_truth_array()[0] + 0.05)
    x = get_flat_params(model)
    module_path = model.structure_energies(db).detach().numpy()
    from atomistic_pinn.training.objective import _flat_to_named

    named = _flat_to_named(model, x)
    functional_path = model.structure_energies_functional(named, db).detach().numpy()
    np.testing.assert_allclose(functional_path, module_path, rtol=1e-14)


def test_baseline_model_energies_are_constant_sums(db) -> None:
    # With zeroed hidden weights and bias b, the baseline outputs b for
    # every atom → E_s = n_atoms · b — hand-checkable.
    b = -0.7
    network = BaselineNN(seed=0)
    last = network.network[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(b)
    model = BaselineEnergyModel(network)
    energies = model.structure_energies(db).detach().numpy()
    np.testing.assert_allclose(energies, b * db.atoms_per_structure(), rtol=1e-14)
    assert model.predicted_parameters(db) is None


def test_self_mean_tether_matches_fixed_target_at_the_mean(db) -> None:
    # mode="self_mean" must equal mode="fixed_bop_prefit" evaluated at the
    # mean of the current predictions (same formula, same constant).
    # Additionally, by the projection theorem the tether against the mean is
    # the smallest possible fixed-target tether — strictly smaller than the
    # tether against the truth whenever predictions are not all identical.
    delta = 0.1
    params = _truth_array()[0] + delta
    model = _pinn_outputting(params)
    x = get_flat_params(model)
    gamma = 0.02

    value_self = float(
        objective_value(model, db, x, tau=0.0, tether=TetherConfig(gamma=gamma, mode="self_mean"))
    )
    predictions = model.predicted_parameters(db)
    mean = predictions.mean(dim=0).detach()
    value_fixed_at_mean = float(
        objective_value(
            model,
            db,
            x,
            tau=0.0,
            tether=TetherConfig(gamma=gamma, mode="fixed_bop_prefit", target=mean),
        )
    )
    assert value_self == pytest.approx(value_fixed_at_mean, rel=1e-14)

    # All predictions are identical here (bias trick): the variance is
    # exactly zero, so the self-mean tether must vanish.
    assert value_self == pytest.approx(
        float(objective_value(model, db, x, tau=0.0, tether=None)), rel=1e-14
    )

    # Projection inequality against a different fixed target.
    value_fixed_at_truth = float(
        objective_value(
            model,
            db,
            x,
            tau=0.0,
            tether=TetherConfig(
                gamma=gamma,
                mode="fixed_bop_prefit",
                target=torch.tensor(_truth_array()[0], dtype=torch.float64),
            ),
        )
    )
    assert value_fixed_at_truth > value_self + gamma * 1e-3


def test_set_flat_params_round_trip(db) -> None:
    model = _pinn_outputting(_truth_array()[0])
    x = get_flat_params(model)
    perturbed = x + 0.01
    set_flat_params(model, perturbed)
    np.testing.assert_array_equal(get_flat_params(model).numpy(), perturbed.numpy())


class TestGradientRoutesAgree:
    """The streaming autograd gradient equals the all-at-once reference.

    The autograd route (labeled §7.8 fallback, used by the real-data runs)
    accumulates per-structure gradients with immediate backward passes to
    cap peak memory; this pins it against the all-at-once graph on the small
    synthetic database — including the τ term, which must flow through the
    leaf-connected parameters (the module-based τ gradient is zero and was
    the pre-fix behaviour).
    """

    @staticmethod
    def _reference_gradient(
        model: PINNEnergyModel | BaselineEnergyModel,
        db: TrainingDatabase,
        x: torch.Tensor,
        tau: float,
        tether: TetherConfig | None,
    ) -> torch.Tensor:
        leaf = x.clone().requires_grad_(True)
        named = _flat_to_named(model, leaf)
        targets = torch.as_tensor(db.shifted_targets(), dtype=torch.float64)
        energies = model.structure_energies_functional(named, db)
        loss = ((energies - targets) ** 2).sum()
        if tau > 0.0:
            reg = sum((p**2).sum() for p in named.values())
            loss = loss + tau * reg
        if tether is not None and tether.gamma > 0.0:
            predictions = model.predicted_parameters(db, named_params=named)
            assert predictions is not None
            if tether.mode == "fixed_bop_prefit":
                assert tether.target is not None
                loss = loss + tether.gamma * ((predictions - tether.target) ** 2).sum()
            else:  # self_mean
                mean = predictions.mean(dim=0).detach()
                loss = loss + tether.gamma * ((predictions - mean) ** 2).sum()
        (g,) = torch.autograd.grad(loss, leaf)
        return g

    def test_streaming_matches_all_at_once_with_tau_and_tether(self, db) -> None:
        from atomistic_pinn.training.objective import objective_function

        model = PINNEnergyModel(PINN(seed=2))
        tau, gamma = 0.05, 1e-3
        prefit = np.array([1.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 1.0])
        tether = TetherConfig(
            gamma=gamma,
            mode="fixed_bop_prefit",
            target=torch.tensor(prefit, dtype=torch.float64),
        )
        f, grad_autograd = objective_function(
            model, db, tau=tau, tether=tether, gradient="autograd"
        )
        x_value = (get_flat_params(model) + 0.01).numpy()
        assert f(x_value) >= 0.0
        reference = self._reference_gradient(
            model, db, torch.tensor(x_value), tau=tau, tether=tether
        )
        np.testing.assert_allclose(grad_autograd(x_value), reference.numpy(), rtol=1e-12)

    def test_tau_gradient_flows_through_leaf(self, db) -> None:
        # Regression: the τ gradient must be non-zero (the module-based
        # formulation contributed exactly zero through the leaf).
        from atomistic_pinn.training.objective import (
            _flat_to_named,
            _weight_regularization,
            objective_function,
        )

        model = PINNEnergyModel(PINN(seed=2))
        tau = 0.05
        _, grad_autograd = objective_function(model, db, tau=tau, tether=None, gradient="autograd")
        x = (get_flat_params(model) + 0.01).numpy()
        assert np.abs(grad_autograd(x)).max() > 0.0
        # And explicitly: the module-based τ term contributes zero to dℰ/dx.
        leaf = torch.tensor(x, requires_grad=True)
        named = _flat_to_named(model, leaf)
        targets = torch.as_tensor(db.shifted_targets(), dtype=torch.float64)
        energies = model.structure_energies_functional(named, db)
        energy_loss = ((energies - targets) ** 2).sum() + tau * _weight_regularization(model)
        (g_module_tau,) = torch.autograd.grad(energy_loss, leaf)
        # Streaming gradient minus its own τ contribution == module-tau one
        # iff the module-based τ gradient is zero: compare against the
        # energy-only streaming gradient + 0.
        _, grad_energy_only = objective_function(
            model, db, tau=0.0, tether=None, gradient="autograd"
        )
        np.testing.assert_allclose(g_module_tau.detach().numpy(), grad_energy_only(x), rtol=1e-10)

    def test_streaming_matches_all_at_once_baseline(self, db) -> None:
        from atomistic_pinn.training.objective import objective_function

        model = BaselineEnergyModel(BaselineNN(seed=2))
        f, grad_autograd = objective_function(model, db, tau=0.02, tether=None, gradient="autograd")
        x_value = (get_flat_params(model) + 0.01).numpy()
        assert f(x_value) >= 0.0
        reference = self._reference_gradient(
            model, db, torch.tensor(x_value), tau=0.02, tether=None
        )
        np.testing.assert_allclose(grad_autograd(x_value), reference.numpy(), rtol=1e-12)
