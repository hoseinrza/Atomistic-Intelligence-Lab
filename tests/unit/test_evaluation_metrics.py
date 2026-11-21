"""Unit tests for the Sprint 4 evaluation metrics (hand-checkable values)."""

import numpy as np
import pytest

from atomistic_pinn.evaluation.metrics import (
    FIDELITY_ALTERNATIVE_DATA,
    energy_metrics,
    force_metrics,
    per_atom_energy_errors,
)

pytestmark = pytest.mark.unit


class TestPerAtomEnergyErrors:
    def test_hand_case(self) -> None:
        predicted = np.array([-5.0, -8.0])
        targets = np.array([-6.0, -8.02])
        n_atoms = np.array([2, 4])
        errors = per_atom_energy_errors(predicted, targets, n_atoms)
        np.testing.assert_allclose(errors, [0.5, 0.005])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            per_atom_energy_errors(np.zeros(2), np.zeros(3), np.ones(2))

    def test_non_positive_atom_count_raises(self) -> None:
        with pytest.raises(ValueError, match="n_atoms must be positive"):
            per_atom_energy_errors(np.zeros(1), np.zeros(1), np.array([0]))


class TestEnergyMetrics:
    def test_hand_case_rmse_mae(self) -> None:
        # Per-atom errors: -2 and +1 meV/atom -> RMSE = sqrt(5/2), MAE = 1.5.
        predicted = np.array([0.0, 0.0])
        targets = np.array([2.0, -1.0]) / 1000.0  # eV, 1 atom each
        metrics = energy_metrics(predicted, targets, np.array([1, 1]))
        assert metrics.rmse_mev_per_atom == pytest.approx(np.sqrt(2.5))
        assert metrics.mae_mev_per_atom == pytest.approx(1.5)
        assert metrics.max_abs_error_mev_per_atom == pytest.approx(2.0)
        assert metrics.n_structures == 2
        assert metrics.n_atoms == 2

    def test_fidelity_label_for_alternative_data(self) -> None:
        metrics = energy_metrics(np.zeros(1), np.zeros(1), np.array([1]))
        record = metrics.with_fidelity(alternative_data=True)
        assert record["fidelity"] == FIDELITY_ALTERNATIVE_DATA
        record_exact = metrics.with_fidelity(alternative_data=False)
        assert record_exact["fidelity"] == "exact reproduction"


class TestForceMetrics:
    def test_hand_case(self) -> None:
        predicted = np.array([[0.1, 0.0, -0.1], [0.2, 0.0, 0.0]])
        reference = np.array([[0.0, 0.0, 0.0], [-0.2, 0.0, 0.0]])
        metrics = force_metrics(predicted, reference, n_structures=1)
        # Component errors: 0.1, 0, -0.1, 0.4, 0, 0
        assert metrics.mae_ev_per_angstrom == pytest.approx(0.6 / 6)
        assert metrics.rmse_ev_per_angstrom == pytest.approx(np.sqrt((0.01 + 0.01 + 0.16) / 6))
        assert metrics.max_abs_error_ev_per_angstrom == pytest.approx(0.4)
        assert metrics.n_atoms == 2
        assert metrics.n_structures == 1

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="force shape mismatch"):
            force_metrics(np.zeros((2, 3)), np.zeros((3, 3)))

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one atom"):
            force_metrics(np.zeros((0, 3)), np.zeros((0, 3)))
