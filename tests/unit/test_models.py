"""Unit tests for the Table 1 networks (Sprint 3, phase 3.1)."""

import pytest
import torch

from atomistic_pinn.exceptions import ConfigurationError
from atomistic_pinn.models import PINN, BaselineNN

pytestmark = pytest.mark.unit


def _fc_param_count(architecture: tuple[int, ...]) -> int:
    return sum(
        architecture[i] * architecture[i + 1] + architecture[i + 1]
        for i in range(len(architecture) - 1)
    )


def test_pinn_parameter_count_matches_table1() -> None:
    model = PINN(seed=0)
    count = sum(p.numel() for p in model.parameters())
    assert count == 1283 == _fc_param_count((60, 15, 15, 8))


def test_baseline_parameter_count_matches_table1() -> None:
    model = BaselineNN(seed=0)
    count = sum(p.numel() for p in model.parameters())
    assert count == 1265 == _fc_param_count((60, 16, 16, 1))


def test_output_shapes() -> None:
    descriptors = torch.zeros((7, 60), dtype=torch.float64)
    assert PINN(seed=1)(descriptors).shape == (7, 8)
    assert BaselineNN(seed=1)(descriptors).shape == (7, 1)


def test_models_run_in_float64() -> None:
    descriptors = torch.zeros((3, 60), dtype=torch.float64)
    assert PINN(seed=0)(descriptors).dtype == torch.float64
    assert BaselineNN(seed=0)(descriptors).dtype == torch.float64


def test_seeded_initialization_is_deterministic() -> None:
    a = PINN(seed=123)
    b = PINN(seed=123)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb)
    c = PINN(seed=124)
    assert any(
        not torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters(), strict=True)
    )


def test_sigmoid_activation_flag_accepted() -> None:
    model = PINN(seed=0, activation="sigmoid")
    out = model(torch.zeros((2, 60), dtype=torch.float64))
    assert out.shape == (2, 8)


def test_unknown_activation_rejected() -> None:
    with pytest.raises(ConfigurationError, match="activation"):
        PINN(seed=0, activation="relu")
