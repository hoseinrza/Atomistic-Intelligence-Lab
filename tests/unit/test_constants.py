"""Unit tests for unit conventions and constants.

Physical constants are checked against ``scipy.constants`` (an independent
reference). Paper-derived constants are checked against the values quoted in
the paper text and tables.
"""

import pytest
import scipy.constants as sc

from atomistic_pinn import constants as c

pytestmark = pytest.mark.unit


def test_boltzmann_constant_matches_scipy() -> None:
    assert c.BOLTZMANN_EV_PER_K == pytest.approx(sc.k / sc.e, rel=1e-8)


def test_avogadro_and_elementary_charge_match_scipy() -> None:
    assert c.AVOGADRO == pytest.approx(sc.N_A, rel=1e-12)
    assert c.ELEMENTARY_CHARGE == pytest.approx(sc.e, rel=1e-12)


def test_aluminum_identifiers() -> None:
    assert c.ALUMINUM_SYMBOL == "Al"
    assert c.ALUMINUM_ATOMIC_NUMBER == 13


def test_unit_conventions_are_declared() -> None:
    assert (c.UNIT_LENGTH, c.UNIT_ENERGY, c.UNIT_FORCE) == ("angstrom", "eV", "eV/angstrom")


def test_descriptor_constants_match_paper() -> None:
    assert c.DESCRIPTOR_LEGENDRE_ORDERS == (0, 1, 2, 4, 6)
    assert c.DESCRIPTOR_SIGMA == pytest.approx(1.0)
    assert c.CUTOFF_RADIUS == pytest.approx(6.0)
    assert c.CUTOFF_D == pytest.approx(1.5)
    assert c.N_RADIAL_CENTERS == 12
    expected = len(c.DESCRIPTOR_LEGENDRE_ORDERS) * c.N_RADIAL_CENTERS
    assert c.N_DESCRIPTORS_PER_ATOM == expected == 60


def test_bop_parameter_contract() -> None:
    assert c.BOP_PARAMETER_NAMES == ("A", "B", "alpha", "beta", "a", "h", "sigma", "lambda")
    assert c.N_BOP_PARAMETERS == len(c.BOP_PARAMETER_NAMES) == 8


def test_architectures_match_paper_table1() -> None:
    assert c.PINN_ARCHITECTURE == (60, 15, 15, 8)
    assert c.NN_ARCHITECTURE == (60, 16, 16, 1)
    assert c.NN_PRIME_ARCHITECTURE == (47, 18, 18, 1)


def test_training_protocol_constants() -> None:
    assert c.N_CV_FOLDS == 10
    assert c.ENERGY_SHIFT_EV_PER_ATOM == pytest.approx(0.38446)


def test_reference_rmse_table_is_complete() -> None:
    assert set(c.TABLE1_REFERENCE_RMSE) == {"NN", "NN_prime", "PINN"}
    for row in c.TABLE1_REFERENCE_RMSE.values():
        assert 0.0 < row["train"] < 5.0
        assert 0.0 < row["val"] < 5.0


def test_force_rmse_references_are_positive() -> None:
    for value in c.FORCE_RMSE_REFERENCE_EV_PER_ANGSTROM.values():
        assert 0.0 < value < 1.0
