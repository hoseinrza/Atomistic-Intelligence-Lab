"""Unit tests for the documented exception hierarchy."""

from atomistic_pinn.exceptions import (
    AtomisticPINNError,
    ConfigurationError,
    DataContractError,
)


def test_all_package_errors_share_a_base_class() -> None:
    for error in (DataContractError, ConfigurationError):
        assert issubclass(error, AtomisticPINNError)


def test_base_class_is_an_exception() -> None:
    assert issubclass(AtomisticPINNError, Exception)


def test_errors_carry_messages() -> None:
    error = DataContractError("structure ids must be unique")
    assert "unique" in str(error)
