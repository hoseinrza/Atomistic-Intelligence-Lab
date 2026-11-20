"""Unit tests for the CLI wiring (Sprint 0: informational commands)."""

from typer.testing import CliRunner

import atomistic_pinn
from atomistic_pinn import constants as c
from atomistic_pinn.cli import app

runner = CliRunner()


def test_version_command_reports_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert atomistic_pinn.__version__ in result.output


def test_paper_command_prints_pinned_constants() -> None:
    result = runner.invoke(app, ["paper"])
    assert result.exit_code == 0
    assert str(c.PINN_ARCHITECTURE) in result.output
    assert str(c.N_DESCRIPTORS_PER_ATOM) in result.output
    assert "0.38446" in result.output


def test_help_is_shown_without_arguments() -> None:
    # Click's no_args_is_help exits with code 2 (usage error convention)
    # while still printing the full help text.
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Usage" in result.output


def test_explicit_help_flag_exits_successfully() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output
