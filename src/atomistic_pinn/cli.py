"""Command-line interface for Atomistic PINN Lab.

Sprint 0 provides informational commands only. The ``train``, ``evaluate``
and ``benchmark`` commands described in the roadmap arrive with the learning
layer (Sprint 3) and the evaluation layer (Sprint 4); see
``docs/architecture.md``.
"""

from __future__ import annotations

import typer

from atomistic_pinn import __version__
from atomistic_pinn import constants as c

app = typer.Typer(
    name="atomistic-pinn",
    help=(
        "Reproduction of the PINN bond-order potential for Al "
        "(Purja Pun et al., Nat. Commun. 10, 2339 (2019))."
    ),
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"atomistic-pinn-lab {__version__}")


@app.command()
def paper() -> None:
    """Print the paper constants pinned in this project."""
    typer.echo("Source: Purja Pun et al., Nature Communications 10, 2339 (2019)")
    typer.echo(f"PINN architecture          : {c.PINN_ARCHITECTURE}")
    typer.echo(f"NN baseline architecture   : {c.NN_ARCHITECTURE}")
    typer.echo(f"Descriptor orders m        : {c.DESCRIPTOR_LEGENDRE_ORDERS}")
    typer.echo(f"Radial centers             : {c.N_RADIAL_CENTERS}")
    typer.echo(f"Descriptors per atom       : {c.N_DESCRIPTORS_PER_ATOM}")
    typer.echo(f"BOP parameters             : {c.BOP_PARAMETER_NAMES}")
    typer.echo(f"Cutoff r_c / d             : {c.CUTOFF_RADIUS} Å / {c.CUTOFF_D} Å")
    typer.echo(f"Energy shift               : {c.ENERGY_SHIFT_EV_PER_ATOM} eV/atom")
    typer.echo(f"Training supercells        : {c.TRAINING_SUPERCELLS}")
    typer.echo(f"Cross-validation folds     : {c.N_CV_FOLDS}")
