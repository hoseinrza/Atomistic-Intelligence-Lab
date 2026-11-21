"""Command-line interface for Atomistic PINN Lab.

Sprint 3 adds ``train``; ``evaluate`` and ``benchmark`` arrive with the
evaluation layer (Sprint 4). Every command fails with a
:class:`~atomistic_pinn.exceptions.AtomisticPINNError`-derived message that
states what is wrong and where the contract is documented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from atomistic_pinn import __version__
from atomistic_pinn import constants as c
from atomistic_pinn.exceptions import AtomisticPINNError

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


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            help="Path to the experiment YAML (e.g. configs/pinn_al.yaml).",
        ),
    ],
) -> None:
    """Train a model from a configuration file and write checkpoint + report."""
    from atomistic_pinn.reporting import write_training_report
    from atomistic_pinn.training.checkpoints import save_checkpoint
    from atomistic_pinn.training.config import load_training_config
    from atomistic_pinn.training.trainer import build_database, run_training

    try:
        resolved = load_training_config(config)
        db = build_database(resolved)
        if db.is_synthetic:
            typer.echo(
                f"[train] database: {db.n_structures} structures "
                f"(synthetic pipeline validation — not DFT data)"
            )
        else:
            typer.echo(
                f"[train] database: {db.n_structures} structures, "
                f"source={resolved.data.source!r} "
                f"(methodological reproduction, alternative data)"
            )
        outcome = run_training(resolved, db)
    except AtomisticPINNError as error:
        typer.secho(f"configuration error: {error}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from error

    checkpoint_path = save_checkpoint(
        Path(resolved.experiment.output_dir) / f"{resolved.experiment.name}.pt",
        outcome,
        resolved.model_dump(),
    )
    report_path = write_training_report(
        Path("results") / resolved.experiment.name,
        outcome,
        db,
        resolved.model_dump(),
        activation=resolved.model.activation or "tanh",
    )
    typer.echo(f"[train] best objective : {outcome.best_objective:.6e} eV²")
    typer.echo(f"[train] RMSE           : {outcome.rmse_mev_per_atom:.3f} meV/atom")
    typer.echo(f"[train] best restart   : {outcome.best_restart} ({outcome.stop_reason})")
    typer.echo(f"[train] checkpoint     : {checkpoint_path}")
    typer.echo(f"[train] report         : {report_path}")
