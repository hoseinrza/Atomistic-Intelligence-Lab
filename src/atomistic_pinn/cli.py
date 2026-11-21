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


@app.command()
def evaluate(
    checkpoint: Annotated[
        Path,
        typer.Option("--checkpoint", exists=True, dir_okay=False, help="Trained .pt checkpoint."),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            help="Training YAML used to build the evaluation dataset.",
        ),
    ],
) -> None:
    """Evaluate a trained checkpoint on its config's dataset (Sprint 4).

    Energy metrics (meV/atom, shifted targets) and — when the dataset
    carries force labels — force metrics from the paper-faithful
    finite-difference route. Never a training quantity: forces are
    post-training only (paper methodology).
    """
    import json

    from atomistic_pinn.evaluation.forces import dataset_force_metrics
    from atomistic_pinn.evaluation.metrics import energy_metrics
    from atomistic_pinn.evaluation.predict import (
        predicted_structure_energies,
        restore_model,
    )
    from atomistic_pinn.training.checkpoints import load_checkpoint
    from atomistic_pinn.training.config import load_training_config
    from atomistic_pinn.training.trainer import build_database

    try:
        resolved = load_training_config(config)
        db = build_database(resolved)
        payload = load_checkpoint(checkpoint)
        model = restore_model(resolved, payload["flat_params"].numpy())
        metrics = energy_metrics(
            predicted_structure_energies(model, db),
            db.shifted_targets(),
            db.atoms_per_structure(),
        )
        record = metrics.with_fidelity(alternative_data=not db.is_synthetic)
        forces_record: dict[str, object] | None = None
        if any(s.forces is not None for s in db.dataset.structures):
            forces = dataset_force_metrics(
                model,
                db.dataset.structures,
                resolved.descriptors.to_descriptor_config(),
            )
            forces_record = {
                "mae_ev_per_angstrom": forces.mae_ev_per_angstrom,
                "rmse_ev_per_angstrom": forces.rmse_ev_per_angstrom,
                "max_abs_error_ev_per_angstrom": forces.max_abs_error_ev_per_angstrom,
                "n_structures": forces.n_structures,
                "method": "finite_difference (paper Methods, post-training only)",
            }
            record["forces"] = forces_record
    except AtomisticPINNError as error:
        typer.secho(f"evaluation error: {error}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from error

    record["checkpoint"] = str(checkpoint)
    record["config_hash"] = payload["config_hash"]
    record["dataset_hash"] = payload["dataset_hash"]
    out_dir = Path("results") / resolved.experiment.name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "evaluate_summary.json"
    summary_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    typer.echo(f"[evaluate] RMSE   : {metrics.rmse_mev_per_atom:.3f} meV/atom")
    typer.echo(f"[evaluate] MAE    : {metrics.mae_mev_per_atom:.3f} meV/atom")
    if forces_record is not None:
        typer.echo(f"[evaluate] |F| RMSE: {forces_record['rmse_ev_per_angstrom']:.4f} eV/Å (FD)")
    typer.echo(f"[evaluate] summary: {summary_path}")


@app.command()
def benchmark(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            help="Cross-validation YAML (training config + cv: section).",
        ),
    ],
) -> None:
    """Run the K-fold cross-validation protocol from one config (Sprint 4).

    Per-structure fold assignment, derived fold seeds, per-fold and averaged
    metrics against the paper reference bands (band, not gate), and the
    per-fold fixed-BOP baseline — all logged with full provenance.
    """
    import json

    from atomistic_pinn.evaluation.cross_validation import run_cross_validation
    from atomistic_pinn.training.config import load_cv_config
    from atomistic_pinn.training.trainer import build_dataset

    try:
        resolved, cv = load_cv_config(config)
        dataset = build_dataset(resolved)
        result = run_cross_validation(
            resolved,
            dataset,
            n_folds=cv.n_folds,
            fold_seed=cv.seed if cv.seed is not None else resolved.experiment.seed,
            shuffle=cv.shuffle,
            max_folds=cv.max_folds,
            alternative_data=not dataset.metadata.extra.get("synthetic", False),
        )
    except AtomisticPINNError as error:
        typer.secho(f"benchmark error: {error}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from error

    out_dir = Path("results") / resolved.experiment.name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")
    typer.echo(
        f"[benchmark] model={result.model_kind} folds={len(result.fold_results)}/{result.n_folds}"
    )
    typer.echo(
        f"[benchmark] validation RMSE: {result.mean_validation_rmse_mev_per_atom:.3f}"
        f" ± {result.std_validation_rmse_mev_per_atom:.3f} meV/atom (reference band 3-4)"
    )
    if result.fixed_bop_mean_validation_rmse_mev_per_atom is not None:
        typer.echo(
            f"[benchmark] fixed-BOP baseline: "
            f"{result.fixed_bop_mean_validation_rmse_mev_per_atom:.3f} meV/atom"
        )
    typer.echo(f"[benchmark] summary: {summary_path}")
