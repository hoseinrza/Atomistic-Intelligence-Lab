"""Synthetic development database generator (Sprint 3 pipeline validation).

Builds small aluminum structures and labels them with energies computed by
the *physics layer itself* under a fixed "truth" parameter set. This gives
the learning pipeline a ground truth to converge to — the PINN must recover
energies of a BOP it can exactly represent — and is **pipeline validation
only**: it is not DFT data and must never be presented as a reproduction
result (fidelity rules, docs/reproduction_protocol.md).
"""

from __future__ import annotations

import numpy as np
import torch

from atomistic_pinn.data.schema import AtomicStructure, Dataset, DatasetMetadata
from atomistic_pinn.physics import atomic_energies

SYNTHETIC_TRUTH_PARAMS = {
    "A": 1.2,
    "B": 1.0,
    "alpha": 1.5,
    "beta": 4.0,
    "a": 1.5,
    "h": 0.0,
    "sigma": 2.0,
    "lambda": 1.5,
}
"""Physical truth parameter set for the synthetic generator (test fixture;
A·α < B·β so repulsion dominates at short range)."""


def generate_synthetic_database(
    n_structures: int = 24,
    seed: int = 7,
    truth_params: dict[str, float] | None = None,
    with_periodic: bool = True,
) -> Dataset:
    """Generate the labeled synthetic development database.

    Composition: ~half FCC conventional cells with random lattice constant
    (periodic, exercises images), ~quarter small non-periodic clusters,
    ~quarter dimers (exercises the short-range regime).

    Args:
        n_structures: Number of structures (≥ 1).
        seed: Structure-sampling seed (independent of training seeds).
        truth_params: The uniform per-atom BOP parameters used to label the
            structures (defaults to :data:`SYNTHETIC_TRUTH_PARAMS`).

    Returns:
        A :class:`~atomistic_pinn.data.schema.Dataset` with
        ``energy_convention = "raw_dft"`` and provenance metadata marking it
        synthetic.
    """
    if n_structures < 1:
        raise ValueError("n_structures must be ≥ 1")
    params = dict(truth_params or SYNTHETIC_TRUTH_PARAMS)
    rng = np.random.default_rng(seed)
    names = ["A", "B", "alpha", "beta", "a", "h", "sigma", "lambda"]
    truth = np.array([[float(params[name]) for name in names]], dtype=np.float64)

    structures: list[AtomicStructure] = []
    for k in range(n_structures):
        kind = k % 4 if with_periodic else 2 + k % 2
        if kind in (0, 1):  # half: periodic FCC cells
            a = float(rng.uniform(3.6, 4.6))
            positions = _fcc_cubic_positions(a)
            cell = np.eye(3) * a
            pbc = [True, True, True]
            group = "fcc_bulk"
        elif kind == 2:  # quarter: non-periodic clusters
            positions = rng.uniform(0.0, 8.0, size=(5, 3))
            cell = np.zeros((3, 3))
            pbc = [False, False, False]
            group = "cluster"
        else:  # quarter: dimers
            r = float(rng.uniform(2.2, 4.5))
            positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
            cell = np.zeros((3, 3))
            pbc = [False, False, False]
            group = "dimer"
        energy = _label_energy(positions, cell, pbc, truth)
        structures.append(
            AtomicStructure(
                id=f"synth_{k:04d}",
                numbers=[13] * len(positions),
                positions=positions.tolist(),
                cell=cell.tolist(),
                pbc=pbc,
                energy=energy,
                metadata={"group": group},
            )
        )

    metadata = DatasetMetadata(
        name="synthetic_bop_dev",
        description=(
            "Pipeline-validation database: structures labeled by the BOP "
            "physics layer under known parameters. NOT DFT data — never a "
            "reproduction result."
        ),
        source="atomistic_pinn.training.synthetic (generated)",
        license="n/a (generated)",
        energy_convention="raw_dft",
        extra={
            "synthetic": True,
            "truth_params": params,
            "seed": seed,
            "with_periodic": with_periodic,
        },
    )
    return Dataset(metadata=metadata, structures=structures)


def _fcc_cubic_positions(a: float) -> np.ndarray:
    """Conventional FCC cell positions."""
    return a * np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )


def _label_energy(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: list[bool],
    truth: np.ndarray,
) -> float:
    """Total energy under the truth parameter set (physics layer)."""
    n_atoms = len(positions)
    per_atom_params = torch.tensor(np.broadcast_to(truth, (n_atoms, truth.shape[1])).copy())
    with torch.no_grad():
        energies = atomic_energies(
            torch.tensor(positions, dtype=torch.float64),
            torch.tensor(cell, dtype=torch.float64),
            pbc,
            per_atom_params,
        )
    return float(energies.sum())
