"""ANI-Al dataset loader (alternative data for the methodological reproduction).

Converts the public ANI-Al DFT database (J. S. Smith et al., "Automated
discovery of a robust interatomic potential for aluminum", *Nat. Commun.*
**12**, 92 (2021); dataset released at ``github.com/atomistic-ml/ani-al``,
``data/Al-data.tgz``, HDF5/ANI format) into the package ``Dataset`` schema
(``atomistic_pinn.data.schema``).

This is the **alternative dataset** selected in ``data/README.md`` — every
result trained on it is a *methodological reproduction (alternative data)*,
never an exact reproduction (``docs/reproduction_protocol.md``).

Dataset facts (verified 2026-11-21 by an independent counting pass over the
downloaded archive, recorded in ``data/README.md``):

* 6,352 configurations, single species {Al}, 55–249 atoms per structure;
* energies and forces per configuration plus one lattice matrix each;
* forces are on the eV/Å scale (mean |F| ≈ 0.10, p95 ≈ 0.42 eV/Å);
* energies are raw DFT total energies in eV whose *absolute reference* sits
  ≈ 1.0 eV/atom above the reference of the paper's VASP setup (paper raw
  value ≈ −3.745 eV/atom; ANI-Al median ≈ −2.72 eV/atom). Absolute offsets
  do not transfer between DFT setups; the paper's uniform +0.38446 eV/atom
  shift therefore remains a *pipeline-side methodological step* of Eq. (13),
  applied exactly as for any ``raw_dft`` dataset by
  ``TrainingDatabase.shifted_targets`` — recorded here as a labeled
  decision, not as a cohesive-energy calibration for this dataset.

The loader is deterministic: files are visited in sorted order, groups in
integer order, configurations in stored order; structure ids are derived
from ``(file, group, configuration)`` only. ``max_structures`` selects a
seeded, reproducible subset for development runs (full-database DFP with
numerical gradients on CPU is documented as intractable — ``docs/sprint3.md``
§4); the subset is a labeled development-split decision, revisited with the
Sprint 4 cross-validation design.
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np

from atomistic_pinn.data.schema import AtomicStructure, Dataset, DatasetMetadata
from atomistic_pinn.exceptions import DataContractError

_AL_Z = 13

_SOURCE_CITATION = (
    "J. S. Smith et al., 'Automated discovery of a robust interatomic "
    "potential for aluminum', Nature Communications 12, 92 (2021); dataset "
    "github.com/atomistic-ml/ani-al (data/Al-data.tgz)"
)

_LICENSE_NOTE = (
    "no explicit license in the source repository (open item, data/README.md); "
    "ColabFit mirror subsets carry CC-BY-4.0 (DOI 10.60732/af254882)"
)

_REFERENCE_NOTE = (
    "raw DFT total energies in eV; absolute reference ~1.0 eV/atom above the "
    "paper's VASP setup (median -2.72 vs paper raw ~-3.745 eV/atom) — offsets "
    "do not transfer between DFT setups; the paper's +0.38446 eV/atom shift "
    "stays a pipeline-side methodological step of Eq. (13)"
)


def dataset_from_ani_al(
    directory: str | Path,
    *,
    max_structures: int | None = None,
    subset_seed: int = 0,
) -> Dataset:
    """Load the ANI-Al HDF5 directory into a validated :class:`Dataset`.

    Args:
        directory: directory containing the extracted ``data-*.h5`` files.
        max_structures: optional seeded subset size (development runs). The
            selection is reproducible under ``subset_seed``; the returned
            dataset keeps the deterministic global id order.
        subset_seed: seed of the subset generator.

    Returns:
        The immutable package dataset (``energy_convention="raw_dft"``).

    Raises:
        DataContractError: on missing files, non-aluminum species,
            inconsistent batch shapes, or non-finite labels.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise DataContractError(
            f"ANI-Al data directory not found: {directory} (contract: "
            "data/README.md alternative-dataset workflow)"
        )
    files = sorted(directory.glob("data-*.h5"))
    if not files:
        raise DataContractError(
            f"no data-*.h5 files under {directory} (expected the extracted ANI-Al archive layout)"
        )

    structures: list[AtomicStructure] = []
    for path in files:
        structures.extend(_structures_from_file(path))

    if max_structures is not None:
        structures = _subset(structures, max_structures, subset_seed)

    metadata = DatasetMetadata(
        name="ANI-Al",
        description=(
            "Single-element aluminum PBE DFT database (crystal and melt "
            "phases, active-learning sampled MD snapshots) with total "
            "energies and forces; alternative dataset for the "
            "methodological reproduction of Purja Pun et al., Nat. Commun. "
            "10, 2339 (2019) — never presented as the paper's own database."
        ),
        source=_SOURCE_CITATION,
        license=_LICENSE_NOTE,
        energy_convention="raw_dft",
        extra={
            "alternative_data": True,
            "reference_note": _REFERENCE_NOTE,
            "n_structures_loaded": len(structures),
        },
    )
    return Dataset(metadata=metadata, structures=structures)


def _structures_from_file(path: Path) -> list[AtomicStructure]:
    """All configurations of one ``data-*.h5`` file, in deterministic order."""
    structures: list[AtomicStructure] = []
    with h5py.File(path, "r") as handle:
        group_keys = sorted(handle.keys(), key=int)
        for group_key in group_keys:
            group = handle[group_key]
            _check_group_shapes(path, group_key, group)
            species = np.asarray(group["species"]).ravel()
            numbers = [_species_to_number(s) for s in species]
            energies = np.asarray(group["energy"], dtype=np.float64)
            coordinates = np.asarray(group["coordinates"], dtype=np.float64)
            cells = np.asarray(group["cell"], dtype=np.float64)
            forces = np.asarray(group["force"], dtype=np.float64)
            jnames = np.asarray(group["Jnames"]).ravel()
            group_index = int(group_key)
            for c in range(energies.shape[0]):
                energy = float(energies[c])
                if not math.isfinite(energy):
                    raise DataContractError(
                        f"{path.name}[{group_key}] configuration {c} has a "
                        "non-finite energy (contract: docs/data_format.md)"
                    )
                positions = coordinates[c].tolist()
                cell = cells[c].tolist()
                pbc = [bool(np.linalg.norm(row) > 0.0) for row in cells[c]]
                structures.append(
                    AtomicStructure(
                        id=f"{path.stem}-g{group_index:03d}-c{c:03d}",
                        numbers=numbers,
                        positions=positions,
                        cell=cell,
                        pbc=pbc,
                        energy=energy,
                        forces=forces[c].tolist(),
                        metadata={
                            "group": "ani_al",
                            "source_id": jnames[c].decode() if c < len(jnames) else "",
                            "source_file": path.name,
                        },
                    )
                )
    return structures


def _check_group_shapes(path: Path, group_key: str, group: h5py.Group) -> None:
    """Verify the ANI batch layout: one (n_conf, n_atoms, ...) block per key."""
    required = ("species", "coordinates", "cell", "energy", "force")
    missing = [k for k in required if k not in group]
    if missing:
        raise DataContractError(
            f"{path.name}[{group_key}] is missing ANI fields {missing} "
            "(expected the ANI-Al HDF5 layout)"
        )
    n_conf = int(np.asarray(group["energy"]).shape[0])
    n_atoms = int(np.asarray(group["species"]).shape[0])
    for key, expected in (
        ("coordinates", (n_conf, n_atoms, 3)),
        ("force", (n_conf, n_atoms, 3)),
        ("cell", (n_conf, 3, 3)),
    ):
        actual = tuple(np.asarray(group[key]).shape)
        if actual != expected:
            raise DataContractError(
                f"{path.name}[{group_key}] field {key} has shape {actual}, "
                f"expected {expected} (contract: ANI-Al HDF5 layout)"
            )


def _species_to_number(symbol: str | bytes) -> int:
    """Map the ANI species symbol to an atomic number (Al only)."""
    if isinstance(symbol, bytes):
        symbol = symbol.decode()
    if symbol != "Al":
        raise DataContractError(
            f"unexpected species {symbol!r}: this loader reproduces the "
            "single-component Al database (contract: data/README.md "
            "criterion 1)"
        )
    return _AL_Z


def _subset(structures: list[AtomicStructure], size: int, seed: int) -> list[AtomicStructure]:
    """Deterministic seeded subset, returned in the global id order."""
    total = len(structures)
    if size <= 0:
        raise DataContractError(
            f"max_structures must be positive, got {size} (dev-run subset "
            "decision, docs/sprint3.md §6)"
        )
    if size >= total:
        return structures
    ids = [s.id for s in structures]
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(total, size=size, replace=False).tolist())
    lookup = dict(zip(ids, structures, strict=True))
    return [lookup[ids[i]] for i in chosen]
