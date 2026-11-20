# Data format and contracts

This document specifies the data contracts implemented in
`src/atomistic_pinn/data/schema.py` and enforced at load time by Pydantic.

## Units

Å (distance), eV (energy), eV/Å (force) — everywhere, without exception.
Stored numbers never carry unit suffixes; units are declared once per dataset
in `DatasetMetadata.units`.

## Structure schema (`AtomicStructure`)

| Field | Type | Unit | Required | Notes |
| --- | --- | --- | --- | --- |
| `id` | `str` | — | yes | unique within the dataset; used for fold assignment |
| `numbers` | `list[int]` | — | yes | atomic numbers, length `n_atoms ≥ 1` |
| `positions` | `list[list[float]]` | Å | yes | shape `(n_atoms, 3)` |
| `cell` | `list[list[float]]` | Å | yes | shape `(3, 3)`, lattice vectors as rows; zero rows for clusters |
| `pbc` | `list[bool]` | — | yes | three flags; default all `False` |
| `energy` | `float \| None` | eV | no | total DFT energy |
| `forces` | `list[list[float]] \| None` | eV/Å | no | shape `(n_atoms, 3)` |
| `metadata` | `dict` | — | no | provenance; recommended keys below |

Validation enforced by the schema: shape consistency of `positions`,
`forces`, `cell`, `pbc`; all coordinates finite; models frozen
(immutable after creation).

## Dataset schema

`Dataset = metadata + list[structures]` with:

* unique `id` across all structures (contract-level check);
* at least one structure;
* `DatasetMetadata` recording `name`, `source` (citation/URL), `license`,
  `version`, and `energy_convention`.

## Energy conventions (critical)

The paper uniformly shifts all DFT energies by **+0.38446 eV per atom**
before training. The convention of this project:

* **Stored energies are raw DFT energies** (`energy_convention = "raw_dft"`,
  the default).
* The shift is applied by the *training pipeline*, once, and logged with the
  run.
* A dataset with pre-shifted energies must declare
  `energy_convention = "shifted"`; the pipeline then skips the shift.

This prevents double-shift bugs, which would silently corrupt training.

## Metadata conventions

`AtomicStructure.metadata` is free-form, but producers should use:

| Key | Meaning |
| --- | --- |
| `group` | structure family: `fcc_bulk`, `hcp_bulk`, `slab`, `grain_boundary`, `stacking_fault`, `vacancy`, `interstitial`, `cluster`, `liquid_md`, ... |
| `source_id` | identifier in the source database |
| `temperature_K` | MD snapshot temperature, if applicable |
| `volume_A3_per_atom` | per-atom volume |

Group labels drive the per-group error analysis and extrapolation tests in
Sprint 4.

## Leakage prevention

* Cross-validation folds are assigned **per structure** (supercell): all
  atoms of a supercell always share one fold.
* Structures generated from the same parent trajectory (e.g. consecutive MD
  frames) are correlated; fold assignment must keep such families together
  when known (recorded via `metadata.group` / `source_id`). This is stricter
  than the paper's random supercell split and will be documented as such if
  applied.

## Storage

* **Interchange:** ASE extxyz (human-inspectable, one file with many
  frames); labels carried through a single-point calculator.
* **Archive (Sprint 1):** HDF5 via `h5py` with datasets
  `positions`, `numbers`, `cell`, `pbc`, `energy`, `forces` and HDF5
  attributes for dataset metadata; deterministic ordering by structure `id`.
* Datasets live under `data/raw/` and `data/processed/`; both are
  git-ignored (`data/README.md`).
