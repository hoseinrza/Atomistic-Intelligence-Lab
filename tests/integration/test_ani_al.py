"""Integration tests for the ANI-Al alternative-dataset loader.

The ``ani_dir`` fixture (shared via ``conftest.py``) reproduces the ANI-Al
HDF5 layout (``data-*.h5`` files with per-group ``species`` / ``coordinates``
/ ``cell`` / ``energy`` / ``force`` blocks) on tiny synthetic data — no
external dataset is downloaded by the test suite. Provenance and selection
rules: ``data/README.md``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from atomistic_pinn.data.ani_al import dataset_from_ani_al
from atomistic_pinn.exceptions import ConfigurationError, DataContractError
from atomistic_pinn.training.config import load_training_config
from atomistic_pinn.training.database import TrainingDatabase
from atomistic_pinn.training.trainer import build_database

pytestmark = pytest.mark.integration

N_ATOMS = 4
N_CONF = 2


class TestConverter:
    def test_loads_all_structures_in_deterministic_order(self, ani_dir: Path) -> None:
        dataset = dataset_from_ani_al(ani_dir)
        assert dataset.n_structures == 8
        ids = [s.id for s in dataset.structures]
        assert ids == sorted(ids)
        assert ids[0] == "data-1-g000-c000"
        assert len(set(ids)) == 8

    def test_labels_and_metadata_pass_through(self, ani_dir: Path) -> None:
        dataset = dataset_from_ani_al(ani_dir)
        structure = dataset.structures[0]
        assert structure.numbers == [13] * N_ATOMS
        assert structure.energy is not None
        assert structure.forces is not None
        assert structure.forces_array is not None
        assert structure.forces_array.shape == (N_ATOMS, 3)
        assert all(structure.pbc)
        assert structure.metadata["source_file"] == "data-1.h5"
        assert structure.metadata["group"] == "ani_al"
        assert "out-0-0.h5" in structure.metadata["source_id"]

    def test_energy_convention_is_raw_dft(self, ani_dir: Path) -> None:
        dataset = dataset_from_ani_al(ani_dir)
        assert dataset.metadata.energy_convention == "raw_dft"
        assert dataset.metadata.name == "ANI-Al"
        assert dataset.metadata.extra["alternative_data"] is True

    def test_values_round_trip_exactly(self, ani_dir: Path) -> None:
        with h5py.File(ani_dir / "data-1.h5", "r") as handle:
            energy = float(np.asarray(handle["0"]["energy"])[0])
            positions = np.asarray(handle["0"]["coordinates"])[0]
        structure = dataset_from_ani_al(ani_dir).structures[0]
        assert structure.energy == energy
        assert np.allclose(structure.positions_array, positions)

    def test_zero_cell_marks_structure_non_periodic(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(3)
        with h5py.File(tmp_path / "data-1.h5", "w") as handle:
            group = handle.create_group("0")
            group.create_dataset("species", data=np.full(N_ATOMS, b"Al", dtype="|S2"))
            group.create_dataset("coordinates", data=rng.uniform(0, 4, (N_CONF, N_ATOMS, 3)))
            group.create_dataset("cell", data=np.zeros((N_CONF, 3, 3)))
            group.create_dataset("energy", data=rng.uniform(-12, -10, N_CONF))
            group.create_dataset("force", data=rng.normal(0, 0.1, (N_CONF, N_ATOMS, 3)))
            group.create_dataset("Jnames", data=np.array(["a", "b"], dtype="|S8"))
        dataset = dataset_from_ani_al(tmp_path)
        assert not any(dataset.structures[0].pbc)

    def test_non_aluminum_species_is_rejected(self, ani_dir: Path) -> None:
        with h5py.File(ani_dir / "data-1.h5", "r+") as handle:
            species = np.asarray(handle["0"]["species"])
            species[0] = b"Cu"
            handle["0"]["species"][...] = species
        with pytest.raises(DataContractError, match="single-component Al"):
            dataset_from_ani_al(ani_dir)

    def test_missing_field_is_rejected(self, ani_dir: Path) -> None:
        with h5py.File(ani_dir / "data-2.h5", "r+") as handle:
            del handle["0"]["force"]
        with pytest.raises(DataContractError, match="missing ANI fields"):
            dataset_from_ani_al(ani_dir)

    def test_non_finite_energy_is_rejected(self, ani_dir: Path) -> None:
        with h5py.File(ani_dir / "data-1.h5", "r+") as handle:
            energies = np.asarray(handle["0"]["energy"])
            energies[1] = np.nan
            handle["0"]["energy"][...] = energies
        with pytest.raises(DataContractError, match="non-finite energy"):
            dataset_from_ani_al(ani_dir)

    def test_missing_directory_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(DataContractError, match="not found"):
            dataset_from_ani_al(tmp_path / "nowhere")


class TestSubset:
    def test_subset_is_reproducible_and_sorted(self, ani_dir: Path) -> None:
        first = dataset_from_ani_al(ani_dir, max_structures=5, subset_seed=42)
        second = dataset_from_ani_al(ani_dir, max_structures=5, subset_seed=42)
        assert first.n_structures == 5
        ids = [s.id for s in first.structures]
        assert ids == sorted(ids)
        assert [s.id for s in second.structures] == ids

    def test_subset_seed_changes_selection(self, ani_dir: Path) -> None:
        a = dataset_from_ani_al(ani_dir, max_structures=4, subset_seed=1)
        b = dataset_from_ani_al(ani_dir, max_structures=4, subset_seed=2)
        assert [s.id for s in a.structures] != [s.id for s in b.structures]

    def test_subset_size_at_least_total_returns_everything(self, ani_dir: Path) -> None:
        dataset = dataset_from_ani_al(ani_dir, max_structures=99)
        assert dataset.n_structures == 8

    def test_subset_size_must_be_positive(self, ani_dir: Path) -> None:
        with pytest.raises(DataContractError, match="max_structures must be positive"):
            dataset_from_ani_al(ani_dir, max_structures=0)


def _ani_config(ani_dir: Path, **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "dataset": None,
        "source": "ani_al",
        "path": str(ani_dir),
        "subset_size": 8,
        "subset_seed": 3,
        "n_folds": 10,
    }
    data.update(overrides)
    return {
        "experiment": {"name": "ani_dev", "seed": 42},
        "model": {
            "type": "pinn",
            "architecture": [60, 15, 15, 8],
            "n_parameters": 1283,
            "activation": "tanh",
            "weight_init": "xavier_uniform",
        },
        "descriptors": {},
        "physics": {"bop_parameters": ["A", "B", "alpha", "beta", "a", "h", "sigma", "lambda"]},
        "training": {
            "objective": "sum_squared_total_energy_error",
            "energy_shift_ev_per_atom": 0.38446,
            "optimizer": "dfp",
            "n_random_restarts": 2,
            "regularization": {"tau": 1e-6, "gamma": 1e-4},
            "forces_in_training": False,
            "gradients": "numerical",
            "line_search": "backtracking_armijo",
            "p_bar_eta": "fixed_bop_prefit",
            "max_iterations": 3,
            "gradient_tolerance": 1e-8,
        },
        "data": data,
    }


class TestConfigAndDatabase:
    def test_ani_al_source_resolves(self, ani_dir: Path, tmp_path: Path) -> None:
        import yaml

        config_path = tmp_path / "ani.yaml"
        config_path.write_text(yaml.safe_dump(_ani_config(ani_dir)))
        resolved = load_training_config(config_path)
        assert resolved.data.source == "ani_al"
        assert resolved.data.subset_size == 8

    def test_ani_al_requires_subset_size(self, ani_dir: Path, tmp_path: Path) -> None:
        import yaml

        config_path = tmp_path / "ani.yaml"
        config_path.write_text(yaml.safe_dump(_ani_config(ani_dir, subset_size=None)))
        with pytest.raises(ConfigurationError, match="subset_size is required"):
            load_training_config(config_path)

    def test_ani_al_requires_existing_path(self, tmp_path: Path) -> None:
        import yaml

        config_path = tmp_path / "ani.yaml"
        config_path.write_text(yaml.safe_dump(_ani_config(tmp_path / "missing", subset_size=8)))
        with pytest.raises(ConfigurationError, match="does not exist"):
            load_training_config(config_path)

    def test_build_database_from_ani_al(self, ani_dir: Path, tmp_path: Path) -> None:
        import yaml

        config_path = tmp_path / "ani.yaml"
        config_path.write_text(yaml.safe_dump(_ani_config(ani_dir, subset_size=5)))
        resolved = load_training_config(config_path)
        db = build_database(resolved)
        assert isinstance(db, TrainingDatabase)
        assert db.n_structures == 5
        assert db.is_synthetic is False
        # Shift discipline: raw +0.38446 eV/atom applied exactly once.
        raw = db.raw_energies
        n_atoms = db.atoms_per_structure()
        targets = db.shifted_targets()
        assert np.allclose(targets, raw + 0.38446 * n_atoms)
        # Descriptor contract: every structure yields the 60-dim fingerprint.
        assert all(
            d.shape == (s.n_atoms, 60) for d, s in zip(db.descriptors, db.geometries, strict=True)
        )
