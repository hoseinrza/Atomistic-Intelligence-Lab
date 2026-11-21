"""Shared integration-test fixtures (Sprint 4)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

N_ATOMS = 4
N_CONF = 2


def _write_group(handle: h5py.File, key: str, rng: np.random.Generator, cell: np.ndarray) -> None:
    """One ANI batch group: N_CONF periodic N_ATOMS-atom Al configurations."""
    group = handle.create_group(key)
    group.create_dataset("species", data=np.full(N_ATOMS, b"Al", dtype="|S2"))
    group.create_dataset("coordinates", data=rng.uniform(0.0, 4.0, size=(N_CONF, N_ATOMS, 3)))
    group.create_dataset("cell", data=np.broadcast_to(cell, (N_CONF, 3, 3)).copy())
    group.create_dataset("energy", data=rng.uniform(-12.0, -10.0, size=N_CONF))
    group.create_dataset("force", data=rng.normal(0.0, 0.1, size=(N_CONF, N_ATOMS, 3)))
    group.create_dataset(
        "Jnames",
        data=np.array([f"/fake/qm/out-{key}-{c}.h5" for c in range(N_CONF)], dtype="|S64"),
    )


@pytest.fixture
def ani_dir(tmp_path: Path) -> Path:
    """Two ANI-style HDF5 files, two groups each (8 structures total)."""
    rng = np.random.default_rng(11)
    cell = 8.0 * np.eye(3)
    for name in ("data-1.h5", "data-2.h5"):
        with h5py.File(tmp_path / name, "w") as handle:
            _write_group(handle, "0", rng, cell)
            _write_group(handle, "1", rng, cell)
    return tmp_path
