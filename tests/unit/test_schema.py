"""Unit tests for the atomic-structure and dataset data contracts."""

import pytest
from pydantic import ValidationError

from atomistic_pinn import constants as c
from atomistic_pinn.data.schema import AtomicStructure, Dataset, DatasetMetadata

pytestmark = pytest.mark.unit


def make_structure(**overrides: object) -> AtomicStructure:
    data: dict[str, object] = {
        "id": "s1",
        "numbers": [13, 13],
        "positions": [[0.0, 0.0, 0.0], [1.43, 1.43, 1.43]],
        "energy": -6.72,
    }
    data.update(overrides)
    return AtomicStructure.model_validate(data)


def test_valid_structure_is_accepted() -> None:
    s = make_structure()
    assert s.n_atoms == 2
    assert s.is_periodic is False
    assert s.energy == pytest.approx(-6.72)
    assert s.forces is None


def test_periodic_structure_is_recognized() -> None:
    s = make_structure(pbc=[True, True, True])
    assert s.is_periodic is True


def test_empty_structure_rejected() -> None:
    with pytest.raises(ValidationError):
        make_structure(numbers=[], positions=[])


def test_position_row_length_enforced() -> None:
    with pytest.raises(ValidationError, match="length 3"):
        make_structure(positions=[[0.0, 0.0], [1.0, 1.0]])


def test_positions_count_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="positions"):
        make_structure(numbers=[13], positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def test_forces_shape_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="forces has"):
        make_structure(forces=[[0.0, 0.0, 0.0]])


def test_force_row_length_enforced() -> None:
    with pytest.raises(ValidationError, match="force row"):
        make_structure(forces=[[0.0, 0.0], [0.0, 0.0, 0.0]])


def test_non_finite_positions_rejected() -> None:
    with pytest.raises(ValidationError, match="not finite"):
        make_structure(positions=[[0.0, 0.0, float("nan")], [1.0, 1.0, 1.0]])


def test_non_finite_cell_rejected() -> None:
    with pytest.raises(ValidationError, match="not finite"):
        make_structure(cell=[[float("inf"), 0, 0], [0, 4.0, 0], [0, 0, 4.0]])


def test_bad_cell_shape_rejected() -> None:
    with pytest.raises(ValidationError, match=r"cell"):
        make_structure(cell=[[4.0, 0.0, 0.0]])


def test_bad_pbc_length_rejected() -> None:
    with pytest.raises(ValidationError, match="pbc"):
        make_structure(pbc=[True, True])


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        make_structure(energy_units="kcal/mol")


def test_frozen_model_rejects_mutation() -> None:
    s = make_structure()
    with pytest.raises(ValidationError):
        s.energy = -1.0  # type: ignore[misc]


def test_json_round_trip() -> None:
    s = make_structure(
        forces=[[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]],
        cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        pbc=[True, True, True],
        metadata={"group": "fcc_bulk", "temperature_K": 300},
    )
    restored = AtomicStructure.model_validate_json(s.model_dump_json())
    assert restored == s


def test_dataset_metadata_defaults_use_package_units() -> None:
    meta = DatasetMetadata(name="test")
    assert meta.units == {
        "length": c.UNIT_LENGTH,
        "energy": c.UNIT_ENERGY,
        "force": c.UNIT_FORCE,
    }
    assert meta.energy_convention == "raw_dft"


def test_invalid_energy_convention_rejected() -> None:
    with pytest.raises(ValidationError):
        DatasetMetadata(name="test", energy_convention="shifted_by_hand")


def test_dataset_counts() -> None:
    s1 = make_structure(id="a")
    s2 = make_structure(
        id="b",
        numbers=[13],
        positions=[[0.0, 0.0, 0.0]],
        energy=None,
    )
    ds = Dataset(metadata=DatasetMetadata(name="test"), structures=[s1, s2])
    assert ds.n_structures == 2
    assert ds.total_atoms == 3
    assert ds.n_labeled_energies == 1


def test_duplicate_structure_ids_rejected() -> None:
    s1 = make_structure(id="a")
    s2 = make_structure(id="a")
    with pytest.raises(ValidationError, match="unique"):
        Dataset(metadata=DatasetMetadata(name="test"), structures=[s1, s2])


def test_empty_dataset_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one structure"):
        Dataset(metadata=DatasetMetadata(name="test"), structures=[])
