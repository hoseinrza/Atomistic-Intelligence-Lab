"""Unit tests for cross-validation fold assignment (Sprint 4)."""

import pytest

from atomistic_pinn.evaluation.folds import assign_folds

pytestmark = pytest.mark.unit

IDS = [f"s{i}" for i in range(12)]


class TestAssignFolds:
    def test_partition_is_complete_disjoint_and_nonempty(self) -> None:
        assignment = assign_folds(IDS, 4, seed=42)
        assert set(assignment.fold_of) == set(IDS)
        counts = [0] * 4
        for fold in assignment.fold_of.values():
            counts[fold] += 1
        assert all(c > 0 for c in counts)
        assert sum(counts) == len(IDS)

    def test_reproducible_under_seed(self) -> None:
        a = assign_folds(IDS, 3, seed=7)
        b = assign_folds(IDS, 3, seed=7)
        assert a.fold_of == b.fold_of
        assert a.fold_seeds == b.fold_seeds

    def test_seed_changes_partition(self) -> None:
        a = assign_folds(IDS, 3, seed=1)
        b = assign_folds(IDS, 3, seed=2)
        assert a.fold_of != b.fold_of

    def test_fold_seeds_are_derived_and_distinct(self) -> None:
        assignment = assign_folds(IDS, 4, seed=42)
        assert len(set(assignment.fold_seeds)) == 4
        assert all(s != 42 for s in assignment.fold_seeds)

    def test_train_validation_split_covers_everything_once(self) -> None:
        assignment = assign_folds(IDS, 3, seed=5)
        seen: set[str] = set()
        for fold in range(3):
            training, validation = assignment.train_validation_split(IDS, fold)
            assert not set(training) & set(validation)
            assert set(validation) == {i for i in IDS if assignment.fold_of[i] == fold}
            seen |= set(validation)
        assert seen == set(IDS)

    def test_rejects_too_few_structures(self) -> None:
        with pytest.raises(ValueError, match="cannot populate"):
            assign_folds(["a", "b"], 3, seed=0)

    def test_rejects_single_fold(self) -> None:
        with pytest.raises(ValueError, match="at least 2 folds"):
            assign_folds(IDS, 1, seed=0)

    def test_rejects_duplicate_ids(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            assign_folds(["a", "a", "b"], 2, seed=0)
