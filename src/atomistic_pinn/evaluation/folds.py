"""Cross-validation fold assignment (Sprint 4, phase 4.1).

Paper protocol (Methods): the database is *randomly partitioned* into 10
subsets; one is held out for validation, the rest are used for training;
the process repeats for every choice of the held-out subset.

Project leakage rules (`docs/data_format.md`, stricter than the paper and
documented as such):

* folds are assigned **per structure** — all atoms of a supercell share one
  fold, so a supercell never spans two folds;
* correlated structure families (e.g. consecutive MD frames, recorded via
  ``metadata.group``/``source_id``) would need to stay together when known —
  the ANI-Al dataset has no such family labels, so plain per-structure
  partitioning applies (documented deviation when applied).

Seeding (`docs/architecture.md`): one master seed per experiment, derived
seeds per fold; the assignment itself is reproducible under the master seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FoldAssignment:
    """Deterministic per-structure fold partition.

    Attributes:
        n_folds: Number of folds K.
        fold_of: Structure id → fold index (0-based, contiguous).
        fold_seeds: Derived seed per fold (for training restarts inside).
    """

    n_folds: int
    fold_of: dict[str, int]
    fold_seeds: tuple[int, ...]

    def train_validation_split(
        self, ids: list[str], fold_index: int
    ) -> tuple[list[str], list[str]]:
        """Ids of (training, validation) for one held-out fold."""
        validation = [i for i in ids if self.fold_of[i] == fold_index]
        training = [i for i in ids if self.fold_of[i] != fold_index]
        return training, validation


def assign_folds(
    ids: list[str], n_folds: int, seed: int, *, shuffle: bool = True
) -> FoldAssignment:
    """Randomly partition structure ids into ``n_folds`` subsets.

    Args:
        ids: Unique structure ids (order defines the id space; the
            partition depends only on (ids, n_folds, seed)).
        n_folds: Number of folds K ≥ 2.
        seed: Master seed of the partition.
        shuffle: Random partition (paper protocol). ``False`` assigns
            contiguously in the given order (deterministic smoke splits).

    Raises:
        ValueError: On fewer structures than folds, duplicate ids, or
            K < 2.
    """
    if n_folds < 2:
        raise ValueError(f"cross-validation needs at least 2 folds, got {n_folds}")
    if len(set(ids)) != len(ids):
        raise ValueError("structure ids must be unique for fold assignment")
    if len(ids) < n_folds:
        raise ValueError(f"{len(ids)} structures cannot populate {n_folds} non-empty folds")
    order = np.arange(len(ids))
    if shuffle:
        order = np.random.default_rng(seed).permutation(order)
    chunks = np.array_split(order, n_folds)
    fold_of: dict[str, int] = {}
    for fold_index, chunk in enumerate(chunks):
        for position in chunk:
            fold_of[ids[int(position)]] = fold_index
    fold_seeds = tuple(
        int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(n_folds)
    )
    return FoldAssignment(n_folds=n_folds, fold_of=fold_of, fold_seeds=fold_seeds)
