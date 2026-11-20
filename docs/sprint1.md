# Sprint 1 — Geometry + descriptors (done)

**Status:** ✅ done (commit `7669e90`). Overview: [sprints.md](sprints.md).
**Next:** [sprint2.md](sprint2.md).

**Goal:** the fixed-length, invariant 60-dimensional fingerprint `G_i` of
every atom's local environment (Eqs. 3–5), plus the geometry primitives the
BOP will need.

## Delivered

* **`geometry/neighbor_list.py`**: all periodic-image neighbor pairs within
  a cutoff — deterministic, full PBC image enumeration (no minimum-image
  shortcuts), raises `DataContractError` on invalid input.
* **`geometry/measure.py`**: distances and angles `θ_ijk` on top of the
  neighbor list.
* **`descriptors/cutoff.py`**: `f_c` of Eq. (5), C³-smooth at `r_c`
  (function and first three derivatives vanish).
* **`descriptors/radial.py`**: truncated Gaussians of Eq. (4).
* **`descriptors/legendre.py`**: Legendre polynomials of the bond angle,
  orders `m ∈ {0, 1, 2, 4, 6}`.
* **`descriptors/builder.py`**: 60-dim builder; float64 accumulation;
  deterministic, sort-free per-atom accumulation order.

## Ledger decisions made (labeled)

| Item | Decision | Where |
| --- | --- | --- |
| §7.1 radial centers | 12 evenly spaced values, 0.5 Å grid, 0.0–5.5 Å | `constants.DEFAULT_RADIAL_CENTERS`, `DescriptorConfig` |
| §7.2 summation range | ordered double sum incl. `j = k` (literal reading; makes `g^(0) = (Σf)²` and `g^(1) = |Σf r̂|²` identities hold); excluded convention behind `include_same_pair` flag | `DescriptorConfig` |
| §7.3 descriptor cutoff | same as BOP: `r_c = 6 Å`, `d = 1.5 Å` | `DescriptorConfig` |

## Gates (all green)

* Rotation, translation, permutation invariance; dimension 60 for any
  periodic or cluster structure (`tests/scientific_validation/
  test_descriptors.py`).
* C³ cutoff smoothness in closed form (`tests/unit/test_cutoff.py`).
* Agreement with a slow, independent reference implementation on random
  configurations (`tests/scientific_validation/
  test_neighbor_list_reference.py`).
* `g^(0)` / `g^(1)` closed-form identities on random structures.
* ASE ↔ neighbor-list integration (`tests/integration/test_neighbor_list.py`).
