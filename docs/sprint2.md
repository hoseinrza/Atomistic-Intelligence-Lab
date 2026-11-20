# Sprint 2 — Bond-order potential (in progress)

**Status:** 🚧 in progress — implementation exists, 2 reference tests red
(see "Current status"). Overview: [sprints.md](sprints.md).
**Next:** [sprint3.md](sprint3.md).

**Goal:** compute `E_i` from neighbor positions and the eight per-atom
parameters with the full physics — repulsion, bond order, screening,
promotion/embedding (Eqs. 6–12). Equations and implementation notes:
`docs/equations.md` §4.

## Deliverables

* **`physics/bop.py`**: atomic energies `E_i` (Eq. 6) and total energy
  `E_s = Σ_i E_i` from `(positions, cell, pbc, p_i (N, 8))`.
* **`physics/screening.py`**: partial screening `S_ijk` (Eq. 10), product
  `S_ij` (Eq. 9); float64 accumulation.
* **`physics/neighbors.py`**: the BOP-side pair enumeration over the
  geometry layer's neighbor list.
* **`physics/forces.py`**: autograd forces (engineering cross-check; the
  paper-faithful force method for metrics stays finite differences).
* The physics layer must stay usable **without** the learning stack — no
  torch model imports (`docs/architecture.md` dependency rules); single
  torch implementation, CPU-compatible.

## Test gates (definition of done)

1. Closed-form single-pair energies.
2. Pairwise additivity of the repulsive term.
3. Screening factor = 1 on the cutoff spheroid; minimal on the bond axis
   (`tests/unit/test_screening.py` — ✅ green).
4. Perfect-crystal energy against a hand-computed value.
5. Total/atomic energy consistency.
6. FD vs. autograd force agreement (two independent methods).
7. Agreement with a slow, independent reference implementation on various
   structures (`tests/scientific_validation/test_bop.py`).

## Current status (updated 2026-11-21)

* Implementation present: `src/atomistic_pinn/physics/`
  (`bop.py`, `screening.py`, `neighbors.py`, `forces.py`).
* `pytest`: **106 passed, 2 failed** — both in
  `tests/scientific_validation/test_bop.py`:
  * `test_matches_independent_reference_on_various_structures`
  * `test_screening_changes_energy_and_reference_agrees`
* Observed discrepancy: main implementation **more negative** than the
  independent reference by ~0.23 eV (~7% relative) on the compared
  structures.
* Leading suspects (attractive-term side, since the pair/repulsive term
  checks pass): application of `S_ij` to the attractive term, the ½
  prefactor convention, the `k ≠ i, j` exclusion in the screening sums,
  and the screening-cutoff interplay at `r_c`.
* Rule: Sprint 2 is not committed until all gates are green — CI must stay
  red-free.
