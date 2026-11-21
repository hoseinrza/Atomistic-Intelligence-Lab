# Sprint 2 — Bond-order potential (done)

**Status:** ✅ done (implementation commit `c56b9f2`; gate-alignment tests and
this record are the follow-up commit). Overview: [sprints.md](sprints.md).
**Next:** [sprint3.md](sprint3.md).

**Goal:** compute `E_i` from neighbor positions and the eight per-atom
parameters with the full physics — repulsion, bond order, screening,
promotion/embedding (Eqs. 6–12). Equations and implementation notes:
`docs/equations.md` §4.

## Delivered

* **`physics/neighbors.py`**: one extended neighbor list at `2·r_c` (the
  screening reach), bond tier (`r < r_c`) by masking, far-sentinel padding
  with a written proof that every padded screening contribution is exactly 1;
  differentiable vector/distance recomputation in torch.
* **`physics/screening.py`**: torch twin of the Eq. (5) cutoff; Eq. (10)
  partial-screening matrix `W[a,b] = 1 − f_c(r_ib + r_ab − r_ia)·exp(−λ²·arg)`
  (diagonal ≡ 1); Eq. (9) as row products. Argument ≥ 0 by the triangle
  inequality; constant on spheroids with poles at the bond atoms.
* **`physics/bop.py`**: Eq. (8) bond counts `z` (`a²·Σ` over bond-tier slots
  `k ≠ j`), Eq. (7) bond order, Eq. (6) pair term, Eq. (11)
  promotion/embedding; per-atom parameters `(N, 8)`; inputs upcast to
  float64; no clamps — `z ≥ 0`, `ρ ≥ 0` by construction.
* **`physics/forces.py`**: finite-difference forces (the paper's method,
  faithful default) + autograd forces (engineering cross-check; documented
  0-fill for empty bond environments where the √ρ gradient is singular).
* The layer imports no learning-stack code (`docs/architecture.md` rules);
  single torch implementation, CPU-compatible.

## Ledger items touched

* §7.15 added: the fitted BOP parameter values of the paper's fixed-BOP
  baseline (Fig. 8) are not published in the main text → re-fit or
  qualitative comparison, never presented as the paper's fit.
* Test parameter sets are synthetic physical fixtures (A·α < B·β for
  short-range repulsive dominance), not paper values.

## Gates (all green)

* Closed-form single-pair energies (`test_dimer_matches_closed_form`).
* Pairwise additivity of the repulsive term via a σ=0 / huge-Bβ fixture
  (`test_repulsive_term_is_pairwise_additive`).
* Screening factor = 1 on the cutoff spheroid, minimal (1 − f_c(0)) on the
  bond axis, 1 far outside (`test_partial_screening_closed_forms`,
  `tests/unit/test_screening.py`).
* Perfect crystal: all pristine-FCC atomic energies identical
  (`test_pristine_fcc_all_atomic_energies_equal`); crystal *values* covered
  by the independent-reference agreement on perturbed FCC — a literal
  hand-computed crystal value is not analytically tractable (hundreds of
  screened neighbors), and the reference transcription is the independent
  check in its place.
* Total/atomic energy consistency (`test_total_energy_equals_sum_of_atomic`).
* FD vs. autograd force agreement on perturbed FCC
  (`test_fd_forces_match_autograd_forces`), plus the closed-form dimer
  derivative (`test_autograd_forces_match_closed_form_dimer_derivative`).
* Agreement with the independent loop-based reference transcription on 5
  structure types × 2 parameter sets, rtol 1e-9
  (`test_matches_independent_reference_on_various_structures`).

## Bug caught by the gates

The reference test exposed `a` instead of `a²` in Eq. 8 during development
(0.14 eV discrepancy, ratio exactly a) — fixed before commit. This is the
independent-reference test doing precisely its job.
