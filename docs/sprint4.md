# Sprint 4 — Evaluation + simulation (in progress — machinery complete)

**Status:** 🚧 in progress — phases 4.1 and 4.3 implemented and tested
(2026-11-21); the final artifacts (real-data dev-run metrics, τ/γ sweep,
CV gate run, 500-atom MD demo) are executing on the labeled dev budget.
Overview: [sprints.md](sprints.md). MD deep-dive: [md_plan.md](md_plan.md).

**Delivered so far (commits `2c508f0` → `5bf3d89`, `b007140`):**

* `evaluation/`: energy metrics (meV/atom, shifted targets, fidelity
  labels), full-pipeline FD forces + full-pipeline autograd forces (torch
  descriptor twin, routes agree rtol 1e-4), per-structure folds, the
  10-fold CV driver wrapping the Sprint 3 trainer, CLI `evaluate` and
  `benchmark` writing provenance-complete summaries to `results/`.
* `simulation/`: `PINNCalculator` (ASE contract, rebuild-on-demand lists,
  labeled 0.2 Å threshold) and the NVE/NVT MD demonstration runners with
  the drift gate and the extrapolation guard.
* `configs/cross_validation.yaml` replaced the Sprint 0 skeleton with the
  executable protocol; `configs/sweep_ani/` holds the phase-3.6 sweep.

## Phase 4.1 — Evaluation (`atomistic_pinn.evaluation`)

* Energy MAE/RMSE in meV/atom vs. the Table 1 reference bands
  (`docs/reproduction_protocol.md`: **band, not gate**, for alternative
  data; fidelity level recorded per number).
* Force MAE/RMSE by **finite differences** (the paper's method) with
  autograd as an independent cross-check — on static configurations.
* 10-fold cross-validation wrapping the Sprint 3 training driver:
  per-structure fold assignment, per-fold and averaged metrics, derived
  seeds per fold, full provenance logging.
* Extrapolation analysis on held-out structure groups (e.g. dislocation,
  MD-sampled configurations — the paper's strongest separation between
  PINN and baseline NN).

## Phase 4.2 — Baselines comparison

| Baseline | Source | Status |
| --- | --- | --- |
| fixed BOP | eight parameters fitted once (paper Fig. 8) — already produced by the Sprint 3 `p̄_η` pre-fit | compare |
| NN′ (PROPhet, Behler–Parrinello) | third-party package | not exactly reproduced (ledger §7.9); optional labeled extension |
| EAM | published Al EAM potential (paper ref. 54) | optional; published parameter file, no refitting |

## Phase 4.3 — Simulation (`atomistic_pinn.simulation`)

* ASE calculator wiring energies and forces into
  `atoms.get_potential_energy()` / `atoms.get_forces()`.
* MD demonstration: force route (autograd, FD-validated), NVE
  energy-conservation gate, seeded NVT run, extrapolation guard — full
  protocol in [md_plan.md](md_plan.md).

## Phase 4.4 — Reporting + reproduction report

* `atomistic_pinn.reporting`: CV summaries, parity plots, error
  histograms, benchmark tables → `results/` (generated, never committed).
* Final report follows `docs/reproduction_protocol.md` reporting
  requirements: resolved configs, git commit, dataset hash, seeds,
  per-fold metrics, checkpoint hashes; all §7 ledger items resolved or
  labeled verbatim.

## Test gates

1. ✅ Cross-validation runs end-to-end from one config with logged fold
   assignments and seeds (`tests/integration/test_cross_validation.py`).
2. ✅ FD and autograd forces agree on static configurations — at the
   pipeline level (`tests/scientific_validation/
   test_pipeline_force_routes.py`) and at the calculator level
   (`tests/integration/test_calculator.py`).
3. ✅ Calculator contract tests; ASE round-trip (`tests/integration/
   test_calculator.py`).
4. ✅ (machinery) NVE energy drift below documented tolerance on seeded
   tiny runs; the production gate runs on the trained checkpoint.
5. 🚧 Metrics reported against reference bands with fidelity level stated —
   lands with the dev-run/sweep/CV artifacts.
