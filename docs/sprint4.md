# Sprint 4 — Evaluation + simulation (planned)

**Status:** planned — starts after [sprint3.md](sprint3.md).
Overview: [sprints.md](sprints.md). MD deep-dive: [md_plan.md](md_plan.md).

**Goal:** prove the reproduction against the paper's reference metrics and
make the potential usable in real atomistic workflows.

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

1. Cross-validation runs end-to-end from one config with logged fold
   assignments and seeds.
2. FD and autograd forces agree on static configurations.
3. Calculator contract tests; ASE round-trip.
4. NVE energy drift below documented tolerance (primary MD physics gate).
5. Metrics reported against reference bands with fidelity level stated.
