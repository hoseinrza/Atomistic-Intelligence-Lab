# Sprint 3 — Networks + training: execution plan

This document turns the Sprint 3 overview (`docs/sprints.md`) into an
executable plan: prerequisites, phases in build order, every ledger decision
that must be resolved before the first training run, tests, and the
definition of done. It assumes Sprint 2 is complete.

Equations: `docs/equations.md`. Uncertainty ledger: `docs/equations.md` §7.
Reproducibility mechanics: `docs/architecture.md`. Gate row:
`docs/reproduction_protocol.md` (sprint 3).

## 0. Prerequisites (Sprint 2 must have delivered)

* `physics/`: differentiable (torch, CPU) BOP computing atomic energies
  `E_i` and total energy `E_s` from `(positions, cell, pbc, per-atom
  parameters p_i (N, 8))` per Eqs. (6)–(12), usable **without** model code.
* `descriptors/`: 60-dim builder from `(positions, cell, pbc)` per Eqs.
  (3)–(5), all invariance and reference tests green.
* FD-vs-autograd force agreement test green (needed again in 3.4's
  contract tests).
* Data layer: `Dataset` / `AtomicStructure` schema + ASE converters
  (`atomistic_pinn.data`), energy-shift convention: +0.38446 eV/atom applied
  in the pipeline, never stored in data.

## 1. Ledger decisions to resolve BEFORE the first training run

None of these may be filled silently (hard project rule). Each resolution is
written into the configs with a comment naming the decision, and into the
final report.

| Item | Ledger | Proposal (labeled, revisitable) |
| --- | --- | --- |
| Activation functions | §7.4 — **RESOLVED** | `tanh` (smooth everywhere — matches the differentiable-physics design; sigmoid kept behind a config flag for comparison) |
| Weight initialization | engineering | seeded `torch.Generator` per restart; scheme (e.g. Xavier/Glorot) named in config; paper says only "random states" |
| τ, γ values | §7.6 — **RESOLVED (initial values, sweep pending)** | documented targeting the 3–4 meV/atom band; chosen values recorded in config + report |
| Number of random restarts | §7.7 — **RESOLVED** | 5 (paper: "several times") |
| DFP line search | §7.8 — **RESOLVED** | textbook DFP update + backtracking Armijo line search; gradients **numerical** per the resolved §7.8 decision (autograd used only as cross-check) |
| Output parameter constraints | §7.5 — **RESOLVED** | literal free outputs; log per-parameter distributions to detect drift into unphysical ranges |
| **p̄_η definition (γ-tether target)** | §7.13 — **RESOLVED (default (a))** | see §3.2 below — default: parameters of the pre-fitted fixed BOP; alternative: running mean of current predictions, behind a config flag |

## 2. Phase 3.1 — Networks (`atomistic_pinn.models`)

* **PINN** (`models/pinn.py`): plain fully connected 60×15×15×8, applied
  per atom: `G_i (60,) → p_i (8,)`; batched `(N, 60) → (N, 8)`. No bias in
  the output assumption is *not* stated anywhere — implement standard
  biases; the parameter-count test below is the arbiter.
* **Baseline NN** (`models/nn.py`): 60×16×16×1, `(N, 60) → (N, 1)` atomic
  energies directly — bypasses the BOP entirely, isolating the physics
  layer's effect.
* Both models consume the same descriptor pipeline; descriptor configs are
  shared with Sprint 1 (no new descriptor code).
* **Parameter counts** (paper Table 1, derivable): PINN 1283, NN 1265 —
  verified by independent arithmetic in tests, not by trusting the config.

**Tests (Phase 3.1 exit):** parameter counts; output shapes `(N, 8)` vs
`(N, 1)`; forward pass deterministic under seed.

## 3. Phase 3.2 — Objective, Eq. (13) (`atomistic_pinn.training.objective`)

```text
ℰ = Σ_s ( E_s - E_s^DFT )²          (unweighted sum over supercells, §7.11 literal)
  + τ Σ_{εκ} ( ‖w_εκ‖² + Σ_κ ‖b_κ‖² )   (all weights and biases)
  + γ Σ_η ( p_η - p̄_η )²               (tether of predicted parameters to p̄_η)
```

* One objective evaluation = **full pass over the entire training database**
  (the paper sums over all supercells; no minibatching is mentioned).
  Engineering note: descriptors depend only on structure → precompute/cache
  them once per supercell; neighbor lists rebuilt per structure.
* `E_s^DFT` enters shifted by +0.38446 eV/atom (pipeline-side, never
  stored).
* Each of the three terms is independently callable and tested on tiny
  hand-checkable examples.

### 3.2a The p̄_η problem (new ledger item §7.13)

The database contains only structures + DFT energies — **no true BOP
parameters** — so "database-averaged value of parameter η" is undefined
without an additional convention. Candidates:

* **(a) Fixed-BOP pre-fit (default):** first fit the eight parameters of
  the *fixed* BOP baseline (paper Fig. 8: same functional form, eight
  parameters fitted once and fixed) by minimizing the energy term of Eq.
  (13) over the database; the fitted values are `p̄_η` — and the fit
  simultaneously delivers the fixed-BOP baseline for Sprint 3/4. One
  computation, two uses.
* **(b) Self-mean:** `p̄_η` = mean of the *current* predicted parameters
  over the training set; the γ term then equals the variance of predicted
  parameters ("all atoms should carry similar parameters"). Behind a config
  flag for comparison.

Decision recorded at Sprint 3 start; revisit if the Supplementary
Information becomes available.

**Tests (Phase 3.2 exit):** three terms verified independently; γ term with
(a) equals a hand-computed value on a toy two-structure set; energy shift
applied exactly once (double-shift is a silent-corruption risk).

## 4. Phase 3.3 — DFP optimizer (`atomistic_pinn.training.dfp`)

* Textbook DFP quasi-Newton (inverse-Hessian approximation updated as
  `H⁺ = H − (H y yᵀ H)/(yᵀ H y) + (s sᵀ)/(yᵀ s)`, with `s = x⁺ − x`,
  `y = g⁺ − g`), warm-started with `H = I`.
* Gradients: **numerical** (resolved §7.8 decision). Cost warning: one
  gradient costs `n_params + 1` objective evaluations (~1284 for the PINN ×
  full-database pass each). Feasibility on CPU is a function of the
  *alternative* dataset size — record the measured cost per iteration in
  the run log; if intractable, the fallback (autograd gradients) is a
  labeled deviation requiring a ledger update of §7.8, never a silent swap.
* Line search: backtracking Armijo on ℰ (labeled, §7.8).
* Restarts: `n_random_restarts` runs from independently seeded random
  inits; best final ℰ kept; derived seed = f(master_seed, restart index).

**Tests (Phase 3.3 exit):** DFP reduces ℰ monotonically (line-search
acceptance enforced) on a quadratic toy problem; H-update matches a
hand-computed DFP step on a 2-D example; restarts reproducible under a
fixed seed.

## 5. Phase 3.4 — Training loop, checkpoints, CLI (`atomistic_pinn.training`, `cli`)

* Training driver: config → resolve ledger decisions (Pydantic validation
  rejects a config whose `null` slots are needed but unnamed) → Phase 3.2a
  pre-fit (if (a) chosen) → restarts × DFP → best checkpoint.
* Cross-validation is **not** in Sprint 3 (Sprint 4 wraps it around this
  driver); Sprint 3 trains on a single documented development split to
  prove the machinery.
* Checkpoints record (architecture.md): model state, DFP state including
  the inverse-Hessian approximation, resolved config, config hash, dataset
  hash, git commit, master seed + derived seeds, restart index, fold id
  (`None` in Sprint 3).
* Reporting (`atomistic_pinn.reporting`, Sprint 3–4 layer): ℰ-vs-iteration
  curves per restart, energy parity plot (predicted vs. shifted DFT),
  per-structure-group error histograms, predicted-parameter distributions
  → `results/` (generated, never committed).
* CLI: `atomistic-pinn train --config configs/pinn_al.yaml` and
  `... baseline_nn.yaml` land here (README Usage promises them in
  Sprints 3–4).

**Tests (Phase 3.4 exit):** checkpoint round-trip restores training state
bit-exactly (including H); a 2-structure × 2-restart run reproduces
bit-identically under the same seed; CLI exits nonzero with the documented
error type when a required config slot is `null`.

## 6. Phase 3.5 — Development run + tuning

* Run both models on the chosen alternative dataset; measure RMSE
  (meV/atom) on the development split against the **reference band** 3–4
  meV/atom (`docs/reproduction_protocol.md`: band, not gate, for
  alternative data).
* Tune τ, γ (documented sweep); confirm PINN vs. baseline-NN comparison is
  apples-to-apples (same descriptors, same split, same seeds).
* Fixed-BOP baseline numbers from Phase 3.2a recorded for the Sprint 4
  comparison table.

## 7. Definition of done

1. All §7 items Sprint 3 depends on (4, 5, 6, 7, 8, 11, 13) are resolved or
   labeled — configs contain **no `null` in a slot Sprint 3 consumes**.
2. All Phase exit tests green: `pytest -m unit` and `-m
   scientific_validation` include the new parameter-count, contract, DFP,
   seed-reproducibility, and checkpoint tests.
3. `atomistic-pinn train` runs end-to-end from one config for both models,
   producing checkpoint + report artifacts with full provenance logging
   (resolved config, git commit, dataset hash, seeds, checkpoint hashes).
4. Development-split RMSE measured and reported with fidelity level
   "methodological reproduction (alternative data)".
5. Docs synced: `docs/architecture.md` module map (learning layer rows),
   `docs/scientific_method.md` §3.3–3.4 statuses, README reproduction table,
   this file's open decisions marked resolved.
