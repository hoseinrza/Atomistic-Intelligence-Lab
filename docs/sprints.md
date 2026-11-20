# Sprint plan

Five sprints (0–4) take the project from an empty repository to a complete,
tested reproduction of the PINN bond-order potential. Status below reflects
the current state of the repository; update this table per sprint.

| Sprint | Theme | Status |
| --- | --- | --- |
| 0 | Project foundation: units, data contracts, CI, docs | ✅ done |
| 1 | Geometry + descriptors (neighbor lists, PBC, 60-dim fingerprint) | ✅ done |
| 2 | Bond-order potential physics layer (Eqs. 6–12) | planned |
| 3 | Networks + training (PINN, NN baseline, Eq. 13 objective, DFP) | planned |
| 4 | Evaluation + simulation (metrics, cross-validation, forces, ASE, MD) | planned |

---

## Sprint 0 — Project foundation (done)

**Goal:** a runnable, tested, documented skeleton so every later sprint
adds physics to an already-solid base.

**Deliverables**

* Unit conventions (Å / eV / eV/Å) and all paper-derived constants pinned in
  `constants.py`.
* Data contracts: schema validation for structures/energies, ASE converters.
* Error hierarchy (`exceptions.py`), Typer CLI (`version`, `paper`).
* CI workflow, config files (`configs/baseline_nn.yaml`,
  `configs/pinn_al.yaml`) with "unspecified in paper" slots left `null`.
* Docs: scientific method, equations + uncertainty ledger, architecture,
  data format, reproduction protocol.

**Exit criteria:** `pytest` green in CI; CLI works; every paper fact in the
ledger is either pinned or explicitly labeled.

## Sprint 1 — Geometry + descriptors (done)

**Goal:** the fixed-length, invariant 60-dimensional fingerprint
`G_i` of every atom's local environment (Eqs. 3–5).

**Deliverables**

* `geometry/`: skin-based neighbor list with full PBC image enumeration,
  distances, angles.
* `descriptors/`: cutoff function (C³-smooth), radial Gaussians, Legendre
  polynomials of bond angles, 60-dim builder.

**Tests / exit criteria**

* Rotation, translation, and permutation invariance.
* Dimension = 60; C³ cutoff smoothness.
* Agreement with a slow, independent reference implementation on random
  configurations.
* Hand-checkable symmetric cases (e.g. perfect FCC site zeroes symmetry-
  forbidden components).
* Deterministic, sort-free accumulation per atom.

## Sprint 2 — Bond-order potential (planned)

**Goal:** compute `E_i` from neighbor positions and the eight per-atom
parameters with the full physics: repulsion, bond order, screening,
promotion/embedding (Eqs. 6–12).

**Deliverables**

* `physics/`: pair term, bond-order term, screening function, promotion
  term; atomic and total energy; usable **without** torch/model code.

**Tests / exit criteria**

* Closed-form single-pair energies; pairwise additivity of the repulsive term.
* Screening factor = 1 outside the cutoff spheroid; minimal on the bond axis.
* Perfect-crystal energy against a hand-computed value.
* Total/atomic energy consistency; finite-difference vs. autograd forces
  (two independent methods).

## Sprint 3 — Networks + training (planned)

**Goal:** learn the eight BOP parameters. PINN 60×15×15×8 (descriptors →
parameters → BOP) vs. baseline NN 60×16×16×1 (descriptors → energy
directly), isolating the effect of the physics layer. Full execution plan
with build order, ledger decisions, and definition of done:
[sprint3_plan.md](sprint3_plan.md).

**Phases (build order)**

1. **Resolve ledger items first** (§7.4 activation, §7.6 τ/γ, §7.7 restarts,
   §7.8 DFP line search, §7.13 p̄_η definition) — proposals are pre-written
   in the plan; nothing is filled silently.
2. **Networks** (`models/`): PINN + baseline NN; parameter counts verified
   against Table 1 by independent arithmetic (1283 / 1265).
3. **Objective** (Eq. 13): energy term + τ weight regularization + γ
   parameter tether; the p̄_η target comes from the fixed-BOP pre-fit
   (which also delivers the fixed-BOP baseline, paper Fig. 8) or the
   self-mean variant behind a flag; energy shift +0.38446 eV/atom applied
   pipeline-side; full-database pass per objective evaluation (no
   minibatching in the paper).
4. **DFP optimizer**: textbook update, numerical gradients (resolved §7.8),
   backtracking-Armijo line search, `n` seeded random restarts, best ℰ kept.
5. **Training loop + checkpoints + CLI**: full-provenance checkpoints
   (DFP state incl. inverse Hessian, hashes, seeds), reporting artifacts,
   `atomistic-pinn train` command; cross-validation stays in Sprint 4.
6. **Development run + tuning**: τ/γ sweep against the 3–4 meV/atom
   reference band on the alternative-data development split.

**Tests / exit criteria**

* Parameter counts (Table 1); output shapes (N, 8) vs. (N, 1); full model
  reproduces the BOP engine exactly for fixed parameters (contract test
  between learning and physics layers).
* DFP reduces the objective monotonically on a quadratic toy problem;
  restarts reproducible under a fixed seed; checkpoint round-trip restores
  training state bit-exactly (including the inverse-Hessian state).
* End-to-end `train` run from one config with full provenance logging.

## Sprint 4 — Evaluation + simulation (planned)

**Goal:** prove the reproduction against paper metrics and make the
potential usable in real atomistic workflows.

**Deliverables**

* `evaluation/`: energy MAE/RMSE in meV/atom vs. Table 1 bands; force
  MAE/RMSE by finite differences (paper's method, autograd as cross-check);
  10-fold cross-validation with per-structure folds; extrapolation analysis
  on held-out structure groups.
* `simulation/`: ASE calculator wiring energies + forces into
  `atoms.get_potential_energy()` / `atoms.get_forces()`; short NVE/NVT MD
  demonstration (detailed protocol, force route, and validation:
  [md_plan.md](md_plan.md)).

**Tests / exit criteria**

* Cross-validation runs end-to-end with logged fold assignments and seeds.
* FD and autograd forces agree; calculator round-trips with ASE.
* Benchmark summary reports compare against the reference bands without
  fabricating values.

---

## After Sprint 4 (stretch, not core)

Property benchmarks beyond energies/forces — thermal expansion, defect
energetics, stacking faults (paper Table 2, Figs. 3–8) — are a stretch goal
after Sprint 4.
