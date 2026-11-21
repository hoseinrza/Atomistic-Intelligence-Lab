# Scientific method

This document explains *what* is reproduced and *why* the methodology works.
The exact equations live in `docs/equations.md`; the software design lives in
`docs/architecture.md`.

Source paper:

> G. P. Purja Pun, R. Batra, R. Ramprasad, and Y. Mishin, *"Physically
> informed artificial neural networks for atomistic modeling of materials"*,
> Nature Communications **10**, 2339 (2019).

---

## 1. The problem the paper solves

Conventional NN potentials map local descriptors directly to atomic energy
(`G_i → E_i`). They interpolate DFT energies extremely well inside the
training domain but extrapolate unphysically outside it (e.g. energies that
develop wiggles instead of the correct repulsive behavior under strong
compression — paper Fig. 6). Traditional physics-based potentials behave
physically when extrapolating but cannot reach ML-level accuracy.

The PINN idea is the compromise: keep the physics, learn the parameters.

```text
NN potential :  G_i ──────────────────────→ E_i          (pure regression)
PINN         :  G_i ──[NN]──→ p_i ──[BOP]──→ E_i          (physics in the loop)
```

Even when the predicted parameters `p_i` are inaccurate for an unseen
environment, the BOP functional form keeps the energy physically meaningful
(repulsion at short range, bond weakening with coordination, screened bonds,
embedding-style promotion energy). The paper demonstrates exactly this
improved transferability for Al.

## 2. Pipeline (what this project implements, sprint by sprint)

```text
DFT supercells ─→ descriptors (60/atom) ─→ NN (60×15×15×8) ─→ 8 BOP params/atom
                                                              │
             training objective (Eq. 13) ←— total energy E_s = Σ_i E_i
                        │
             DFP optimization + random restarts
                        │
             10-fold cross-validation
                        │
             energy/force evaluation (forces by finite differences)
                        │
             ASE calculator → property checks → MD
```

## 3. Component summaries

Each component lists: mathematical purpose → paper equation → numerical
strategy → tests. Full equations: `docs/equations.md`.

### 3.1 Descriptors (Sprint 1)

* **Purpose:** a fixed-length, rotation/translation/permutation-invariant
  fingerprint of each atom's local environment, rich enough to distinguish
  crystal, defect, surface, and liquid environments.
* **Equations:** (3)–(5) — sums of Legendre polynomials of bond angles,
  weighted by truncated Gaussians of the two bond lengths.
* **Numerical strategy:** restrict the double sum to neighbors inside the
  cutoff (the cutoff function zeroes everything else anyway); accumulate in
  float64; sort-free deterministic accumulation order per atom.
* **Tests:** the three invariances; dimension 60; C³ cutoff smoothness;
  agreement with a slow, independent reference implementation on random
  configurations; descriptor values for hand-checkable symmetric cases
  (e.g. perfect FCC site has `g^(1) = g^(3) = ...` symmetry-forbidden
  components equal to zero by spherical symmetry).

### 3.2 Bond-order potential (Sprint 2)

* **Purpose:** compute `E_i` from neighbor positions and the eight per-atom
  parameters with the physics of repulsion, bond order, screening, and
  promotion/embedding energy (Eqs. 6–12).
* **Numerical strategy:** the square-root arguments (`1 + z_ij`, `ρ_i`) are
  provably non-negative; screening products are accumulated in float64;
  neighbor-list driven so cost scales as O(N · n_neighbors²) for screening.
* **Tests:** closed-form single-pair energies; pairwise-additivity of the
  repulsive term; screening factor equals 1 outside the cutoff spheroid and
  is minimal for an atom on the bond axis; energy of a perfect crystal
  against a hand-computed value; total-energy/atomic-energy consistency;
  finite-difference forces against autograd forces (two independent
  methods).

### 3.3 Networks (Sprint 3 — ✅ implemented, pipeline-validated)

* **Purpose:** PINN (60×15×15×8) maps descriptors to the eight BOP
  parameters; the baseline NN (60×16×16×1) maps descriptors directly to
  atomic energy, isolating the effect of the physics layer.
* **Uncertainty:** activation functions are not specified in the paper
  (`docs/equations.md` §7 item 4).
* **Tests:** parameter counts match Table 1; output shape (N, 8) vs (N, 1);
  the full model reproduces the BOP engine exactly when given fixed
  parameters (contract test between the learning and physics layers).

### 3.4 Training (Sprint 3 — ✅ implemented, pipeline-validated)

* **Purpose:** minimize Eq. (13) — squared total-energy errors, weight
  regularization (τ), and a tether of predicted parameters to their
  database averages (γ).
* **Method:** Davidson–Fletcher–Powell quasi-Newton optimization from
  several random starts, best objective kept; energies shifted by
  +0.38446 eV/atom before training. No force term: the paper trains on
  energies only.
* **Tests:** DFP reduces the objective monotonically on a quadratic toy
  problem; restarts reproduce results under a fixed seed; checkpoint
  round-trip restores training state bit-exactly.

### 3.5 Evaluation and simulation (Sprint 4)

* Energy MAE/RMSE in meV/atom against Table 1 reference bands; force MAE/RMSE
  by finite differences (paper's method) with autograd as a cross-check;
  10-fold cross-validation with per-structure fold assignment; extrapolation
  analysis on held-out structure groups; ASE calculator wiring energies and
  (finite-difference) forces into `atoms.get_potential_energy()` and
  `atoms.get_forces()`; a short NVE/NVT MD demonstration.

## 4. Baselines

| Baseline | Description | Status |
| --- | --- | --- |
| NN 60×16×16×1 | same descriptors, direct energy regression | Sprint 3 |
| NN′ 47×18×18×1 | third-party PROPhet package, Behler–Parrinello features | not exactly reproduced (deviation note) |
| fixed BOP | same functional form, eight parameters fitted once and fixed (paper Fig. 8) | Sprint 3/4 |
| EAM | published Al EAM potential (paper ref. 54) | optional; uses a published parameter file, no refitting |

## 5. Ledger: paper facts vs. decisions vs. deviations

Maintaining this distinction is a hard requirement of the project.

| Item | Type | Choice / value | Where |
| --- | --- | --- | --- |
| Units Å / eV / eV/Å | paper convention | adopted | `constants.py` |
| r_c = 6 Å, d = 1.5 Å | paper fact | pinned | `constants.py` |
| σ = 1.0, m ∈ {0,1,2,4,6}, 12 r₀ | paper fact | pinned | `constants.py` |
| 12 radial center coordinates | **unspecified** | RESOLVED (S1): 0.5 Å grid 0.0-5.5, labeled decision | `constants.py`, `docs/equations.md` §7.1 |
| j = k in descriptor sum | **unspecified** | RESOLVED (S1): ordered sum incl. j=k (literal; g^(1)=\|Σf r̂\|² identity holds), flag for alternatives | `docs/equations.md` §7.2 |
| Descriptor cutoff | **unspecified** | RESOLVED (S1): same r_c=6 Å, d=1.5 Å as BOP | `docs/equations.md` §7.3 |
| Activation functions | **unspecified** | deferred to Sprint 3, labeled | `docs/equations.md` §7.4 |
| τ, γ, restart count | **unspecified** | tuned/documented later, labeled | `docs/equations.md` §7.6–7 |
| p̄_η (γ-tether target) | **unspecified** | RESOLVED at S3 start: fixed-BOP pre-fit parameters (default) or self-mean (flag) | `docs/equations.md` §7.13, `docs/sprint3.md` |
| DFP implementation | paper fact + textbook variant | DFP w/ numerical gradients | `docs/equations.md` §7.8 |
| Force evaluation by FD | paper fact | faithful default (static configurations) | Sprint 4 |
| MD demo forces + ensemble setup | unspecified / engineering | labeled decisions: autograd forces (FD-validated), NVE gate + short NVT | `docs/md_plan.md` |
| PyTorch implementation of physics layer | engineering | single torch implementation, CPU-compatible | `docs/architecture.md` |
| Alternative dataset | deviation | methodological reproduction only | `data/README.md` |
| Per-atom-normalized objective | experimental deviation | comparison experiments only | `docs/equations.md` §7.11 |

## 6. What is explicitly *not* reproduced

* The authors' in-house MPI/OpenMP training and simulation codes (paper
  mentions them and a forthcoming HPC package); this project is
  single-node PyTorch.
* The NN′ pipeline of the PROPhet package (unless later added as a labeled
  experimental extension).
* The exact DFT dataset (not public; see `data/README.md`).
* Property benchmarks beyond energies/forces (thermal expansion, defect
  energetics, stacking faults — paper Table 2/Fig. 3–8) are a *stretch
  goal* after Sprint 4, not part of the core reproduction.
