# Equations of the PINN–BOP model

This document is the single source of truth for the mathematics reproduced by
this project. Every equation is transcribed from:

> G. P. Purja Pun, R. Batra, R. Ramprasad, and Y. Mishin, *"Physically
> informed artificial neural networks for atomistic modeling of materials"*,
> Nature Communications **10**, 2339 (2019), DOI
> [10.1038/s41467-019-10343-5](https://doi.org/10.1038/s41467-019-10343-5).

Equation numbers below refer to that paper. Units throughout: Å, eV, eV/Å.

---

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| `N` | number of atoms in a supercell |
| `E_s` | total energy of supercell `s`; `E_s = Σ_i E_i` |
| `E_s^DFT` | DFT reference energy of supercell `s` |
| `r_ij` | distance between atoms `i` and `j` |
| `θ_ijk` | angle at atom `i` between the bonds `ij` and `ik` |
| `f(r)` | truncated Gaussian, Eq. (4) |
| `f_c(r)` | cutoff function, Eq. (5) |
| `P_m(x)` | Legendre polynomial of order `m` |
| `G_i^l` (l = 1..60) | local structural descriptors (fingerprints) of atom `i` |
| `p_i = (A_i, B_i, α_i, β_i, a_i, h_i, σ_i, λ_i)` | the eight BOP parameters of atom `i`, predicted by the PINN network |
| `p̄_η` | database-averaged value of BOP parameter `η` |
| `w`, `b` | network weights and biases |
| `τ`, `γ` | regularization strengths (values not specified in the paper) |

## 2. The two-step mapping — Eq. (1)

```text
E_i = E_i( r_{i1}, ..., r_{in}, p_i( G_i^1, ..., G_i^k ) )
```

The neural network maps descriptors to parameters (`G_i → p_i`) per atom; a
physics-based BOP then maps (neighbor positions, `p_i`) to energy. The paper
argues this intermediate step does not degrade accuracy (a feedforward NN can
represent any real-valued function, paper refs. 48–49) while making
extrapolation physics-guided: even inaccurate predicted parameters keep the
energy physically meaningful through the BOP functional form.

An extension, Eq. (2), splits the parameters into local (NN-predicted) `a_i`
and global (fixed) `b`. It is **not** part of the 2019 reproduction scope; it
is recorded here for completeness only.

## 3. Local structural descriptors — Eqs. (3)–(5)

**Eq. (3)** — angular descriptors as sums over neighbor pairs:

```text
g_i^(m) = Σ_{j,k} P_m( cos θ_ijk ) · f(r_ij) · f(r_ik),    m = 0, 1, 2, ...
```

**Eq. (4)** — truncated Gaussian of width σ centered at r₀:

```text
f(r) = (1/σ³) · exp[ -(r - r₀)² / σ² ] · f_c(r)
```

**Eq. (5)** — cutoff function, smooth to third order at the cutoff:

```text
f_c(r) = { (r - r_c)⁴ / ( d⁴ + (r - r_c)⁴ )   for r ≤ r_c
         { 0                                  for r > r_c
```

**Paper values** (Methods): σ = 1.0; m ∈ {0, 1, 2, 4, 6}; 12 values of r₀;
total k = 5 × 12 = **60 descriptors per atom**. Cutoff parameters
"were fixed at r_c = 6 Å and d = 1.5 Å" (Methods, BOP section).

Physical reading given by the paper: `g^(0)` measures the local atomic
density; `g^(1)` behaves like a dipole moment of unit charges at positions
`j, k` (zero for spherically symmetric environments); `g^(2)` is related to
the quadrupole moment. Polynomials up to m = 6 were found necessary to
represent diverse environments.

**Exact identities of the double-sum form** (used as independent checks in
the scientific tests): reading Σ_{j,k} as an ordered double sum including
j = k gives, per radial center c,

```text
g_i^(0)[c] = ( Σ_j f_c(r_ij) )²            (squared local density)
g_i^(1)[c] = | Σ_j f_c(r_ij) r̂_ij |²       (squared dipole moment)
```

The second identity proves rigorously that `g^(1) ≥ 0` and vanishes exactly
for spherically balanced environments — the paper's dipole interpretation.
Both identities are verified on random structures in
`tests/scientific_validation/test_descriptors.py`.

Properties exploited by the Sprint 1 tests:

* translation, rotation, and permutation invariance;
* `f_c` and its first three derivatives vanish at `r = r_c` (C³ smoothness;
  verifiable in closed form — let x = r − r_c ≤ 0, then
  f_c = x⁴/(d⁴+x⁴), f_c′ = 4x³d⁴/(d⁴+x⁴)², f_c″ = 4d⁴x²(3d⁴−5x⁴)/(d⁴+x⁴)³,
  all → 0 as x → 0⁻, and f_c‴ → 0 likewise);
* descriptor dimension is 60 for any periodic or cluster structure.

## 4. Bond-order potential — Eqs. (6)–(12)

**Eq. (6)** — atomic energy: pair term + promotion term.

```text
E_i = ½ Σ_{j≠i} [ exp( -A_i α_i r_ij ) - S_ij · b_ij · exp( -B_i β_i r_ij ) ] · f_c(r_ij) + E_i^(p)
```

**Eq. (7)** — bond order:

```text
b_ij = (1 + z_ij)^(-1/2)
```

**Eq. (8)** — effective bond count (screened, angular):

```text
z_ij = a_i² Σ_{k≠i,j} S_ik ( cos θ_jik + h_i )² f_c(r_ik)
```

**Eq. (9)** — screening factor as a product of partial screenings:

```text
S_ij = Π_{k≠i,j} S_ijk
```

**Eq. (10)** — partial screening by neighbor `k` of bond `ij`:

```text
S_ijk = 1 - f_c( r_ik + r_jk - r_ij ) · exp( -λ_i² ( r_ik + r_jk - r_ij ) )
```

**Eq. (11)** — promotion / embedding energy:

```text
E_i^(p) = -σ_i ( Σ_{j≠i} S_ij b_ij f_c(r_ij) )^(1/2)
```

**Eq. (12)** — EAM interpretation of the same term:

```text
F(ρ_i) = -σ_i ρ_i^(1/2),    ρ_i = Σ_{j≠i} S_ij b_ij f_c(r_ij)
```

**Physical roles** (as assigned by the paper): the first exponential in
Eq. (6) is pairwise repulsion; the second is the attractive term whose
strength is reduced by the bond order (more bonds → smaller `b_ij` → weaker
`ij` bond) and by screening from surrounding atoms; `E^(p)` is the promotion
energy of covalent bonding (e.g. s²p² → sp³ hybridization for group IV) and
simultaneously the EAM embedding function, so the same form covers covalent
and metallic bonding — this is why the BOP "reduces to the EAM formalism in
the limit of metallic bonding".

**Implementation notes for Sprint 2:**

* All eight parameters are per-atom quantities (the network predicts one
  parameter set per atom). Sums in Eqs. (6)–(11) carry the *center* atom's
  parameters; the ½ prefactors restore pair symmetry after summing `E_i` over
  all atoms.
* Numerical safety: `z_ij ≥ 0` (squares of real numbers times non-negative
  `f_c`, `S_ik ∈ (0, 1]`) and `ρ_i ≥ 0` by construction, so the square roots
  in Eqs. (7) and (11) never receive negative arguments.
* No division by `r` appears anywhere in Eqs. (3)–(12): coincident atoms
  produce finite energies (though they are unphysical inputs and will be
  rejected by validation in practice).
* `r_ik + r_jk − r_ij` is constant on spheroids with poles at `i` and `j`;
  the paper uses this as the physical justification of Eq. (10): atoms on the
  cutoff spheroid have `S_ijk = 1` (no screening), atoms on the `i`–`j` line
  have `r_ik + r_jk − r_ij → 0` and strongly screen (almost break) the bond.
* Exponentials have non-positive arguments for `r ≥ 0` (`A, α, B, β, λ` are
  expected positive), so they are bounded by 1: no overflow risk, only
  harmless underflow to 0.

## 5. Training objective — Eq. (13)

```text
ℰ = Σ_s ( E_s - E_s^DFT )²
    + τ Σ_{εκ} ( ‖w_εκ‖² + Σ_κ ‖b_κ‖² )
    + γ Σ_η ( p_η - p̄_η )²
```

The second term is weight/bias regularization (overfitting control, strength
τ); the third pulls each predicted BOP parameter toward its database-average
`p̄_η` (strength γ).

**Paper facts about the training procedure:**

* Minimized by the **Davidson–Fletcher–Powell (DFP)** algorithm of
  unconstrained optimization; repeated several times from different random
  states, keeping the solution with smallest ℰ.
* **10-fold cross-validation**: the database is randomly partitioned into 10
  subsets; one subset is held out for validation, the rest are used for
  training; the process is repeated 10 times.
* Before training, all DFT energies are uniformly shifted by
  **+0.38446 eV per atom** to match the experimental cohesive energy of Al,
  3.36 eV/atom. The original `E_s^DFT` values were *not* corrected to remove
  the free-atom energy.
* **Forces are not used** for training or validation; they are computed after
  training by the **finite-difference method**.

## 6. Reference numbers (Table 1 and Results text)

| Model | Architecture | Parameters | RMSE train (meV/atom) | RMSE validation |
| --- | --- | --- | --- | --- |
| NN  | 60×16×16×1 | 1265 | 3.36 | 3.85 |
| NN′ | 47×18×18×1 (Behler–Parrinello features, PROPhet) | 1225 | 3.62 | 3.54 |
| PINN | 60×15×15×8 | 1283 | 3.46 | 3.59 |

Force RMSE (eV/Å), quoted approximately in the Results text: validation —
PINN ≈ 0.1 vs NN ≈ 0.2; testing — PINN ≈ 0.1 vs NN ≈ 0.4 (edge dislocation)
and ≈ 0.6 (HCP MD).

The parameter counts are *derivable*: they equal the number of weights and
biases of plain fully connected layers, which `tests/scientific_validation/`
verifies.

## 7. Known uncertainties (resolve before relying on them)

Items the paper's main text does **not** specify. Each must be resolved (via
the Supplementary Information, an author request, or an explicitly labeled
implementation decision) before Sprint 4 reporting. None of them may be
silently filled in.

1. **Radial centers r₀.** 12 values are used but their coordinates are not
   given. **RESOLVED as a labeled implementation decision** (Sprint 1):
   evenly spaced 0.5 Å grid, r₀ ∈ {0.0, 0.5, ..., 5.5} Å
   (`constants.DEFAULT_RADIAL_CENTERS`, exposed via `DescriptorConfig`).
   Revisit if the Supplementary Information becomes available.
2. **Descriptor summation range.** Whether the double sum Σ_{j,k} includes
   j = k, and whether pairs are ordered, is not stated. **RESOLVED as a
   labeled implementation decision** (Sprint 1): full ordered double sum
   over neighbors within cutoff, including j = k (literal reading). This
   reading makes the exact identities of §3 hold (g^(0) = (Σf)²,
   g^(1) = |Σf r̂|²) and is implemented in `DescriptorConfig.include_same_pair`;
   the excluded-pair convention is available behind the same flag for
   comparison experiments.
3. **Descriptor cutoff.** **RESOLVED as a labeled implementation decision**
   (Sprint 1): identical to the BOP cutoff (r_c = 6 Å, d = 1.5 Å), since the
   paper fixes these values in the BOP section and uses the same f_c form in
   Eq. (4).
4. **Activation functions** of the two hidden layers: not specified.
   Candidates: tanh, sigmoid. Decision (labeled) deferred to Sprint 3.
5. **Output parameter constraints:** none are stated. Physically, A, B, α, β,
   a, σ, λ are expected positive and h ∈ (−1, 1). The literal implementation
   uses free outputs; any constraint scheme will be documented as a
   deviation.
6. **τ and γ:** values not given; they will be tuned to reach the 3–4
   meV/atom target and documented.
7. **Number of random restarts:** the paper says "several times"; a concrete
   number will be chosen and documented.
8. **DFP variant details** (line search, gradient computation): not given.
   Plan: textbook DFP with numerically computed gradients (labeled
   implementation decision). **UPDATE 2026-11-21 (labeled fallback, as
   pre-authorized in `docs/sprint3.md` §4):** measured cost on the ANI-Al
   alternative-data dev subset (16 structures, 2,048 atoms): one objective
   evaluation ≈ 11.3 s ⇒ one numerical gradient (1,285 evaluations) ≈ 4.0 h
   ⇒ even a 20-iteration, 2-restart run ≈ 160+ h — intractable on CPU.
   The alternative-data dev runs therefore use **autograd gradients**
   (`configs/*_ani.yaml`, `training.gradients: autograd`); the numerical
   route remains the resolved §7.8 default for small databases and paper-scale
   sanity checks. Route equivalence is validated by the scientific tests
   (module-vs-functional energy equality, `tests/scientific_validation/
   test_objective.py`; FD-vs-autograd force agreement, Sprint 2 gates), and
   the measured cost is recorded in the run log. A numerical-gradient run on
   a reduced subset stays available for cross-checking the dev-run optimum.
9. **NN′ baseline** relies on the third-party PROPhet package and
   Behler–Parrinello symmetry functions; it will not be reproduced exactly.
   Options: skip it, or implement BP symmetry functions as an explicitly
   labeled experimental extension.
10. **Dataset.** The 3,649-supercell training/validation database and the
    26,425-supercell testing database are not publicly downloadable (paper:
    Supplementary Information / on request). **Status 2026-11-21:** the
    Supplementary Information was obtained; its group-by-group database
    composition is transcribed and verified in `docs/dataset_composition.md`
    (composition only — no positions/energies). The alternative dataset for
    the methodological reproduction is selected: ANI-Al (Smith et al. 2021,
    6,352 single-element Al PBE calculations) — survey and provenance in
    `data/README.md`. Results remain "methodological reproduction
    (alternative data)".
11. **Objective normalization.** Eq. (13) as written sums *unweighted*
    squared total-energy errors over supercells (large supercells dominate).
    We implement it literally; any per-atom normalization will be an
    explicitly labeled deviation used only for comparison experiments.
12. **MD demonstration setup.** The paper's thermostat, timestep, run
    lengths, and force route *during* MD trajectories are not stated in the
    main text. Plan (labeled engineering decisions, `docs/md_plan.md`):
    autograd forces for dynamics — validated against the paper-faithful
    finite-difference forces on static configurations — with NVE
    energy conservation as the primary correctness gate and a short seeded
    NVT run as the demonstration. Training remains energy-only, so the
    paper's energy-only methodology is unaffected.
13. **Definition of `p̄_η` (γ-tether target, Eq. 13).** The database contains
    only structures + DFT energies — no true BOP parameters — so the
    "database-averaged value of parameter η" is undefined without an extra
    convention. Candidates (decide at Sprint 3 start, labeled;
    `docs/sprint3.md` §3.2a): (a) parameters of the once-fitted fixed
    BOP (paper Fig. 8) — default, also delivers that baseline;
    (b) running mean of the current predicted parameters over the training
    set (the γ term then equals the variance of predicted parameters) —
    behind a config flag. Revisit if the Supplementary Information becomes
    available.
15. **Fitted BOP parameter values (fixed-BOP baseline, Fig. 8).** The
    authors fitted the fixed-parameter BOP baseline to the same DFT
    database, but the resulting eight parameter values are not given in the
    main text (possibly in the Supplementary Information). Consequences for
    Sprint 3/4: the fixed-BOP baseline will either be *refitted here*
    (reported as "re-fit", never as the paper's fit — related to §7.13
    option (a)) or compared only qualitatively. The physics-layer tests use
    synthetic physical parameter sets (A·α < B·β for short-range repulsive
    dominance; positive σ, a, λ) — these are fixtures, not paper values.
