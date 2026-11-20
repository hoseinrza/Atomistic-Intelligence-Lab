# Molecular dynamics plan (Sprint 4)

This document specifies the molecular-dynamics (MD) component of Sprint 4:
what role MD plays in this reproduction, how forces are obtained, the
demonstration protocol, and how it is validated. The software design lives
in `docs/architecture.md`; the physics equations in `docs/equations.md`.

## 1. Role of MD in this reproduction

**Paper facts.** The paper's MD runs (thermal expansion, melting-related
checks, and the HCP-MD test configurations sampled from trajectories) were
performed with the authors' in-house simulation codes. Those codes are
explicitly *not* reproduced (`docs/scientific_method.md` §6), and the
property benchmarks are a *stretch goal* after Sprint 4, not part of the
core reproduction.

**This project.** MD is a *demonstration*: proof that the trained PINN
potential is usable in a standard atomistic workflow (`atoms.get_potential_energy()`
/ `atoms.get_forces()` through ASE) and stays physically stable over a short
trajectory. No paper metric is claimed from it.

## 2. The force problem

The paper's stated force method is **finite differences** (post-training
only). Central differences on every Cartesian coordinate cost `6N` energy
evaluations per configuration. For a 500-atom MD step with, say, 10⁴ steps,
that is `3 × 10⁷` BOP evaluations — intractable for dynamics, and the
paper's force route *during* trajectories is not stated in the main text
(registered as uncertainty item 12 in `docs/equations.md` §7).

**Decision (labeled engineering deviation):** MD runs use **autograd
forces** — the gradient of total energy w.r.t. positions, computed through
descriptors → network → BOP — while every *static* force metric reported
for the reproduction stays computed by the paper-faithful finite
differences. Agreement of the two methods on static configurations is
itself a scientific test (Sprint 2/4 test list), so the deviation is
validated, and training remains energy-only regardless (paper methodology
unaffected).

## 3. Forces through the physics layer

* A neighbor list is built once per force evaluation and held **fixed**
  while the gradient flows; the C³-smooth cutoff function (Sprint 1) is
  what makes this differentiation well-defined near `r_c`.
* The list is rebuilt when any atom has moved more than a rebuild threshold
  from the positions it was built at (skin-based rebuild-on-demand,
  `docs/architecture.md`). The threshold value is a labeled engineering
  decision; the current `NeighborList.build` API is stateless, so the
  calculator tracks max displacement between calls.
* Forces are accumulated per atom as `-dE/dR_i`; translational invariance
  gives ΣF = 0, which is checked in tests.

## 4. Demonstration protocol

All numeric values below are **proposed engineering decisions** (not paper
facts), tunable in a single config (`configs/md_demo.yaml`, values `null`
until Sprint 4 fills them with a named decision).

| Stage | Ensemble | System | Purpose |
| --- | --- | --- | --- |
| A | — | static FCC supercell | ASE calculator wiring; FD-vs-autograd force agreement |
| B | NVE | FCC Al, 5×5×5 conventional cells (500 atoms), dt = 1 fs, ~10 ps | strongest correctness gate: total-energy drift |
| C | NVT (Langevin, seeded) | same system, T = 300 K, ~5 ps | temperature stability; usable trajectory demo |

Proposed defaults: initial velocities from Maxwell–Boltzmann at 300 K with
zero total momentum; Langevin friction documented once chosen; trajectory
frames + per-frame energy/temperature logged to `results/` (generated,
never committed).

**Extrapolation guard.** Per frame, coordination numbers and nearest-neighbor
statistics are logged; a frame whose environments fall outside the training
domain's descriptor range is flagged in the log rather than silently used
(`docs/reproduction_protocol.md` fidelity rules).

## 5. Validation and tests

* **Calculator contract:** shapes/types of energy and forces; calculator
  energy equals a direct engine call on the same frame.
* **FD vs autograd:** agreement on random and strained configurations
  (scientific_validation marker, two independent methods).
* **NVE energy conservation:** fixed seed, fixed step count; total-energy
  drift below a documented tolerance — this is the primary physics gate.
* **ΣF = 0** within tolerance on every logged frame.
* **Reproducibility:** identical trajectory under the same master seed
  (Langevin noise drawn from a seeded generator); checkpoint/config/dataset
  hashes recorded per run.

## 6. Explicitly out of scope (stretch, after Sprint 4)

* Thermal expansion (lattice parameter vs. T) and melting-point estimates
  (paper Table 2 / Figs. 3–8) — stretch goal, methodological comparison only.
* Any MD on the paper's exact database (unavailable; see `data/README.md`).
