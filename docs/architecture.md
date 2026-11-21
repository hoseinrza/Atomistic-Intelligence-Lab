# Architecture

## Design principles

1. **Scientific fidelity first.** The physics layer implements exactly the
   equations in `docs/equations.md`; anything unspecified there is flagged,
   never silently substituted.
2. **Separation of concerns.** Data, geometry, descriptors, physics,
   learning, evaluation, simulation, and reporting are independent layers
   with a one-directional dependency graph (below).
3. **Testability.** Every scientific claim is backed by a test against an
   independent reference (closed-form cases, `scipy.constants`, ASE geometry
   utilities, finite differences vs. autograd).
4. **Reproducibility.** Every stochastic step is seeded; every experiment
   records config, git commit, dataset hash, and fold assignments.
5. **No speculative abstraction.** Modules are added when a sprint needs
   them.

## Layers and module map

| Layer | Package | Contents | Sprint |
| --- | --- | --- | --- |
| Data | `atomistic_pinn.data` | schema (data contracts), ASE converters; loaders, HDF5 storage, splits | 0, 1, 4 |
| Geometry | `atomistic_pinn.geometry` | neighbor lists, PBC, distances, angles | 1 |
| Descriptors | `atomistic_pinn.descriptors` | Gaussian radial functions, Legendre polynomials, cutoff, 60-d builder | 1 |
| Physics | `atomistic_pinn.physics` | BOP pair/bond-order/screening/promotion terms, atomic + total energy | 2 |
| Learning | `atomistic_pinn.models`, `atomistic_pinn.training` | PINN + NN architectures, Eq. (13) objective, DFP, restarts, checkpoints | ✅ 3 |
| Evaluation | `atomistic_pinn.evaluation` | energy/force metrics, cross-validation, extrapolation analysis | 4 |
| Simulation | `atomistic_pinn.simulation` | ASE calculator, basic MD | 4 |
| Reporting | `atomistic_pinn.reporting` | training curves, parity plots, error histograms, benchmark summaries | 3–4 |
| Cross-cutting | `constants`, `exceptions`, `cli` | units, paper constants, error hierarchy, Typer CLI | 0+ |

## Dependency rules

```text
constants, exceptions   (no internal dependencies)
        ↑
data, geometry          (constants only)
        ↑
descriptors, physics    (constants, data, geometry)
        ↑
models, training        (descriptors, physics, data)
        ↑
evaluation, simulation  (models, physics, data)
        ↑
reporting, cli          (everything above)
```

* The physics layer must be usable **without** the learning stack, so that
  physics tests and the ASE calculator do not import torch-based model code.
* The CLI never contains scientific logic; it only orchestrates layers.

## Physics implementation strategy

* The BOP and descriptors will be implemented once, on PyTorch tensors
  (CPU-compatible), so the same code serves training (autograd through the
  network *and* through the BOP) and simulation (`.detach().numpy()` at the
  ASE boundary).
* The paper computes forces by **finite differences**; that remains the
  faithful default. Autograd forces are an engineering improvement used as an
  independent cross-check in tests (FD vs. autograd agreement is itself a
  scientific test).
* Neighbor lists are the performance bottleneck (screening is
  O(N·n²)); Sprint 1 introduces a skin-based list rebuilt on demand, never
  silently stale.

## Testing strategy

Three pytest markers (`pyproject.toml`):

* `unit` — fast, no external data (constants, schema validation).
* `integration` — components working together (converters + ASE; later
  model + BOP contract tests).
* `scientific_validation` — independent-reference checks: closed-form
  energies, invariance properties, cutoff smoothness, FD/autograd force
  agreement, cross-validated metric reproduction.

Policy: **never fabricate reference values.** References come from closed
form, independent libraries, or two independently implemented methods that
must agree.

## Error handling

* `AtomisticPINNError` hierarchy (`exceptions.py`); every user-facing error
  states what is wrong and where the contract is documented.
* Configurations are validated (Pydantic) before any run starts; a config
  that fills an "unspecified in paper" slot must name the deviation
  explicitly or it is rejected.

## Reproducibility mechanics

* Seeds: one master seed per experiment, derived seeds per fold and per
  restart, all logged.
* Checkpoints record: model state, optimizer state, config hash, dataset
  hash, git commit, fold id.
* `results/` artifacts are generated, never committed; `experiments/` holds
  the run scripts that produce them.
