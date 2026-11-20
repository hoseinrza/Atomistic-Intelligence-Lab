# Sprint 0 — Project foundation (done)

**Status:** ✅ done (commit `83ed7fc`). Overview: [sprints.md](sprints.md).
**Next:** [sprint1.md](sprint1.md).

**Goal:** a runnable, tested, documented skeleton so every later sprint adds
physics to an already-solid base.

## Delivered

* **Units & constants** (`src/atomistic_pinn/constants.py`): Å / eV / eV/Å
  conventions, physical constants, and paper-derived constants
  (`r_c = 6 Å`, `d = 1.5 Å`, `σ = 1.0`, `m ∈ {0,1,2,4,6}`, 12 `r₀`,
  energy shift +0.38446 eV/atom).
* **Data contracts** (`src/atomistic_pinn/data/`): Pydantic schema
  (`AtomicStructure`, `Dataset`, `DatasetMetadata`) with validation; ASE
  converters (`structure_from_ase` / `structure_to_ase`) with round-trip
  tests.
* **Error hierarchy** (`exceptions.py`): `AtomisticPINNError` base; every
  user-facing error states what is wrong and where the contract is
  documented.
* **CLI** (`cli.py`): `atomistic-pinn version`, `atomistic-pinn paper`
  (`train` / `evaluate` / `benchmark` arrive in Sprints 3–4).
* **Configs** (`configs/pinn_al.yaml`, `configs/baseline_nn.yaml`): paper
  facts pinned; every paper-unspecified slot left `null` and commented.
* **CI** (`.github/workflows/tests.yml`): pytest with the three markers.
* **Docs**: `equations.md` (+ §7 uncertainty ledger), `scientific_method.md`,
  `architecture.md`, `data_format.md`, `reproduction_protocol.md`.

## Gates (all green)

* `pytest` green in CI; markers `unit` / `integration` /
  `scientific_validation` enforced (`--strict-markers`).
* Physical constants verified against `scipy.constants`
  (`tests/unit/test_constants.py`).
* Recorded paper facts (parameter counts, energy shift, dataset sizes)
  verified by independent arithmetic
  (`tests/scientific_validation/test_paper_facts.py`).
* ASE ↔ schema round trip (`tests/integration/test_converters.py`).

## Explicitly out of scope in Sprint 0

Any physics or learning code — layers land in the sprint that needs them
(`docs/architecture.md`, "no speculative abstraction").
