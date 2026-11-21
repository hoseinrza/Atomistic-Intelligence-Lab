# Atomistic PINN Lab

A research-grade, modular Python framework that reproduces the **physically
informed neural network (PINN) bond-order potential** for aluminum introduced
in:

> G. P. Purja Pun, R. Batra, R. Ramprasad & Y. Mishin,
> *Physically informed artificial neural networks for atomistic modeling of
> materials*, **Nature Communications 10**, 2339 (2019).
> [doi:10.1038/s41467-019-10343-5](https://doi.org/10.1038/s41467-019-10343-5)

This is a scientific **reproduction** project: it implements the paper's
methodology as faithfully as the published information allows, keeps every
deviation explicit and labeled, and treats scientific correctness,
numerical correctness, and transparent documentation as higher priorities
than speed of implementation.

## The idea in one line

Instead of regressing atomic energies directly (`G_i → E_i`, conventional NN
potentials), a small neural network maps each atom's 60 local descriptors to
the **eight parameters of an analytical bond-order potential**
(`G_i → p_i → E_i`), so that extrapolation outside the training domain
remains governed by physics instead of neural-network guesswork.

## Reproduction scope

| Paper element | Reference | Status |
| --- | --- | --- |
| Unit conventions & data contracts | — | ✅ Sprint 0 |
| 60-dim Legendre/Gaussian descriptors (Eqs. 3–5) | Methods | ✅ Sprint 1 |
| Neighbor lists / PBC geometry | Methods | ✅ Sprint 1 |
| Analytical BOP with 8 NN-predicted parameters (Eqs. 6–12) | Methods | ✅ Sprint 2 |
| PINN network 60×15×15×8 + NN baseline 60×16×16×1 | Table 1 | ✅ Sprint 3 |
| DFP optimization + random restarts + Eq. (13) objective | Methods | ✅ Sprint 3 (real-data runs on ANI-Al executing) |
| 10-fold cross-validation | Methods | ✅ Sprint 4 machinery (dev-gate run executing) |
| Force evaluation by finite differences (post-training only) | Methods | ✅ Sprint 4 (full pipeline; autograd cross-check validated) |
| ASE calculator + basic MD | — | ✅ Sprint 4 machinery (NVE drift gate + seeded NVT; 500-atom demo pending trained checkpoint) |

Explicitly **not** done: forces are never used as a training or validation
target (paper methodology); DFP is not silently replaced by Adam/L-BFGS;
equations unspecified by the paper are flagged, not invented
(see [docs/equations.md](docs/equations.md) §7).

## Reproduction fidelity

The paper's DFT database (3,649 training supercells / 127,592 atoms;
26,425 testing supercells / 2,376,388 atoms) is **not publicly
downloadable** — it is available only via the paper's Supplementary
Information tables or on request from the corresponding author. Until it is
obtained, all training results are **methodological reproductions on
documented alternative data**, never exact reproductions. The learning
pipeline itself is validated on a **labeled synthetic database** (energies
produced by the physics layer under known parameters — pipeline validation
only, never presented as a reproduction result). Details and the
acquisition plan: [data/README.md](data/README.md) and
[docs/reproduction_protocol.md](docs/reproduction_protocol.md).

## Installation

Requires Python 3.11+.

```bash
git clone <this repository>
cd Atomistic-Intelligence-Lab
python3 -m venv .venv && source .venv/bin/activate
# CPU-only torch is sufficient (the paper itself trained on CPUs):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Usage

```bash
atomistic-pinn version   # package version
atomistic-pinn paper     # paper constants pinned in this project
atomistic-pinn train --config configs/pinn_al.yaml
```

`train` (Sprint 3) resolves the config's ledger decisions, builds the
database, runs the pre-fit + seeded-restart DFP training, and writes a
provenance-complete checkpoint plus a report under `results/`. The shipped
configs point at the labeled **synthetic** development database (pipeline
validation — not DFT data); point `data.source` at the alternative dataset
once acquired ([data/README.md](data/README.md)).

`evaluate` and `benchmark` (cross-validation, force metrics, MD) arrive with
Sprint 4:

```bash
atomistic-pinn evaluate --model checkpoints/pinn_al.pt
atomistic-pinn benchmark --config configs/cross_validation.yaml
```

## Project layout

```text
├── AGENTS.md         working conventions (commit discipline, fidelity rules)
├── configs/          experiment configurations (paper facts + labeled decisions)
├── data/             no datasets in git; see data/README.md
├── docs/             scientific method, equations, architecture, sprint records
├── src/atomistic_pinn/
│   ├── constants.py  units + physical + paper-derived constants
│   ├── exceptions.py error hierarchy
│   ├── cli.py        Typer CLI (version, paper, train)
│   ├── data/         schema (data contracts) + ASE converters
│   ├── geometry/     neighbor lists (full PBC image enumeration) + angles
│   ├── descriptors/  cutoff, radial Gaussians, Legendre, 60-dim builder
│   ├── physics/      BOP engine (Eqs. 6–12, torch) + FD/autograd forces
│   ├── models/       PINN 60×15×15×8 and baseline NN 60×16×16×1 (Table 1)
│   ├── training/     Eq. (13) objective, DFP, fixed-BOP pre-fit, trainer,
│   │                 provenance checkpoints, config validation, synthetic DB
│   └── reporting/    training reports (metrics + figures)
├── tests/
│   ├── unit/                 constants, schema, cutoff/Legendre, DFP, models
│   ├── integration/          ASE ↔ schema, neighbor lists, CLI runs
│   └── scientific_validation/ paper facts, invariances, closed forms,
│                              independent references, training gates
└── .github/workflows/tests.yml
```

Full module map and dependency rules: [docs/architecture.md](docs/architecture.md).

## Documentation

| Document | Content |
| --- | --- |
| [docs/equations.md](docs/equations.md) | all paper equations transcribed + the uncertainty ledger |
| [docs/scientific_method.md](docs/scientific_method.md) | why the method works, component-by-component |
| [docs/architecture.md](docs/architecture.md) | layers, module map, dependency rules, testing strategy |
| [docs/data_format.md](docs/data_format.md) | data contracts, units, energy conventions, leakage rules |
| [docs/reproduction_protocol.md](docs/reproduction_protocol.md) | fidelity levels, dataset plan, target metrics, reporting rules |
| [docs/dataset_composition.md](docs/dataset_composition.md) | transcribed + verified composition of the paper's DFT database (SI Tables 1–3) |
| [docs/md_plan.md](docs/md_plan.md) | molecular dynamics plan (Sprint 4): force route, protocol, validation |
| [docs/sprint0.md](docs/sprint0.md) … [docs/sprint4.md](docs/sprint4.md) | one document per sprint: delivered records (0–3), live plans (4) |

## Testing

```bash
pytest                 # everything
pytest -m unit         # fast subset
pytest -m scientific_validation
```

Physical constants are verified against `scipy.constants`; the recorded
paper facts (parameter counts, energy shift, dataset sizes) are verified by
independent arithmetic; later sprints add invariance, closed-form, and
finite-difference-vs-autograd tests. Reference values are never fabricated.

## Citation

See [CITATION.cff](CITATION.cff): please cite the original paper (and this
repository) if you use this software.

## License

[MIT](LICENSE). The reproduced methodology is from the open-access paper
(CC BY 4.0).
