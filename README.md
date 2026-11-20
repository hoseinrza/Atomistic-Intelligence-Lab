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
| 60-dim Legendre/Gaussian descriptors (Eqs. 3–5) | Methods | planned — Sprint 1 |
| Neighbor lists / PBC geometry | Methods | planned — Sprint 1 |
| Analytical BOP with 8 NN-predicted parameters (Eqs. 6–12) | Methods | planned — Sprint 2 |
| PINN network 60×15×15×8 + NN baseline 60×16×16×1 | Table 1 | planned — Sprint 3 |
| DFP optimization + random restarts + Eq. (13) objective | Methods | planned — Sprint 3 |
| 10-fold cross-validation | Methods | planned — Sprint 4 |
| Force evaluation by finite differences (post-training only) | Methods | planned — Sprint 4 |
| ASE calculator + basic MD | — | planned — Sprint 4 |

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
documented alternative data**, never exact reproductions. Details and the
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

## Usage (Sprint 0)

```bash
atomistic-pinn version   # package version
atomistic-pinn paper     # paper constants pinned in this project
```

`train`, `evaluate`, and `benchmark` commands arrive with Sprints 3–4:

```bash
atomistic-pinn train --config configs/pinn_al.yaml
atomistic-pinn evaluate --model checkpoints/pinn_al.pt
atomistic-pinn benchmark --config configs/cross_validation.yaml
```

## Project layout

```text
├── configs/          experiment configurations (paper values, unknowns = null)
├── data/             no datasets in git; see data/README.md
├── docs/             scientific method, equations, architecture, protocols
├── src/atomistic_pinn/
│   ├── constants.py  units + physical + paper-derived constants
│   ├── exceptions.py error hierarchy
│   ├── cli.py        Typer CLI
│   └── data/         schema (data contracts) + ASE converters
├── tests/
│   ├── unit/                 constants, schema validation
│   ├── integration/          ASE ↔ schema round trips
│   └── scientific_validation/ paper-facts cross-checks; grows per sprint
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
