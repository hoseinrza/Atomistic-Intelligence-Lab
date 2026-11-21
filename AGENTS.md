# Working conventions (agent memory)

Instructions for any agent (or human) working in this repository. Keep this
file current; it is the durable record of how the owner wants the project run.

## Commit discipline (owner instruction, 2026-11-21)

**Commit after every change.** Every completed unit of work — a phase, a fix,
a doc update, a test group — is committed immediately as its own small,
descriptively-messaged commit. Do not accumulate large uncommitted batches.

## Sprint workflow

* Work follows `docs/sprints.md` and the per-sprint plans
  (`docs/sprint0.md` … `docs/sprint4.md`, plus `docs/sprint3.md`,
  `docs/md_plan.md`).
* Start a sprint only on the owner's explicit go-ahead ("شروع کن" / explicit
  instruction). Report per sprint: files, functionality, test results,
  limitations, next step.
* Status tables in `docs/sprints.md` / per-sprint records are updated as part
  of each sprint.

## Scientific fidelity rules (project constitution)

* Source of truth: Purja Pun et al., Nat. Commun. 10:2339 (2019); equations
  transcribed in `docs/equations.md`.
* Values the paper does not specify are **never filled silently** — they are
  ledger items (`docs/equations.md` §7) resolved as explicitly labeled
  decisions, named in configs with their §7.x item.
* Never fabricate reference values; validate against independent references
  (closed forms, scipy/ASE, two independently implemented methods).
* Units: Å, eV, eV/Å everywhere (`src/atomistic_pinn/constants.py`).
* Fidelity levels: results on non-original data are always reported as
  "methodological reproduction (alternative data)", never as exact
  reproduction. Synthetic databases are pipeline validation only.
* Forces are never a training target (paper methodology).
* Every stochastic step is seeded; configs are validated; errors carry the
  ledger/contract pointer.

## Language

* The owner communicates in Persian (Farsi). Code, comments, docs, and
  commit messages are in English; replies to the owner are in Persian.

## Environment

* Python 3.11 in `.venv` (uv-managed; see README Installation).
* Quality gates before every commit: `ruff format .`, `ruff check .`,
  `mypy`, `pytest` — all must be green.
* `data/raw/`, `data/processed/`, `checkpoints/`, `results/` are
  git-ignored; datasets and trained models are never committed.
