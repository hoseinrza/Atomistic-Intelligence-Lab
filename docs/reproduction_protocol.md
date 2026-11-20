# Reproduction protocol

## Fidelity levels

| Level | Meaning | Requirements |
| --- | --- | --- |
| **Exact reproduction** | paper equations + paper's own database + paper hyperparameters | original 3,649-supercell database and 26,425-supercell test set obtained from the authors |
| **Methodological reproduction** | paper equations + documented alternative Al DFT data | alternative dataset meeting the criteria in `data/README.md`; deviations from unspecified paper details explicitly labeled |

**Rule:** results from alternative data are always reported as
"methodological reproduction (alternative data)" and never as an exact
reproduction. The fidelity level of every produced number is recorded in the
report that generates it.

## Dataset acquisition plan

1. Download the paper's Supplementary Information (Supplementary Tables 1–3
   describe the composition of the training and testing databases) from the
   journal page; extract the structure-group composition.
2. Request the databases from the corresponding author
   (Y. Mishin, George Mason University; see paper Data availability) and log
   the outcome in `data/README.md`.
3. If unsuccessful, survey alternative public Al DFT datasets against the
   criteria in `data/README.md`; select one, record the rationale and the
   provenance here, and proceed with the methodological reproduction.

## Protocol steps (sprint-mapped)

| Sprint | Step | Gate |
| --- | --- | --- |
| 0 | units, constants, data contracts, CI | all tests green |
| 1 | geometry + descriptors | invariance tests, cutoff smoothness, dim-60, slow-reference agreement |
| 2 | BOP engine | closed-form single-pair energies, screening spheroid behavior, FD vs. autograd forces, crystal energy consistency |
| 3 | PINN + NN + Eq. (13) + DFP + restarts | objective decreases; restart reproducibility under seed; checkpoint round-trip |
| 4 | 10-fold CV, test evaluation, FD forces, reports, ASE calculator, MD demo | metrics against reference bands (below); full experiment reproducible from one config |

## Target metrics (from the paper)

Energy RMSE (meV/atom), Table 1:

| Model | train | validation |
| --- | --- | --- |
| NN  | 3.36 | 3.85 |
| NN′ | 3.62 | 3.54 |
| PINN | 3.46 | 3.59 |

Paper's stated goal band: 3–4 meV/atom for both training and validation.

Force RMSE (eV/Å, paper text, approximate): validation — PINN ≈ 0.1 vs
NN ≈ 0.2; testing — PINN ≈ 0.1 vs NN ≈ 0.4 (dislocation) / 0.6 (HCP MD).

For the methodological reproduction (alternative data), these are **reference
bands, not pass/fail gates**: differences will be measured, reported, and
discussed with the dataset differences in mind.

## Reporting requirements

Every experiment run must log:

* the full resolved configuration (YAML),
* git commit hash and package version,
* master seed and all derived seeds (folds, restarts),
* dataset hash and the energy convention used,
* per-fold and averaged metrics,
* checkpoint file hashes.

Reports (parity plots, error histograms, per-group errors, CV summaries) are
generated into `results/` by `atomistic_pinn.reporting` and are
git-ignored.

## Open verification items before Sprint 4

All items in `docs/equations.md` §7 must be either resolved against the
Supplementary Information / author response, or explicitly labeled as
implementation decisions in the final report. Unresolved items at reporting
time are listed verbatim in the reproduction report.
