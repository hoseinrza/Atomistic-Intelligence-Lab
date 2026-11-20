# Data

This directory intentionally contains **no datasets**; datasets are never
committed to the repository.

## Original paper dataset (status)

The paper's aluminum DFT database (3,649 training/validation supercells with
127,592 atoms; 26,425 testing supercells with 2,376,388 atoms) is **not
distributed with the paper**. Per the paper's Data availability statement,
supporting data are "available in the Supplementary Information file or from
the corresponding author upon reasonable request" (corresponding author:
Y. Mishin, George Mason University).

Action items:

- [ ] Obtain the Supplementary Information (Supplementary Tables 1-3 describe
  the dataset composition) from the journal page.
- [ ] Request the training/testing database from the corresponding author and
  record the outcome here (date, response, licensing).

Until one of these succeeds, any training results produced by this project are
**methodological reproductions on alternative data**, never exact
reproductions (see `docs/reproduction_protocol.md`).

## Alternative dataset workflow (methodological reproduction)

If the original data cannot be obtained, training will use an alternative
aluminum DFT dataset that satisfies these requirements:

1. Single-element Al, one consistent DFT setup across all structures
   (exchange-correlation functional, plane-wave cutoff, k-point density).
2. Raw total energies in eV with the package energy convention
   (see `docs/data_format.md`) and documented provenance.
3. Configuration diversity comparable to the paper: crystals under
   isotropic/uniaxial strain, surfaces and slabs, grain boundaries and
   stacking faults, point defects, small clusters, MD snapshots including
   high-temperature and part-liquid configurations.
4. Forces optional: they are not used for training (paper methodology) but are
   useful for post-hoc evaluation.
5. A license permitting redistribution of derived scientific results.

Candidate datasets will be surveyed in Sprint 1; the selection and its
rationale will be recorded in `docs/reproduction_protocol.md`.

## Layout (planned)

```text
data/raw/        unmodified downloaded or generated source data (git-ignored)
data/processed/  validated datasets in the package schema (git-ignored)
```
