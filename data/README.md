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

- [x] Obtain the Supplementary Information (Supplementary Tables 1-3 describe
  the dataset composition) from the journal page. **Done 2026-11-21**: the SI
  PDF was downloaded (git-ignored copy: `data/raw/si_2019/si.pdf`); the full
  group-by-group composition is transcribed and arithmetically verified in
  [`docs/dataset_composition.md`](../docs/dataset_composition.md). Note: the
  SI contains composition only — no positions, no energies. The SI also
  reveals the provenance: most of the database was drawn from the
  Botu & Ramprasad database (PRB 92, 094306 / IJQC 115, 1074), which is
  itself not publicly distributed (verified 2026-11-21 — only the fitted
  "AGNI" LAMMPS potential files are public).
- [ ] Request the training/testing database from the corresponding author and
  record the outcome here (date, response, licensing).

Until one of these succeeds, any training results produced by this project are
**methodological reproductions on alternative data**, never exact
reproductions (see `docs/reproduction_protocol.md`).

## Assessed external datasets (2026-11-21)

| Source | What it offers | Verdict |
| --- | --- | --- |
| Venkatraman, Wilson & Montes de Oca Zapiain, *npj Comput. Mater.* **11**, 24 (2025) (physics-informed charge prediction for Al corrosion) | ReaxFF MD (LAMMPS) trajectories of Al/Al2O3/5M-chloride systems (4 species, ~41,000 atoms) with QEq **partial charges**; data "upon reasonable request" | **Not usable** for this reproduction: multi-species system (Al/O/H/Cl) vs. our single-component Al BOP (Eqs. 3–12), target property is per-atom charge (not total energy), energies are classical ReaxFF (not DFT), and data are on-request only. Kept only as a possible *methodological* reference (active learning / uncertainty) — out of scope. |
| Botu & Ramprasad database (the paper's actual data source, per the SI) | The very DFT database the 2019 paper drew from | **Not public** (verified 2026-11-21). Author request to R. Ramprasad (Georgia Tech) could be attempted alongside the Y. Mishin request. |

## Alternative dataset survey and selection (2026-11-21)

Survey per the criteria above (single-element Al, one consistent DFT setup,
raw eV energies, diversity, license). Public candidates considered:

| Candidate | Content | Criterion check |
| --- | --- | --- |
| **ANI-Al** (J. S. Smith et al., *Nat. Commun.* **12**, 92 (2021), "Automated discovery of a robust interatomic potential for aluminum") | 6,352 DFT calculations, **single-element Al**, 55–249 atoms per structure, PBE (VASP), energies + forces + cells; crystal and melt phases, active-learning sampled MD snapshots | 1 ✅ one consistent setup; 2 ✅ raw total energies (eV-scale, unshifted; unit convention to be pinned in the converter tests); 3 partial — crystals + melt + hot MD, but **no slabs/grain boundaries/stacking faults** (documented deviation); 4 ✅ forces present (unused for training, useful for evaluation); 5 ⚠️ GitHub repo has **no explicit license**; the ColabFit mirror subsets carry CC-BY-4.0 |
| MP-ALOE / OMat24 (ColabFit mirrors) | ~10⁶ DFT calculations incl. Al | Fails criterion 1 for this use (mixed structure provenance / multi-element focus); far larger than needed; keep as fallback |
| Materials Project / OQMD / JARVIS | Static relaxed structures with energies | Fails criterion 3 (no MD/liquid/defect-trajectory diversity) |

**Selection: ANI-Al** (Smith et al. 2021).

* Access route A (primary, complete): GitHub `atomistic-ml/ani-al`,
  `data/Al-data.tgz` — HDF5 (ANI format), one group per configuration with
  `species` / `coordinates` / `cell` / `energy` / `force` / `fermi` /
  provenance paths. Downloaded 2026-11-21 to `data/raw/ani_al/`
  (git-ignored), extracted 43 HDF5 files, verified by an independent
  counting pass: **6,352 configurations, species = {Al} only, 55–249
  atoms/structure, energies + forces present** (matches the paper's stated
  6,352 DFT calculations).
* Access route B (DOI-citable, CC-BY-4.0): ColabFit Exchange subsets
  `ANI-Al_NC2021-train` (2,779 configs; DOI 10.60732/af254882) and
  `ANI-Al_NC2021-test` (DOI 10.60732/d1e27447).
* License caveat (criterion 5): recorded as an open item — ask the authors
  for explicit dataset licensing when contacting them; until then, derived
  results are attributed per the source publication.
* Fidelity rule: all training results on this dataset are reported as
  "methodological reproduction (alternative data)" — never as exact
  reproduction (`docs/reproduction_protocol.md`).

Next step: build the ANI-Al → package-schema converter
(`atomistic_pinn.data`), validate it (units, energy convention, schema
round-trip), then run the Sprint 3 phase-3.5 development training on the
alternative data.

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
