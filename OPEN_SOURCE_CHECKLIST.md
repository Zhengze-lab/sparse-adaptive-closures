# Open-Source Release Checklist

## Required Before Public Release

- [x] Add the selected open-source license.
- [x] Add `environment.yml`.
- [x] Add `DATA_SOURCES.md`.
- [x] Add `KNOWN_LIMITATIONS.md`.
- [x] Add and run `python scripts/smoke_test.py` in the staging folder.
- [x] Add and run `python scripts/check_release_package.py --write-manifest`.
- [x] Update `CITATION.cff` with the final repository URL.
- [x] Create a public release tag.
- [x] Archive the release on Zenodo and add the DOI.
- [ ] Check third-party dataset redistribution terms.
- [ ] Confirm `DATA_SOURCES.md` against the final public dataset links.
- [ ] Run `python scripts/smoke_test.py` from a clean checkout.
- [ ] Run `python scripts/run_e3_local_linearization_pendulum.py` from a clean checkout.
- [ ] Run `python scripts/run_slot_screening.py` from a clean checkout.
- [ ] Run `python scripts/run_ablation_suite.py` from a clean checkout.

## Current Staging Validation

- `python scripts/smoke_test.py`: passed.
- `python scripts/run_e6_cascaded_tanks_slot_oe.py`: passed.
- `python scripts/run_ablation_suite.py`: passed.
- `python scripts/check_release_package.py --write-manifest`: passed.
- Raw third-party archives are not bundled; `data/raw/` contains only policy notes.
- No Python bytecode caches or regenerable TIFF/PNG exports are bundled.

## Suggested Repository Settings

- Default branch: `main`
- Release tag format: `v1.0.0`
- Archive title: `Sparse adaptive closures for regime-aware gray-box ODEs: experimental code`
- Include generated synthetic data and frozen metrics in the archive.
- Exclude raw third-party datasets unless redistribution is explicitly permitted.
