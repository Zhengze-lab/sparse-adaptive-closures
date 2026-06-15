# Release Notes

## v0.1.2

Date: 2026-06-15

This release is created as a non-prerelease GitHub Release after Zenodo
integration was enabled. It is intended to trigger Zenodo archiving and DOI
minting for the experiment-code repository.

### Validation

- `python scripts/check_release_package.py --write-manifest`: passed.
- `python scripts/smoke_test.py`: passed before release preparation.

### Remaining Metadata

- Add the Zenodo DOI to `CITATION.cff`, `README.md`, and release notes after
  Zenodo finishes archiving this release.
- Confirm external dataset source links and redistribution notes before the
  final archival release.

## v0.1.1-staging

Date: 2026-06-15

This staging release is created after enabling the GitHub repository in Zenodo
so that Zenodo can archive the repository and mint a DOI. The experiment code,
frozen outputs, release QA script, manifest, data-source notes, and MIT license
are unchanged in scope from the staging package.

### Validation

- `python scripts/check_release_package.py --write-manifest`: passed.
- `python scripts/smoke_test.py`: passed before release preparation.

### Remaining Metadata

- Add the Zenodo DOI to `CITATION.cff`, `README.md`, and release notes after
  Zenodo finishes archiving this release.
- Confirm external dataset source links and redistribution notes before the
  final `v1.0.0` release.

## v0.1.0-staging

Date: 2026-06-15

This staging release packages the experiment code and reproducibility artifacts
for sparse adaptive closures in regime-aware gray-box ODE models.

### Included

- Synthetic ODE experiments E1--E5.
- Public real-system diagnostics E6--E9.
- Battery ECM and Silverbox boundary diagnostics.
- Coefficient-slot screening protocol.
- Ablation suite for slot choice, input choice, library family, noise backend,
  public-system comparators, and physical constraints.
- Frozen CSV/JSON outputs and PDF/SVG figures.
- Provenance records for run configuration and data sources.
- MIT license, Python dependency files, data-source notes, known limitations,
  and a lightweight smoke test.

### Excluded

- Large third-party raw datasets.
- Local caches and downloaded archives.
- Regenerable raster exports.
- Article-writing and publishing-administration materials.

### Staging Validation

- `python scripts/smoke_test.py`: passed.
- `python scripts/run_e6_cascaded_tanks_slot_oe.py`: passed.
- `python scripts/run_ablation_suite.py`: passed.

### Remaining Release Metadata

- Create a release tag.
- Archive the release and add the DOI.
- Confirm external dataset source links and redistribution notes.
