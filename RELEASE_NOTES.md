# Release Notes

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

### Before Public Release

- Replace `TO_BE_ADDED` in `CITATION.cff` with the public repository URL.
- Create a release tag.
- Archive the release and add the DOI.
- Confirm external dataset source links and redistribution notes.
