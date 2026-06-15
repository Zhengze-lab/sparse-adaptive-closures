# Known Limitations

This repository is intended for reproducing the experiments, not for serving as
a polished general-purpose modeling package.

## Scope

- The code targets coefficient-slot and sparse-closure experiments for ODE
  gray-box models.
- It does not automatically discover arbitrary physical slots in arbitrary real
  systems.
- Slot screening is implemented for the synthetic E3--E5 cases and reinforced
  for selected public real-system diagnostics.

## Data

- Large raw third-party datasets are not bundled.
- Some public-system scripts require provider-side downloads, credentials, or a
  local cache.
- Frozen derived results are included so that users can audit outputs even
  before rerunning every external-data experiment.

## Modeling

- Public real-system cases include both positive cases and boundary cases.
- Boundary cases are retained to show when a single coefficient slot is too
  weak for missing state structure, calibration shift, or unmeasured
  disturbances.
- Several scripts are research prototypes with explicit metrics and provenance,
  not stable library APIs.

## Runtime

- The smoke test is lightweight and uses only synthetic experiments.
- Full ablations and public-system scripts can take substantially longer and may
  depend on network access for data acquisition.
