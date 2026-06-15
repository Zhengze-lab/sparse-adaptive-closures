# Reproduction Guide

This guide gives a practical route for reproducing the experimental outputs.

## 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate sparse-adaptive-closures
```

## 2. Quick Smoke Test

This route avoids third-party raw datasets and should finish quickly.

```bash
python scripts/smoke_test.py
```

Expected key outputs:

- `results/e3_local_linearization_pendulum/metrics.json`
- `results/slot_screening/slot_screening_decisions.csv`

## 3. Core Synthetic Cases

These cases require no third-party raw data.

```bash
python scripts/run_e3_local_linearization_pendulum.py
python scripts/run_e4_velocity_drag.py
python scripts/run_e5_monod_rate_slot.py
python scripts/run_slot_screening.py
```

Expected key outputs:

- `results/e3_local_linearization_pendulum/metrics.json`
- `results/e4_velocity_drag/metrics.json`
- `results/e5_monod_rate_slot/metrics.json`
- `results/slot_screening/slot_screening_decisions.csv`

## 4. Full Ablation Suite

```bash
python scripts/run_ablation_suite.py
```

Expected key outputs:

- `results/ablation_suite/*.csv`

The ablation suite uses frozen derived outputs for some public-system and
battery diagnostics. If a required derived output is missing, rerun the
corresponding experiment listed below.

## 5. Public Real-System Cases

The public real-system scripts may need provider-side downloads or local caches:

```bash
python scripts/run_e6_cascaded_tanks_slot_oe.py
python scripts/run_e7_emps_friction_slot.py
python scripts/run_e8_pmsm_temperature_slot.py
python scripts/run_e9_ornl_building_thermal_slot.py
python scripts/run_public_real_protocol_reinforcement.py
```

See `DATA_SOURCES.md`, `data/raw/README.md`, and the corresponding
`data/provenance/*` files for source records.

## 6. Battery and Silverbox Boundary Cases

Battery and Silverbox scripts are retained for completeness but require third-party datasets and can take longer:

```bash
python scripts/audit_battery_lfp_ocv_dyn.py
python scripts/run_batt2_ocv_calibration.py
python scripts/run_batt3c_discharge_ocv_ecm.py
python scripts/run_batt4a_filtered_r0_slot_ecm.py
python scripts/run_batt4c_narrow_dynamic_pilot.py
python scripts/run_batt4d_cell_transfer_alignment.py
python scripts/audit_silverbox_data.py
python scripts/run_silverbox_sb2b_windowed_oe.py
```

## Notes

- Random seeds and run metadata are recorded in `data/provenance/`.
- Frozen CSV/JSON outputs are bundled for auditability.
- Large raw external data and regenerable raster exports are intentionally excluded.
- Known limitations are listed in `KNOWN_LIMITATIONS.md`.
- Release-package QA can be run with `python scripts/check_release_package.py`.
- File checksums are listed in `MANIFEST.sha256`.
