# Raw Data Policy

Large third-party raw datasets are not bundled in this experimental-code repository.
See `../../DATA_SOURCES.md` for the consolidated source table.

Reasons:

- several public datasets are large;
- redistribution terms differ by provider;
- raw data should be fetched from the original source of record when possible.

The scripts and provenance files record the dataset sources used by the experiments. The main external sources are:

| Case | Dataset | Source record |
| --- | --- | --- |
| E6 Cascaded Tanks | Cascaded Tanks benchmark | Dataset DOI `10.4121/12960104`; source records in `data/provenance/e6_cascaded_tanks_slot_oe_provenance.json` |
| E7 EMPS | Electro-mechanical positioning system benchmark | Source records in `data/provenance/e7_emps_friction_slot_provenance.json` |
| E8 PMSM thermal | Electric Motor Temperature dataset | Source records in `data/provenance/e8_pmsm_temperature_slot_provenance.json` |
| E9 ORNL building | ORNL flexible research platform building data | Dataset DOI `10.6084/m9.figshare.20520438` |
| Battery ECM | LiFePO4 OCV/DYN dataset | Mendeley Data DOI `10.17632/p8kf893yv3.1` |
| Silverbox | Nonlinear Benchmarks Silverbox | ECC benchmark DOI `10.23919/ECC.2013.6669201` |

Before public release, add exact download commands or provider-specific instructions here if redistribution is not allowed.
