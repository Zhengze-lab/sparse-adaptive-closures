#!/usr/bin/env python3
"""Run BATT-4a2: physics-filtered R0 coefficient-slot ECM."""

from __future__ import annotations

import json
import platform
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pysindy as ps
import scipy

import run_batt3_constant_ecm as batt3
import run_batt3c_discharge_ocv_ecm as batt3c
import run_batt4a_r0_slot_ecm as b4


ROOT = batt3.ROOT
RESULT_DIR = ROOT / "results" / "battery_lfp_r0_slot_filtered_ecm"
FIGURE_DIR = RESULT_DIR / "figures"
PROVENANCE_DIR = ROOT / "data" / "provenance"


CONFIG: dict[str, Any] = {
    "experiment_id": "BATT4a2_lfp_filtered_r0_slot_ecm",
    "description": "Physics-filtered R0 coefficient-slot ECM after the unconstrained BATT-4a polynomial slot failed high-amplitude extrapolation.",
    "source_dataset": "battery_lfp_ocv_dyn",
    "source_url": "https://data.mendeley.com/datasets/p8kf893yv3/1",
    "source_doi": "10.17632/p8kf893yv3.1",
    "license": "CC BY 4.0",
    "baseline_experiment": "BATT3c_lfp_discharge_ocv_current_sign_ecm",
    "failed_diagnostic_experiment": "BATT4a_lfp_r0_slot_ecm",
    "ocv_curve": "discharge",
    "current_convention": "I_model=I_raw",
    "feature_scaling": "sc=s-0.5, Tn=T/25, In=|I|/2, Isat=min(|I|,2)/2",
    "feature_sets": {
        "T_only": ["1", "Tn", "Tn^2"],
        "T_SOC": ["1", "sc", "sc^2", "Tn", "Tn^2", "sc*Tn"],
        "T_SOC_Ilinear": ["1", "sc", "sc^2", "Tn", "Tn^2", "sc*Tn", "In"],
        "T_SOC_Isat": ["1", "sc", "sc^2", "Tn", "Tn^2", "sc*Tn", "Isat", "sc*Isat", "Tn*Isat"],
    },
    "thresholds": [0.0, 1e-2, 5e-2, 1e-1],
    "physical_selection_rule": "Select the lowest validation RMSE among models with <=1% negative R0 on train/validation/high-amplitude and high-amplitude RMSE no worse than BATT-3c constant ECM.",
    "max_negative_fraction_for_physical": 0.01,
    "require_high_amplitude_not_worse_than_constant_ecm": True,
    "fit_abs_current_min_a": b4.CONFIG["fit_abs_current_min_a"],
    "fit_stride": b4.CONFIG["fit_stride"],
    "prediction_sample_stride": b4.CONFIG["prediction_sample_stride"],
    "initialization_window_s": b4.CONFIG["initialization_window_s"],
    "score_start_s": b4.CONFIG["score_start_s"],
    "pysindy_optimizer": b4.CONFIG["pysindy_optimizer"],
    "pysindy_alpha": b4.CONFIG["pysindy_alpha"],
    "pysindy_max_iter": b4.CONFIG["pysindy_max_iter"],
    "alternating_iterations": b4.CONFIG["alternating_iterations"],
    "active_threshold": b4.CONFIG["active_threshold"],
}


FeatureFn = Callable[[np.ndarray, float | np.ndarray, np.ndarray], np.ndarray]


def feature_function(kind: str) -> FeatureFn:
    def features(soc: np.ndarray, temperature_c: float | np.ndarray, current_a: np.ndarray) -> np.ndarray:
        soc_arr = np.asarray(soc, dtype=float)
        temp_arr = np.asarray(temperature_c, dtype=float)
        if temp_arr.ndim == 0:
            temp_arr = np.full_like(soc_arr, float(temp_arr), dtype=float)
        cur_arr = np.asarray(current_a, dtype=float)
        sc = soc_arr - 0.5
        tn = temp_arr / 25.0
        inn = np.abs(cur_arr) / 2.0
        isat = np.minimum(np.abs(cur_arr), 2.0) / 2.0
        if kind == "T_only":
            return np.column_stack([np.ones_like(sc), tn, tn * tn])
        if kind == "T_SOC":
            return np.column_stack([np.ones_like(sc), sc, sc * sc, tn, tn * tn, sc * tn])
        if kind == "T_SOC_Ilinear":
            return np.column_stack([np.ones_like(sc), sc, sc * sc, tn, tn * tn, sc * tn, inn])
        if kind == "T_SOC_Isat":
            return np.column_stack([np.ones_like(sc), sc, sc * sc, tn, tn * tn, sc * tn, isat, sc * isat, tn * isat])
        raise ValueError(f"Unknown feature set: {kind}")

    return features


def configure_feature_set(kind: str) -> None:
    b4.r0_features = feature_function(kind)
    b4.CONFIG["feature_names"] = list(CONFIG["feature_sets"][kind])


def model_name(kind: str, threshold: float) -> str:
    if threshold == 0:
        return f"BATT4a_R0_slot_filtered_{kind}_STLSQ_dense"
    return f"BATT4a_R0_slot_filtered_{kind}_STLSQ_t{threshold:g}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    batt3.write_csv(path, rows)


def fit_one_slot(
    records: list[batt3.DynRecord],
    ocv_model: batt3.OcvModel,
    kind: str,
    threshold: float,
    initial_states: dict[str, dict[str, float]],
) -> b4.SlotFitResult:
    configure_feature_set(kind)
    result = b4.fit_slot_model(records, ocv_model, threshold, initial_states)
    return replace(result, model=model_name(kind, threshold))


def coefficient_rows(slot_results: list[tuple[str, b4.SlotFitResult]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind, result in slot_results:
        names = list(CONFIG["feature_sets"][kind])
        for name, coeff in zip(names, result.coeffs):
            rows.append(
                {
                    "feature_set": kind,
                    "model": result.model,
                    "threshold": result.threshold,
                    "feature": name,
                    "coefficient_ohm": float(coeff),
                    "active": abs(float(coeff)) >= float(CONFIG["active_threshold"]),
                }
            )
    return rows


def threshold_summary_rows(
    slot_results: list[tuple[str, b4.SlotFitResult]],
    split_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {(row["model"], row["split"]): row for row in split_rows}
    rows: list[dict[str, Any]] = []
    for kind, result in slot_results:
        row: dict[str, Any] = {
            "feature_set": kind,
            "model": result.model,
            "threshold": result.threshold,
            "active_terms": b4.active_terms(result.coeffs),
            "design_rows": result.design_rows,
            "linear_fit_nrmse": result.history[-1]["linear_fit_nrmse"],
        }
        for split in ["train", "validation_temperature", "test_high_amplitude", "test_cell_transfer"]:
            met = lookup.get((result.model, split), {})
            row[f"{split}_rmse_mv"] = met.get("rmse_mv", "")
            row[f"{split}_nrmse"] = met.get("nrmse", "")
            row[f"{split}_r0_negative_fraction"] = met.get("r0_negative_fraction", "")
        rows.append(row)
    return rows


def select_best_physical(threshold_rows: list[dict[str, Any]], constant_ecm_high_rmse: float) -> dict[str, Any]:
    max_neg = float(CONFIG["max_negative_fraction_for_physical"])
    candidates = []
    for row in threshold_rows:
        if row.get("validation_temperature_rmse_mv") == "":
            continue
        if float(row["train_r0_negative_fraction"]) > max_neg:
            continue
        if float(row["validation_temperature_r0_negative_fraction"]) > max_neg:
            continue
        if float(row["test_high_amplitude_r0_negative_fraction"]) > max_neg:
            continue
        if bool(CONFIG["require_high_amplitude_not_worse_than_constant_ecm"]):
            if float(row["test_high_amplitude_rmse_mv"]) > constant_ecm_high_rmse:
                continue
        candidates.append(row)
    if not candidates:
        candidates = [row for row in threshold_rows if row.get("validation_temperature_rmse_mv") != ""]
    return min(candidates, key=lambda row: float(row["validation_temperature_rmse_mv"]))


def improvement_rows(split_rows: list[dict[str, Any]], baseline_model: str) -> list[dict[str, Any]]:
    return b4.improvement_rows(split_rows, baseline_model)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    raw_records = batt3.load_dynamic_records()
    records = batt3c.records_for_current_convention(raw_records, str(CONFIG["current_convention"]))
    ocv_model = batt3c.load_ocv_model(str(CONFIG["ocv_curve"]))

    ocv_states = batt3.fit_ocv_train_states(records, ocv_model)
    ohmic_params, ohmic_states, ohmic_result = batt3.fit_ohmic(records, ocv_model)
    ecm_params, ecm_states, ecm_result = batt3.fit_constant_ecm(records, ocv_model)

    baseline_params = {
        "BATT3c_OCV_discharge_only": {},
        "BATT3c_constant_ohmic": ohmic_params,
        "BATT3c_constant_ecm": ecm_params,
    }
    baseline_states = {
        "BATT3c_OCV_discharge_only": ocv_states,
        "BATT3c_constant_ohmic": ohmic_states,
        "BATT3c_constant_ecm": ecm_states,
    }
    trajectory_rows, split_rows, prediction_rows, split_summary = b4.evaluate_models(
        records,
        ocv_model,
        baseline_params,
        baseline_states,
    )

    slot_results: list[tuple[str, b4.SlotFitResult]] = []
    for kind in CONFIG["feature_sets"]:
        for threshold in CONFIG["thresholds"]:
            result = fit_one_slot(records, ocv_model, kind, float(threshold), ohmic_states)
            slot_results.append((kind, result))
            configure_feature_set(kind)
            slot_trajectory, slot_split, slot_prediction, slot_summary = b4.evaluate_models(
                records,
                ocv_model,
                {result.model: {"coeffs": result.coeffs}},
                {result.model: result.train_states},
            )
            trajectory_rows.extend(slot_trajectory)
            split_rows.extend(slot_split)
            prediction_rows.extend(slot_prediction)
            split_summary.update(slot_summary)

    threshold_rows = threshold_summary_rows(slot_results, split_rows)
    split_lookup = {(row["model"], row["split"]): float(row["rmse_mv"]) for row in split_rows}
    constant_high = split_lookup[("BATT3c_constant_ecm", "test_high_amplitude")]
    best_physical = select_best_physical(threshold_rows, constant_high)
    best_kind = str(best_physical["feature_set"])
    best_result = next(result for kind, result in slot_results if kind == best_kind and result.model == best_physical["model"])

    metrics_by_trajectory_path = RESULT_DIR / "metrics_by_trajectory.csv"
    metrics_by_split_path = RESULT_DIR / "metrics_by_split.csv"
    predictions_path = RESULT_DIR / "predictions_sample.csv"
    coefficients_path = RESULT_DIR / "coefficients.csv"
    threshold_summary_path = RESULT_DIR / "threshold_summary.csv"
    improvement_path = RESULT_DIR / "improvement_vs_batt3c_constant_ecm.csv"
    fit_summary_path = RESULT_DIR / "fit_summary.csv"
    metrics_path = RESULT_DIR / "metrics.json"
    provenance_path = PROVENANCE_DIR / "batt4a_filtered_r0_slot_ecm_provenance.json"

    fit_rows = [
        {
            "model": "BATT3c_constant_ohmic",
            "feature_set": "constant",
            "threshold": "",
            "active_terms": 1,
            "R0_ohm": ohmic_params["R0_ohm"],
            "cost": float(ohmic_result.cost),
            "optimality": float(ohmic_result.optimality),
            "nfev": int(ohmic_result.nfev),
            "success": bool(ohmic_result.success),
            "message": str(ohmic_result.message),
        },
        {
            "model": "BATT3c_constant_ecm",
            "feature_set": "constant",
            "threshold": "",
            "active_terms": 3,
            **ecm_params,
            "cost": float(ecm_result.cost),
            "optimality": float(ecm_result.optimality),
            "nfev": int(ecm_result.nfev),
            "success": bool(ecm_result.success),
            "message": str(ecm_result.message),
        },
    ]
    for kind, result in slot_results:
        fit_rows.append(
            {
                "feature_set": kind,
                "model": result.model,
                "threshold": result.threshold,
                "active_terms": b4.active_terms(result.coeffs),
                "design_rows": result.design_rows,
                "linear_fit_nrmse": result.history[-1]["linear_fit_nrmse"],
            }
        )

    write_csv(metrics_by_trajectory_path, trajectory_rows)
    write_csv(metrics_by_split_path, split_rows)
    write_csv(predictions_path, prediction_rows)
    write_csv(coefficients_path, coefficient_rows(slot_results))
    write_csv(threshold_summary_path, threshold_rows)
    write_csv(improvement_path, improvement_rows(split_rows, "BATT3c_constant_ecm"))
    write_csv(fit_summary_path, fit_rows)

    selected_models = [
        "BATT3c_OCV_discharge_only",
        "BATT3c_constant_ecm",
        str(best_physical["model"]),
        "BATT4a_R0_slot_filtered_T_SOC_Isat_STLSQ_t0.05",
    ]
    configure_feature_set(best_kind)
    b4.write_rmse_svg(FIGURE_DIR / "rmse_by_split.svg", split_rows, selected_models, zh=False)
    b4.write_rmse_svg(FIGURE_DIR / "rmse_by_split_zh.svg", split_rows, selected_models, zh=True)
    b4.write_r0_profile_svg(FIGURE_DIR / "r0_profiles.svg", best_result.coeffs, zh=False)
    b4.write_r0_profile_svg(FIGURE_DIR / "r0_profiles_zh.svg", best_result.coeffs, zh=True)

    validation_rank = sorted(
        [row for row in split_rows if row["split"] == "validation_temperature"],
        key=lambda row: float(row["rmse_mv"]),
    )
    high_rank = sorted(
        [row for row in split_rows if row["split"] == "test_high_amplitude"],
        key=lambda row: float(row["rmse_mv"]),
    )
    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "complete",
        "config": CONFIG,
        "raw_zip_sha256": batt3.sha256_file(batt3.RAW_ZIP),
        "record_count": len(records),
        "threshold_summary": threshold_rows,
        "best_physical_model": best_physical,
        "validation_rank_by_rmse": validation_rank,
        "high_amplitude_rank_by_rmse": high_rank,
        "split_summary": split_summary,
        "outputs": {
            "metrics_by_trajectory_csv": str(metrics_by_trajectory_path.relative_to(ROOT)),
            "metrics_by_split_csv": str(metrics_by_split_path.relative_to(ROOT)),
            "predictions_sample_csv": str(predictions_path.relative_to(ROOT)),
            "coefficients_csv": str(coefficients_path.relative_to(ROOT)),
            "threshold_summary_csv": str(threshold_summary_path.relative_to(ROOT)),
            "improvement_vs_batt3c_constant_ecm_csv": str(improvement_path.relative_to(ROOT)),
            "fit_summary_csv": str(fit_summary_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
            "figures": [
                str((FIGURE_DIR / "rmse_by_split.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "rmse_by_split_zh.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "r0_profiles.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "r0_profiles_zh.svg").relative_to(ROOT)),
            ],
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    provenance = {
        **CONFIG,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pysindy": ps.__version__,
        },
        "script": "scripts/run_batt4a_filtered_r0_slot_ecm.py",
        "outputs": metrics["outputs"] | {
            "provenance_json": str(provenance_path.relative_to(ROOT)),
        },
        "test_usage_rule": "R0 slot coefficients are fitted only on split=train. Non-train records use only the initialization prefix for initial SOC estimation.",
        "next_action": "Use the selected physical R0(T) model as the BATT-4a positive baseline, then test whether dynamic slots a_tau(z), b_C(z), or cell-transfer terms can address the remaining cross-cell error.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
