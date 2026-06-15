#!/usr/bin/env python3
"""Run BATT-4d: cell-transfer/SOC/OCV alignment diagnostics."""

from __future__ import annotations

import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.optimize import least_squares

import run_batt3_constant_ecm as batt3
import run_batt3c_discharge_ocv_ecm as batt3c
import run_batt4a_r0_slot_ecm as b4
import run_batt4a_filtered_r0_slot_ecm as b4f
import run_batt4b_r0_residual_diagnostics as b4b
import run_batt4c_narrow_dynamic_pilot as b4c


ROOT = batt3.ROOT
RESULT_DIR = ROOT / "results" / "battery_lfp_cell_transfer_alignment"
FIGURE_DIR = RESULT_DIR / "figures"
PROVENANCE_DIR = ROOT / "data" / "provenance"
BATT4C_METRICS = ROOT / "results" / "battery_lfp_narrow_dynamic_pilot" / "metrics.json"


CONFIG: dict[str, Any] = {
    "experiment_id": "BATT4d_lfp_cell_transfer_alignment",
    "description": "Diagnose whether the A003 cell-transfer failure is dominated by SOC/OCV/cell offset rather than R0(T) or q_tau dynamics.",
    "source_dataset": "battery_lfp_ocv_dyn",
    "source_url": "https://data.mendeley.com/datasets/p8kf893yv3/1",
    "source_doi": "10.17632/p8kf893yv3.1",
    "license": "CC BY 4.0",
    "baseline_experiment": "BATT4a2_lfp_filtered_r0_slot_ecm",
    "dynamic_experiment": "BATT4c_lfp_narrow_dynamic_pilot",
    "target_split": "test_cell_transfer",
    "target_record": "A003_25_+25C",
    "selected_r0_model": b4b.CONFIG["selected_r0_model"],
    "selected_feature_set": b4b.CONFIG["selected_feature_set"],
    "ocv_curve": "discharge",
    "current_convention": "I_model=I_raw",
    "score_start_s": batt3.CONFIG["score_start_s"],
    "initialization_window_s": batt3.CONFIG["initialization_window_s"],
    "prediction_sample_stride": 60,
    "soc_delta_grid": [-0.60, 0.20, 0.005],
    "s0_grid": [0.20, 0.99, 0.01],
    "q_grid_ah": [1.60, 3.20, 0.05],
    "diagnostic_warning": "Rows with uses_scored_target=true are oracle diagnostics and must not be reported as deployable prediction models.",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    batt3.write_csv(path, rows)


def grid_values(start: float, stop: float, step: float) -> np.ndarray:
    n = int(round((stop - start) / step))
    values = start + step * np.arange(n + 1, dtype=float)
    return np.round(values, 10)


def read_dynamic_selection() -> tuple[float, float]:
    with BATT4C_METRICS.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    selected = metrics["selected_by_validation"]
    return float(selected["tau_s"]), float(selected["dynamic_gain_ohm"])


def load_base_context() -> dict[str, Any]:
    b4f.configure_feature_set(str(CONFIG["selected_feature_set"]))
    raw_records = batt3.load_dynamic_records()
    records = batt3c.records_for_current_convention(raw_records, str(CONFIG["current_convention"]))
    target = next(record for record in records if record.label == str(CONFIG["target_record"]))
    ocv_model = batt3c.load_ocv_model(str(CONFIG["ocv_curve"]))
    coeffs = b4b.read_selected_r0_coeffs()
    states = b4b.fit_r0t_states(records, ocv_model, coeffs)
    state = states[target.label]
    base_pred, soc, r0 = b4.predict_slot_voltage(target, ocv_model, coeffs, float(state["s0"]))
    tau_s, gain = read_dynamic_selection()
    q_tau = b4b.filtered_current(target.time_s, target.current_a, tau_s)
    dynamic_pred = base_pred - gain * q_tau
    score_idx = batt3.score_indices(target)
    prefix_idx = np.flatnonzero(target.time_s <= float(CONFIG["initialization_window_s"]))
    valid_score = np.isfinite(base_pred[score_idx]) & np.isfinite(target.voltage_v[score_idx])
    valid_prefix = np.isfinite(base_pred[prefix_idx]) & np.isfinite(target.voltage_v[prefix_idx])
    return {
        "records": records,
        "record": target,
        "ocv_model": ocv_model,
        "coeffs": coeffs,
        "state": state,
        "base_pred": base_pred,
        "dynamic_pred": dynamic_pred,
        "soc": soc,
        "r0": r0,
        "q_tau": q_tau,
        "tau_s": tau_s,
        "dynamic_gain_ohm": gain,
        "score_idx": score_idx[valid_score],
        "prefix_idx": prefix_idx[valid_prefix],
    }


def prediction_from_soc(
    record: batt3.DynRecord,
    ocv_model: batt3.OcvModel,
    coeffs: np.ndarray,
    soc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r0 = b4.r0_from_coeffs(soc, float(record.temperature_c), record.current_a, coeffs)
    pred = ocv_model(soc, float(record.temperature_c)) - r0 * record.current_a
    return pred, r0


def metric_row(
    name: str,
    record: batt3.DynRecord,
    pred: np.ndarray,
    idx: np.ndarray,
    *,
    correction_family: str,
    calibration_scope: str,
    uses_scored_target: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    y = record.voltage_v[idx]
    p = pred[idx]
    row = {
        "model": name,
        "correction_family": correction_family,
        "calibration_scope": calibration_scope,
        "uses_scored_target": bool(uses_scored_target),
        "n_eval": int(len(idx)),
        **batt3.error_metrics(y, p),
    }
    if extra:
        row.update(extra)
    return row


def add_constant_offset(pred: np.ndarray, residual_v: np.ndarray) -> tuple[np.ndarray, float]:
    offset = float(np.mean(residual_v))
    return pred - offset, offset


def scan_soc_delta(context: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    record = context["record"]
    score_idx = context["score_idx"]
    ocv_model = context["ocv_model"]
    coeffs = context["coeffs"]
    base_soc = context["soc"]
    rows: list[dict[str, Any]] = []
    best_no_bias: dict[str, Any] | None = None
    best_with_bias: dict[str, Any] | None = None
    best_no_bias_pred: np.ndarray | None = None
    best_with_bias_pred: np.ndarray | None = None
    for delta in grid_values(*[float(v) for v in CONFIG["soc_delta_grid"]]):
        soc = np.clip(base_soc + float(delta), 0.001, 0.999)
        pred, _ = prediction_from_soc(record, ocv_model, coeffs, soc)
        residual = pred[score_idx] - record.voltage_v[score_idx]
        corrected, bias = add_constant_offset(pred, residual)
        no_bias_met = batt3.error_metrics(record.voltage_v[score_idx], pred[score_idx])
        with_bias_met = batt3.error_metrics(record.voltage_v[score_idx], corrected[score_idx])
        row = {
            "soc_delta": float(delta),
            "soc_min": float(np.min(soc[score_idx])),
            "soc_max": float(np.max(soc[score_idx])),
            "score_bias_offset_mv": 1000.0 * bias,
            "rmse_no_bias_mv": no_bias_met["rmse_mv"],
            "bias_no_bias_mv": no_bias_met["bias_mv"],
            "rmse_with_score_bias_mv": with_bias_met["rmse_mv"],
            "bias_with_score_bias_mv": with_bias_met["bias_mv"],
        }
        rows.append(row)
        if best_no_bias is None or row["rmse_no_bias_mv"] < best_no_bias["rmse_no_bias_mv"]:
            best_no_bias = row
            best_no_bias_pred = pred
        if best_with_bias is None or row["rmse_with_score_bias_mv"] < best_with_bias["rmse_with_score_bias_mv"]:
            best_with_bias = row
            best_with_bias_pred = corrected
    assert best_no_bias is not None and best_with_bias is not None
    assert best_no_bias_pred is not None and best_with_bias_pred is not None
    return rows, best_no_bias, best_with_bias, best_no_bias_pred, best_with_bias_pred


def scan_s0_capacity(context: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    record = context["record"]
    score_idx = context["score_idx"]
    ocv_model = context["ocv_model"]
    coeffs = context["coeffs"]
    rows: list[dict[str, Any]] = []
    best_no_bias: dict[str, Any] | None = None
    best_with_bias: dict[str, Any] | None = None
    best_no_bias_pred: np.ndarray | None = None
    best_with_bias_pred: np.ndarray | None = None
    for q_ah in grid_values(*[float(v) for v in CONFIG["q_grid_ah"]]):
        for s0 in grid_values(*[float(v) for v in CONFIG["s0_grid"]]):
            soc = np.clip(float(s0) - record.net_discharge_ah / float(q_ah), 0.001, 0.999)
            pred, _ = prediction_from_soc(record, ocv_model, coeffs, soc)
            residual = pred[score_idx] - record.voltage_v[score_idx]
            corrected, bias = add_constant_offset(pred, residual)
            no_bias_met = batt3.error_metrics(record.voltage_v[score_idx], pred[score_idx])
            with_bias_met = batt3.error_metrics(record.voltage_v[score_idx], corrected[score_idx])
            row = {
                "s0": float(s0),
                "q_assumed_ah": float(q_ah),
                "soc_min": float(np.min(soc[score_idx])),
                "soc_max": float(np.max(soc[score_idx])),
                "score_bias_offset_mv": 1000.0 * bias,
                "rmse_no_bias_mv": no_bias_met["rmse_mv"],
                "bias_no_bias_mv": no_bias_met["bias_mv"],
                "rmse_with_score_bias_mv": with_bias_met["rmse_mv"],
                "bias_with_score_bias_mv": with_bias_met["bias_mv"],
            }
            rows.append(row)
            if best_no_bias is None or row["rmse_no_bias_mv"] < best_no_bias["rmse_no_bias_mv"]:
                best_no_bias = row
                best_no_bias_pred = pred
            if best_with_bias is None or row["rmse_with_score_bias_mv"] < best_with_bias["rmse_with_score_bias_mv"]:
                best_with_bias = row
                best_with_bias_pred = corrected
    assert best_no_bias is not None and best_with_bias is not None
    assert best_no_bias_pred is not None and best_with_bias_pred is not None
    return rows, best_no_bias, best_with_bias, best_no_bias_pred, best_with_bias_pred


def fit_prefix_s0_offset(context: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    record = context["record"]
    prefix_idx = context["prefix_idx"]
    ocv_model = context["ocv_model"]
    coeffs = context["coeffs"]
    state = context["state"]
    x0 = np.array([float(state["s0"]), 0.0], dtype=float)
    lb = np.array([0.05, -1.0], dtype=float)
    ub = np.array([0.99, 1.0], dtype=float)

    def residual(x: np.ndarray) -> np.ndarray:
        soc = np.clip(float(x[0]) - record.net_discharge_ah / float(batt3.CONFIG["q_nominal_ah"]), 0.001, 0.999)
        pred, _ = prediction_from_soc(record, ocv_model, coeffs, soc)
        return pred[prefix_idx] + float(x[1]) - record.voltage_v[prefix_idx]

    result = least_squares(
        residual,
        x0,
        bounds=(lb, ub),
        loss=str(batt3.CONFIG["least_squares_loss"]),
        f_scale=float(batt3.CONFIG["least_squares_f_scale_v"]),
        max_nfev=100,
        x_scale="jac",
    )
    soc = np.clip(float(result.x[0]) - record.net_discharge_ah / float(batt3.CONFIG["q_nominal_ah"]), 0.001, 0.999)
    pred, _ = prediction_from_soc(record, ocv_model, coeffs, soc)
    pred = pred + float(result.x[1])
    return pred, {"prefix_s0": float(result.x[0]), "prefix_voltage_offset_mv": 1000.0 * float(result.x[1]), "prefix_cost": float(result.cost)}


def ocv_curve_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    record = context["record"]
    coeffs = context["coeffs"]
    rows: list[dict[str, Any]] = []
    for curve in ["discharge", "mean", "charge"]:
        ocv_model = batt3c.load_ocv_model(curve)
        guess = batt3.initial_soc_guess(record, ocv_model, 0.0)
        state = b4.fit_slot_state(
            record,
            ocv_model,
            coeffs,
            guess,
            end_time_s=float(CONFIG["initialization_window_s"]),
            stride=1,
        )
        pred, soc, _ = b4.predict_slot_voltage(record, ocv_model, coeffs, float(state["s0"]))
        rows.append(
            metric_row(
                f"ocv_curve_{curve}",
                record,
                pred,
                context["score_idx"],
                correction_family="ocv_curve_switch",
                calibration_scope="prefix_0_300s",
                uses_scored_target=False,
                extra={
                    "ocv_curve": curve,
                    "s0": float(state["s0"]),
                    "soc_min": float(np.min(soc[context["score_idx"]])),
                    "soc_max": float(np.max(soc[context["score_idx"]])),
                    "init_cost": float(state["init_cost"]),
                },
            )
        )
    return rows


def prediction_sample_rows(
    context: dict[str, Any],
    selected_predictions: list[tuple[str, np.ndarray]],
) -> list[dict[str, Any]]:
    record = context["record"]
    stride = int(CONFIG["prediction_sample_stride"])
    idx = np.arange(0, len(record.time_s), stride, dtype=int)
    rows: list[dict[str, Any]] = []
    for name, pred in selected_predictions:
        for j in idx:
            rows.append(
                {
                    "model": name,
                    "record": record.label,
                    "time_s": float(record.time_s[j]),
                    "current_a_model": float(record.current_a[j]),
                    "voltage_v": float(record.voltage_v[j]),
                    "prediction_v": float(pred[j]),
                    "error_mv": 1000.0 * float(pred[j] - record.voltage_v[j]),
                    "baseline_soc": float(context["soc"][j]),
                    "baseline_r0_ohm": float(context["r0"][j]),
                }
            )
    return rows


def write_bar_svg(path: Path, rows: list[dict[str, Any]], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_rows = sorted(rows, key=lambda row: float(row["rmse_mv"]))
    width = 1100.0
    left = 360.0
    top = 66.0
    row_h = 30.0
    bar_max = max(float(row["rmse_mv"]) for row in plot_rows)
    bars: list[str] = []
    for i, row in enumerate(plot_rows):
        y = top + i * row_h
        value = float(row["rmse_mv"])
        bar_w = value / bar_max * 620.0 if bar_max else 0.0
        label = str(row["model"])
        fill = "#2563eb" if not row["uses_scored_target"] else "#dc2626"
        bars.append(f'<text x="{left - 12}" y="{y + 15}" text-anchor="end" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#374151">{label}</text>')
        bars.append(f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="18" fill="{fill}"/>')
        bars.append(f'<text x="{left + bar_w + 8:.2f}" y="{y + 14}" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">{value:.1f}</text>')
    title = "BATT-4d cell-transfer correction diagnostics" if not zh else "BATT-4d 跨电芯校正诊断"
    legend = "blue=prefix only, red=oracle diagnostic" if not zh else "蓝色=仅前缀，红色=oracle诊断"
    height = top + row_h * len(plot_rows) + 62.0
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">
  <rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>
  <text x="52" y="34" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  <text x="52" y="54" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#6b7280">{legend}</text>
  {''.join(bars)}
  <text x="{left + 310}" y="{height - 18:.2f}" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#374151">RMSE (mV)</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    context = load_base_context()
    record = context["record"]
    score_idx = context["score_idx"]
    prefix_idx = context["prefix_idx"]
    base_pred = context["base_pred"]
    dynamic_pred = context["dynamic_pred"]

    correction_rows: list[dict[str, Any]] = []
    correction_rows.append(
        metric_row(
            "BATT4a2_R0T_prefix_s0",
            record,
            base_pred,
            score_idx,
            correction_family="baseline",
            calibration_scope="prefix_s0_only",
            uses_scored_target=False,
            extra={"s0": float(context["state"]["s0"])},
        )
    )
    correction_rows.append(
        metric_row(
            "BATT4c_qtau50_prefix_s0",
            record,
            dynamic_pred,
            score_idx,
            correction_family="selected_dynamic",
            calibration_scope="train_gain_prefix_s0",
            uses_scored_target=False,
            extra={"tau_s": context["tau_s"], "dynamic_gain_ohm": context["dynamic_gain_ohm"]},
        )
    )

    prefix_residual = base_pred[prefix_idx] - record.voltage_v[prefix_idx]
    prefix_bias_pred, prefix_bias = add_constant_offset(base_pred, prefix_residual)
    correction_rows.append(
        metric_row(
            "prefix_voltage_offset_only",
            record,
            prefix_bias_pred,
            score_idx,
            correction_family="voltage_offset",
            calibration_scope="prefix_0_300s",
            uses_scored_target=False,
            extra={"voltage_offset_mv": -1000.0 * prefix_bias},
        )
    )

    score_residual = base_pred[score_idx] - record.voltage_v[score_idx]
    score_bias_pred, score_bias = add_constant_offset(base_pred, score_residual)
    correction_rows.append(
        metric_row(
            "oracle_score_voltage_offset",
            record,
            score_bias_pred,
            score_idx,
            correction_family="voltage_offset",
            calibration_scope="score_oracle",
            uses_scored_target=True,
            extra={"voltage_offset_mv": -1000.0 * score_bias},
        )
    )

    prefix_s0_offset_pred, prefix_s0_offset = fit_prefix_s0_offset(context)
    correction_rows.append(
        metric_row(
            "prefix_s0_plus_voltage_offset",
            record,
            prefix_s0_offset_pred,
            score_idx,
            correction_family="soc_and_voltage_offset",
            calibration_scope="prefix_0_300s",
            uses_scored_target=False,
            extra=prefix_s0_offset,
        )
    )

    soc_delta_rows, best_delta_no_bias, best_delta_with_bias, best_delta_pred, best_delta_bias_pred = scan_soc_delta(context)
    correction_rows.append(
        metric_row(
            "oracle_soc_delta_only",
            record,
            best_delta_pred,
            score_idx,
            correction_family="soc_shift",
            calibration_scope="score_oracle",
            uses_scored_target=True,
            extra={"soc_delta": best_delta_no_bias["soc_delta"]},
        )
    )
    correction_rows.append(
        metric_row(
            "oracle_soc_delta_plus_voltage_offset",
            record,
            best_delta_bias_pred,
            score_idx,
            correction_family="soc_shift_and_voltage_offset",
            calibration_scope="score_oracle",
            uses_scored_target=True,
            extra={
                "soc_delta": best_delta_with_bias["soc_delta"],
                "voltage_offset_mv": -float(best_delta_with_bias["score_bias_offset_mv"]),
            },
        )
    )

    s0q_rows, best_s0q_no_bias, best_s0q_with_bias, best_s0q_pred, best_s0q_bias_pred = scan_s0_capacity(context)
    correction_rows.append(
        metric_row(
            "oracle_s0_capacity_only",
            record,
            best_s0q_pred,
            score_idx,
            correction_family="s0_capacity",
            calibration_scope="score_oracle",
            uses_scored_target=True,
            extra={"s0": best_s0q_no_bias["s0"], "q_assumed_ah": best_s0q_no_bias["q_assumed_ah"]},
        )
    )
    correction_rows.append(
        metric_row(
            "oracle_s0_capacity_plus_voltage_offset",
            record,
            best_s0q_bias_pred,
            score_idx,
            correction_family="s0_capacity_and_voltage_offset",
            calibration_scope="score_oracle",
            uses_scored_target=True,
            extra={
                "s0": best_s0q_with_bias["s0"],
                "q_assumed_ah": best_s0q_with_bias["q_assumed_ah"],
                "voltage_offset_mv": -float(best_s0q_with_bias["score_bias_offset_mv"]),
            },
        )
    )

    ocv_rows = ocv_curve_rows(context)
    sample_rows = prediction_sample_rows(
        context,
        [
            ("BATT4a2_R0T_prefix_s0", base_pred),
            ("prefix_voltage_offset_only", prefix_bias_pred),
            ("oracle_score_voltage_offset", score_bias_pred),
            ("oracle_soc_delta_plus_voltage_offset", best_delta_bias_pred),
            ("oracle_s0_capacity_plus_voltage_offset", best_s0q_bias_pred),
        ],
    )

    correction_summary_path = RESULT_DIR / "correction_summary.csv"
    soc_delta_scan_path = RESULT_DIR / "soc_delta_scan.csv"
    s0_capacity_scan_path = RESULT_DIR / "s0_capacity_scan.csv"
    ocv_curve_summary_path = RESULT_DIR / "ocv_curve_summary.csv"
    prediction_sample_path = RESULT_DIR / "prediction_sample.csv"
    metrics_path = RESULT_DIR / "metrics.json"
    provenance_path = PROVENANCE_DIR / "batt4d_cell_transfer_alignment_provenance.json"

    write_csv(correction_summary_path, correction_rows)
    write_csv(soc_delta_scan_path, soc_delta_rows)
    write_csv(s0_capacity_scan_path, s0q_rows)
    write_csv(ocv_curve_summary_path, ocv_rows)
    write_csv(prediction_sample_path, sample_rows)
    write_bar_svg(FIGURE_DIR / "cell_transfer_correction_rmse.svg", correction_rows, zh=False)
    write_bar_svg(FIGURE_DIR / "cell_transfer_correction_rmse_zh.svg", correction_rows, zh=True)

    best_prefix_only = min((row for row in correction_rows if not row["uses_scored_target"]), key=lambda row: float(row["rmse_mv"]))
    best_oracle = min((row for row in correction_rows if row["uses_scored_target"]), key=lambda row: float(row["rmse_mv"]))
    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "complete",
        "config": CONFIG,
        "target_record": {
            "label": record.label,
            "filename": record.filename,
            "cell_id": record.cell_id,
            "temperature_c": record.temperature_c,
            "amplitude_token": record.amplitude_token,
            "n_score": int(len(score_idx)),
            "prefix_n": int(len(prefix_idx)),
            "baseline_s0": float(context["state"]["s0"]),
            "baseline_soc_score_min": float(np.min(context["soc"][score_idx])),
            "baseline_soc_score_max": float(np.max(context["soc"][score_idx])),
        },
        "r0_coefficients_ohm": context["coeffs"].tolist(),
        "dynamic_selection": {
            "tau_s": context["tau_s"],
            "dynamic_gain_ohm": context["dynamic_gain_ohm"],
        },
        "correction_summary": correction_rows,
        "ocv_curve_summary": ocv_rows,
        "best_prefix_only": best_prefix_only,
        "best_oracle_diagnostic": best_oracle,
        "interpretation": {
            "prefix_only_result": "Prefix-only voltage/SOC calibration does not solve A003 because the first 300 s are not representative of the scored discharge trajectory.",
            "oracle_result": "Large improvement from score-oracle SOC/capacity/voltage alignment indicates a cross-cell OCV/SOC alignment problem, not an R0(T)-only or q_tau-only problem.",
            "release_usage": "Use BATT-4d as a diagnostic boundary for cross-cell generalization; do not report oracle rows as deployable prediction models.",
        },
        "outputs": {
            "correction_summary_csv": str(correction_summary_path.relative_to(ROOT)),
            "soc_delta_scan_csv": str(soc_delta_scan_path.relative_to(ROOT)),
            "s0_capacity_scan_csv": str(s0_capacity_scan_path.relative_to(ROOT)),
            "ocv_curve_summary_csv": str(ocv_curve_summary_path.relative_to(ROOT)),
            "prediction_sample_csv": str(prediction_sample_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
            "figures": [
                str((FIGURE_DIR / "cell_transfer_correction_rmse.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "cell_transfer_correction_rmse_zh.svg").relative_to(ROOT)),
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
        },
        "script": "scripts/run_batt4d_cell_transfer_alignment.py",
        "inputs": {
            "raw_zip": str(batt3.RAW_ZIP.relative_to(ROOT)),
            "ocv_grid_csv": str(batt3.OCV_GRID_CSV.relative_to(ROOT)),
            "batt4a2_coefficients_csv": str((b4b.BATT4A2_RESULT_DIR / "coefficients.csv").relative_to(ROOT)),
            "batt4c_metrics_json": str(BATT4C_METRICS.relative_to(ROOT)),
        },
        "outputs": metrics["outputs"] | {
            "provenance_json": str(provenance_path.relative_to(ROOT)),
        },
        "test_usage_rule": "BATT-4d is diagnostic. Prefix-only rows use only the first 300 s for calibration; oracle rows use scored A003 residuals and are not deployable model-selection evidence.",
        "next_action": "If cross-cell performance is required, obtain or construct cell-specific OCV/SOC calibration before expanding R0 or dynamic coefficient slots.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
