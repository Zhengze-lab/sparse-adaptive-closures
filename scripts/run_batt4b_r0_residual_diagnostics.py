#!/usr/bin/env python3
"""Run BATT-4b: residual diagnostics after the physics-filtered R0(T) slot."""

from __future__ import annotations

import csv
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

import run_batt3_constant_ecm as batt3
import run_batt3c_discharge_ocv_ecm as batt3c
import run_batt4a_r0_slot_ecm as b4
import run_batt4a_filtered_r0_slot_ecm as b4f


ROOT = batt3.ROOT
RESULT_DIR = ROOT / "results" / "battery_lfp_r0_residual_diagnostics"
FIGURE_DIR = RESULT_DIR / "figures"
PROVENANCE_DIR = ROOT / "data" / "provenance"
BATT4A2_RESULT_DIR = ROOT / "results" / "battery_lfp_r0_slot_filtered_ecm"


CONFIG: dict[str, Any] = {
    "experiment_id": "BATT4b_lfp_r0_residual_diagnostics",
    "description": "Residual and dynamic-slot diagnostics after the physics-filtered BATT-4a R0(T) coefficient slot.",
    "source_dataset": "battery_lfp_ocv_dyn",
    "source_url": "https://data.mendeley.com/datasets/p8kf893yv3/1",
    "source_doi": "10.17632/p8kf893yv3.1",
    "license": "CC BY 4.0",
    "baseline_experiment": "BATT4a2_lfp_filtered_r0_slot_ecm",
    "selected_r0_model": "BATT4a_R0_slot_filtered_T_only_STLSQ_dense",
    "selected_feature_set": "T_only",
    "ocv_curve": "discharge",
    "current_convention": "I_model=I_raw",
    "feature_scaling": "Tn=T/25",
    "score_start_s": batt3.CONFIG["score_start_s"],
    "initialization_window_s": batt3.CONFIG["initialization_window_s"],
    "fit_stride": batt3.CONFIG["fit_stride"],
    "prediction_sample_stride": batt3.CONFIG["prediction_sample_stride"],
    "dynamic_tau_grid_s": [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0],
    "residual_feature_names": [
        "current_a",
        "abs_current_a",
        "soc",
        "temperature_c",
        "time_since_score_s",
        "q_tau_20s",
        "q_tau_50s",
        "q_tau_100s",
        "q_tau_200s",
        "q_tau_500s",
        "q_tau_1000s",
    ],
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    batt3.write_csv(path, rows)


def read_selected_r0_coeffs() -> np.ndarray:
    path = BATT4A2_RESULT_DIR / "coefficients.csv"
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["model"] == str(CONFIG["selected_r0_model"])
                and row["feature_set"] == str(CONFIG["selected_feature_set"])
            ):
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No coefficients found for {CONFIG['selected_r0_model']}")
    rows.sort(key=lambda row: list(b4f.CONFIG["feature_sets"][str(CONFIG["selected_feature_set"])]).index(row["feature"]))
    return np.array([float(row["coefficient_ohm"]) for row in rows], dtype=float)


def filtered_current(time_s: np.ndarray, current_a: np.ndarray, tau_s: float) -> np.ndarray:
    q = np.empty_like(current_a, dtype=float)
    q[0] = 0.0
    tau = max(float(tau_s), 1e-9)
    for k in range(len(current_a) - 1):
        dt = max(0.0, float(time_s[k + 1] - time_s[k]))
        alpha = math.exp(-dt / tau)
        q[k + 1] = alpha * q[k] + (1.0 - alpha) * current_a[k]
    return q


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 0.0 or y_std <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def one_feature_regression(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3 or float(np.var(x)) <= 0.0:
        return {
            "n": int(x.size),
            "corr": float("nan"),
            "slope_mv_per_unit": float("nan"),
            "intercept_mv": float("nan"),
            "r2": float("nan"),
        }
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(x.size),
        "corr": pearson_corr(x, y),
        "slope_mv_per_unit": float(coef[1]),
        "intercept_mv": float(coef[0]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan"),
    }


def rmse_mv(error_v: np.ndarray) -> float:
    return 1000.0 * float(np.sqrt(np.mean(error_v * error_v)))


def fit_r0t_states(
    records: list[batt3.DynRecord],
    ocv_model: batt3.OcvModel,
    coeffs: np.ndarray,
) -> dict[str, dict[str, float]]:
    states: dict[str, dict[str, float]] = {}
    for record in records:
        guess = batt3.initial_soc_guess(record, ocv_model, 0.03)
        if record.split == "train":
            states[record.label] = b4.fit_slot_state(
                record,
                ocv_model,
                coeffs,
                guess,
                end_time_s=None,
                stride=int(CONFIG["fit_stride"]),
            )
        else:
            states[record.label] = b4.fit_slot_state(
                record,
                ocv_model,
                coeffs,
                guess,
                end_time_s=float(CONFIG["initialization_window_s"]),
                stride=1,
            )
    return states


def collect_residual_rows(
    records: list[batt3.DynRecord],
    ocv_model: batt3.OcvModel,
    coeffs: np.ndarray,
    states: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    sample_rows: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    taus = [float(v) for v in CONFIG["dynamic_tau_grid_s"]]
    for record in records:
        state = states[record.label]
        pred, soc, r0 = b4.predict_slot_voltage(record, ocv_model, coeffs, float(state["s0"]))
        error_v = pred - record.voltage_v
        idx = batt3.score_indices(record)
        valid = (
            np.isfinite(pred[idx])
            & np.isfinite(record.voltage_v[idx])
            & np.isfinite(soc[idx])
            & np.isfinite(r0[idx])
        )
        score_idx = idx[valid]
        tau_features = {f"q_tau_{int(tau)}s": filtered_current(record.time_s, record.current_a, tau) for tau in taus}
        item: dict[str, np.ndarray] = {
            "record": np.array([record.label] * len(score_idx), dtype=object),
            "split": np.array([record.split] * len(score_idx), dtype=object),
            "time_s": record.time_s[score_idx],
            "time_since_score_s": record.time_s[score_idx] - float(CONFIG["score_start_s"]),
            "current_a": record.current_a[score_idx],
            "abs_current_a": np.abs(record.current_a[score_idx]),
            "soc": soc[score_idx],
            "temperature_c": np.full(len(score_idx), float(record.temperature_c), dtype=float),
            "r0_ohm": r0[score_idx],
            "voltage_v": record.voltage_v[score_idx],
            "prediction_v": pred[score_idx],
            "residual_v": error_v[score_idx],
            "residual_mv": 1000.0 * error_v[score_idx],
        }
        for name, values in tau_features.items():
            item[name] = values[score_idx]
        arrays.append(item)

        sample_idx = np.arange(0, len(record.time_s), int(CONFIG["prediction_sample_stride"]), dtype=int)
        for j in sample_idx:
            sample_rows.append(
                {
                    "split": record.split,
                    "record": record.label,
                    "time_s": float(record.time_s[j]),
                    "temperature_c": int(record.temperature_c),
                    "current_a_model": float(record.current_a[j]),
                    "voltage_v": float(record.voltage_v[j]),
                    "prediction_v": float(pred[j]),
                    "residual_mv": 1000.0 * float(error_v[j]),
                    "soc": float(soc[j]),
                    "r0_ohm": float(r0[j]),
                }
            )
    return sample_rows, arrays


def concat_by_split(arrays: list[dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    splits = sorted({str(split) for item in arrays for split in item["split"]})
    out: dict[str, dict[str, np.ndarray]] = {}
    for split in splits:
        selected = [item for item in arrays if str(item["split"][0]) == split]
        keys = [key for key in selected[0] if key not in {"record", "split"}]
        out[split] = {key: np.concatenate([item[key] for item in selected]) for key in keys}
    return out


def record_summary_rows(arrays: list[dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in arrays:
        residual_v = item["residual_v"]
        residual_centered = residual_v - float(np.mean(residual_v))
        rows.append(
            {
                "split": str(item["split"][0]),
                "record": str(item["record"][0]),
                "n_score": int(residual_v.size),
                "temperature_c": float(item["temperature_c"][0]),
                "mean_current_a": float(np.mean(item["current_a"])),
                "mean_abs_current_a": float(np.mean(item["abs_current_a"])),
                "soc_min": float(np.min(item["soc"])),
                "soc_max": float(np.max(item["soc"])),
                "r0_min_ohm": float(np.min(item["r0_ohm"])),
                "r0_max_ohm": float(np.max(item["r0_ohm"])),
                "rmse_mv": rmse_mv(residual_v),
                "bias_mv": 1000.0 * float(np.mean(residual_v)),
                "mae_mv": 1000.0 * float(np.mean(np.abs(residual_v))),
                "bias_removed_rmse_mv": rmse_mv(residual_centered),
                "bias_fraction_of_rmse": abs(float(np.mean(residual_v))) / float(np.sqrt(np.mean(residual_v * residual_v))),
            }
        )
    return rows


def residual_feature_rows(split_arrays: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, item in split_arrays.items():
        y = item["residual_mv"]
        for feature in CONFIG["residual_feature_names"]:
            stats = one_feature_regression(item[str(feature)], y)
            rows.append({"split": split, "feature": feature, **stats})
    return rows


def split_metric_rows(arrays: list[dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    split_arrays = concat_by_split(arrays)
    rows: list[dict[str, Any]] = []
    for split, item in split_arrays.items():
        residual_v = item["residual_v"]
        centered = residual_v - float(np.mean(residual_v))
        rows.append(
            {
                "model": str(CONFIG["selected_r0_model"]),
                "split": split,
                "n_score": int(residual_v.size),
                "rmse_mv": rmse_mv(residual_v),
                "bias_mv": 1000.0 * float(np.mean(residual_v)),
                "mae_mv": 1000.0 * float(np.mean(np.abs(residual_v))),
                "bias_removed_rmse_mv": rmse_mv(centered),
                "r0_min_ohm": float(np.min(item["r0_ohm"])),
                "r0_max_ohm": float(np.max(item["r0_ohm"])),
                "r0_negative_fraction": float(np.mean(item["r0_ohm"] < 0.0)),
            }
        )
    return rows


def fit_dynamic_proxy(
    split_arrays: dict[str, dict[str, np.ndarray]],
    tau_s: float,
    with_intercept: bool,
) -> dict[str, Any]:
    feature = f"q_tau_{int(tau_s)}s"
    train = split_arrays["train"]
    x_train = train[feature]
    y_train = train["residual_v"]
    finite = np.isfinite(x_train) & np.isfinite(y_train)
    x_train = x_train[finite]
    y_train = y_train[finite]
    if with_intercept:
        design = np.column_stack([np.ones_like(x_train), x_train])
    else:
        design = x_train[:, None]
    coef, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    row: dict[str, Any] = {
        "tau_s": float(tau_s),
        "with_intercept": bool(with_intercept),
        "intercept_v": float(coef[0]) if with_intercept else 0.0,
        "dynamic_gain_ohm": float(coef[1]) if with_intercept else float(coef[0]),
        "train_fit_r2": one_feature_regression(x_train, 1000.0 * y_train)["r2"],
    }
    for split, item in split_arrays.items():
        x = item[feature]
        residual = item["residual_v"]
        if with_intercept:
            residual_hat = float(coef[0]) + float(coef[1]) * x
        else:
            residual_hat = float(coef[0]) * x
        corrected = residual - residual_hat
        base_rmse = rmse_mv(residual)
        corrected_rmse = rmse_mv(corrected)
        row[f"{split}_base_rmse_mv"] = base_rmse
        row[f"{split}_corrected_rmse_mv"] = corrected_rmse
        row[f"{split}_improvement_percent"] = 100.0 * (base_rmse - corrected_rmse) / base_rmse if base_rmse else float("nan")
    return row


def dynamic_proxy_rows(split_arrays: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tau in CONFIG["dynamic_tau_grid_s"]:
        rows.append(fit_dynamic_proxy(split_arrays, float(tau), with_intercept=False))
        rows.append(fit_dynamic_proxy(split_arrays, float(tau), with_intercept=True))
    return rows


def max_feature_by_r2(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    candidates = [row for row in rows if row["split"] == split and np.isfinite(float(row["r2"]))]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: float(row["r2"]))


def best_dynamic_proxy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row["with_intercept"] is False]
    return min(candidates, key=lambda row: float(row["validation_temperature_corrected_rmse_mv"]))


def write_feature_r2_svg(path: Path, rows: list[dict[str, Any]], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split = "validation_temperature"
    selected = [row for row in rows if row["split"] == split]
    features = [str(row["feature"]) for row in selected]
    values = [max(0.0, float(row["r2"])) if np.isfinite(float(row["r2"])) else 0.0 for row in selected]
    width = 980.0
    left = 220.0
    top = 68.0
    row_h = 28.0
    max_v = max(values) if values else 1.0
    max_v = max(max_v, 1e-9)
    bars: list[str] = []
    for i, (feature, value) in enumerate(zip(features, values)):
        y = top + i * row_h
        bar_w = value / max_v * 620.0
        bars.append(f'<text x="{left - 12}" y="{y + 15}" text-anchor="end" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#374151">{feature}</text>')
        bars.append(f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="18" fill="#2563eb"/>')
        bars.append(f'<text x="{left + bar_w + 8:.2f}" y="{y + 14}" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">{value:.3f}</text>')
    title = "BATT-4b validation residual single-feature R2" if not zh else "BATT-4b 留出温度 residual 单特征 R2"
    height = top + row_h * max(1, len(features)) + 42.0
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">
  <rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>
  <text x="52" y="34" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  {''.join(bars)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_dynamic_proxy_svg(path: Path, rows: list[dict[str, Any]], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [row for row in rows if row["with_intercept"] is False]
    taus = np.array([float(row["tau_s"]) for row in selected], dtype=float)
    val = np.array([float(row["validation_temperature_corrected_rmse_mv"]) for row in selected], dtype=float)
    high = np.array([float(row["test_high_amplitude_corrected_rmse_mv"]) for row in selected], dtype=float)
    base_val = float(selected[0]["validation_temperature_base_rmse_mv"])
    base_high = float(selected[0]["test_high_amplitude_base_rmse_mv"])
    left, top, panel_w, panel_h = 82.0, 62.0, 680.0, 300.0
    y_max = max(float(np.max(val)), float(np.max(high)), base_val, base_high) * 1.1
    y_min = 0.0
    x_min = float(np.min(taus))
    x_max = float(np.max(taus))

    def points(y: np.ndarray) -> str:
        xp = left + (taus - x_min) / (x_max - x_min) * panel_w
        yp = top + panel_h - (y - y_min) / (y_max - y_min) * panel_h
        return " ".join(f"{float(xx):.2f},{float(yy):.2f}" for xx, yy in zip(xp, yp))

    val_points = points(val)
    high_points = points(high)
    title = "BATT-4b dynamic residual proxy scan" if not zh else "BATT-4b 动态 residual proxy 扫描"
    xlab = "tau (s)" if not zh else "tau (秒)"
    ylab = "Corrected RMSE (mV)" if not zh else "校正后 RMSE (mV)"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="940" height="430" viewBox="0 0 940 430">
  <rect width="940" height="430" fill="#ffffff"/>
  <text x="82" y="34" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  <rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" y2="{top + panel_h}" stroke="#6b7280"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_h}" stroke="#6b7280"/>
  <polyline points="{val_points}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
  <polyline points="{high_points}" fill="none" stroke="#dc2626" stroke-width="2.5"/>
  <line x1="{left}" y1="{top + panel_h - base_val / y_max * panel_h:.2f}" x2="{left + panel_w}" y2="{top + panel_h - base_val / y_max * panel_h:.2f}" stroke="#2563eb" stroke-width="1.3" stroke-dasharray="5 5"/>
  <line x1="{left}" y1="{top + panel_h - base_high / y_max * panel_h:.2f}" x2="{left + panel_w}" y2="{top + panel_h - base_high / y_max * panel_h:.2f}" stroke="#dc2626" stroke-width="1.3" stroke-dasharray="5 5"/>
  <text x="{left + panel_w / 2}" y="402" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{xlab}</text>
  <text x="24" y="{top + panel_h / 2}" transform="rotate(-90 24 {top + panel_h / 2})" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{ylab}</text>
  <rect x="790" y="94" width="18" height="11" fill="#2563eb"/>
  <text x="816" y="104" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">validation</text>
  <rect x="790" y="122" width="18" height="11" fill="#dc2626"/>
  <text x="816" y="132" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">high amplitude</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    b4f.configure_feature_set(str(CONFIG["selected_feature_set"]))
    raw_records = batt3.load_dynamic_records()
    records = batt3c.records_for_current_convention(raw_records, str(CONFIG["current_convention"]))
    ocv_model = batt3c.load_ocv_model(str(CONFIG["ocv_curve"]))
    coeffs = read_selected_r0_coeffs()
    states = fit_r0t_states(records, ocv_model, coeffs)
    sample_rows, arrays = collect_residual_rows(records, ocv_model, coeffs, states)
    split_arrays = concat_by_split(arrays)

    split_rows = split_metric_rows(arrays)
    record_rows = record_summary_rows(arrays)
    feature_rows = residual_feature_rows(split_arrays)
    dynamic_rows = dynamic_proxy_rows(split_arrays)

    residual_samples_path = RESULT_DIR / "residual_samples.csv"
    split_metrics_path = RESULT_DIR / "split_metrics.csv"
    record_summary_path = RESULT_DIR / "record_residual_summary.csv"
    feature_summary_path = RESULT_DIR / "residual_feature_summary.csv"
    dynamic_proxy_path = RESULT_DIR / "dynamic_proxy_scan.csv"
    metrics_path = RESULT_DIR / "metrics.json"
    provenance_path = PROVENANCE_DIR / "batt4b_r0_residual_diagnostics_provenance.json"

    write_csv(residual_samples_path, sample_rows)
    write_csv(split_metrics_path, split_rows)
    write_csv(record_summary_path, record_rows)
    write_csv(feature_summary_path, feature_rows)
    write_csv(dynamic_proxy_path, dynamic_rows)
    write_feature_r2_svg(FIGURE_DIR / "validation_residual_feature_r2.svg", feature_rows, zh=False)
    write_feature_r2_svg(FIGURE_DIR / "validation_residual_feature_r2_zh.svg", feature_rows, zh=True)
    write_dynamic_proxy_svg(FIGURE_DIR / "dynamic_proxy_scan.svg", dynamic_rows, zh=False)
    write_dynamic_proxy_svg(FIGURE_DIR / "dynamic_proxy_scan_zh.svg", dynamic_rows, zh=True)

    split_lookup = {row["split"]: row for row in split_rows}
    best_features = {split: max_feature_by_r2(feature_rows, split) for split in split_lookup}
    best_proxy = best_dynamic_proxy(dynamic_rows)
    cell_record = next((row for row in record_rows if row["split"] == "test_cell_transfer"), {})
    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "complete",
        "config": CONFIG,
        "r0_coefficients_ohm": coeffs.tolist(),
        "split_metrics": split_rows,
        "best_single_feature_by_split": best_features,
        "best_dynamic_proxy_without_intercept": best_proxy,
        "cell_transfer_record_summary": cell_record,
        "diagnostic_interpretation": {
            "dynamic_slot_evidence": "Use dynamic_proxy_scan.csv to decide whether a train-fitted first-order current filter consistently reduces validation and high-amplitude residuals.",
            "cell_transfer_evidence": "Compare cell-transfer RMSE and bias-removed RMSE; a large reduction after removing record bias indicates cell/SOC/OCV offset rather than an R0-only issue.",
            "next_action_rule": "Add a_tau(z) or b_C(z) only if the dynamic proxy improves validation and high-amplitude without relying on intercept leakage; handle cell transfer separately if record bias dominates.",
        },
        "outputs": {
            "residual_samples_csv": str(residual_samples_path.relative_to(ROOT)),
            "split_metrics_csv": str(split_metrics_path.relative_to(ROOT)),
            "record_residual_summary_csv": str(record_summary_path.relative_to(ROOT)),
            "residual_feature_summary_csv": str(feature_summary_path.relative_to(ROOT)),
            "dynamic_proxy_scan_csv": str(dynamic_proxy_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
            "figures": [
                str((FIGURE_DIR / "validation_residual_feature_r2.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "validation_residual_feature_r2_zh.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "dynamic_proxy_scan.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "dynamic_proxy_scan_zh.svg").relative_to(ROOT)),
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
        "script": "scripts/run_batt4b_r0_residual_diagnostics.py",
        "inputs": {
            "batt4a2_coefficients_csv": str((BATT4A2_RESULT_DIR / "coefficients.csv").relative_to(ROOT)),
            "raw_zip": str(batt3.RAW_ZIP.relative_to(ROOT)),
            "ocv_grid_csv": str(batt3.OCV_GRID_CSV.relative_to(ROOT)),
        },
        "outputs": metrics["outputs"] | {
            "provenance_json": str(provenance_path.relative_to(ROOT)),
        },
        "test_usage_rule": "The R0(T) coefficients come from BATT-4a2 train-only fitting. BATT-4b uses non-train prefixes only for SOC initialization and performs post-hoc diagnostics.",
        "next_action": "Use the residual diagnostics to decide whether to run a dynamic-slot pilot for a_tau/b_C or to prioritize cell-transfer/SOC/OCV correction.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
