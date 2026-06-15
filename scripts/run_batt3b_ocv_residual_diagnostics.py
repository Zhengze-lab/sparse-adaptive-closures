#!/usr/bin/env python3
"""Run BATT-3b: OCV-mode, hysteresis, and residual diagnostics."""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

import run_batt3_constant_ecm as batt3


ROOT = batt3.ROOT
RESULT_DIR = ROOT / "results" / "battery_lfp_ocv_residual_diagnostics"
FIGURE_DIR = RESULT_DIR / "figures"
PROVENANCE_DIR = ROOT / "data" / "provenance"


CONFIG: dict[str, Any] = {
    "experiment_id": "BATT3b_lfp_ocv_residual_diagnostics",
    "description": "Diagnose whether battery voltage residuals are dominated by OCV mode, hysteresis, SOC initialization, or current-dependent ECM structure.",
    "source_dataset": "battery_lfp_ocv_dyn",
    "source_url": "https://data.mendeley.com/datasets/p8kf893yv3/1",
    "source_doi": "10.17632/p8kf893yv3.1",
    "license": "CC BY 4.0",
    "raw_zip": str(batt3.RAW_ZIP.relative_to(ROOT)),
    "ocv_grid_csv": str(batt3.OCV_GRID_CSV.relative_to(ROOT)),
    "dynamic_script": batt3.CONFIG["dyn_script"],
    "fit_stride": 20,
    "prediction_sample_stride": 30,
    "initialization_window_s": batt3.CONFIG["initialization_window_s"],
    "score_start_s": batt3.CONFIG["score_start_s"],
    "current_deadband_a": 0.02,
    "ocv_training_temperatures_c": batt3.CONFIG["ocv_training_temperatures_c"],
    "heldout_temperature_validation_c": batt3.CONFIG["heldout_temperature_validation_c"],
    "variants": [
        "ocv_mean",
        "ocv_charge",
        "ocv_discharge",
        "direction_switch",
        "fitted_hysteresis_gain",
    ],
}


@dataclass(frozen=True)
class VariantState:
    s0: float
    hysteresis_gain: float = 1.0
    init_cost: float | None = None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    batt3.write_csv(path, rows)


def load_ocv_models() -> dict[str, batt3.OcvModel]:
    rows: list[dict[str, str]] = []
    with batt3.OCV_GRID_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train_temps = np.array(CONFIG["ocv_training_temperatures_c"], dtype=float)
    soc_grid = np.array(sorted({float(row["soc"]) for row in rows}), dtype=float)
    models: dict[str, batt3.OcvModel] = {}
    for curve in ["mean", "charge", "discharge"]:
        lookup = {
            (int(row["temperature_c"]), float(row["soc"])): float(row["voltage"])
            for row in rows
            if row["curve"] == curve
            and int(row["temperature_c"]) in set(int(v) for v in train_temps)
        }
        matrix = np.empty((len(train_temps), len(soc_grid)), dtype=float)
        for i, temp in enumerate(train_temps):
            for j, soc in enumerate(soc_grid):
                matrix[i, j] = lookup[(int(temp), float(soc))]
        models[curve] = batt3.OcvModel(
            interpolator=RegularGridInterpolator(
                (train_temps, soc_grid),
                matrix,
                method="linear",
                bounds_error=False,
                fill_value=None,
            ),
            soc_grid=soc_grid,
            temperatures_c=train_temps,
        )
    return models


def direction_code(current_a: np.ndarray) -> np.ndarray:
    deadband = float(CONFIG["current_deadband_a"])
    code = np.zeros_like(current_a, dtype=float)
    code[current_a > deadband] = -1.0
    code[current_a < -deadband] = 1.0
    return code


def direction_label(current_a: float) -> str:
    deadband = float(CONFIG["current_deadband_a"])
    if current_a > deadband:
        return "discharge"
    if current_a < -deadband:
        return "charge"
    return "rest"


def ocv_for_variant(
    variant: str,
    models: dict[str, batt3.OcvModel],
    soc: np.ndarray,
    temperature_c: float,
    current_a: np.ndarray,
    hysteresis_gain: float = 1.0,
) -> np.ndarray:
    mean = models["mean"](soc, temperature_c)
    if variant == "ocv_mean":
        return mean
    if variant == "ocv_charge":
        return models["charge"](soc, temperature_c)
    if variant == "ocv_discharge":
        return models["discharge"](soc, temperature_c)
    charge = models["charge"](soc, temperature_c)
    discharge = models["discharge"](soc, temperature_c)
    if variant == "direction_switch":
        code = direction_code(current_a)
        out = mean.copy()
        out[code > 0.0] = charge[code > 0.0]
        out[code < 0.0] = discharge[code < 0.0]
        return out
    if variant == "fitted_hysteresis_gain":
        half_gap = 0.5 * (charge - discharge)
        return mean + hysteresis_gain * direction_code(current_a) * half_gap
    raise ValueError(f"Unknown OCV diagnostic variant: {variant}")


def predict_variant(
    record: batt3.DynRecord,
    models: dict[str, batt3.OcvModel],
    variant: str,
    state: VariantState,
) -> tuple[np.ndarray, np.ndarray]:
    soc = batt3.soc_from_s0(record, state.s0)
    pred = ocv_for_variant(
        variant,
        models,
        soc,
        float(record.temperature_c),
        record.current_a,
        state.hysteresis_gain,
    )
    return pred, soc


def residual_for_state(
    record: batt3.DynRecord,
    models: dict[str, batt3.OcvModel],
    variant: str,
    s0: float,
    hysteresis_gain: float = 1.0,
) -> np.ndarray:
    pred, soc = predict_variant(record, models, variant, VariantState(s0=s0, hysteresis_gain=hysteresis_gain))
    return np.concatenate([pred - record.voltage_v, batt3.range_residual(soc)])


def fit_s0_on_record(
    record: batt3.DynRecord,
    models: dict[str, batt3.OcvModel],
    variant: str,
    hysteresis_gain: float = 1.0,
    prefix_only: bool = False,
) -> VariantState:
    if prefix_only:
        idx = batt3.sample_indices(record, 1, end_time_s=float(CONFIG["initialization_window_s"]))
    else:
        idx = batt3.sample_indices(record, int(CONFIG["fit_stride"]))
    fit_record = batt3.subset_record(record, idx, "_diagnostic_fit")
    x0 = np.array([batt3.initial_soc_guess(record, models["mean"], 0.0)], dtype=float)

    def residual(x: np.ndarray) -> np.ndarray:
        return residual_for_state(fit_record, models, variant, float(x[0]), hysteresis_gain)

    result = least_squares(
        residual,
        x0,
        bounds=(np.array([0.05]), np.array([0.99])),
        loss=str(batt3.CONFIG["least_squares_loss"]),
        f_scale=float(batt3.CONFIG["least_squares_f_scale_v"]),
        max_nfev=80,
        x_scale="jac",
    )
    return VariantState(s0=float(result.x[0]), hysteresis_gain=hysteresis_gain, init_cost=float(result.cost))


def fit_train_states(
    records: list[batt3.DynRecord],
    models: dict[str, batt3.OcvModel],
    variant: str,
    hysteresis_gain: float = 1.0,
) -> dict[str, VariantState]:
    return {
        record.label: fit_s0_on_record(record, models, variant, hysteresis_gain=hysteresis_gain, prefix_only=False)
        for record in records
        if record.split == "train"
    }


def fit_hysteresis_gain(
    records: list[batt3.DynRecord],
    models: dict[str, batt3.OcvModel],
) -> tuple[float, dict[str, VariantState], Any]:
    train = [record for record in records if record.split == "train"]
    x0 = [0.5]
    lb = [0.0]
    ub = [1.0]
    for record in train:
        x0.append(batt3.initial_soc_guess(record, models["mean"], 0.0))
        lb.append(0.05)
        ub.append(0.99)
    fit_records = [
        batt3.subset_record(record, batt3.sample_indices(record, int(CONFIG["fit_stride"])), "_hyst_fit")
        for record in train
    ]

    def residual(x: np.ndarray) -> np.ndarray:
        gain = float(x[0])
        parts: list[np.ndarray] = []
        for i, fit_record in enumerate(fit_records):
            parts.append(residual_for_state(fit_record, models, "fitted_hysteresis_gain", float(x[1 + i]), gain))
        return np.concatenate(parts)

    result = least_squares(
        residual,
        np.array(x0, dtype=float),
        bounds=(np.array(lb, dtype=float), np.array(ub, dtype=float)),
        loss=str(batt3.CONFIG["least_squares_loss"]),
        f_scale=float(batt3.CONFIG["least_squares_f_scale_v"]),
        max_nfev=120,
        x_scale="jac",
    )
    gain = float(result.x[0])
    states = {
        record.label: VariantState(s0=float(result.x[1 + i]), hysteresis_gain=gain)
        for i, record in enumerate(train)
    }
    return gain, states, result


def error_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    residual = y - pred
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.std(y))
    return {
        "rmse_mv": 1000.0 * rmse,
        "mae_mv": 1000.0 * float(np.mean(np.abs(err))),
        "max_abs_mv": 1000.0 * float(np.max(np.abs(err))),
        "bias_error_mv": 1000.0 * float(np.mean(err)),
        "mean_residual_mv": 1000.0 * float(np.mean(residual)),
        "nrmse": rmse / denom if denom > 0 else float("nan"),
    }


def corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    xs = x[mask]
    ys = y[mask]
    if float(np.std(xs)) <= 1e-12 or float(np.std(ys)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def linear_slope_and_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3 or float(np.std(x[mask])) <= 1e-12:
        return float("nan"), float("nan")
    xx = x[mask]
    yy = y[mask]
    a, b = np.polyfit(xx, yy, 1)
    pred = a * xx + b
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(r2)


def summarize_residual_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), str(row["split"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (model, split), sub in sorted(grouped.items()):
        residual = np.array([float(row["residual_mv"]) for row in sub], dtype=float)
        soc = np.array([float(row["soc"]) for row in sub], dtype=float)
        current = np.array([float(row["current_a_discharge_positive"]) for row in sub], dtype=float)
        abs_current = np.abs(current)
        temperature = np.array([float(row["temperature_c"]) for row in sub], dtype=float)
        time_s = np.array([float(row["time_s"]) for row in sub], dtype=float)
        current_slope, current_r2 = linear_slope_and_r2(current, residual)
        abs_current_slope, abs_current_r2 = linear_slope_and_r2(abs_current, residual)
        soc_slope, soc_r2 = linear_slope_and_r2(soc, residual)
        summary.append(
            {
                "model": model,
                "split": split,
                "n_sample": int(len(sub)),
                "residual_mean_mv": float(np.mean(residual)),
                "residual_std_mv": float(np.std(residual)),
                "corr_residual_soc": corr(soc, residual),
                "corr_residual_current": corr(current, residual),
                "corr_residual_abs_current": corr(abs_current, residual),
                "corr_residual_temperature": corr(temperature, residual),
                "corr_residual_time": corr(time_s, residual),
                "current_slope_mv_per_a": current_slope,
                "current_linear_r2": current_r2,
                "abs_current_slope_mv_per_a": abs_current_slope,
                "abs_current_linear_r2": abs_current_r2,
                "soc_slope_mv_per_soc": soc_slope,
                "soc_linear_r2": soc_r2,
            }
        )
    return summary


def summarize_direction(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["split"]), str(row["current_direction"]))
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (model, split, direction), sub in sorted(grouped.items()):
        y = np.array([float(row["voltage_v"]) for row in sub], dtype=float)
        pred = np.array([float(row["prediction_v"]) for row in sub], dtype=float)
        metrics = error_metrics(y, pred)
        out.append(
            {
                "model": model,
                "split": split,
                "current_direction": direction,
                "n_sample": int(len(sub)),
                **metrics,
            }
        )
    return out


def summarize_soc_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = np.linspace(0.0, 1.0, 11)
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        soc = float(row["soc"])
        bin_id = int(np.clip(np.searchsorted(bins, soc, side="right") - 1, 0, len(bins) - 2))
        grouped.setdefault((str(row["model"]), str(row["split"]), bin_id), []).append(row)
    out: list[dict[str, Any]] = []
    for (model, split, bin_id), sub in sorted(grouped.items()):
        residual = np.array([float(row["residual_mv"]) for row in sub], dtype=float)
        out.append(
            {
                "model": model,
                "split": split,
                "soc_bin_left": float(bins[bin_id]),
                "soc_bin_right": float(bins[bin_id + 1]),
                "n_sample": int(len(sub)),
                "residual_mean_mv": float(np.mean(residual)),
                "residual_std_mv": float(np.std(residual)),
            }
        )
    return out


def evaluate_variants(
    records: list[batt3.DynRecord],
    models: dict[str, batt3.OcvModel],
    state_by_variant: dict[str, dict[str, VariantState]],
    hysteresis_gain: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    metrics_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    split_acc: dict[tuple[str, str], dict[str, list[float]]] = {}
    for variant in CONFIG["variants"]:
        for record in records:
            state = state_by_variant.get(variant, {}).get(record.label)
            init_method = "training_fit"
            if state is None:
                state = fit_s0_on_record(
                    record,
                    models,
                    variant,
                    hysteresis_gain=hysteresis_gain if variant == "fitted_hysteresis_gain" else 1.0,
                    prefix_only=True,
                )
                init_method = "prefix_output_error"
            pred, soc = predict_variant(record, models, variant, state)
            score_idx = batt3.score_indices(record)
            valid = np.isfinite(pred[score_idx]) & np.isfinite(record.voltage_v[score_idx])
            y = record.voltage_v[score_idx][valid]
            pred_score = pred[score_idx][valid]
            soc_score = soc[score_idx][valid]
            metrics = error_metrics(y, pred_score)
            metrics_rows.append(
                {
                    "model": variant,
                    "split": record.split,
                    "record": record.label,
                    "filename": record.filename,
                    "temperature_c": record.temperature_c,
                    "amplitude_token": record.amplitude_token,
                    "n_score": int(len(y)),
                    "init_method": init_method,
                    "s0": float(state.s0),
                    "hysteresis_gain": float(state.hysteresis_gain),
                    "soc_min": float(np.min(soc_score)),
                    "soc_max": float(np.max(soc_score)),
                    "soc_outside_0_1_fraction": float(np.mean((soc_score < 0.0) | (soc_score > 1.0))),
                    "soc_outside_ocv_grid_fraction": float(np.mean((soc_score < 0.02) | (soc_score > 0.98))),
                    "stable_fraction": float(np.mean(valid)) if len(valid) else 0.0,
                    **metrics,
                }
            )
            split_acc.setdefault((variant, record.split), {"y": [], "pred": []})
            split_acc[(variant, record.split)]["y"].extend(float(v) for v in y)
            split_acc[(variant, record.split)]["pred"].extend(float(v) for v in pred_score)

            sample_idx = np.arange(0, len(record.time_s), int(CONFIG["prediction_sample_stride"]), dtype=int)
            for j in sample_idx:
                sample_rows.append(
                    {
                        "model": variant,
                        "split": record.split,
                        "record": record.label,
                        "time_s": float(record.time_s[j]),
                        "temperature_c": record.temperature_c,
                        "current_a_discharge_positive": float(record.current_a[j]),
                        "current_direction": direction_label(float(record.current_a[j])),
                        "voltage_v": float(record.voltage_v[j]),
                        "prediction_v": float(pred[j]),
                        "error_mv": 1000.0 * float(pred[j] - record.voltage_v[j]),
                        "residual_mv": 1000.0 * float(record.voltage_v[j] - pred[j]),
                        "soc": float(soc[j]),
                    }
                )
    split_rows: list[dict[str, Any]] = []
    split_summary: dict[str, dict[str, Any]] = {}
    for (variant, split), values in sorted(split_acc.items()):
        y = np.array(values["y"], dtype=float)
        pred = np.array(values["pred"], dtype=float)
        row = {"model": variant, "split": split, "n_score": int(len(y)), **error_metrics(y, pred)}
        split_rows.append(row)
        split_summary[f"{variant}:{split}"] = row
    return metrics_rows, split_rows, sample_rows, split_summary


def svg_polyline(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    return batt3.svg_polyline(x, y, x_min, x_max, y_min, y_max, left, top, width, height)


def write_rmse_svg(path: Path, split_rows: list[dict[str, Any]], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    splits = ["train", "validation_temperature", "test_high_amplitude", "test_cell_transfer"]
    models = list(CONFIG["variants"])
    label_en = {
        "ocv_mean": "OCV mean",
        "ocv_charge": "OCV charge",
        "ocv_discharge": "OCV discharge",
        "direction_switch": "Direction switch",
        "fitted_hysteresis_gain": "Fitted hysteresis",
        "train": "Train",
        "validation_temperature": "Held-out T",
        "test_high_amplitude": "High amp.",
        "test_cell_transfer": "Cell transfer",
    }
    label_zh = {
        "ocv_mean": "平均 OCV",
        "ocv_charge": "充电 OCV",
        "ocv_discharge": "放电 OCV",
        "direction_switch": "方向切换",
        "fitted_hysteresis_gain": "拟合滞回",
        "train": "训练",
        "validation_temperature": "留出温度",
        "test_high_amplitude": "高倍率",
        "test_cell_transfer": "跨电芯",
    }
    labels = label_zh if zh else label_en
    colors = ["#4b5563", "#2563eb", "#16a34a", "#dc2626", "#7c3aed"]
    lookup = {(row["model"], row["split"]): float(row["rmse_mv"]) for row in split_rows}
    y_max = max(lookup.values()) * 1.12
    left, top, width, height = 84.0, 70.0, 760.0, 320.0
    group_w = width / len(splits)
    bar_w = 19.0
    gap = 4.0
    bars: list[str] = []
    for i, split in enumerate(splits):
        base_x = left + i * group_w + group_w / 2 - (len(models) * bar_w + (len(models) - 1) * gap) / 2
        for j, model in enumerate(models):
            value = lookup.get((model, split), 0.0)
            bar_h = value / y_max * height
            x = base_x + j * (bar_w + gap)
            y = top + height - bar_h
            bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{colors[j]}"/>')
        bars.append(f'<text x="{left + i * group_w + group_w / 2:.2f}" y="420" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#374151">{labels[split]}</text>')
    legend: list[str] = []
    for i, model in enumerate(models):
        y = 92 + i * 24
        legend.append(f'<rect x="858" y="{y - 12}" width="17" height="11" fill="{colors[i]}"/>')
        legend.append(f'<text x="883" y="{y - 2}" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="11" fill="#111827">{labels[model]}</text>')
    title = "BATT-3b OCV-mode RMSE comparison" if not zh else "BATT-3b OCV 模式 RMSE 对比"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="460" viewBox="0 0 1040 460">
  <rect width="1040" height="460" fill="#ffffff"/>
  <text x="84" y="36" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  <rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#6b7280"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#6b7280"/>
  <text x="25" y="{top + height / 2}" transform="rotate(-90 25 {top + height / 2})" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">RMSE (mV)</text>
  {''.join(bars)}
  {''.join(legend)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_soc_residual_svg(path: Path, soc_bin_rows: list[dict[str, Any]], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    models = ["ocv_mean", "direction_switch", "fitted_hysteresis_gain"]
    split = "validation_temperature"
    sub = [row for row in soc_bin_rows if row["split"] == split and row["model"] in models]
    if not sub:
        return
    x_min, x_max = 0.0, 1.0
    y_values = [float(row["residual_mean_mv"]) for row in sub]
    y_min = min(y_values) - 30.0
    y_max = max(y_values) + 30.0
    left, top, width, height = 82.0, 70.0, 740.0, 320.0
    colors = {"ocv_mean": "#4b5563", "direction_switch": "#dc2626", "fitted_hysteresis_gain": "#7c3aed"}
    labels_en = {"ocv_mean": "OCV mean", "direction_switch": "Direction switch", "fitted_hysteresis_gain": "Fitted hysteresis"}
    labels_zh = {"ocv_mean": "平均 OCV", "direction_switch": "方向切换", "fitted_hysteresis_gain": "拟合滞回"}
    labels = labels_zh if zh else labels_en
    lines: list[str] = []
    legend: list[str] = []
    for i, model in enumerate(models):
        rows = sorted([row for row in sub if row["model"] == model], key=lambda r: float(r["soc_bin_left"]))
        x = np.array([0.5 * (float(row["soc_bin_left"]) + float(row["soc_bin_right"])) for row in rows], dtype=float)
        y = np.array([float(row["residual_mean_mv"]) for row in rows], dtype=float)
        lines.append(f'<polyline points="{svg_polyline(x, y, x_min, x_max, y_min, y_max, left, top, width, height)}" fill="none" stroke="{colors[model]}" stroke-width="2.2"/>')
        ly = 94 + i * 24
        legend.append(f'<line x1="850" y1="{ly}" x2="884" y2="{ly}" stroke="{colors[model]}" stroke-width="2.2"/>')
        legend.append(f'<text x="893" y="{ly + 4}" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">{labels[model]}</text>')
    zero_y = top + height - (0.0 - y_min) / (y_max - y_min) * height
    title = "Validation residual by SOC bin" if not zh else "留出温度 residual 随 SOC 分箱"
    y_label = "mean residual V-measured minus prediction (mV)" if not zh else "平均 residual：实测-预测 (mV)"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1030" height="460" viewBox="0 0 1030 460">
  <rect width="1030" height="460" fill="#ffffff"/>
  <text x="82" y="36" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  <rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{left}" y1="{zero_y:.2f}" x2="{left + width}" y2="{zero_y:.2f}" stroke="#9ca3af" stroke-dasharray="5,4"/>
  <line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#6b7280"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#6b7280"/>
  {''.join(lines)}
  <text x="{left + width / 2}" y="430" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">SOC</text>
  <text x="25" y="{top + height / 2}" transform="rotate(-90 25 {top + height / 2})" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#374151">{y_label}</text>
  {''.join(legend)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    records = batt3.load_dynamic_records()
    models = load_ocv_models()
    state_by_variant: dict[str, dict[str, VariantState]] = {}
    for variant in ["ocv_mean", "ocv_charge", "ocv_discharge", "direction_switch"]:
        state_by_variant[variant] = fit_train_states(records, models, variant, hysteresis_gain=1.0)
    hysteresis_gain, hysteresis_states, hysteresis_result = fit_hysteresis_gain(records, models)
    state_by_variant["fitted_hysteresis_gain"] = hysteresis_states

    metrics_rows, split_rows, sample_rows, split_summary = evaluate_variants(
        records,
        models,
        state_by_variant,
        hysteresis_gain,
    )
    residual_feature_rows = summarize_residual_features(sample_rows)
    direction_rows = summarize_direction(sample_rows)
    soc_bin_rows = summarize_soc_bins(sample_rows)

    metrics_by_trajectory_path = RESULT_DIR / "metrics_by_trajectory.csv"
    metrics_by_split_path = RESULT_DIR / "metrics_by_split.csv"
    residual_samples_path = RESULT_DIR / "residual_samples.csv"
    residual_features_path = RESULT_DIR / "residual_feature_summary.csv"
    residual_direction_path = RESULT_DIR / "residual_by_current_direction.csv"
    residual_soc_bins_path = RESULT_DIR / "residual_by_soc_bin.csv"
    metrics_path = RESULT_DIR / "metrics.json"
    provenance_path = PROVENANCE_DIR / "batt3b_ocv_residual_diagnostics_provenance.json"

    write_csv(metrics_by_trajectory_path, metrics_rows)
    write_csv(metrics_by_split_path, split_rows)
    write_csv(residual_samples_path, sample_rows)
    write_csv(residual_features_path, residual_feature_rows)
    write_csv(residual_direction_path, direction_rows)
    write_csv(residual_soc_bins_path, soc_bin_rows)
    write_rmse_svg(FIGURE_DIR / "ocv_mode_rmse_by_split.svg", split_rows, zh=False)
    write_rmse_svg(FIGURE_DIR / "ocv_mode_rmse_by_split_zh.svg", split_rows, zh=True)
    write_soc_residual_svg(FIGURE_DIR / "validation_residual_by_soc.svg", soc_bin_rows, zh=False)
    write_soc_residual_svg(FIGURE_DIR / "validation_residual_by_soc_zh.svg", soc_bin_rows, zh=True)

    validation_rank = sorted(
        [row for row in split_rows if row["split"] == "validation_temperature"],
        key=lambda row: float(row["rmse_mv"]),
    )
    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "complete",
        "config": CONFIG,
        "raw_zip_sha256": batt3.sha256_file(batt3.RAW_ZIP),
        "record_count": len(records),
        "hysteresis_gain": hysteresis_gain,
        "hysteresis_fit": {
            "cost": float(hysteresis_result.cost),
            "optimality": float(hysteresis_result.optimality),
            "nfev": int(hysteresis_result.nfev),
            "success": bool(hysteresis_result.success),
            "message": str(hysteresis_result.message),
        },
        "validation_rank_by_rmse": validation_rank,
        "split_summary": split_summary,
        "outputs": {
            "metrics_by_trajectory_csv": str(metrics_by_trajectory_path.relative_to(ROOT)),
            "metrics_by_split_csv": str(metrics_by_split_path.relative_to(ROOT)),
            "residual_samples_csv": str(residual_samples_path.relative_to(ROOT)),
            "residual_feature_summary_csv": str(residual_features_path.relative_to(ROOT)),
            "residual_by_current_direction_csv": str(residual_direction_path.relative_to(ROOT)),
            "residual_by_soc_bin_csv": str(residual_soc_bins_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
            "figures": [
                str((FIGURE_DIR / "ocv_mode_rmse_by_split.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "ocv_mode_rmse_by_split_zh.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "validation_residual_by_soc.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "validation_residual_by_soc_zh.svg").relative_to(ROOT)),
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
        "script": "scripts/run_batt3b_ocv_residual_diagnostics.py",
        "outputs": metrics["outputs"] | {
            "provenance_json": str(provenance_path.relative_to(ROOT)),
        },
        "test_usage_rule": "Global hysteresis gain is fitted only on split=train. Non-train records use a short prefix for initial SOC estimation and are scored after the initialization window.",
        "next_action": "Use diagnostics to decide whether BATT-4 should target OCV/hysteresis slots before RC coefficient slots.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
