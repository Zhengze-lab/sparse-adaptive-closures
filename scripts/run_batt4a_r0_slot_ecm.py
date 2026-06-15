#!/usr/bin/env python3
"""Run BATT-4a: sparse R0(SOC,T,|I|) coefficient-slot ECM."""

from __future__ import annotations

import json
import platform
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pysindy as ps
import scipy
from scipy.optimize import least_squares

import run_batt3_constant_ecm as batt3
import run_batt3c_discharge_ocv_ecm as batt3c


ROOT = batt3.ROOT
RESULT_DIR = ROOT / "results" / "battery_lfp_r0_slot_ecm"
FIGURE_DIR = RESULT_DIR / "figures"
PROVENANCE_DIR = ROOT / "data" / "provenance"


CONFIG: dict[str, Any] = {
    "experiment_id": "BATT4a_lfp_r0_slot_ecm",
    "description": "Coefficient-slot ECM with R0(SOC,T,|I|), fixed OCV_discharge and I_model=I_raw from BATT-3c.",
    "source_dataset": "battery_lfp_ocv_dyn",
    "source_url": "https://data.mendeley.com/datasets/p8kf893yv3/1",
    "source_doi": "10.17632/p8kf893yv3.1",
    "license": "CC BY 4.0",
    "baseline_experiment": "BATT3c_lfp_discharge_ocv_current_sign_ecm",
    "ocv_curve": "discharge",
    "current_convention": "I_model=I_raw",
    "r0_slot": "R0(s,T,|I|)",
    "feature_definition": "sc=s-0.5, Tn=T/25, In=|I|/2",
    "feature_names": [
        "1",
        "sc",
        "sc^2",
        "Tn",
        "Tn^2",
        "sc*Tn",
        "In",
        "sc*In",
        "Tn*In",
        "In^2",
    ],
    "soc_center": 0.5,
    "temperature_scale_c": 25.0,
    "current_scale_a": 2.0,
    "fit_abs_current_min_a": 0.10,
    "fit_stride": batt3.CONFIG["fit_stride"],
    "prediction_sample_stride": batt3.CONFIG["prediction_sample_stride"],
    "initialization_window_s": batt3.CONFIG["initialization_window_s"],
    "score_start_s": batt3.CONFIG["score_start_s"],
    "least_squares_loss": batt3.CONFIG["least_squares_loss"],
    "least_squares_f_scale_v": batt3.CONFIG["least_squares_f_scale_v"],
    "state_refit_max_nfev": 80,
    "alternating_iterations": 3,
    "pysindy_optimizer": "STLSQ",
    "pysindy_alpha": 1e-8,
    "pysindy_max_iter": 30,
    "pysindy_normalize_columns": False,
    "thresholds": [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
    "active_threshold": 1e-7,
    "ocv_training_temperatures_c": batt3.CONFIG["ocv_training_temperatures_c"],
    "heldout_temperature_validation_c": batt3.CONFIG["heldout_temperature_validation_c"],
}


@dataclass(frozen=True)
class SlotFitResult:
    model: str
    threshold: float
    coeffs: np.ndarray
    train_states: dict[str, dict[str, float]]
    history: list[dict[str, Any]]
    design_rows: int


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    batt3.write_csv(path, rows)


def model_name_for_threshold(threshold: float) -> str:
    if threshold == 0:
        return "BATT4a_R0_slot_STLSQ_dense"
    return f"BATT4a_R0_slot_STLSQ_t{threshold:g}"


def r0_features(soc: np.ndarray, temperature_c: float | np.ndarray, current_a: np.ndarray) -> np.ndarray:
    soc_arr = np.asarray(soc, dtype=float)
    temp_arr = np.asarray(temperature_c, dtype=float)
    if temp_arr.ndim == 0:
        temp_arr = np.full_like(soc_arr, float(temp_arr), dtype=float)
    current_arr = np.asarray(current_a, dtype=float)
    sc = soc_arr - float(CONFIG["soc_center"])
    tn = temp_arr / float(CONFIG["temperature_scale_c"])
    inn = np.abs(current_arr) / float(CONFIG["current_scale_a"])
    return np.column_stack(
        [
            np.ones_like(sc),
            sc,
            sc * sc,
            tn,
            tn * tn,
            sc * tn,
            inn,
            sc * inn,
            tn * inn,
            inn * inn,
        ]
    )


def r0_from_coeffs(soc: np.ndarray, temperature_c: float | np.ndarray, current_a: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return r0_features(soc, temperature_c, current_a) @ coeffs


def predict_slot_voltage(
    record: batt3.DynRecord,
    ocv_model: batt3.OcvModel,
    coeffs: np.ndarray,
    s0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    soc = batt3.soc_from_s0(record, s0)
    ocv = ocv_model(soc, float(record.temperature_c))
    r0 = r0_from_coeffs(soc, float(record.temperature_c), record.current_a, coeffs)
    return ocv - r0 * record.current_a, soc, r0


def build_slot_design(
    records: list[batt3.DynRecord],
    ocv_model: batt3.OcvModel,
    states: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    stride = int(CONFIG["fit_stride"])
    min_abs_i = float(CONFIG["fit_abs_current_min_a"])
    for record in records:
        if record.split != "train":
            continue
        state = states[record.label]
        idx = batt3.sample_indices(record, stride)
        fit_record = batt3.subset_record(record, idx, "_fit")
        soc = batt3.soc_from_s0(fit_record, float(state["s0"]))
        ocv = ocv_model(soc, float(fit_record.temperature_c))
        design = fit_record.current_a[:, None] * r0_features(
            soc,
            float(fit_record.temperature_c),
            fit_record.current_a,
        )
        target = ocv - fit_record.voltage_v
        finite = (
            np.all(np.isfinite(design), axis=1)
            & np.isfinite(target)
            & np.isfinite(soc)
            & (np.abs(fit_record.current_a) >= min_abs_i)
        )
        rows.append(design[finite])
        targets.append(target[finite])
    if not rows:
        raise RuntimeError("No rows available for R0 slot design.")
    return np.vstack(rows), np.concatenate(targets)


def fit_slot_coefficients(design: np.ndarray, target: np.ndarray, threshold: float) -> np.ndarray:
    optimizer = ps.STLSQ(
        threshold=float(threshold),
        alpha=float(CONFIG["pysindy_alpha"]),
        max_iter=int(CONFIG["pysindy_max_iter"]),
        normalize_columns=bool(CONFIG["pysindy_normalize_columns"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimizer.fit(design, target.reshape(-1, 1))
    return np.asarray(optimizer.coef_, dtype=float).reshape(-1)


def fit_slot_state(
    record: batt3.DynRecord,
    ocv_model: batt3.OcvModel,
    coeffs: np.ndarray,
    initial_s0: float,
    end_time_s: float | None,
    stride: int,
) -> dict[str, float]:
    idx = batt3.sample_indices(record, stride, end_time_s=end_time_s)
    fit_record = batt3.subset_record(record, idx, "_slot_state")
    x0 = np.array([float(np.clip(initial_s0, 0.05, 0.99))], dtype=float)

    def residual(x: np.ndarray) -> np.ndarray:
        pred, soc, _ = predict_slot_voltage(fit_record, ocv_model, coeffs, float(x[0]))
        return np.concatenate([pred - fit_record.voltage_v, batt3.range_residual(soc)])

    result = least_squares(
        residual,
        x0,
        bounds=(np.array([0.05]), np.array([0.99])),
        loss=str(CONFIG["least_squares_loss"]),
        f_scale=float(CONFIG["least_squares_f_scale_v"]),
        max_nfev=int(CONFIG["state_refit_max_nfev"]),
        x_scale="jac",
    )
    return {"s0": float(result.x[0]), "v10": 0.0, "init_cost": float(result.cost)}


def refit_train_states(
    records: list[batt3.DynRecord],
    ocv_model: batt3.OcvModel,
    coeffs: np.ndarray,
    previous_states: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for record in records:
        if record.split != "train":
            continue
        previous = previous_states.get(record.label)
        if previous is None:
            s0 = batt3.initial_soc_guess(record, ocv_model, float(np.mean(coeffs[:1])) if coeffs.size else 0.0)
        else:
            s0 = float(previous["s0"])
        out[record.label] = fit_slot_state(
            record,
            ocv_model,
            coeffs,
            s0,
            end_time_s=None,
            stride=int(CONFIG["fit_stride"]),
        )
    return out


def linear_fit_nrmse(design: np.ndarray, target: np.ndarray, coeffs: np.ndarray) -> float:
    pred = design @ coeffs
    denom = float(np.std(target))
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - target) ** 2)) / denom)


def active_terms(coeffs: np.ndarray) -> int:
    return int(np.sum(np.abs(coeffs) >= float(CONFIG["active_threshold"])))


def fit_slot_model(
    records: list[batt3.DynRecord],
    ocv_model: batt3.OcvModel,
    threshold: float,
    initial_states: dict[str, dict[str, float]],
) -> SlotFitResult:
    states = {key: dict(value) for key, value in initial_states.items()}
    coeffs = np.zeros(len(CONFIG["feature_names"]), dtype=float)
    history: list[dict[str, Any]] = []
    design_rows = 0
    for iteration in range(int(CONFIG["alternating_iterations"])):
        design, target = build_slot_design(records, ocv_model, states)
        design_rows = int(design.shape[0])
        coeffs = fit_slot_coefficients(design, target, threshold)
        states = refit_train_states(records, ocv_model, coeffs, states)
        history.append(
            {
                "iteration": iteration + 1,
                "threshold": threshold,
                "active_terms": active_terms(coeffs),
                "design_rows": design_rows,
                "linear_fit_nrmse": linear_fit_nrmse(design, target, coeffs),
            }
        )
    return SlotFitResult(
        model=model_name_for_threshold(threshold),
        threshold=threshold,
        coeffs=coeffs,
        train_states=states,
        history=history,
        design_rows=design_rows,
    )


def fit_initial_state_for_model(
    record: batt3.DynRecord,
    ocv_model: batt3.OcvModel,
    model: str,
    params: dict[str, Any],
) -> dict[str, float]:
    if model.startswith("BATT4a_R0_slot"):
        guess = batt3.initial_soc_guess(record, ocv_model, 0.03)
        return fit_slot_state(
            record,
            ocv_model,
            np.asarray(params["coeffs"], dtype=float),
            guess,
            end_time_s=float(CONFIG["initialization_window_s"]),
            stride=1,
        )
    if model == "BATT3c_OCV_discharge_only":
        return batt3.fit_initial_state(record, ocv_model, "ocv_only", {})
    if model == "BATT3c_constant_ohmic":
        return batt3.fit_initial_state(record, ocv_model, "ohmic_only", params)
    if model == "BATT3c_constant_ecm":
        return batt3.fit_initial_state(record, ocv_model, "constant_ecm", params)
    raise ValueError(f"Unknown model: {model}")


def predict_model(
    record: batt3.DynRecord,
    ocv_model: batt3.OcvModel,
    model: str,
    params: dict[str, Any],
    state: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s0 = float(state["s0"])
    if model.startswith("BATT4a_R0_slot"):
        pred, soc, r0 = predict_slot_voltage(record, ocv_model, np.asarray(params["coeffs"], dtype=float), s0)
        return pred, soc, r0
    if model == "BATT3c_OCV_discharge_only":
        pred, soc, _ = batt3.predict_voltage(record, ocv_model, "ocv_only", {}, s0, 0.0)
        return pred, soc, np.zeros_like(pred)
    if model == "BATT3c_constant_ohmic":
        pred, soc, _ = batt3.predict_voltage(record, ocv_model, "ohmic_only", params, s0, 0.0)
        return pred, soc, np.full_like(pred, float(params["R0_ohm"]))
    if model == "BATT3c_constant_ecm":
        pred, soc, _ = batt3.predict_voltage(
            record,
            ocv_model,
            "constant_ecm",
            params,
            s0,
            float(state.get("v10", 0.0)),
        )
        return pred, soc, np.full_like(pred, float(params["R0_ohm"]))
    raise ValueError(f"Unknown model: {model}")


def evaluate_models(
    records: list[batt3.DynRecord],
    ocv_model: batt3.OcvModel,
    model_params: dict[str, dict[str, Any]],
    train_states: dict[str, dict[str, dict[str, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    trajectory_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    split_acc: dict[tuple[str, str], dict[str, list[float]]] = {}
    for model, params in model_params.items():
        for record in records:
            state = train_states.get(model, {}).get(record.label)
            init_method = "training_fit"
            if state is None:
                state = fit_initial_state_for_model(record, ocv_model, model, params)
                init_method = "prefix_output_error"
            pred, soc, r0 = predict_model(record, ocv_model, model, params, state)
            idx = batt3.score_indices(record)
            valid = np.isfinite(pred[idx]) & np.isfinite(record.voltage_v[idx]) & np.isfinite(soc[idx])
            score_y = record.voltage_v[idx][valid]
            score_pred = pred[idx][valid]
            score_soc = soc[idx][valid]
            score_r0 = r0[idx][valid]
            met = batt3.error_metrics(score_y, score_pred)
            trajectory_rows.append(
                {
                    "model": model,
                    "split": record.split,
                    "record": record.label,
                    "filename": record.filename,
                    "temperature_c": record.temperature_c,
                    "amplitude_token": record.amplitude_token,
                    "n_score": int(len(score_y)),
                    "init_method": init_method,
                    "s0": float(state["s0"]),
                    "soc_min": float(np.min(score_soc)),
                    "soc_max": float(np.max(score_soc)),
                    "r0_min_ohm": float(np.min(score_r0)),
                    "r0_max_ohm": float(np.max(score_r0)),
                    "r0_negative_fraction": float(np.mean(score_r0 < 0.0)),
                    "stable_fraction": float(np.mean(valid)) if len(valid) else 0.0,
                    **met,
                }
            )
            key = (model, record.split)
            split_acc.setdefault(key, {"y": [], "pred": [], "r0": []})
            split_acc[key]["y"].extend(float(v) for v in score_y)
            split_acc[key]["pred"].extend(float(v) for v in score_pred)
            split_acc[key]["r0"].extend(float(v) for v in score_r0)

            sample_idx = np.arange(0, len(record.time_s), int(CONFIG["prediction_sample_stride"]), dtype=int)
            for j in sample_idx:
                prediction_rows.append(
                    {
                        "model": model,
                        "split": record.split,
                        "record": record.label,
                        "time_s": float(record.time_s[j]),
                        "temperature_c": record.temperature_c,
                        "current_a_model": float(record.current_a[j]),
                        "voltage_v": float(record.voltage_v[j]),
                        "prediction_v": float(pred[j]),
                        "error_mv": 1000.0 * float(pred[j] - record.voltage_v[j]),
                        "soc": float(soc[j]),
                        "r0_ohm": float(r0[j]),
                    }
                )

    split_rows: list[dict[str, Any]] = []
    split_summary: dict[str, dict[str, Any]] = {}
    for (model, split), values in sorted(split_acc.items()):
        y = np.array(values["y"], dtype=float)
        pred = np.array(values["pred"], dtype=float)
        r0 = np.array(values["r0"], dtype=float)
        met = batt3.error_metrics(y, pred)
        row = {
            "model": model,
            "split": split,
            "n_score": int(len(y)),
            "r0_min_ohm": float(np.min(r0)),
            "r0_max_ohm": float(np.max(r0)),
            "r0_negative_fraction": float(np.mean(r0 < 0.0)),
            **met,
        }
        split_rows.append(row)
        split_summary[f"{model}:{split}"] = row
    return trajectory_rows, split_rows, prediction_rows, split_summary


def threshold_summary_rows(
    slot_results: list[SlotFitResult],
    split_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    split_lookup = {(row["model"], row["split"]): row for row in split_rows}
    rows: list[dict[str, Any]] = []
    for result in slot_results:
        last = result.history[-1]
        row: dict[str, Any] = {
            "model": result.model,
            "threshold": result.threshold,
            "active_terms": active_terms(result.coeffs),
            "design_rows": result.design_rows,
            "linear_fit_nrmse": last["linear_fit_nrmse"],
        }
        for split in ["train", "validation_temperature", "test_high_amplitude", "test_cell_transfer"]:
            met = split_lookup.get((result.model, split), {})
            row[f"{split}_rmse_mv"] = met.get("rmse_mv", "")
            row[f"{split}_nrmse"] = met.get("nrmse", "")
            row[f"{split}_r0_negative_fraction"] = met.get("r0_negative_fraction", "")
        rows.append(row)
    return rows


def coefficient_rows(slot_results: list[SlotFitResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = list(CONFIG["feature_names"])
    for result in slot_results:
        for name, coeff in zip(names, result.coeffs):
            rows.append(
                {
                    "model": result.model,
                    "threshold": result.threshold,
                    "feature": name,
                    "coefficient_ohm": float(coeff),
                    "active": abs(float(coeff)) >= float(CONFIG["active_threshold"]),
                }
            )
    return rows


def improvement_rows(split_rows: list[dict[str, Any]], baseline_model: str) -> list[dict[str, Any]]:
    lookup = {(row["model"], row["split"]): float(row["rmse_mv"]) for row in split_rows}
    models = sorted({row["model"] for row in split_rows})
    splits = sorted({row["split"] for row in split_rows})
    rows: list[dict[str, Any]] = []
    for model in models:
        if model == baseline_model:
            continue
        for split in splits:
            base = lookup.get((baseline_model, split))
            value = lookup.get((model, split))
            if base is None or value is None:
                continue
            rows.append(
                {
                    "model": model,
                    "baseline_model": baseline_model,
                    "split": split,
                    "baseline_rmse_mv": base,
                    "model_rmse_mv": value,
                    "improvement_percent": 100.0 * (base - value) / base if base else float("nan"),
                }
            )
    return rows


def select_best_slot_model(threshold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        row
        for row in threshold_rows
        if row.get("validation_temperature_rmse_mv") != ""
        and float(row.get("train_r0_negative_fraction", 1.0)) <= 0.01
        and float(row.get("validation_temperature_r0_negative_fraction", 1.0)) <= 0.01
    ]
    if not candidates:
        candidates = [row for row in threshold_rows if row.get("validation_temperature_rmse_mv") != ""]
    return min(candidates, key=lambda row: float(row["validation_temperature_rmse_mv"]))


def label_for_model(model: str, zh: bool = False) -> str:
    labels_en = {
        "BATT3c_OCV_discharge_only": "BATT-3c discharge OCV only",
        "BATT3c_constant_ohmic": "BATT-3c constant ohmic",
        "BATT3c_constant_ecm": "BATT-3c constant ECM",
        "BATT4a_R0_slot_STLSQ_dense": "BATT-4a R0 slot dense",
    }
    labels_zh = {
        "BATT3c_OCV_discharge_only": "BATT-3c 仅放电 OCV",
        "BATT3c_constant_ohmic": "BATT-3c 常数欧姆",
        "BATT3c_constant_ecm": "BATT-3c 常系数 ECM",
        "BATT4a_R0_slot_STLSQ_dense": "BATT-4a R0 槽 dense",
    }
    labels = labels_zh if zh else labels_en
    if model in labels:
        return labels[model]
    if model.startswith("BATT4a_R0_slot_STLSQ_t"):
        suffix = model.replace("BATT4a_R0_slot_STLSQ_t", "")
        return (f"BATT-4a R0 槽 STLSQ 阈值 {suffix}" if zh else f"BATT-4a R0 slot STLSQ threshold {suffix}")
    return model


def write_rmse_svg(
    path: Path,
    split_rows: list[dict[str, Any]],
    selected_models: list[str],
    zh: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    splits = ["train", "validation_temperature", "test_high_amplitude", "test_cell_transfer"]
    split_labels_en = {
        "train": "Train",
        "validation_temperature": "Held-out T",
        "test_high_amplitude": "High amp.",
        "test_cell_transfer": "Cell transfer",
    }
    split_labels_zh = {
        "train": "训练",
        "validation_temperature": "留出温度",
        "test_high_amplitude": "高倍率",
        "test_cell_transfer": "跨电芯",
    }
    split_labels = split_labels_zh if zh else split_labels_en
    colors = ["#6b7280", "#2563eb", "#7c3aed", "#dc2626", "#f97316"]
    lookup = {(row["model"], row["split"]): float(row["rmse_mv"]) for row in split_rows}
    y_max = max(lookup.get((model, split), 0.0) for model in selected_models for split in splits) * 1.12
    left, top, width, height = 86.0, 70.0, 820.0, 320.0
    group_w = width / len(splits)
    bar_w = 18.0
    gap = 4.0
    bars: list[str] = []
    for i, split in enumerate(splits):
        base_x = left + i * group_w + group_w / 2 - (len(selected_models) * bar_w + (len(selected_models) - 1) * gap) / 2
        for j, model in enumerate(selected_models):
            value = lookup.get((model, split), 0.0)
            bar_h = value / y_max * height if y_max > 0 else 0.0
            x = base_x + j * (bar_w + gap)
            y = top + height - bar_h
            bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{colors[j % len(colors)]}"/>')
        bars.append(f'<text x="{left + i * group_w + group_w / 2:.2f}" y="420" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#374151">{split_labels[split]}</text>')
    legend: list[str] = []
    for i, model in enumerate(selected_models):
        y = 86 + i * 25
        legend.append(f'<rect x="928" y="{y - 12}" width="17" height="11" fill="{colors[i % len(colors)]}"/>')
        legend.append(f'<text x="953" y="{y - 2}" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="11" fill="#111827">{label_for_model(model, zh=zh)}</text>')
    title = "BATT-4a R0 coefficient-slot RMSE comparison" if not zh else "BATT-4a R0 系数槽 RMSE 对比"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="460" viewBox="0 0 1260 460">
  <rect width="1260" height="460" fill="#ffffff"/>
  <text x="86" y="36" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  <rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#6b7280"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#6b7280"/>
  <text x="25" y="{top + height / 2}" transform="rotate(-90 25 {top + height / 2})" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">RMSE (mV)</text>
  {''.join(bars)}
  {''.join(legend)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


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
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    xp = left + (x - x_min) / (x_max - x_min) * width
    yp = top + height - (y - y_min) / (y_max - y_min) * height
    return " ".join(f"{float(xx):.2f},{float(yy):.2f}" for xx, yy in zip(xp, yp))


def write_r0_profile_svg(path: Path, coeffs: np.ndarray, zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    soc_grid = np.linspace(0.05, 0.95, 160)
    current_grid = np.linspace(0.0, 10.0, 160)
    temps = [-25.0, 25.0, 45.0]
    colors = ["#2563eb", "#dc2626", "#059669"]
    left1, top, panel_w, panel_h = 82.0, 70.0, 430.0, 280.0
    left2 = 610.0
    y_values = []
    for temp in temps:
        y_values.append(r0_from_coeffs(soc_grid, temp, np.full_like(soc_grid, 1.0), coeffs))
    y_values.append(r0_from_coeffs(np.full_like(current_grid, 0.5), 25.0, current_grid, coeffs))
    y_min = min(float(np.min(v)) for v in y_values) - 0.005
    y_max = max(float(np.max(v)) for v in y_values) + 0.005
    items: list[str] = []
    for color, temp in zip(colors, temps):
        values = r0_from_coeffs(soc_grid, temp, np.full_like(soc_grid, 1.0), coeffs)
        points = svg_polyline(soc_grid, values, 0.05, 0.95, y_min, y_max, left1, top, panel_w, panel_h)
        items.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    values_i = r0_from_coeffs(np.full_like(current_grid, 0.5), 25.0, current_grid, coeffs)
    points_i = svg_polyline(current_grid, values_i, 0.0, 10.0, y_min, y_max, left2, top, panel_w, panel_h)
    items.append(f'<polyline points="{points_i}" fill="none" stroke="#7c3aed" stroke-width="2"/>')
    legend: list[str] = []
    for i, (color, temp) in enumerate(zip(colors, temps)):
        y = 382 + i * 22
        label = f"T={temp:g} C, |I|=1 A"
        legend.append(f'<rect x="82" y="{y - 12}" width="18" height="11" fill="{color}"/>')
        legend.append(f'<text x="108" y="{y - 2}" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">{label}</text>')
    legend.append('<rect x="610" y="370" width="18" height="11" fill="#7c3aed"/>')
    legend.append('<text x="636" y="380" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">SOC=0.5, T=25 C</text>')
    title = "BATT-4a recovered R0 profiles" if not zh else "BATT-4a 恢复的 R0 函数剖面"
    left_title = "R0 vs SOC" if not zh else "R0 随 SOC 变化"
    right_title = "R0 vs |I|" if not zh else "R0 随 |I| 变化"
    x1 = "SOC" if not zh else "SOC"
    x2 = "|I| (A)" if not zh else "|I| (A)"
    ylab = "R0 (ohm)" if not zh else "R0 (欧姆)"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="450" viewBox="0 0 1120 450">
  <rect width="1120" height="450" fill="#ffffff"/>
  <text x="82" y="34" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  <text x="{left1}" y="57" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="14" fill="#111827">{left_title}</text>
  <text x="{left2}" y="57" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="14" fill="#111827">{right_title}</text>
  <rect x="{left1}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#ffffff" stroke="#d1d5db"/>
  <rect x="{left2}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{left1}" y1="{top + panel_h}" x2="{left1 + panel_w}" y2="{top + panel_h}" stroke="#6b7280"/>
  <line x1="{left1}" y1="{top}" x2="{left1}" y2="{top + panel_h}" stroke="#6b7280"/>
  <line x1="{left2}" y1="{top + panel_h}" x2="{left2 + panel_w}" y2="{top + panel_h}" stroke="#6b7280"/>
  <line x1="{left2}" y1="{top}" x2="{left2}" y2="{top + panel_h}" stroke="#6b7280"/>
  <text x="25" y="{top + panel_h / 2}" transform="rotate(-90 25 {top + panel_h / 2})" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{ylab}</text>
  <text x="{left1 + panel_w / 2}" y="372" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{x1}</text>
  <text x="{left2 + panel_w / 2}" y="372" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{x2}</text>
  {''.join(items)}
  {''.join(legend)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


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
    slot_initial_states = {key: dict(value) for key, value in ohmic_states.items()}
    slot_results = [
        fit_slot_model(records, ocv_model, float(threshold), slot_initial_states)
        for threshold in CONFIG["thresholds"]
    ]

    model_params: dict[str, dict[str, Any]] = {
        "BATT3c_OCV_discharge_only": {},
        "BATT3c_constant_ohmic": ohmic_params,
        "BATT3c_constant_ecm": ecm_params,
    }
    train_states: dict[str, dict[str, dict[str, float]]] = {
        "BATT3c_OCV_discharge_only": ocv_states,
        "BATT3c_constant_ohmic": ohmic_states,
        "BATT3c_constant_ecm": ecm_states,
    }
    for result in slot_results:
        model_params[result.model] = {"coeffs": result.coeffs}
        train_states[result.model] = result.train_states

    trajectory_rows, split_rows, prediction_rows, split_summary = evaluate_models(
        records,
        ocv_model,
        model_params,
        train_states,
    )
    threshold_rows = threshold_summary_rows(slot_results, split_rows)
    best_slot = select_best_slot_model(threshold_rows)
    selected_models = [
        "BATT3c_OCV_discharge_only",
        "BATT3c_constant_ohmic",
        "BATT3c_constant_ecm",
        str(best_slot["model"]),
    ]
    if "BATT4a_R0_slot_STLSQ_dense" not in selected_models:
        selected_models.append("BATT4a_R0_slot_STLSQ_dense")

    metrics_by_trajectory_path = RESULT_DIR / "metrics_by_trajectory.csv"
    metrics_by_split_path = RESULT_DIR / "metrics_by_split.csv"
    predictions_path = RESULT_DIR / "predictions_sample.csv"
    coefficients_path = RESULT_DIR / "coefficients.csv"
    threshold_summary_path = RESULT_DIR / "threshold_summary.csv"
    improvement_constant_ecm_path = RESULT_DIR / "improvement_vs_batt3c_constant_ecm.csv"
    fit_summary_path = RESULT_DIR / "fit_summary.csv"
    metrics_path = RESULT_DIR / "metrics.json"
    provenance_path = PROVENANCE_DIR / "batt4a_r0_slot_ecm_provenance.json"

    fit_rows = [
        {
            "model": "BATT3c_constant_ohmic",
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
    for result in slot_results:
        last = result.history[-1]
        fit_rows.append(
            {
                "model": result.model,
                "threshold": result.threshold,
                "active_terms": active_terms(result.coeffs),
                "design_rows": result.design_rows,
                "linear_fit_nrmse": last["linear_fit_nrmse"],
            }
        )

    write_csv(metrics_by_trajectory_path, trajectory_rows)
    write_csv(metrics_by_split_path, split_rows)
    write_csv(predictions_path, prediction_rows)
    write_csv(coefficients_path, coefficient_rows(slot_results))
    write_csv(threshold_summary_path, threshold_rows)
    write_csv(improvement_constant_ecm_path, improvement_rows(split_rows, "BATT3c_constant_ecm"))
    write_csv(fit_summary_path, fit_rows)

    best_result = next(result for result in slot_results if result.model == best_slot["model"])
    write_rmse_svg(FIGURE_DIR / "rmse_by_split.svg", split_rows, selected_models, zh=False)
    write_rmse_svg(FIGURE_DIR / "rmse_by_split_zh.svg", split_rows, selected_models, zh=True)
    write_r0_profile_svg(FIGURE_DIR / "r0_profiles.svg", best_result.coeffs, zh=False)
    write_r0_profile_svg(FIGURE_DIR / "r0_profiles_zh.svg", best_result.coeffs, zh=True)

    validation_rows = sorted(
        [row for row in split_rows if row["split"] == "validation_temperature"],
        key=lambda row: float(row["rmse_mv"]),
    )
    high_amp_rows = sorted(
        [row for row in split_rows if row["split"] == "test_high_amplitude"],
        key=lambda row: float(row["rmse_mv"]),
    )
    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "complete",
        "config": CONFIG,
        "raw_zip_sha256": batt3.sha256_file(batt3.RAW_ZIP),
        "record_count": len(records),
        "slot_threshold_summary": threshold_rows,
        "best_slot_by_validation": best_slot,
        "validation_rank_by_rmse": validation_rows,
        "high_amplitude_rank_by_rmse": high_amp_rows,
        "split_summary": split_summary,
        "outputs": {
            "metrics_by_trajectory_csv": str(metrics_by_trajectory_path.relative_to(ROOT)),
            "metrics_by_split_csv": str(metrics_by_split_path.relative_to(ROOT)),
            "predictions_sample_csv": str(predictions_path.relative_to(ROOT)),
            "coefficients_csv": str(coefficients_path.relative_to(ROOT)),
            "threshold_summary_csv": str(threshold_summary_path.relative_to(ROOT)),
            "improvement_vs_batt3c_constant_ecm_csv": str(improvement_constant_ecm_path.relative_to(ROOT)),
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
        "script": "scripts/run_batt4a_r0_slot_ecm.py",
        "outputs": metrics["outputs"] | {
            "provenance_json": str(provenance_path.relative_to(ROOT)),
        },
        "test_usage_rule": "R0 slot coefficients are fitted only on split=train. Non-train records use only the initialization prefix for initial SOC estimation.",
        "baseline_rule": "Compare BATT-4a against the BATT-3c OCV_discharge + I_model=I_raw constant ECM baseline.",
        "next_action": "If R0 slot improves validation without worsening high-amplitude behavior, proceed to dynamic slots; otherwise diagnose feature library, positivity constraints, and current-amplitude extrapolation.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
