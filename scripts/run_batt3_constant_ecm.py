#!/usr/bin/env python3
"""Run BATT-3: constant first-order ECM output-error baseline."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = ROOT / "data" / "raw" / "external" / "battery_lfp_ocv_dyn" / "p8kf893yv3_v1.zip"
OCV_GRID_CSV = ROOT / "data" / "processed" / "battery_lfp_ocv_dyn" / "batt2_ocv_grid.csv"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "battery_lfp_constant_ecm"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG: dict[str, Any] = {
    "experiment_id": "BATT3_lfp_constant_first_order_ecm",
    "description": "Constant first-order Thevenin/RC ECM output-error baseline using BATT-2 OCV_mean(s,T).",
    "source_dataset": "battery_lfp_ocv_dyn",
    "source_url": "https://data.mendeley.com/datasets/p8kf893yv3/1",
    "source_doi": "10.17632/p8kf893yv3.1",
    "license": "CC BY 4.0",
    "raw_zip": str(RAW_ZIP.relative_to(ROOT)),
    "ocv_grid_csv": str(OCV_GRID_CSV.relative_to(ROOT)),
    "dyn_root": "DYNData",
    "dyn_script": "script1",
    "q_nominal_ah": 2.5,
    "ocv_curve": "mean",
    "ocv_training_temperatures_c": [-25, -5, 5, 15, 25, 45],
    "heldout_temperature_validation_c": [-15, 35],
    "stress_test_amplitude_token": 50,
    "fit_stride": 20,
    "prediction_sample_stride": 30,
    "initialization_window_s": 300.0,
    "score_start_s": 300.0,
    "least_squares_loss": "soft_l1",
    "least_squares_f_scale_v": 0.02,
    "max_nfev_ohmic": 80,
    "max_nfev_ecm": 120,
    "model_sign_convention": "raw dataset current is converted to discharge-positive current I=-current_raw; Vhat=OCV(s,T)-R0*I-v1.",
}


@dataclass(frozen=True)
class DynRecord:
    filename: str
    label: str
    cell_id: str
    amplitude_token: int
    temperature_c: int
    split: str
    time_s: np.ndarray
    current_a: np.ndarray
    voltage_v: np.ndarray
    net_discharge_ah: np.ndarray


@dataclass(frozen=True)
class OcvModel:
    interpolator: RegularGridInterpolator
    soc_grid: np.ndarray
    temperatures_c: np.ndarray

    def __call__(self, soc: np.ndarray, temperature_c: float) -> np.ndarray:
        clipped_soc = np.clip(soc, float(self.soc_grid[0]), float(self.soc_grid[-1]))
        points = np.column_stack([np.full_like(clipped_soc, temperature_c, dtype=float), clipped_soc])
        return np.asarray(self.interpolator(points), dtype=float)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_temperature_c(filename: str) -> int:
    match = re.search(r"_([NP])(\d{2})(?:_|\.)", filename)
    if not match:
        raise ValueError(f"Cannot parse temperature from {filename}")
    sign = -1 if match.group(1) == "N" else 1
    return sign * int(match.group(2))


def parse_dyn_identity(filename: str) -> tuple[str, int, int]:
    stem = Path(filename).stem
    match = re.search(r"(A\d{3})_DYN_(\d+)_([NP]\d{2})$", stem)
    if not match:
        raise ValueError(f"Cannot parse dynamic identity from {filename}")
    return match.group(1), int(match.group(2)), parse_temperature_c(filename)


def field_array(obj: Any, name: str) -> np.ndarray:
    return np.asarray(getattr(obj, name), dtype=float).ravel()


def split_for_record(cell_id: str, amplitude_token: int, temperature_c: int) -> str:
    if cell_id != "A002":
        return "test_cell_transfer"
    if amplitude_token == int(CONFIG["stress_test_amplitude_token"]):
        return "test_high_amplitude"
    if temperature_c in set(int(v) for v in CONFIG["heldout_temperature_validation_c"]):
        return "validation_temperature"
    return "train"


def load_dynamic_records() -> list[DynRecord]:
    records: list[DynRecord] = []
    with zipfile.ZipFile(RAW_ZIP) as zf:
        names = sorted(
            name
            for name in zf.namelist()
            if name.startswith("DYNdata/A002_DYN/")
            and name.lower().endswith(".mat")
        )
        for name in names:
            mat = loadmat(io.BytesIO(zf.read(name)), squeeze_me=True, struct_as_record=False)
            root = mat[str(CONFIG["dyn_root"])]
            script = getattr(root, str(CONFIG["dyn_script"]))
            cell_id, amplitude_token, temperature_c = parse_dyn_identity(name)

            time_s = field_array(script, "time")
            current_raw = field_array(script, "current")
            voltage_v = field_array(script, "voltage")
            chg_ah = field_array(script, "chgAh")
            dis_ah = field_array(script, "disAh")

            # The protocol uses discharge-positive current. The raw files use the
            # opposite sign in OCV discharge segments, so flip the sign here.
            current_a = -current_raw
            net_discharge_ah = (dis_ah - dis_ah[0]) - (chg_ah - chg_ah[0])
            mask = (
                np.isfinite(time_s)
                & np.isfinite(current_a)
                & np.isfinite(voltage_v)
                & np.isfinite(net_discharge_ah)
                & (voltage_v > 1.5)
                & (voltage_v < 4.0)
            )
            time_s = time_s[mask]
            current_a = current_a[mask]
            voltage_v = voltage_v[mask]
            net_discharge_ah = net_discharge_ah[mask]
            order = np.argsort(time_s)
            time_s = time_s[order]
            current_a = current_a[order]
            voltage_v = voltage_v[order]
            net_discharge_ah = net_discharge_ah[order]
            time_s = time_s - time_s[0]
            label = f"{cell_id}_{amplitude_token:02d}_{temperature_c:+d}C"
            records.append(
                DynRecord(
                    filename=name,
                    label=label,
                    cell_id=cell_id,
                    amplitude_token=amplitude_token,
                    temperature_c=temperature_c,
                    split=split_for_record(cell_id, amplitude_token, temperature_c),
                    time_s=time_s,
                    current_a=current_a,
                    voltage_v=voltage_v,
                    net_discharge_ah=net_discharge_ah,
                )
            )
    return records


def load_ocv_model() -> OcvModel:
    rows: list[dict[str, str]] = []
    with OCV_GRID_CSV.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["curve"] == str(CONFIG["ocv_curve"]):
                rows.append(row)
    train_temps = np.array(CONFIG["ocv_training_temperatures_c"], dtype=float)
    soc_values = sorted({float(row["soc"]) for row in rows})
    soc_grid = np.array(soc_values, dtype=float)
    lookup = {
        (int(row["temperature_c"]), float(row["soc"])): float(row["voltage"])
        for row in rows
        if int(row["temperature_c"]) in set(int(v) for v in train_temps)
    }
    matrix = np.empty((len(train_temps), len(soc_grid)), dtype=float)
    for i, temp in enumerate(train_temps):
        for j, soc in enumerate(soc_grid):
            matrix[i, j] = lookup[(int(temp), float(soc))]
    interpolator = RegularGridInterpolator(
        (train_temps, soc_grid),
        matrix,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    return OcvModel(interpolator=interpolator, soc_grid=soc_grid, temperatures_c=train_temps)


def sample_indices(record: DynRecord, stride: int, end_time_s: float | None = None) -> np.ndarray:
    if end_time_s is None:
        idx = np.arange(0, len(record.time_s), stride, dtype=int)
    else:
        idx = np.flatnonzero(record.time_s <= end_time_s)[::stride]
    if idx.size == 0:
        return np.array([0], dtype=int)
    return idx


def subset_record(record: DynRecord, idx: np.ndarray, suffix: str = "") -> DynRecord:
    label = record.label if not suffix else f"{record.label}{suffix}"
    return DynRecord(
        filename=record.filename,
        label=label,
        cell_id=record.cell_id,
        amplitude_token=record.amplitude_token,
        temperature_c=record.temperature_c,
        split=record.split,
        time_s=record.time_s[idx],
        current_a=record.current_a[idx],
        voltage_v=record.voltage_v[idx],
        net_discharge_ah=record.net_discharge_ah[idx],
    )


def score_indices(record: DynRecord) -> np.ndarray:
    idx = np.flatnonzero(record.time_s >= float(CONFIG["score_start_s"]))
    if idx.size == 0:
        return np.arange(len(record.time_s), dtype=int)
    return idx


def soc_from_s0(record: DynRecord, s0: float) -> np.ndarray:
    return s0 - record.net_discharge_ah / float(CONFIG["q_nominal_ah"])


def range_residual(soc: np.ndarray) -> np.ndarray:
    lower = np.maximum(0.0, 0.02 - soc)
    upper = np.maximum(0.0, soc - 0.98)
    return 4.0 * np.concatenate([lower, upper])


def initial_soc_guess(record: DynRecord, ocv_model: OcvModel, r0: float = 0.0) -> float:
    n = min(120, len(record.time_s))
    target_ocv = float(np.median(record.voltage_v[:n] + r0 * record.current_a[:n]))
    grid = np.linspace(0.04, 0.98, 350)
    ocv_values = ocv_model(grid, float(record.temperature_c))
    return float(grid[int(np.argmin(np.abs(ocv_values - target_ocv)))])


def simulate_v1(time_s: np.ndarray, current_a: np.ndarray, a_tau: float, b_c: float, v10: float) -> np.ndarray:
    v1 = np.empty_like(current_a, dtype=float)
    v1[0] = v10
    for k in range(len(current_a) - 1):
        dt = max(0.0, float(time_s[k + 1] - time_s[k]))
        alpha = math.exp(-a_tau * dt)
        if a_tau > 1e-12:
            gamma = (b_c / a_tau) * (1.0 - alpha)
        else:
            gamma = b_c * dt
        v1[k + 1] = alpha * v1[k] + gamma * current_a[k]
    return v1


def predict_voltage(
    record: DynRecord,
    ocv_model: OcvModel,
    model: str,
    global_params: dict[str, float],
    s0: float,
    v10: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    soc = soc_from_s0(record, s0)
    ocv = ocv_model(soc, float(record.temperature_c))
    if model == "ocv_only":
        return ocv, soc, np.zeros_like(ocv)
    r0 = float(global_params["R0_ohm"])
    if model == "ohmic_only":
        return ocv - r0 * record.current_a, soc, np.zeros_like(ocv)
    if model == "constant_ecm":
        v1 = simulate_v1(
            record.time_s,
            record.current_a,
            float(global_params["a_tau_1_per_s"]),
            float(global_params["b_C_v_per_a_s"]),
            v10,
        )
        return ocv - r0 * record.current_a - v1, soc, v1
    raise ValueError(f"Unknown model: {model}")


def fit_ohmic(records: list[DynRecord], ocv_model: OcvModel) -> tuple[dict[str, float], dict[str, dict[str, float]], Any]:
    train = [record for record in records if record.split == "train"]
    stride = int(CONFIG["fit_stride"])
    fit_pairs = [(record, subset_record(record, sample_indices(record, stride), "_fit")) for record in train]
    x0 = [0.02]
    lb = [0.0]
    ub = [0.5]
    for record in train:
        x0.append(initial_soc_guess(record, ocv_model, 0.02))
        lb.append(0.05)
        ub.append(0.99)

    def residual(x: np.ndarray) -> np.ndarray:
        r0 = float(x[0])
        parts: list[np.ndarray] = []
        for i, (_, fit_record) in enumerate(fit_pairs):
            s0 = float(x[1 + i])
            soc = soc_from_s0(fit_record, s0)
            ocv = ocv_model(soc, float(fit_record.temperature_c))
            pred = ocv - r0 * fit_record.current_a
            parts.append(pred - fit_record.voltage_v)
            parts.append(range_residual(soc))
        return np.concatenate(parts)

    result = least_squares(
        residual,
        np.array(x0, dtype=float),
        bounds=(np.array(lb, dtype=float), np.array(ub, dtype=float)),
        loss=str(CONFIG["least_squares_loss"]),
        f_scale=float(CONFIG["least_squares_f_scale_v"]),
        max_nfev=int(CONFIG["max_nfev_ohmic"]),
        x_scale="jac",
    )
    params = {"R0_ohm": float(result.x[0])}
    states = {record.label: {"s0": float(result.x[1 + i]), "v10": 0.0} for i, record in enumerate(train)}
    return params, states, result


def fit_ocv_train_states(records: list[DynRecord], ocv_model: OcvModel) -> dict[str, dict[str, float]]:
    train = [record for record in records if record.split == "train"]
    stride = int(CONFIG["fit_stride"])
    states: dict[str, dict[str, float]] = {}
    for record in train:
        fit_record = subset_record(record, sample_indices(record, stride), "_fit")
        x0 = np.array([initial_soc_guess(record, ocv_model, 0.0)], dtype=float)

        def residual(x: np.ndarray) -> np.ndarray:
            soc = soc_from_s0(fit_record, float(x[0]))
            pred = ocv_model(soc, float(fit_record.temperature_c))
            return np.concatenate([pred - fit_record.voltage_v, range_residual(soc)])

        result = least_squares(
            residual,
            x0,
            bounds=(np.array([0.05]), np.array([0.99])),
            loss=str(CONFIG["least_squares_loss"]),
            f_scale=float(CONFIG["least_squares_f_scale_v"]),
            max_nfev=80,
            x_scale="jac",
        )
        states[record.label] = {"s0": float(result.x[0]), "v10": 0.0}
    return states


def fit_constant_ecm(records: list[DynRecord], ocv_model: OcvModel) -> tuple[dict[str, float], dict[str, dict[str, float]], Any]:
    train = [record for record in records if record.split == "train"]
    stride = int(CONFIG["fit_stride"])
    fit_pairs = [(record, subset_record(record, sample_indices(record, stride), "_fit")) for record in train]
    x0 = [0.02, 0.005, 0.0001]
    lb = [0.0, 1e-5, 1e-7]
    ub = [0.5, 0.2, 0.05]
    for record in train:
        x0.extend([initial_soc_guess(record, ocv_model, 0.02), 0.0])
        lb.extend([0.05, -0.5])
        ub.extend([0.99, 0.5])

    def residual(x: np.ndarray) -> np.ndarray:
        params = {
            "R0_ohm": float(x[0]),
            "a_tau_1_per_s": float(x[1]),
            "b_C_v_per_a_s": float(x[2]),
        }
        parts: list[np.ndarray] = []
        offset = 3
        for i, (_, fit_record) in enumerate(fit_pairs):
            s0 = float(x[offset + 2 * i])
            v10 = float(x[offset + 2 * i + 1])
            pred, soc, _ = predict_voltage(fit_record, ocv_model, "constant_ecm", params, s0, v10)
            parts.append(pred - fit_record.voltage_v)
            parts.append(range_residual(soc))
        return np.concatenate(parts)

    result = least_squares(
        residual,
        np.array(x0, dtype=float),
        bounds=(np.array(lb, dtype=float), np.array(ub, dtype=float)),
        loss=str(CONFIG["least_squares_loss"]),
        f_scale=float(CONFIG["least_squares_f_scale_v"]),
        max_nfev=int(CONFIG["max_nfev_ecm"]),
        x_scale="jac",
    )
    params = {
        "R0_ohm": float(result.x[0]),
        "a_tau_1_per_s": float(result.x[1]),
        "b_C_v_per_a_s": float(result.x[2]),
        "tau_s": float(1.0 / result.x[1]),
        "R1_equiv_ohm": float(result.x[2] / result.x[1]),
        "C1_equiv_f": float(1.0 / result.x[2]),
    }
    offset = 3
    states = {
        record.label: {
            "s0": float(result.x[offset + 2 * i]),
            "v10": float(result.x[offset + 2 * i + 1]),
        }
        for i, record in enumerate(train)
    }
    return params, states, result


def fit_initial_state(
    record: DynRecord,
    ocv_model: OcvModel,
    model: str,
    global_params: dict[str, float],
) -> dict[str, float]:
    end_time = float(CONFIG["initialization_window_s"])
    idx = sample_indices(record, 1, end_time_s=end_time)
    fit_record = subset_record(record, idx, "_init")
    s_guess = initial_soc_guess(record, ocv_model, float(global_params.get("R0_ohm", 0.0)))
    if model == "constant_ecm":
        x0 = np.array([s_guess, 0.0], dtype=float)
        lb = np.array([0.05, -0.5], dtype=float)
        ub = np.array([0.99, 0.5], dtype=float)

        def residual(x: np.ndarray) -> np.ndarray:
            pred, soc, _ = predict_voltage(fit_record, ocv_model, model, global_params, float(x[0]), float(x[1]))
            return np.concatenate([pred - fit_record.voltage_v, range_residual(soc)])

        result = least_squares(
            residual,
            x0,
            bounds=(lb, ub),
            loss=str(CONFIG["least_squares_loss"]),
            f_scale=float(CONFIG["least_squares_f_scale_v"]),
            max_nfev=80,
            x_scale="jac",
        )
        return {"s0": float(result.x[0]), "v10": float(result.x[1]), "init_cost": float(result.cost)}

    x0 = np.array([s_guess], dtype=float)

    def residual_s(x: np.ndarray) -> np.ndarray:
        pred, soc, _ = predict_voltage(fit_record, ocv_model, model, global_params, float(x[0]), 0.0)
        return np.concatenate([pred - fit_record.voltage_v, range_residual(soc)])

    result = least_squares(
        residual_s,
        x0,
        bounds=(np.array([0.05]), np.array([0.99])),
        loss=str(CONFIG["least_squares_loss"]),
        f_scale=float(CONFIG["least_squares_f_scale_v"]),
        max_nfev=80,
        x_scale="jac",
    )
    return {"s0": float(result.x[0]), "v10": 0.0, "init_cost": float(result.cost)}


def error_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.std(y))
    return {
        "rmse_mv": 1000.0 * rmse,
        "mae_mv": 1000.0 * mae,
        "max_abs_mv": 1000.0 * float(np.max(np.abs(err))),
        "bias_mv": 1000.0 * float(np.mean(err)),
        "nrmse": rmse / denom if denom > 0 else float("nan"),
    }


def evaluate_models(
    records: list[DynRecord],
    ocv_model: OcvModel,
    model_params: dict[str, dict[str, float]],
    train_states: dict[str, dict[str, dict[str, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    split_acc: dict[tuple[str, str], dict[str, list[float]]] = {}
    model_order = ["ocv_only", "ohmic_only", "constant_ecm"]
    for model in model_order:
        params = model_params[model]
        for record in records:
            state = train_states.get(model, {}).get(record.label)
            init_method = "training_fit"
            if state is None:
                state = fit_initial_state(record, ocv_model, model, params)
                init_method = "prefix_output_error"
            pred, soc, v1 = predict_voltage(
                record,
                ocv_model,
                model,
                params,
                float(state["s0"]),
                float(state.get("v10", 0.0)),
            )
            idx = score_indices(record)
            valid = np.isfinite(pred[idx]) & np.isfinite(record.voltage_v[idx])
            score_y = record.voltage_v[idx][valid]
            score_pred = pred[idx][valid]
            score_soc = soc[idx][valid]
            met = error_metrics(score_y, score_pred)
            metrics_rows.append(
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
                    "v10": float(state.get("v10", 0.0)),
                    "soc_min": float(np.min(score_soc)),
                    "soc_max": float(np.max(score_soc)),
                    "stable_fraction": float(np.mean(valid)) if len(valid) else 0.0,
                    **met,
                }
            )
            key = (model, record.split)
            split_acc.setdefault(key, {"y": [], "pred": []})
            split_acc[key]["y"].extend(float(v) for v in score_y)
            split_acc[key]["pred"].extend(float(v) for v in score_pred)

            sample_idx = np.arange(0, len(record.time_s), int(CONFIG["prediction_sample_stride"]), dtype=int)
            for j in sample_idx:
                prediction_rows.append(
                    {
                        "model": model,
                        "split": record.split,
                        "record": record.label,
                        "time_s": float(record.time_s[j]),
                        "temperature_c": record.temperature_c,
                        "current_a_discharge_positive": float(record.current_a[j]),
                        "voltage_v": float(record.voltage_v[j]),
                        "prediction_v": float(pred[j]),
                        "error_mv": 1000.0 * float(pred[j] - record.voltage_v[j]),
                        "soc": float(soc[j]),
                        "v1_v": float(v1[j]),
                    }
                )

    split_rows: list[dict[str, Any]] = []
    split_summary: dict[str, dict[str, Any]] = {}
    for (model, split), values in sorted(split_acc.items()):
        y = np.array(values["y"], dtype=float)
        pred = np.array(values["pred"], dtype=float)
        met = error_metrics(y, pred)
        row = {
            "model": model,
            "split": split,
            "n_score": int(len(y)),
            **met,
        }
        split_rows.append(row)
        split_summary[f"{model}:{split}"] = row
    return metrics_rows, split_rows, prediction_rows, split_summary


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


def write_rmse_svg(path: Path, split_rows: list[dict[str, Any]], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_splits = ["train", "validation_temperature", "test_high_amplitude", "test_cell_transfer"]
    models = ["ocv_only", "ohmic_only", "constant_ecm"]
    labels_en = {
        "ocv_only": "OCV only",
        "ohmic_only": "Ohmic only",
        "constant_ecm": "Constant ECM",
        "train": "Train",
        "validation_temperature": "Held-out T",
        "test_high_amplitude": "High amp.",
        "test_cell_transfer": "Cell transfer",
    }
    labels_zh = {
        "ocv_only": "仅 OCV",
        "ohmic_only": "欧姆项",
        "constant_ecm": "常系数 ECM",
        "train": "训练",
        "validation_temperature": "留出温度",
        "test_high_amplitude": "高倍率",
        "test_cell_transfer": "跨电芯",
    }
    labels = labels_zh if zh else labels_en
    colors = {"ocv_only": "#4b5563", "ohmic_only": "#2563eb", "constant_ecm": "#dc2626"}
    lookup = {(row["model"], row["split"]): float(row["rmse_mv"]) for row in split_rows}
    y_max = max(lookup.values()) * 1.12
    left, top, width, height = 86.0, 68.0, 720.0, 320.0
    group_w = width / len(selected_splits)
    bar_w = 32.0
    gap = 8.0
    bars = []
    for i, split in enumerate(selected_splits):
        base_x = left + i * group_w + group_w / 2 - (1.5 * bar_w + gap)
        for j, model in enumerate(models):
            value = lookup.get((model, split), 0.0)
            bar_h = value / y_max * height
            x = base_x + j * (bar_w + gap)
            y = top + height - bar_h
            bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{colors[model]}"/>')
            bars.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 5:.2f}" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="10" fill="#111827">{value:.1f}</text>')
        bars.append(f'<text x="{left + i * group_w + group_w / 2:.2f}" y="420" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#374151">{labels[split]}</text>')
    legend = []
    for i, model in enumerate(models):
        y = 94 + i * 26
        legend.append(f'<rect x="830" y="{y - 12}" width="18" height="12" fill="{colors[model]}"/>')
        legend.append(f'<text x="856" y="{y - 2}" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">{labels[model]}</text>')
    title = "BATT-3 voltage RMSE by split" if not zh else "BATT-3 端电压 RMSE 分组结果"
    y_label = "RMSE (mV)" if not zh else "RMSE (mV)"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="980" height="460" viewBox="0 0 980 460">
  <rect width="980" height="460" fill="#ffffff"/>
  <text x="86" y="34" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  <rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#6b7280"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#6b7280"/>
  <text x="26" y="{top + height / 2}" transform="rotate(-90 26 {top + height / 2})" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{y_label}</text>
  {''.join(bars)}
  {''.join(legend)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_prediction_svg(path: Path, prediction_rows: list[dict[str, Any]], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validation_records = sorted({row["record"] for row in prediction_rows if row["split"] == "validation_temperature"})
    chosen = validation_records[:2]
    if not chosen:
        return
    models = ["voltage_v", "ocv_only", "ohmic_only", "constant_ecm"]
    colors = {"voltage_v": "#111827", "ocv_only": "#6b7280", "ohmic_only": "#2563eb", "constant_ecm": "#dc2626"}
    label_en = {"voltage_v": "Measured", "ocv_only": "OCV only", "ohmic_only": "Ohmic only", "constant_ecm": "Constant ECM"}
    label_zh = {"voltage_v": "实测", "ocv_only": "仅 OCV", "ohmic_only": "欧姆项", "constant_ecm": "常系数 ECM"}
    labels = label_zh if zh else label_en
    title = "BATT-3 held-out-temperature voltage prediction" if not zh else "BATT-3 留出温度端电压预测"
    x_label = "time (s)" if not zh else "时间 (s)"
    y_label = "terminal voltage (V)" if not zh else "端电压 (V)"
    panel_w, panel_h = 760.0, 165.0
    left, top0 = 80.0, 70.0
    panels = []
    for pidx, record_name in enumerate(chosen):
        top = top0 + pidx * 210.0
        sub = [row for row in prediction_rows if row["record"] == record_name]
        measured_rows = [row for row in sub if row["model"] == "ocv_only"]
        x = np.array([float(row["time_s"]) for row in measured_rows], dtype=float)
        y_meas = np.array([float(row["voltage_v"]) for row in measured_rows], dtype=float)
        y_series = [y_meas]
        model_arrays: dict[str, np.ndarray] = {"voltage_v": y_meas}
        for model in ["ocv_only", "ohmic_only", "constant_ecm"]:
            rows = [row for row in sub if row["model"] == model]
            arr = np.array([float(row["prediction_v"]) for row in rows], dtype=float)
            model_arrays[model] = arr
            y_series.append(arr)
        y_min = min(float(np.min(arr)) for arr in y_series) - 0.02
        y_max = max(float(np.max(arr)) for arr in y_series) + 0.02
        x_min, x_max = float(np.min(x)), float(np.max(x))
        panels.append(f'<text x="{left}" y="{top - 13}" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#111827">{record_name}</text>')
        panels.append(f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#ffffff" stroke="#d1d5db"/>')
        panels.append(f'<line x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" y2="{top + panel_h}" stroke="#6b7280"/>')
        panels.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_h}" stroke="#6b7280"/>')
        for model in models:
            dash = ' stroke-dasharray="5,4"' if model in {"ocv_only", "ohmic_only"} else ""
            stroke_w = "1.6" if model != "voltage_v" else "1.3"
            panels.append(
                f'<polyline points="{svg_polyline(x, model_arrays[model], x_min, x_max, y_min, y_max, left, top, panel_w, panel_h)}" '
                f'fill="none" stroke="{colors[model]}" stroke-width="{stroke_w}"{dash}/>'
            )
    legend = []
    for i, model in enumerate(models):
        y = 96 + i * 25
        dash = ' stroke-dasharray="5,4"' if model in {"ocv_only", "ohmic_only"} else ""
        legend.append(f'<line x1="858" y1="{y}" x2="895" y2="{y}" stroke="{colors[model]}" stroke-width="1.8"{dash}/>')
        legend.append(f'<text x="904" y="{y + 4}" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">{labels[model]}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1030" height="520" viewBox="0 0 1030 520">
  <rect width="1030" height="520" fill="#ffffff"/>
  <text x="80" y="34" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  {''.join(panels)}
  <text x="{left + panel_w / 2}" y="498" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{x_label}</text>
  <text x="25" y="260" transform="rotate(-90 25 260)" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{y_label}</text>
  {''.join(legend)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    if not RAW_ZIP.exists():
        raise FileNotFoundError(f"Missing raw zip: {RAW_ZIP}")
    if not OCV_GRID_CSV.exists():
        raise FileNotFoundError(f"Missing BATT-2 OCV grid: {OCV_GRID_CSV}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    records = load_dynamic_records()
    ocv_model = load_ocv_model()
    train_records = [record for record in records if record.split == "train"]
    if not train_records:
        raise RuntimeError("No BATT-3 training records were found.")

    ocv_states = fit_ocv_train_states(records, ocv_model)
    ohmic_params, ohmic_states, ohmic_result = fit_ohmic(records, ocv_model)
    ecm_params, ecm_states, ecm_result = fit_constant_ecm(records, ocv_model)
    model_params = {
        "ocv_only": {},
        "ohmic_only": ohmic_params,
        "constant_ecm": ecm_params,
    }
    train_states = {
        "ocv_only": ocv_states,
        "ohmic_only": ohmic_states,
        "constant_ecm": ecm_states,
    }
    metrics_rows, split_rows, prediction_rows, split_summary = evaluate_models(
        records,
        ocv_model,
        model_params,
        train_states,
    )

    metrics_by_traj_path = RESULT_DIR / "metrics_by_trajectory.csv"
    metrics_by_split_path = RESULT_DIR / "metrics_by_split.csv"
    prediction_path = RESULT_DIR / "predictions_sample.csv"
    metrics_json_path = RESULT_DIR / "metrics.json"
    provenance_path = PROVENANCE_DIR / "batt3_constant_ecm_provenance.json"

    write_csv(metrics_by_traj_path, metrics_rows)
    write_csv(metrics_by_split_path, split_rows)
    write_csv(prediction_path, prediction_rows)
    write_rmse_svg(FIGURE_DIR / "rmse_by_split.svg", split_rows, zh=False)
    write_rmse_svg(FIGURE_DIR / "rmse_by_split_zh.svg", split_rows, zh=True)
    write_prediction_svg(FIGURE_DIR / "heldout_temperature_voltage_prediction.svg", prediction_rows, zh=False)
    write_prediction_svg(FIGURE_DIR / "heldout_temperature_voltage_prediction_zh.svg", prediction_rows, zh=True)

    record_summary = [
        {
            "record": record.label,
            "filename": record.filename,
            "cell_id": record.cell_id,
            "amplitude_token": record.amplitude_token,
            "temperature_c": record.temperature_c,
            "split": record.split,
            "n_samples": int(len(record.time_s)),
            "duration_s": float(record.time_s[-1] - record.time_s[0]),
            "current_min_a": float(np.min(record.current_a)),
            "current_max_a": float(np.max(record.current_a)),
            "voltage_min_v": float(np.min(record.voltage_v)),
            "voltage_max_v": float(np.max(record.voltage_v)),
            "net_discharge_ah_min": float(np.min(record.net_discharge_ah)),
            "net_discharge_ah_max": float(np.max(record.net_discharge_ah)),
        }
        for record in records
    ]

    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "complete",
        "config": CONFIG,
        "raw_zip_sha256": sha256_file(RAW_ZIP),
        "record_count": len(records),
        "train_record_count": len(train_records),
        "record_summary": record_summary,
        "fit_summary": {
            "ohmic_only": {
                "params": ohmic_params,
                "cost": float(ohmic_result.cost),
                "optimality": float(ohmic_result.optimality),
                "nfev": int(ohmic_result.nfev),
                "success": bool(ohmic_result.success),
                "message": str(ohmic_result.message),
            },
            "constant_ecm": {
                "params": ecm_params,
                "cost": float(ecm_result.cost),
                "optimality": float(ecm_result.optimality),
                "nfev": int(ecm_result.nfev),
                "success": bool(ecm_result.success),
                "message": str(ecm_result.message),
            },
        },
        "train_initial_states": train_states,
        "split_summary": split_summary,
        "outputs": {
            "metrics_by_trajectory_csv": str(metrics_by_traj_path.relative_to(ROOT)),
            "metrics_by_split_csv": str(metrics_by_split_path.relative_to(ROOT)),
            "predictions_sample_csv": str(prediction_path.relative_to(ROOT)),
            "metrics_json": str(metrics_json_path.relative_to(ROOT)),
            "figures": [
                str((FIGURE_DIR / "rmse_by_split.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "rmse_by_split_zh.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "heldout_temperature_voltage_prediction.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "heldout_temperature_voltage_prediction_zh.svg").relative_to(ROOT)),
            ],
        },
    }
    metrics_json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    provenance = {
        **CONFIG,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "script": "scripts/run_batt3_constant_ecm.py",
        "outputs": metrics["outputs"] | {
            "provenance_json": str(provenance_path.relative_to(ROOT)),
        },
        "test_usage_rule": "Global ECM parameters are fitted only on split=train. Non-train records use only a short prefix for initial-state estimation, then score after the initialization window.",
        "next_action": "Use BATT-3 as the constant baseline for BATT-4 coefficient-slot ECM.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
