#!/usr/bin/env python3
"""Run SB-2b: windowed output-error diagnostics for Silverbox."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata as metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nonlinear_benchmarks as nb
import numpy as np
import scipy
from scipy.optimize import least_squares

import run_silverbox_sb1_linear_oe as sb1
import run_silverbox_sb2_slot_oe as sb2


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "external" / "silverbox"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "silverbox_sb2b_windowed_oe"
FIGURE_DIR = RESULT_DIR / "figures"
SB1_METRICS_PATH = ROOT / "results" / "silverbox_sb1_linear_oe" / "metrics.json"


CONFIG: dict[str, Any] = {
    "experiment_id": "SB2b_silverbox_windowed_output_error",
    "description": "Windowed output-error diagnostics for Silverbox linear and stiffness-slot models.",
    "source_dataset": "silverbox_data_audit",
    "dataloader": "nonlinear_benchmarks.Silverbox",
    "train_fit_fraction": 0.8,
    "window_length": 300,
    "state_initialization_window": 50,
    "residual_stride": 5,
    "fit_window_count": 12,
    "validation_window_count": 8,
    "test_window_count": 8,
    "linear_max_nfev": 45,
    "slot_max_nfev": 35,
    "init_max_nfev": 50,
    "max_abs_state": 5.0,
    "max_abs_b_scaled": 10.0,
    "max_abs_beta2_scaled": 2.0,
    "max_abs_x20_scaled": 50.0,
    "slot_beta2_starts_scaled": [0.0],
    "model_form_linear": "x1_dot=x2; x2_dot=-c*x2-k*x1+b*(u-u0); y=x1+y0",
    "model_form_slot": "x1_dot=x2; x2_dot=-c*x2-x1*(beta0+beta2*x1^2)+b*(u-u0); y=x1+y0",
    "test_usage_rule": "Official test records are used only after choosing the diagnostic setup from train_val.",
}


def package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not_installed"


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
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_sb1_model() -> dict[str, Any]:
    return json.loads(SB1_METRICS_PATH.read_text(encoding="utf-8"))["model"]


def make_windows(u: np.ndarray, y: np.ndarray, n_windows: int, window_length: int) -> list[dict[str, Any]]:
    if len(y) < window_length:
        raise ValueError("Record is shorter than the requested window length.")
    max_start = len(y) - window_length
    if n_windows <= 1:
        starts = np.array([0], dtype=int)
    else:
        starts = np.linspace(0, max_start, n_windows).round().astype(int)
    windows = []
    for idx, start in enumerate(starts):
        end = int(start + window_length)
        windows.append(
            {
                "window_id": int(idx),
                "start_index": int(start),
                "end_index": end,
                "u": np.asarray(u[start:end], dtype=float),
                "y": np.asarray(y[start:end], dtype=float),
            }
        )
    return windows


def rmse_mv_window(y_true: np.ndarray, y_pred: np.ndarray, n_init: int) -> float:
    return sb1.rmse_mv(y_true, y_pred, n_init)


def nrmse_window(y_true: np.ndarray, y_pred: np.ndarray, n_init: int) -> float:
    return sb1.nrmse(y_true, y_pred, n_init)


def unpack_linear(params: np.ndarray, b_scale: float, x2_scale: float, n_windows: int) -> tuple[float, float, float, np.ndarray]:
    c = float(np.exp(params[0]))
    k = float(np.exp(params[1]))
    b = float(params[2] * b_scale)
    x20 = np.asarray(params[3 : 3 + n_windows], dtype=float) * x2_scale
    return c, k, b, x20


def fit_windowed_linear(
    windows: list[dict[str, Any]],
    dt: float,
    sb1_model: dict[str, Any],
    y_scale: float,
    x2_scale: float,
) -> dict[str, Any]:
    n_init = int(CONFIG["state_initialization_window"])
    stride = int(CONFIG["residual_stride"])
    n_windows = len(windows)
    b_scale = max(abs(float(sb1_model["b"])), 1.0)
    start = np.concatenate(
        [
            np.array([np.log(float(sb1_model["c"])), np.log(float(sb1_model["k"])), float(sb1_model["b"]) / b_scale]),
            np.zeros(n_windows, dtype=float),
        ]
    )

    def residual(params: np.ndarray) -> np.ndarray:
        c, k, b, x20_values = unpack_linear(params, b_scale, x2_scale, n_windows)
        pieces = []
        for window, x20 in zip(windows, x20_values):
            y = window["y"]
            pred = sb1.simulate_linear_oe(
                window["u"],
                float(y[0]),
                float(x20),
                dt,
                c,
                k,
                b,
                float(sb1_model["u_offset"]),
                float(sb1_model["y_offset"]),
            )
            pieces.append((pred[n_init::stride] - y[n_init::stride]) / y_scale)
        return np.concatenate(pieces)

    lower = np.concatenate(
        [
            np.array([np.log(1e-3), np.log(1e-3), -float(CONFIG["max_abs_b_scaled"])]),
            np.full(n_windows, -float(CONFIG["max_abs_x20_scaled"])),
        ]
    )
    upper = np.concatenate(
        [
            np.array([np.log(1e5), np.log(1e8), float(CONFIG["max_abs_b_scaled"])]),
            np.full(n_windows, float(CONFIG["max_abs_x20_scaled"])),
        ]
    )
    result = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        max_nfev=int(CONFIG["linear_max_nfev"]),
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    c, k, b, x20_values = unpack_linear(result.x, b_scale, x2_scale, n_windows)
    return {
        "model": "windowed_linear",
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "c": c,
        "k": k,
        "b": b,
        "x20_fit_mean": float(np.mean(x20_values)),
        "x20_fit_std": float(np.std(x20_values)),
        "b_scale": b_scale,
        "params_scaled": [float(v) for v in result.x[:3]],
    }


def unpack_slot(params: np.ndarray, beta2_scale: float, b_scale: float, x2_scale: float, n_windows: int) -> tuple[float, float, float, float, np.ndarray]:
    c = float(np.exp(params[0]))
    beta0 = float(np.exp(params[1]))
    beta2 = float(params[2] * beta2_scale)
    b = float(params[3] * b_scale)
    x20 = np.asarray(params[4 : 4 + n_windows], dtype=float) * x2_scale
    return c, beta0, beta2, b, x20


def fit_windowed_slot(
    windows: list[dict[str, Any]],
    dt: float,
    sb1_model: dict[str, Any],
    y_scale: float,
    x2_scale: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_init = int(CONFIG["state_initialization_window"])
    stride = int(CONFIG["residual_stride"])
    n_windows = len(windows)
    b_scale = max(abs(float(sb1_model["b"])), 1.0)
    beta2_scale = max(float(sb1_model["k"]) / max(y_scale * y_scale, 1e-12), 1.0)
    candidate_rows = []
    best: dict[str, Any] | None = None

    for beta2_start in CONFIG["slot_beta2_starts_scaled"]:
        start = np.concatenate(
            [
                np.array(
                    [
                        np.log(float(sb1_model["c"])),
                        np.log(float(sb1_model["k"])),
                        float(beta2_start),
                        float(sb1_model["b"]) / b_scale,
                    ],
                    dtype=float,
                ),
                np.zeros(n_windows, dtype=float),
            ]
        )

        def residual(params: np.ndarray) -> np.ndarray:
            c, beta0, beta2, b, x20_values = unpack_slot(params, beta2_scale, b_scale, x2_scale, n_windows)
            pieces = []
            for window, x20 in zip(windows, x20_values):
                y = window["y"]
                pred, stable = sb2.simulate_slot_oe(
                    window["u"],
                    float(y[0]),
                    float(x20),
                    dt,
                    c,
                    beta0,
                    beta2,
                    b,
                    float(sb1_model["u_offset"]),
                    float(sb1_model["y_offset"]),
                    float(CONFIG["max_abs_state"]),
                )
                if not stable or not np.all(np.isfinite(pred[n_init::stride])):
                    pieces.append(np.full(len(y[n_init::stride]), 1e6, dtype=float))
                else:
                    pieces.append((pred[n_init::stride] - y[n_init::stride]) / y_scale)
            return np.concatenate(pieces)

        lower = np.concatenate(
            [
                np.array(
                    [
                        np.log(1e-3),
                        np.log(1e-3),
                        -float(CONFIG["max_abs_beta2_scaled"]),
                        -float(CONFIG["max_abs_b_scaled"]),
                    ]
                ),
                np.full(n_windows, -float(CONFIG["max_abs_x20_scaled"])),
            ]
        )
        upper = np.concatenate(
            [
                np.array(
                    [
                        np.log(1e5),
                        np.log(1e8),
                        float(CONFIG["max_abs_beta2_scaled"]),
                        float(CONFIG["max_abs_b_scaled"]),
                    ]
                ),
                np.full(n_windows, float(CONFIG["max_abs_x20_scaled"])),
            ]
        )
        result = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=int(CONFIG["slot_max_nfev"]),
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        c, beta0, beta2, b, x20_values = unpack_slot(result.x, beta2_scale, b_scale, x2_scale, n_windows)
        row = {
            "model": "windowed_slot",
            "start": f"beta2_scaled_{beta2_start:g}",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "c": c,
            "beta0": beta0,
            "beta2": beta2,
            "beta2_scaled": beta2 / beta2_scale,
            "b": b,
            "x20_fit_mean": float(np.mean(x20_values)),
            "x20_fit_std": float(np.std(x20_values)),
            "b_scale": b_scale,
            "beta2_scale": beta2_scale,
            "params_scaled": [float(v) for v in result.x[:4]],
        }
        candidate_rows.append(row)
        if best is None or float(row["cost"]) < float(best["cost"]):
            best = row

    if best is None:
        raise RuntimeError("No slot candidate was fitted.")
    return best, candidate_rows


def init_velocity_linear(window: dict[str, Any], params: dict[str, Any], dt: float, sb1_model: dict[str, Any], y_scale: float, x2_scale: float) -> float:
    n_init = int(CONFIG["state_initialization_window"])
    u = window["u"][:n_init]
    y = window["y"][:n_init]

    def residual(x2_scaled: np.ndarray) -> np.ndarray:
        pred = sb1.simulate_linear_oe(
            u,
            float(y[0]),
            float(x2_scaled[0] * x2_scale),
            dt,
            float(params["c"]),
            float(params["k"]),
            float(params["b"]),
            float(sb1_model["u_offset"]),
            float(sb1_model["y_offset"]),
        )
        return (pred - y) / y_scale

    result = least_squares(residual, np.array([0.0]), bounds=(np.array([-100.0]), np.array([100.0])), max_nfev=int(CONFIG["init_max_nfev"]))
    return float(result.x[0] * x2_scale)


def init_velocity_slot(window: dict[str, Any], params: dict[str, Any], dt: float, sb1_model: dict[str, Any], y_scale: float, x2_scale: float) -> float:
    n_init = int(CONFIG["state_initialization_window"])
    u = window["u"][:n_init]
    y = window["y"][:n_init]

    def residual(x2_scaled: np.ndarray) -> np.ndarray:
        pred, stable = sb2.simulate_slot_oe(
            u,
            float(y[0]),
            float(x2_scaled[0] * x2_scale),
            dt,
            float(params["c"]),
            float(params["beta0"]),
            float(params["beta2"]),
            float(params["b"]),
            float(sb1_model["u_offset"]),
            float(sb1_model["y_offset"]),
            float(CONFIG["max_abs_state"]),
        )
        if not stable:
            return np.full(len(y), 1e6, dtype=float)
        return (pred - y) / y_scale

    result = least_squares(residual, np.array([0.0]), bounds=(np.array([-100.0]), np.array([100.0])), max_nfev=int(CONFIG["init_max_nfev"]))
    return float(result.x[0] * x2_scale)


def evaluate_windows(
    record: str,
    split: str,
    windows: list[dict[str, Any]],
    model_name: str,
    params: dict[str, Any],
    dt: float,
    sb1_model: dict[str, Any],
    y_scale: float,
    x2_scale: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_init = int(CONFIG["state_initialization_window"])
    rows = []
    for window in windows:
        y = window["y"]
        if model_name == "windowed_linear":
            x20 = init_velocity_linear(window, params, dt, sb1_model, y_scale, x2_scale)
            pred = sb1.simulate_linear_oe(
                window["u"],
                float(y[0]),
                x20,
                dt,
                float(params["c"]),
                float(params["k"]),
                float(params["b"]),
                float(sb1_model["u_offset"]),
                float(sb1_model["y_offset"]),
            )
            stable = True
        elif model_name == "windowed_slot":
            x20 = init_velocity_slot(window, params, dt, sb1_model, y_scale, x2_scale)
            pred, stable = sb2.simulate_slot_oe(
                window["u"],
                float(y[0]),
                x20,
                dt,
                float(params["c"]),
                float(params["beta0"]),
                float(params["beta2"]),
                float(params["b"]),
                float(sb1_model["u_offset"]),
                float(sb1_model["y_offset"]),
                float(CONFIG["max_abs_state"]),
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        rmse = float("nan") if not stable else rmse_mv_window(y, pred, n_init)
        norm_rmse = float("nan") if not stable else nrmse_window(y, pred, n_init)
        rows.append(
            {
                "record": record,
                "split": split,
                "model": model_name,
                "window_id": window["window_id"],
                "start_index": window["start_index"],
                "end_index": window["end_index"],
                "n_init": n_init,
                "stable": bool(stable),
                "x20_initialized": x20,
                "rmse_mv": rmse,
                "nrmse": norm_rmse,
            }
        )

    rmse_values = np.array([float(row["rmse_mv"]) for row in rows], dtype=float)
    nrmse_values = np.array([float(row["nrmse"]) for row in rows], dtype=float)
    finite_rmse = rmse_values[np.isfinite(rmse_values)]
    finite_nrmse = nrmse_values[np.isfinite(nrmse_values)]
    if len(finite_rmse) == 0:
        rmse_mean = rmse_median = rmse_std = float("nan")
    else:
        rmse_mean = float(np.mean(finite_rmse))
        rmse_median = float(np.median(finite_rmse))
        rmse_std = float(np.std(finite_rmse))
    if len(finite_nrmse) == 0:
        nrmse_mean = nrmse_median = float("nan")
    else:
        nrmse_mean = float(np.mean(finite_nrmse))
        nrmse_median = float(np.median(finite_nrmse))
    summary = {
        "record": record,
        "split": split,
        "model": model_name,
        "n_windows": len(rows),
        "window_length": int(CONFIG["window_length"]),
        "n_init": n_init,
        "forecast_horizon": int(CONFIG["window_length"] - CONFIG["state_initialization_window"]),
        "mean_rmse_mv": rmse_mean,
        "median_rmse_mv": rmse_median,
        "std_rmse_mv": rmse_std,
        "mean_nrmse": nrmse_mean,
        "median_nrmse": nrmse_median,
        "stable_fraction": float(np.mean([bool(row["stable"]) for row in rows])),
    }
    return summary, rows


def add_improvements(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    linear_by_record = {
        (row["record"], row["split"]): float(row["mean_rmse_mv"])
        for row in summary_rows
        if row["model"] == "windowed_linear"
    }
    out = []
    for row in summary_rows:
        row = dict(row)
        baseline = linear_by_record.get((row["record"], row["split"]), float("nan"))
        current = float(row["mean_rmse_mv"])
        if row["model"] == "windowed_slot" and np.isfinite(baseline) and baseline:
            row["improvement_vs_windowed_linear_percent"] = 100.0 * (baseline - current) / baseline
        else:
            row["improvement_vs_windowed_linear_percent"] = ""
        out.append(row)
    return out


def write_metric_bar_svg(path: Path, summary_rows: list[dict[str, Any]], language: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = ["validation", "test_multisine", "test_arrow_full", "test_arrow_no_extrapolation"]
    models = ["windowed_linear", "windowed_slot"]
    values = {
        (row["record"], row["model"]): float(row["mean_rmse_mv"])
        for row in summary_rows
        if row["record"] in records
    }
    width, height = 980, 520
    left, top, plot_w, plot_h = 90.0, 70.0, 820.0, 330.0
    finite_values = [value for value in values.values() if np.isfinite(value)]
    ymax = (max(finite_values) if finite_values else 1.0) * 1.12
    colors = {"windowed_linear": "#2563eb", "windowed_slot": "#dc2626"}
    if language == "zh":
        title = "Silverbox SB-2b 窗口化 output-error 诊断"
        y_label = "平均 RMSE (mV)"
        model_labels = {"windowed_linear": "窗口化线性 OE", "windowed_slot": "窗口化刚度槽 OE"}
    else:
        title = "Silverbox SB-2b Windowed Output-Error Diagnostics"
        y_label = "Mean RMSE (mV)"
        model_labels = {"windowed_linear": "Windowed linear OE", "windowed_slot": "Windowed stiffness-slot OE"}
    group_w = plot_w / len(records)
    bar_w = group_w * 0.28
    bars = []
    labels = []
    for i, record in enumerate(records):
        cx = left + group_w * (i + 0.5)
        for j, model in enumerate(models):
            value = values[(record, model)]
            if not np.isfinite(value):
                value = 0.0
            x = cx + (j - 0.5) * bar_w * 1.35
            bar_h = value / ymax * plot_h
            bars.append(
                f'<rect x="{x - bar_w / 2:.2f}" y="{top + plot_h - bar_h:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{colors[model]}"/>'
            )
            bars.append(
                f'<text x="{x:.2f}" y="{top + plot_h - bar_h - 6:.2f}" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="11" fill="#111827">{value:.2f}</text>'
            )
        labels.append(
            f'<text x="{cx:.2f}" y="{top + plot_h + 28:.2f}" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="11" fill="#374151">{record}</text>'
        )
    legend = []
    for idx, model in enumerate(models):
        lx = 610 + idx * 170
        legend.append(f'<rect x="{lx}" y="28" width="18" height="12" fill="{colors[model]}"/>')
        legend.append(
            f'<text x="{lx + 26}" y="39" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">{model_labels[model]}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  <text x="80" y="38" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  {''.join(legend)}
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#6b7280"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#6b7280"/>
  <text x="26" y="{top + plot_h / 2}" transform="rotate(-90 26 {top + plot_h / 2})" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{y_label}</text>
  {''.join(bars)}
  {''.join(labels)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    sb1_model = load_sb1_model()
    train_val, test = nb.Silverbox(dir_placement=str(RAW_ROOT), force_download=False)
    records = {
        "test_multisine": test[0],
        "test_arrow_full": test[1],
        "test_arrow_no_extrapolation": test[2],
    }

    u_all = np.asarray(train_val.u, dtype=float)
    y_all = np.asarray(train_val.y, dtype=float)
    dt = float(train_val.sampling_time)
    split_idx = int(len(y_all) * float(CONFIG["train_fit_fraction"]))
    u_fit, y_fit = u_all[:split_idx], y_all[:split_idx]
    u_val, y_val = u_all[split_idx:], y_all[split_idx:]
    y_scale = float(sb1_model["y_scale"])
    x2_scale = float(sb1_model["x2_scale"])

    fit_windows = make_windows(u_fit, y_fit, int(CONFIG["fit_window_count"]), int(CONFIG["window_length"]))
    linear_params = fit_windowed_linear(fit_windows, dt, sb1_model, y_scale, x2_scale)
    slot_params, slot_candidate_rows = fit_windowed_slot(fit_windows, dt, sb1_model, y_scale, x2_scale)

    eval_window_groups = {
        "train_fit": ("internal_fit", fit_windows),
        "validation": ("internal_validation", make_windows(u_val, y_val, int(CONFIG["validation_window_count"]), int(CONFIG["window_length"]))),
        "test_multisine": (
            "official_test",
            make_windows(np.asarray(records["test_multisine"].u, dtype=float), np.asarray(records["test_multisine"].y, dtype=float), int(CONFIG["test_window_count"]), int(CONFIG["window_length"])),
        ),
        "test_arrow_full": (
            "official_test",
            make_windows(np.asarray(records["test_arrow_full"].u, dtype=float), np.asarray(records["test_arrow_full"].y, dtype=float), int(CONFIG["test_window_count"]), int(CONFIG["window_length"])),
        ),
        "test_arrow_no_extrapolation": (
            "official_test",
            make_windows(
                np.asarray(records["test_arrow_no_extrapolation"].u, dtype=float),
                np.asarray(records["test_arrow_no_extrapolation"].y, dtype=float),
                int(CONFIG["test_window_count"]),
                int(CONFIG["window_length"]),
            ),
        ),
    }

    summary_rows = []
    window_rows = []
    for record, (split, windows) in eval_window_groups.items():
        for model_name, params in [("windowed_linear", linear_params), ("windowed_slot", slot_params)]:
            summary, rows = evaluate_windows(record, split, windows, model_name, params, dt, sb1_model, y_scale, x2_scale)
            summary_rows.append(summary)
            window_rows.extend(rows)
    summary_rows = add_improvements(summary_rows)

    parameter_rows = [
        {key: value for key, value in linear_params.items() if key != "params_scaled"},
        {key: value for key, value in slot_params.items() if key != "params_scaled"},
    ]
    write_csv(RESULT_DIR / "parameters.csv", parameter_rows)
    write_csv(RESULT_DIR / "slot_candidate_starts.csv", [{key: value for key, value in row.items() if key != "params_scaled"} for row in slot_candidate_rows])
    write_csv(RESULT_DIR / "summary_by_record_model.csv", summary_rows)
    write_csv(RESULT_DIR / "window_metrics.csv", window_rows)

    write_metric_bar_svg(FIGURE_DIR / "windowed_rmse_by_record.svg", summary_rows, "en")
    write_metric_bar_svg(FIGURE_DIR / "windowed_rmse_by_record_zh.svg", summary_rows, "zh")
    sb2.write_coefficient_svg(
        FIGURE_DIR / "stiffness_coefficient.svg",
        float(slot_params["beta0"]),
        float(slot_params["beta2"]),
        float(np.min(y_all - float(sb1_model["y_offset"]))),
        float(np.max(y_all - float(sb1_model["y_offset"]))),
        "Silverbox SB-2b Windowed Stiffness Coefficient",
        {"x": "Centered output y-y0 (V)", "y": "p_k(y)"},
    )
    sb2.write_coefficient_svg(
        FIGURE_DIR / "stiffness_coefficient_zh.svg",
        float(slot_params["beta0"]),
        float(slot_params["beta2"]),
        float(np.min(y_all - float(sb1_model["y_offset"]))),
        float(np.max(y_all - float(sb1_model["y_offset"]))),
        "Silverbox SB-2b 窗口化刚度系数函数",
        {"x": "中心化输出 y-y0 (V)", "y": "p_k(y)"},
    )

    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "linear_model": linear_params,
        "slot_model": slot_params,
        "slot_candidate_starts": slot_candidate_rows,
        "summary_by_record_model": summary_rows,
    }
    metrics_path = RESULT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    provenance_path = PROVENANCE_DIR / "silverbox_sb2b_windowed_oe_provenance.json"
    provenance = {
        **CONFIG,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "nonlinear-benchmarks": package_version("nonlinear-benchmarks"),
            "numpy": package_version("numpy"),
            "scipy": scipy.__version__,
        },
        "source_files": {
            "raw_audit_provenance": "data/provenance/silverbox_data_audit_provenance.json",
            "sb1_metrics": "results/silverbox_sb1_linear_oe/metrics.json",
            "sb2_metrics": "results/silverbox_sb2_slot_oe/metrics.json",
        },
        "outputs": {
            "parameters_csv": "results/silverbox_sb2b_windowed_oe/parameters.csv",
            "slot_candidate_starts_csv": "results/silverbox_sb2b_windowed_oe/slot_candidate_starts.csv",
            "summary_by_record_model_csv": "results/silverbox_sb2b_windowed_oe/summary_by_record_model.csv",
            "window_metrics_csv": "results/silverbox_sb2b_windowed_oe/window_metrics.csv",
            "metrics_json": "results/silverbox_sb2b_windowed_oe/metrics.json",
            "hashes_json": "results/silverbox_sb2b_windowed_oe/hashes.json",
            "figures": "results/silverbox_sb2b_windowed_oe/figures/",
        },
        "test_usage_rule": CONFIG["test_usage_rule"],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    hash_targets = [
        RESULT_DIR / "parameters.csv",
        RESULT_DIR / "slot_candidate_starts.csv",
        RESULT_DIR / "summary_by_record_model.csv",
        RESULT_DIR / "window_metrics.csv",
        metrics_path,
        provenance_path,
    ]
    hashes = {f"{path.relative_to(ROOT)}_sha256": sha256_file(path) for path in hash_targets}
    hashes_path = RESULT_DIR / "hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
