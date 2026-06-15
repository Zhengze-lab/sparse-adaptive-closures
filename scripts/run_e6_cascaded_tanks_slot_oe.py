#!/usr/bin/env python3
"""Run E6: Cascaded Tanks level-adaptive outflow slot output-error experiment."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import nonlinear_benchmarks as nb
import numpy as np
import scipy
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "external" / "cascaded_tanks"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "e6_cascaded_tanks_slot_oe"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG: dict[str, Any] = {
    "experiment_id": "E6_cascaded_tanks_slot_output_error",
    "description": "Real cascaded-tanks output-error test: local linear outflow vs level-adaptive square-root outflow slot.",
    "source_dataset": "Cascaded Tanks benchmark",
    "source_url": "https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks",
    "source_data_url": "https://data.4tu.nl/file/d4810b78-6cdd-48fe-8950-9bd601e5f47f/3b697e42-01a4-4979-a370-813a456c36f5",
    "dataloader": "nonlinear_benchmarks.Cascaded_Tanks",
    "citation": (
        "M. Schoukens, P. Mattsson, T. Wigren, and J. P. Noel. Cascaded tanks benchmark "
        "combining soft and hard nonlinearities. Workshop on Nonlinear System Identification "
        "Benchmarks, Brussels, 2016."
    ),
    "sampling_time_s": 4.0,
    "state_initialization_window": 50,
    "fit_stride": 4,
    "prediction_sample_stride": 4,
    "fit_max_nfev": 80,
    "init_max_nfev": 80,
    "models": {
        "B0_linear_outflow": "dx1=-a1*x1+b*u; dx2=a2*x1-a3*x2; y=x2",
        "B3_sqrt_outflow_slot": "dx1=-k1*sqrt(x1)+b*u; dx2=k2*sqrt(x1)-k3*sqrt(x2); y=x2",
    },
    "test_usage_rule": "The benchmark test record is used only for final evaluation; the first 50 samples are used for state initialization.",
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_records() -> dict[str, dict[str, Any]]:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    train, test = nb.Cascaded_Tanks(dir_placement=str(RAW_ROOT), force_download=False)
    return {
        "estimation": {
            "record": train,
            "u": np.asarray(train.u, dtype=float),
            "y": np.asarray(train.y, dtype=float),
            "sampling_time": float(train.sampling_time),
            "state_initialization_window": 0,
        },
        "test": {
            "record": test,
            "u": np.asarray(test.u, dtype=float),
            "y": np.asarray(test.y, dtype=float),
            "sampling_time": float(test.sampling_time),
            "state_initialization_window": int(getattr(test, "state_initialization_window_length", CONFIG["state_initialization_window"])),
        },
    }


def rhs_linear(x: np.ndarray, u: float, params: dict[str, float]) -> np.ndarray:
    x1, x2 = float(x[0]), float(x[1])
    return np.array(
        [
            -params["a1"] * x1 + params["b"] * u,
            params["a2"] * x1 - params["a3"] * x2,
        ],
        dtype=float,
    )


def rhs_sqrt(x: np.ndarray, u: float, params: dict[str, float]) -> np.ndarray:
    x1, x2 = max(float(x[0]), 0.0), max(float(x[1]), 0.0)
    return np.array(
        [
            -params["k1"] * math.sqrt(x1) + params["b"] * u,
            params["k2"] * math.sqrt(x1) - params["k3"] * math.sqrt(x2),
        ],
        dtype=float,
    )


def rk4_step(rhs: Callable[[np.ndarray, float, dict[str, float]], np.ndarray], x: np.ndarray, u: float, dt: float, params: dict[str, float]) -> np.ndarray:
    k1 = rhs(x, u, params)
    k2 = rhs(x + 0.5 * dt * k1, u, params)
    k3 = rhs(x + 0.5 * dt * k2, u, params)
    k4 = rhs(x + dt * k3, u, params)
    nxt = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if not np.all(np.isfinite(nxt)):
        return np.array([100.0, 100.0], dtype=float)
    return np.clip(nxt, 0.0, 100.0)


def simulate(
    u: np.ndarray,
    y0: float,
    x10: float,
    dt: float,
    model: str,
    params: dict[str, float],
) -> np.ndarray:
    rhs = rhs_linear if model == "B0_linear_outflow" else rhs_sqrt
    yhat = np.empty(len(u), dtype=float)
    x = np.array([max(x10, 0.0), max(y0, 0.0)], dtype=float)
    yhat[0] = x[1]
    for idx in range(len(u) - 1):
        x = rk4_step(rhs, x, float(u[idx]), dt, params)
        yhat[idx + 1] = x[1]
    return yhat


def unpack(model: str, raw: np.ndarray) -> tuple[dict[str, float], float]:
    values = np.exp(np.clip(raw, -18.0, 8.0))
    if model == "B0_linear_outflow":
        names = ["a1", "a2", "a3", "b", "x10"]
    else:
        names = ["k1", "k2", "k3", "b", "x10"]
    params = {name: float(value) for name, value in zip(names[:-1], values[:-1])}
    return params, float(values[-1])


def error_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_init: int) -> dict[str, float]:
    y = y_true[n_init:]
    pred = y_pred[n_init:]
    residual = pred - y
    rmse = float(np.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.std(y)) or 1.0
    nrmse = rmse / denom
    fit_percent = 100.0 * (1.0 - nrmse)
    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "fit_percent": fit_percent,
        "bias": float(np.mean(residual)),
        "max_abs_error": float(np.max(np.abs(residual))),
    }


def fit_model(model: str, u: np.ndarray, y: np.ndarray, dt: float, starts: list[np.ndarray]) -> dict[str, Any]:
    y_scale = float(np.std(y[int(CONFIG["state_initialization_window"]) :])) or 1.0
    n_init = int(CONFIG["state_initialization_window"])
    stride = int(CONFIG["fit_stride"])

    def residual(raw: np.ndarray) -> np.ndarray:
        params, x10 = unpack(model, raw)
        pred = simulate(u, y[0], x10, dt, model, params)
        return (pred[n_init::stride] - y[n_init::stride]) / y_scale

    best: dict[str, Any] | None = None
    for idx, start in enumerate(starts):
        result = least_squares(
            residual,
            start,
            max_nfev=int(CONFIG["fit_max_nfev"]),
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        params, x10 = unpack(model, result.x)
        pred = simulate(u, y[0], x10, dt, model, params)
        metrics = error_metrics(y, pred, n_init)
        candidate = {
            "start_index": idx,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "params": params,
            "x10_estimation": x10,
            "raw_params": [float(v) for v in result.x],
            "prediction": pred,
            "metrics": metrics,
        }
        if best is None or metrics["nrmse"] < best["metrics"]["nrmse"]:
            best = candidate
    assert best is not None
    return best


def make_starts(model: str, y: np.ndarray, u: np.ndarray) -> list[np.ndarray]:
    y_mean = max(float(np.mean(y)), 1e-3)
    u_mean = max(float(np.mean(u)), 1e-3)
    starts: list[np.ndarray] = []
    if model == "B0_linear_outflow":
        bases = [
            [0.003, 0.003, 0.003, 0.006 * y_mean / u_mean, y[0]],
            [0.008, 0.006, 0.006, 0.012 * y_mean / u_mean, y_mean],
        ]
    else:
        sqrt_y = math.sqrt(y_mean)
        bases = [
            [0.02, 0.02, 0.02, 0.01 * sqrt_y / u_mean, y[0]],
            [0.05, 0.04, 0.04, 0.025 * sqrt_y / u_mean, y_mean],
        ]
    for base in bases:
        starts.append(np.log(np.maximum(np.array(base, dtype=float), 1e-8)))
    return starts


def initialize_x10(
    model: str,
    params: dict[str, float],
    u: np.ndarray,
    y: np.ndarray,
    dt: float,
    x10_start: float,
    n_init: int,
) -> float:
    n_init = min(max(int(n_init), 2), len(y))
    y_scale = float(np.std(y[:n_init])) or 1.0

    def residual(log_x10: np.ndarray) -> np.ndarray:
        x10 = float(np.exp(np.clip(log_x10[0], -18.0, 8.0)))
        pred = simulate(u[:n_init], y[0], x10, dt, model, params)
        return (pred - y[:n_init]) / y_scale

    result = least_squares(
        residual,
        np.array([math.log(max(x10_start, 1e-8))], dtype=float),
        max_nfev=int(CONFIG["init_max_nfev"]),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    return float(np.exp(np.clip(result.x[0], -18.0, 8.0)))


def parameter_rows(model: str, fit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in fit["params"].items():
        rows.append({"model": model, "parameter": key, "value": float(value)})
    rows.append({"model": model, "parameter": "x10_estimation", "value": float(fit["x10_estimation"])})
    rows.append({"model": model, "parameter": "start_index", "value": int(fit["start_index"])})
    return rows


def collect_raw_files(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(raw_root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def write_rollout_svg(path: Path, records: dict[str, dict[str, Any]], predictions: dict[tuple[str, str], np.ndarray], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 520
    margin = {"left": 70, "right": 20, "top": 45, "bottom": 55}
    panel_gap = 40
    panel_width = (width - margin["left"] - margin["right"] - panel_gap) / 2.0
    panel_height = height - margin["top"] - margin["bottom"]
    colors = {"measured": "#000000", "B0_linear_outflow": "#D55E00", "B3_sqrt_outflow_slot": "#0072B2"}
    dashes = {"measured": "", "B0_linear_outflow": "7 4", "B3_sqrt_outflow_slot": "3 2"}
    labels_en = {"measured": "Measured output", "B0_linear_outflow": "B0 fixed linear outflow", "B3_sqrt_outflow_slot": "B3 level-adaptive outflow slot"}
    labels_zh = {"measured": "实测输出", "B0_linear_outflow": "B0 固定线性出流", "B3_sqrt_outflow_slot": "B3 液位自适应出流槽"}
    labels = labels_zh if zh else labels_en
    title = "E6 串联水箱 output-error rollout" if zh else "E6 Cascaded Tanks output-error rollout"

    all_y = np.concatenate([records[name]["y"] for name in records])
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    def sx(idx: int, n: int, panel_idx: int) -> float:
        x0 = margin["left"] + panel_idx * (panel_width + panel_gap)
        return x0 + panel_width * idx / max(n - 1, 1)

    def sy(value: float) -> float:
        return margin["top"] + panel_height * (1.0 - (value - y_min) / (y_max - y_min))

    def polyline(values: np.ndarray, panel_idx: int) -> str:
        step = max(1, len(values) // 400)
        pts = [f"{sx(i, len(values), panel_idx):.2f},{sy(float(values[i])):.2f}" for i in range(0, len(values), step)]
        return " ".join(pts)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="25" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">{title}</text>',
    ]
    for panel_idx, split in enumerate(["estimation", "test"]):
        x0 = margin["left"] + panel_idx * (panel_width + panel_gap)
        title_split = "估计集" if (zh and split == "estimation") else "测试集" if zh else split.capitalize()
        parts.append(f'<text x="{x0 + panel_width/2:.1f}" y="{margin["top"] - 12}" text-anchor="middle" font-family="Arial" font-size="13">{title_split}</text>')
        parts.append(f'<rect x="{x0:.2f}" y="{margin["top"]:.2f}" width="{panel_width:.2f}" height="{panel_height:.2f}" fill="none" stroke="#666"/>')
        y = records[split]["y"]
        parts.append(
            f'<polyline points="{polyline(y, panel_idx)}" fill="none" stroke="{colors["measured"]}" stroke-width="1.8"/>'
        )
        for model in ["B0_linear_outflow", "B3_sqrt_outflow_slot"]:
            pred = predictions[(model, split)]
            parts.append(
                f'<polyline points="{polyline(pred, panel_idx)}" fill="none" stroke="{colors[model]}" stroke-width="2.0" stroke-dasharray="{dashes[model]}"/>'
            )
        if split == "test":
            n_init = int(CONFIG["state_initialization_window"])
            x_init = sx(n_init, len(y), panel_idx)
            parts.append(f'<line x1="{x_init:.2f}" y1="{margin["top"]:.2f}" x2="{x_init:.2f}" y2="{margin["top"] + panel_height:.2f}" stroke="#999" stroke-dasharray="4 3"/>')
    legend_x, legend_y = margin["left"], height - 20
    for idx, key in enumerate(["measured", "B0_linear_outflow", "B3_sqrt_outflow_slot"]):
        x = legend_x + idx * 265
        parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x+35}" y2="{legend_y}" stroke="{colors[key]}" stroke-width="2" stroke-dasharray="{dashes[key]}"/>')
        parts.append(f'<text x="{x+42}" y="{legend_y+4}" font-family="Arial" font-size="12">{labels[key]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records()
    train = records["estimation"]
    dt = float(train["sampling_time"])
    n_init = int(CONFIG["state_initialization_window"])

    fits: dict[str, dict[str, Any]] = {}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    parameter_table: list[dict[str, Any]] = []

    for model in ["B0_linear_outflow", "B3_sqrt_outflow_slot"]:
        fit = fit_model(model, train["u"], train["y"], dt, make_starts(model, train["y"], train["u"]))
        fits[model] = fit
        parameter_table.extend(parameter_rows(model, fit))
        for split, record in records.items():
            if split == "estimation":
                x10 = float(fit["x10_estimation"])
                score_init = n_init
            else:
                score_init = int(record["state_initialization_window"]) or n_init
                x10 = initialize_x10(
                    model,
                    fit["params"],
                    record["u"],
                    record["y"],
                    float(record["sampling_time"]),
                    float(fit["x10_estimation"]),
                    score_init,
                )
                parameter_table.append({"model": model, "parameter": f"x10_{split}_initialized", "value": float(x10)})
            pred = simulate(record["u"], float(record["y"][0]), x10, float(record["sampling_time"]), model, fit["params"])
            predictions[(model, split)] = pred
            metrics = error_metrics(record["y"], pred, score_init)
            metric_rows.append(
                {
                    "model": model,
                    "split": split,
                    "n_samples": int(len(record["y"])),
                    "score_start_index": int(score_init),
                    "sampling_time_s": float(record["sampling_time"]),
                    **metrics,
                }
            )

    metric_by_key = {(row["model"], row["split"]): row for row in metric_rows}
    b0_test = float(metric_by_key[("B0_linear_outflow", "test")]["nrmse"])
    b3_test = float(metric_by_key[("B3_sqrt_outflow_slot", "test")]["nrmse"])
    improvement = 100.0 * (b0_test - b3_test) / b0_test if b0_test else float("nan")

    prediction_rows: list[dict[str, Any]] = []
    stride = int(CONFIG["prediction_sample_stride"])
    for split, record in records.items():
        for idx in range(0, len(record["y"]), stride):
            row: dict[str, Any] = {
                "split": split,
                "index": idx,
                "time_s": float(idx * record["sampling_time"]),
                "input_u": float(record["u"][idx]),
                "measured_y": float(record["y"][idx]),
            }
            for model in ["B0_linear_outflow", "B3_sqrt_outflow_slot"]:
                row[f"{model}_prediction"] = float(predictions[(model, split)][idx])
            prediction_rows.append(row)

    raw_files = collect_raw_files(RAW_ROOT)
    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "summary": {
            "test_b0_nrmse": b0_test,
            "test_b3_nrmse": b3_test,
            "test_improvement_percent": improvement,
        },
        "metric_rows": metric_rows,
        "parameters": parameter_table,
        "raw_files": raw_files,
    }

    write_csv(RESULT_DIR / "metrics_by_split.csv", metric_rows)
    write_csv(RESULT_DIR / "parameters.csv", parameter_table)
    write_csv(RESULT_DIR / "prediction_sample.csv", prediction_rows)
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_rollout_svg(FIGURE_DIR / "rollout.svg", records, predictions, zh=False)
    write_rollout_svg(FIGURE_DIR / "rollout_zh.svg", records, predictions, zh=True)

    provenance = {
        **CONFIG,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "nonlinear-benchmarks": package_version("nonlinear-benchmarks"),
        },
        "outputs": {
            "metrics_json": str((RESULT_DIR / "metrics.json").relative_to(ROOT)),
            "metrics_by_split_csv": str((RESULT_DIR / "metrics_by_split.csv").relative_to(ROOT)),
            "parameters_csv": str((RESULT_DIR / "parameters.csv").relative_to(ROOT)),
            "prediction_sample_csv": str((RESULT_DIR / "prediction_sample.csv").relative_to(ROOT)),
            "rollout_svg": str((FIGURE_DIR / "rollout.svg").relative_to(ROOT)),
            "rollout_zh_svg": str((FIGURE_DIR / "rollout_zh.svg").relative_to(ROOT)),
        },
        "raw_files": raw_files,
    }
    provenance_path = PROVENANCE_DIR / "e6_cascaded_tanks_slot_oe_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
