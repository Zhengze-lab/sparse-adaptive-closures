#!/usr/bin/env python3
"""Run BATT-2: OCV(SOC, temperature) calibration for the LiFePO4 ECM case."""

from __future__ import annotations

import csv
import hashlib
import io
import json
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


ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = ROOT / "data" / "raw" / "external" / "battery_lfp_ocv_dyn" / "p8kf893yv3_v1.zip"
PROCESSED_DIR = ROOT / "data" / "processed" / "battery_lfp_ocv_dyn"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "battery_lfp_ocv_calibration"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG: dict[str, Any] = {
    "experiment_id": "BATT2_lfp_ocv_calibration",
    "description": "OCV(SOC,T) calibration from LiFePO4 OCV data for the battery ECM route.",
    "source_dataset": "battery_lfp_ocv_dyn",
    "source_url": "https://data.mendeley.com/datasets/p8kf893yv3/1",
    "source_doi": "10.17632/p8kf893yv3.1",
    "license": "CC BY 4.0",
    "raw_zip": str(RAW_ZIP.relative_to(ROOT)),
    "soc_grid_start": 0.02,
    "soc_grid_end": 0.98,
    "soc_grid_count": 97,
    "used_scripts": {
        "script1": "discharge curve; SOC = 1 - normalized disAh",
        "script3": "charge curve; SOC = normalized chgAh",
    },
    "excluded_ocv_files": "A002_02 repeat OCV files are excluded from first-pass calibration and reserved for repeatability checks.",
    "heldout_temperatures_c": [-15, 35],
    "training_temperatures_c": [-25, -5, 5, 15, 25, 45],
    "models": [
        "linear_temperature_interpolation",
        "least_squares_polynomial_soc7_temp2",
    ],
}


@dataclass(frozen=True)
class OcvCurve:
    filename: str
    temperature_c: int
    curve: str
    soc: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    capacity_ah: float


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_temperature_c(filename: str) -> int:
    match = re.search(r"_([NP])(\d{2})(?:_|\.)", filename)
    if not match:
        raise ValueError(f"Cannot parse temperature from {filename}")
    sign = -1 if match.group(1) == "N" else 1
    return sign * int(match.group(2))


def field_array(obj: Any, name: str) -> np.ndarray:
    return np.asarray(getattr(obj, name), dtype=float).ravel()


def extract_curve(filename: str, root: Any, script_name: str) -> OcvCurve:
    script = getattr(root, script_name)
    temperature = parse_temperature_c(filename)
    voltage = field_array(script, "voltage")
    current = field_array(script, "current")
    if script_name == "script1":
        curve = "discharge"
        ah = field_array(script, "disAh")
        ah0 = float(np.nanmin(ah))
        capacity = float(np.nanmax(ah) - ah0)
        soc = 1.0 - (ah - ah0) / capacity
    elif script_name == "script3":
        curve = "charge"
        ah = field_array(script, "chgAh")
        ah0 = float(np.nanmin(ah))
        capacity = float(np.nanmax(ah) - ah0)
        soc = (ah - ah0) / capacity
    else:
        raise ValueError(f"Unsupported OCV script: {script_name}")

    mask = (
        np.isfinite(soc)
        & np.isfinite(voltage)
        & np.isfinite(current)
        & (soc >= -0.02)
        & (soc <= 1.02)
        & (voltage > 1.5)
        & (voltage < 4.0)
    )
    soc = np.clip(soc[mask], 0.0, 1.0)
    voltage = voltage[mask]
    current = current[mask]
    order = np.argsort(soc)
    return OcvCurve(
        filename=filename,
        temperature_c=temperature,
        curve=curve,
        soc=soc[order],
        voltage=voltage[order],
        current=current[order],
        capacity_ah=capacity,
    )


def load_ocv_curves() -> list[OcvCurve]:
    curves: list[OcvCurve] = []
    with zipfile.ZipFile(RAW_ZIP) as zf:
        names = sorted(
            name
            for name in zf.namelist()
            if name.startswith("OCVdata/")
            and name.lower().endswith(".mat")
            and "/A002_02_" not in name
        )
        for name in names:
            mat = loadmat(io.BytesIO(zf.read(name)), squeeze_me=True, struct_as_record=False)
            root = mat["OCVData"]
            curves.append(extract_curve(name, root, "script1"))
            curves.append(extract_curve(name, root, "script3"))
    return curves


def interpolate_curve(curve: OcvCurve, soc_grid: np.ndarray) -> np.ndarray:
    soc = curve.soc
    voltage = curve.voltage
    unique_soc, inverse = np.unique(soc, return_inverse=True)
    if len(unique_soc) != len(soc):
        sums = np.zeros(len(unique_soc), dtype=float)
        counts = np.zeros(len(unique_soc), dtype=float)
        np.add.at(sums, inverse, voltage)
        np.add.at(counts, inverse, 1.0)
        voltage_unique = sums / np.maximum(counts, 1.0)
    else:
        voltage_unique = voltage
    return np.interp(soc_grid, unique_soc, voltage_unique)


def build_grid(curves: list[OcvCurve], soc_grid: np.ndarray) -> tuple[list[dict[str, Any]], dict[tuple[int, str], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    grid: dict[tuple[int, str], np.ndarray] = {}
    by_temp: dict[int, dict[str, np.ndarray]] = {}
    for curve in curves:
        values = interpolate_curve(curve, soc_grid)
        grid[(curve.temperature_c, curve.curve)] = values
        by_temp.setdefault(curve.temperature_c, {})[curve.curve] = values
        for soc, voltage in zip(soc_grid, values):
            rows.append(
                {
                    "temperature_c": curve.temperature_c,
                    "curve": curve.curve,
                    "soc": float(soc),
                    "voltage": float(voltage),
                    "source_file": curve.filename,
                    "capacity_ah": curve.capacity_ah,
                }
            )
    for temperature, curves_by_kind in by_temp.items():
        if "charge" in curves_by_kind and "discharge" in curves_by_kind:
            mean_values = 0.5 * (curves_by_kind["charge"] + curves_by_kind["discharge"])
            grid[(temperature, "mean")] = mean_values
            for soc, voltage in zip(soc_grid, mean_values):
                rows.append(
                    {
                        "temperature_c": temperature,
                        "curve": "mean",
                        "soc": float(soc),
                        "voltage": float(voltage),
                        "source_file": "charge_discharge_average",
                        "capacity_ah": "",
                    }
                )
    return rows, grid


def design_matrix(soc: np.ndarray, temperature_c: np.ndarray) -> np.ndarray:
    t_norm = (temperature_c - 10.0) / 35.0
    columns = []
    names = []
    for temp_degree in range(3):
        for soc_degree in range(8):
            columns.append((t_norm**temp_degree) * (soc**soc_degree))
            names.append(f"Tn^{temp_degree}*soc^{soc_degree}")
    return np.column_stack(columns), names


def rmse_mv(y: np.ndarray, yhat: np.ndarray) -> float:
    err = yhat - y
    return 1000.0 * float(np.sqrt(np.mean(err * err)))


def max_abs_mv(y: np.ndarray, yhat: np.ndarray) -> float:
    return 1000.0 * float(np.max(np.abs(yhat - y)))


def evaluate_models(
    grid: dict[tuple[int, str], np.ndarray],
    soc_grid: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    train_temps = np.array(CONFIG["training_temperatures_c"], dtype=float)
    heldout_temps = [int(v) for v in CONFIG["heldout_temperatures_c"]]
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    model_summary: dict[str, dict[str, Any]] = {}

    for curve in ["discharge", "charge", "mean"]:
        train_matrix = np.vstack([grid[(int(temp), curve)] for temp in train_temps])
        interpolator = RegularGridInterpolator(
            (train_temps, soc_grid),
            train_matrix,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        train_soc = np.tile(soc_grid, len(train_temps))
        train_temp = np.repeat(train_temps, len(soc_grid))
        train_y = train_matrix.ravel()
        phi, feature_names = design_matrix(train_soc, train_temp)
        coeffs, *_ = np.linalg.lstsq(phi, train_y, rcond=None)

        for model in CONFIG["models"]:
            all_true = []
            all_pred = []
            for temp in heldout_temps:
                y_true = grid[(temp, curve)]
                if model == "linear_temperature_interpolation":
                    points = np.column_stack([np.full_like(soc_grid, temp, dtype=float), soc_grid])
                    y_pred = np.asarray(interpolator(points), dtype=float)
                elif model == "least_squares_polynomial_soc7_temp2":
                    phi_test, _ = design_matrix(soc_grid, np.full_like(soc_grid, temp, dtype=float))
                    y_pred = phi_test @ coeffs
                else:
                    raise ValueError(f"Unknown model: {model}")
                all_true.append(y_true)
                all_pred.append(y_pred)
                metrics_rows.append(
                    {
                        "model": model,
                        "curve": curve,
                        "temperature_c": temp,
                        "rmse_mv": rmse_mv(y_true, y_pred),
                        "max_abs_mv": max_abs_mv(y_true, y_pred),
                        "n_points": len(soc_grid),
                    }
                )
                for soc, true_v, pred_v in zip(soc_grid, y_true, y_pred):
                    prediction_rows.append(
                        {
                            "model": model,
                            "curve": curve,
                            "temperature_c": temp,
                            "soc": float(soc),
                            "voltage_true": float(true_v),
                            "voltage_pred": float(pred_v),
                            "error_mv": 1000.0 * float(pred_v - true_v),
                        }
                    )
            y_all = np.concatenate(all_true)
            pred_all = np.concatenate(all_pred)
            metrics_rows.append(
                {
                    "model": model,
                    "curve": curve,
                    "temperature_c": "heldout_all",
                    "rmse_mv": rmse_mv(y_all, pred_all),
                    "max_abs_mv": max_abs_mv(y_all, pred_all),
                    "n_points": len(y_all),
                }
            )
            model_summary[f"{model}:{curve}"] = {
                "rmse_mv": rmse_mv(y_all, pred_all),
                "max_abs_mv": max_abs_mv(y_all, pred_all),
                "n_points": int(len(y_all)),
            }
        model_summary[f"poly_features:{curve}"] = {
            "feature_names": feature_names,
            "coefficients": [float(v) for v in coeffs],
        }
    return metrics_rows, prediction_rows, model_summary


def color_palette(n: int) -> list[str]:
    base = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
    return [base[i % len(base)] for i in range(n)]


def polyline(x: np.ndarray, y: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float, left: float, top: float, width: float, height: float) -> str:
    xp = left + (x - x_min) / (x_max - x_min) * width
    yp = top + height - (y - y_min) / (y_max - y_min) * height
    return " ".join(f"{float(xx):.2f},{float(yy):.2f}" for xx, yy in zip(xp, yp))


def write_ocv_curve_svg(path: Path, grid: dict[tuple[int, str], np.ndarray], soc_grid: np.ndarray, zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temps = sorted(int(temp) for temp in CONFIG["training_temperatures_c"] + CONFIG["heldout_temperatures_c"])
    values = [grid[(temp, "mean")] for temp in temps]
    y_min = min(float(np.min(v)) for v in values) - 0.03
    y_max = max(float(np.max(v)) for v in values) + 0.03
    left, top, width, height = 78.0, 70.0, 760.0, 330.0
    colors = color_palette(len(temps))
    title = "OCV-SOC curves by temperature" if not zh else "不同温度下的 OCV-SOC 曲线"
    x_label = "SOC" if not zh else "SOC"
    y_label = "OCV (V)" if not zh else "开路电压 OCV (V)"
    legend_title = "Temperature" if not zh else "温度"
    lines = []
    legend = []
    for idx, (temp, voltage) in enumerate(zip(temps, values)):
        dash = "6,4" if temp in CONFIG["heldout_temperatures_c"] else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<polyline points="{polyline(soc_grid, voltage, 0.0, 1.0, y_min, y_max, left, top, width, height)}" '
            f'fill="none" stroke="{colors[idx]}" stroke-width="2.1"{dash_attr}/>'
        )
        y_leg = 92 + idx * 22
        legend.append(f'<line x1="860" y1="{y_leg}" x2="890" y2="{y_leg}" stroke="{colors[idx]}" stroke-width="2.1"{dash_attr}/>')
        legend.append(f'<text x="898" y="{y_leg + 4}" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">{temp} C</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="980" height="470" viewBox="0 0 980 470">
  <rect width="980" height="470" fill="#ffffff"/>
  <text x="78" y="36" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  <rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#6b7280"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#6b7280"/>
  {''.join(lines)}
  <text x="{left + width / 2}" y="445" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{x_label}</text>
  <text x="25" y="{top + height / 2}" transform="rotate(-90 25 {top + height / 2})" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{y_label}</text>
  <text x="860" y="68" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" font-weight="700" fill="#111827">{legend_title}</text>
  {''.join(legend)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_error_svg(path: Path, prediction_rows: list[dict[str, Any]], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in prediction_rows
        if row["model"] == "linear_temperature_interpolation" and row["curve"] == "mean"
    ]
    temps = sorted({int(row["temperature_c"]) for row in rows})
    left, top, width, height = 78.0, 70.0, 760.0, 330.0
    y_values = np.array([float(row["error_mv"]) for row in rows])
    y_min = float(np.min(y_values)) - 2.0
    y_max = float(np.max(y_values)) + 2.0
    colors = color_palette(len(temps))
    title = "Held-out OCV interpolation error" if not zh else "留出温度 OCV 插值误差"
    y_label = "Prediction error (mV)" if not zh else "预测误差 (mV)"
    lines = []
    legend = []
    for idx, temp in enumerate(temps):
        sub = [row for row in rows if int(row["temperature_c"]) == temp]
        x = np.array([float(row["soc"]) for row in sub])
        y = np.array([float(row["error_mv"]) for row in sub])
        lines.append(
            f'<polyline points="{polyline(x, y, 0.0, 1.0, y_min, y_max, left, top, width, height)}" '
            f'fill="none" stroke="{colors[idx]}" stroke-width="2.1"/>'
        )
        y_leg = 92 + idx * 22
        legend.append(f'<line x1="860" y1="{y_leg}" x2="890" y2="{y_leg}" stroke="{colors[idx]}" stroke-width="2.1"/>')
        legend.append(f'<text x="898" y="{y_leg + 4}" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">{temp} C</text>')
    zero_y = top + height - (0.0 - y_min) / (y_max - y_min) * height
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="980" height="470" viewBox="0 0 980 470">
  <rect width="980" height="470" fill="#ffffff"/>
  <text x="78" y="36" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  <rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{left}" y1="{zero_y:.2f}" x2="{left + width}" y2="{zero_y:.2f}" stroke="#9ca3af" stroke-dasharray="5,4"/>
  <line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#6b7280"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#6b7280"/>
  {''.join(lines)}
  <text x="{left + width / 2}" y="445" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">SOC</text>
  <text x="25" y="{top + height / 2}" transform="rotate(-90 25 {top + height / 2})" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{y_label}</text>
  {''.join(legend)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    if not RAW_ZIP.exists():
        raise FileNotFoundError(f"Missing raw zip: {RAW_ZIP}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    soc_grid = np.linspace(float(CONFIG["soc_grid_start"]), float(CONFIG["soc_grid_end"]), int(CONFIG["soc_grid_count"]))
    curves = load_ocv_curves()
    curve_rows: list[dict[str, Any]] = []
    for curve in curves:
        curve_rows.append(
            {
                "source_file": curve.filename,
                "temperature_c": curve.temperature_c,
                "curve": curve.curve,
                "n_samples": int(len(curve.soc)),
                "capacity_ah": curve.capacity_ah,
                "current_min": float(np.min(curve.current)),
                "current_max": float(np.max(curve.current)),
                "voltage_min": float(np.min(curve.voltage)),
                "voltage_max": float(np.max(curve.voltage)),
            }
        )

    grid_rows, grid = build_grid(curves, soc_grid)
    metrics_rows, prediction_rows, model_summary = evaluate_models(grid, soc_grid)

    ocv_curves_path = PROCESSED_DIR / "batt2_ocv_curves.csv"
    ocv_grid_path = PROCESSED_DIR / "batt2_ocv_grid.csv"
    metrics_by_curve_path = RESULT_DIR / "metrics_by_curve.csv"
    heldout_predictions_path = RESULT_DIR / "heldout_predictions.csv"
    metrics_path = RESULT_DIR / "metrics.json"
    provenance_path = PROVENANCE_DIR / "batt2_ocv_calibration_provenance.json"

    write_csv(ocv_curves_path, curve_rows)
    write_csv(ocv_grid_path, grid_rows)
    write_csv(metrics_by_curve_path, metrics_rows)
    write_csv(heldout_predictions_path, prediction_rows)

    write_ocv_curve_svg(FIGURE_DIR / "ocv_curves_by_temperature.svg", grid, soc_grid, zh=False)
    write_ocv_curve_svg(FIGURE_DIR / "ocv_curves_by_temperature_zh.svg", grid, soc_grid, zh=True)
    write_error_svg(FIGURE_DIR / "heldout_ocv_error.svg", prediction_rows, zh=False)
    write_error_svg(FIGURE_DIR / "heldout_ocv_error_zh.svg", prediction_rows, zh=True)

    interpolation_mean = model_summary["linear_temperature_interpolation:mean"]
    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "complete",
        "config": CONFIG,
        "raw_zip_sha256": sha256_file(RAW_ZIP),
        "curve_count": len(curves),
        "temperatures_c": sorted({curve.temperature_c for curve in curves}),
        "soc_grid_count": int(len(soc_grid)),
        "primary_model": "linear_temperature_interpolation:mean",
        "primary_heldout_rmse_mv": interpolation_mean["rmse_mv"],
        "primary_heldout_max_abs_mv": interpolation_mean["max_abs_mv"],
        "model_summary": model_summary,
        "outputs": {
            "ocv_curves_csv": str(ocv_curves_path.relative_to(ROOT)),
            "ocv_grid_csv": str(ocv_grid_path.relative_to(ROOT)),
            "metrics_by_curve_csv": str(metrics_by_curve_path.relative_to(ROOT)),
            "heldout_predictions_csv": str(heldout_predictions_path.relative_to(ROOT)),
            "figures": [
                str((FIGURE_DIR / "ocv_curves_by_temperature.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "ocv_curves_by_temperature_zh.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "heldout_ocv_error.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "heldout_ocv_error_zh.svg").relative_to(ROOT)),
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
        "script": "scripts/run_batt2_ocv_calibration.py",
        "outputs": metrics["outputs"] | {
            "metrics_json": str(metrics_path.relative_to(ROOT)),
            "provenance_json": str(provenance_path.relative_to(ROOT)),
        },
        "test_usage_rule": "Temperatures -15 C and 35 C are held out from calibration and used only for BATT-2 OCV validation.",
        "next_action": "Use the calibrated OCV baseline to build BATT-3 constant first-order ECM output-error baseline.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
