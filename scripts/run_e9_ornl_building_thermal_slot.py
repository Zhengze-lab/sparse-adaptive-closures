#!/usr/bin/env python3
"""Run E9: ORNL building thermal coefficient-slot boundary experiment."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata as metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "external" / "ornl_building_thermal"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "e9_ornl_building_thermal_slot"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG: dict[str, Any] = {
    "experiment_id": "E9_ornl_building_thermal_slot",
    "description": "ORNL multizone building thermal RC test: fixed envelope coefficient vs weather-adaptive envelope slot.",
    "source_dataset": "Datasets of a Multizone Office Building under Different HVAC System Operation Scenarios",
    "source_doi": "10.6084/m9.figshare.20520438.v3",
    "source_url": "https://figshare.com/articles/dataset/Scientific_Journal_Data_Empirical_HVAC_Operation/20520438",
    "license": "CC BY 4.0",
    "train_scenario": "Base_Heating",
    "test_scenario": "Base_Cooling",
    "sample_time_s": 60.0,
    "savgol_window": 121,
    "savgol_polyorder": 3,
    "edge_trim": 80,
    "prediction_sample_stride": 20,
    "models": {
        "B0_fixed_envelope": "dT/dt = a_env*(Tout-T) + a_hvac*AF*(Tsup-T) + a_solar*GloSolar + b",
        "B3_weather_adaptive_envelope_slot": "dT/dt = p_env(WS,GloSolar)*(Tout-T) + a_hvac*AF*(Tsup-T) + a_solar*GloSolar + b",
        "B1_full_linear_features": "wider linear feature model over envelope, HVAC, solar, wind, humidity",
    },
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


def read_numeric_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["-", ""])
    df = df.iloc[1:].reset_index(drop=True)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    for col in df.columns:
        if col != "TIMESTAMP":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_scenario(name: str) -> dict[str, np.ndarray]:
    building = read_numeric_csv(RAW_ROOT / f"Building_{name}.csv")
    weather = read_numeric_csv(RAW_ROOT / f"Weather_{name}.csv")
    df = pd.merge(building, weather, on="TIMESTAMP", how="inner")
    room_cols = [col for col in df.columns if col.startswith("T_Room_")]
    zone = df[room_cols].mean(axis=1).to_numpy(dtype=float)
    tout = df["T_out"].to_numpy(dtype=float)
    tsup = df["T_Sup_RTU"].to_numpy(dtype=float)
    af = df["AF_RTU"].to_numpy(dtype=float)
    solar = df["Glo_Solar"].to_numpy(dtype=float)
    wind = df["WS"].to_numpy(dtype=float)
    rh = df["RH_out"].to_numpy(dtype=float)
    hvac = af * (tsup - zone)
    finite = np.isfinite(zone) & np.isfinite(tout) & np.isfinite(hvac) & np.isfinite(solar) & np.isfinite(wind) & np.isfinite(rh)
    zone = zone[finite]
    tout = tout[finite]
    hvac = hvac[finite]
    solar = solar[finite]
    wind = wind[finite]
    rh = rh[finite]
    n = len(zone)
    window = min(int(CONFIG["savgol_window"]), n // 2 * 2 - 1)
    if window % 2 == 0:
        window -= 1
    dzone = savgol_filter(
        zone,
        window_length=window,
        polyorder=int(CONFIG["savgol_polyorder"]),
        deriv=1,
        delta=float(CONFIG["sample_time_s"]),
        mode="interp",
    )
    edge = min(int(CONFIG["edge_trim"]), (n - 3) // 2)
    sl = slice(edge, n - edge)
    return {
        "scenario": name,
        "zone": zone[sl],
        "dzone": dzone[sl],
        "tout": tout[sl],
        "hvac": hvac[sl],
        "solar": solar[sl],
        "wind": wind[sl],
        "rh": rh[sl],
        "n": int(len(zone[sl])),
    }


def scale_from(train: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "hvac": float(np.std(train["hvac"])) or 1.0,
        "solar": float(np.std(train["solar"])) or 1.0,
        "wind": float(np.std(train["wind"])) or 1.0,
        "rh": float(np.std(train["rh"])) or 1.0,
    }


def design(model: str, data: dict[str, np.ndarray], scale: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    delta_env = data["tout"] - data["zone"]
    hvac = data["hvac"] / scale["hvac"]
    solar = data["solar"] / scale["solar"]
    wind = data["wind"] / scale["wind"]
    rh = data["rh"] / scale["rh"]
    if model == "B0_fixed_envelope":
        names = ["1", "Tout-Tzone", "HVAC", "Solar"]
        theta = np.column_stack([np.ones_like(delta_env), delta_env, hvac, solar])
    elif model == "B3_weather_adaptive_envelope_slot":
        names = ["1", "Tout-Tzone", "(Tout-Tzone)*wind", "(Tout-Tzone)*solar", "HVAC", "Solar"]
        theta = np.column_stack([np.ones_like(delta_env), delta_env, delta_env * wind, delta_env * solar, hvac, solar])
    elif model == "B1_full_linear_features":
        names = ["1", "Tout-Tzone", "(Tout-Tzone)*wind", "(Tout-Tzone)*solar", "HVAC", "Solar", "wind", "RH", "Tzone"]
        theta = np.column_stack([np.ones_like(delta_env), delta_env, delta_env * wind, delta_env * solar, hvac, solar, wind, rh, data["zone"]])
    else:
        raise ValueError(model)
    return theta, names


def fit_model(model: str, train: dict[str, np.ndarray], scale: dict[str, float]) -> dict[str, Any]:
    theta, names = design(model, train, scale)
    coeffs, *_ = np.linalg.lstsq(theta, train["dzone"], rcond=None)
    return {"model": model, "names": names, "coefficients": coeffs}


def predict_derivative(fit: dict[str, Any], data: dict[str, np.ndarray], scale: dict[str, float]) -> np.ndarray:
    theta, _ = design(fit["model"], data, scale)
    return theta @ fit["coefficients"]


def derivative_from_state(fit: dict[str, Any], row: dict[str, float], zone_temp: float, scale: dict[str, float]) -> float:
    c = fit["coefficients"]
    delta = row["tout"] - zone_temp
    hvac = row["af_hvac"] / scale["hvac"]
    solar = row["solar"] / scale["solar"]
    wind = row["wind"] / scale["wind"]
    rh = row["rh"] / scale["rh"]
    if fit["model"] == "B0_fixed_envelope":
        return float(c[0] + c[1] * delta + c[2] * hvac + c[3] * solar)
    if fit["model"] == "B3_weather_adaptive_envelope_slot":
        return float(c[0] + c[1] * delta + c[2] * delta * wind + c[3] * delta * solar + c[4] * hvac + c[5] * solar)
    if fit["model"] == "B1_full_linear_features":
        return float(c[0] + c[1] * delta + c[2] * delta * wind + c[3] * delta * solar + c[4] * hvac + c[5] * solar + c[6] * wind + c[7] * rh + c[8] * zone_temp)
    raise ValueError(fit["model"])


def rollout(fit: dict[str, Any], data: dict[str, np.ndarray], scale: dict[str, float]) -> np.ndarray:
    dt = float(CONFIG["sample_time_s"])
    pred = np.empty_like(data["zone"], dtype=float)
    pred[0] = float(data["zone"][0])
    for idx in range(len(pred) - 1):
        row = {
            "tout": float(data["tout"][idx]),
            "af_hvac": float(data["hvac"][idx]),
            "solar": float(data["solar"][idx]),
            "wind": float(data["wind"][idx]),
            "rh": float(data["rh"][idx]),
        }
        k1 = derivative_from_state(fit, row, float(pred[idx]), scale)
        k2 = derivative_from_state(fit, row, float(pred[idx] + 0.5 * dt * k1), scale)
        nxt = pred[idx] + dt * k2
        pred[idx + 1] = float(np.clip(nxt if np.isfinite(nxt) else pred[idx], -50.0, 80.0))
    return pred


def error_metrics(true: np.ndarray, pred: np.ndarray, prefix: str) -> dict[str, float]:
    err = pred - true
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.std(true)) or 1.0
    return {
        f"{prefix}_rmse_c": rmse,
        f"{prefix}_nrmse": rmse / denom,
        f"{prefix}_mae_c": float(np.mean(np.abs(err))),
        f"{prefix}_bias_c": float(np.mean(err)),
    }


def write_rollout_svg(path: Path, test: dict[str, np.ndarray], preds: dict[str, np.ndarray], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 920, 460
    left, right, top, bottom = 64, 20, 45, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    all_y = np.concatenate([test["zone"], *preds.values()])
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad
    colors = {"Measured": "#000000", "B0_fixed_envelope": "#D55E00", "B3_weather_adaptive_envelope_slot": "#0072B2"}
    dashes = {"Measured": "", "B0_fixed_envelope": "8 4", "B3_weather_adaptive_envelope_slot": ""}
    labels = {
        "Measured": "实测平均室温" if zh else "Measured mean room temperature",
        "B0_fixed_envelope": "B0 固定围护系数" if zh else "B0 fixed envelope",
        "B3_weather_adaptive_envelope_slot": "B3 天气自适应围护槽" if zh else "B3 weather-adaptive envelope slot",
    }
    title = "E9 ORNL 建筑热动态 rollout" if zh else "E9 ORNL building thermal rollout"

    def sx(i: int, n: int) -> float:
        return left + plot_w * i / max(n - 1, 1)

    def sy(v: float) -> float:
        return top + plot_h * (1.0 - (v - y_min) / (y_max - y_min))

    def poly(values: np.ndarray) -> str:
        step = max(1, len(values) // 700)
        return " ".join(f"{sx(i, len(values)):.2f},{sy(float(values[i])):.2f}" for i in range(0, len(values), step))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">{title}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#666"/>',
        f'<polyline points="{poly(test["zone"])}" fill="none" stroke="{colors["Measured"]}" stroke-width="1.7"/>',
    ]
    for model in ["B0_fixed_envelope", "B3_weather_adaptive_envelope_slot"]:
        parts.append(f'<polyline points="{poly(preds[model])}" fill="none" stroke="{colors[model]}" stroke-width="2" stroke-dasharray="{dashes[model]}"/>')
    lx, ly = left, height - 24
    for i, key in enumerate(["Measured", "B0_fixed_envelope", "B3_weather_adaptive_envelope_slot"]):
        x = lx + i * 270
        parts.append(f'<line x1="{x}" y1="{ly}" x2="{x+32}" y2="{ly}" stroke="{colors[key]}" stroke-width="2" stroke-dasharray="{dashes[key]}"/>')
        parts.append(f'<text x="{x+38}" y="{ly+4}" font-family="Arial" font-size="12">{labels[key]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def collect_raw_files() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(RAW_ROOT.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    train = load_scenario(str(CONFIG["train_scenario"]))
    test = load_scenario(str(CONFIG["test_scenario"]))
    scale = scale_from(train)
    models = ["B0_fixed_envelope", "B3_weather_adaptive_envelope_slot", "B1_full_linear_features"]
    fits = {model: fit_model(model, train, scale) for model in models}
    coefficient_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    rollouts: dict[str, np.ndarray] = {}
    for model, fit in fits.items():
        for name, coeff in zip(fit["names"], fit["coefficients"]):
            coefficient_rows.append({"model": model, "feature": name, "coefficient": float(coeff)})
        for split, data in [("train", train), ("test", test)]:
            d_pred = predict_derivative(fit, data, scale)
            pred = rollout(fit, data, scale)
            if split == "test":
                rollouts[model] = pred
            metric_rows.append(
                {
                    "model": model,
                    "split": split,
                    "scenario": data["scenario"],
                    "n_samples": int(data["n"]),
                    **error_metrics(data["dzone"], d_pred, "derivative"),
                    **error_metrics(data["zone"], pred, "rollout"),
                }
            )
    metric_by_key = {(row["model"], row["split"]): row for row in metric_rows}
    b0 = float(metric_by_key[("B0_fixed_envelope", "test")]["rollout_nrmse"])
    b3 = float(metric_by_key[("B3_weather_adaptive_envelope_slot", "test")]["rollout_nrmse"])
    b1 = float(metric_by_key[("B1_full_linear_features", "test")]["rollout_nrmse"])
    summary = {
        "train_scenario": CONFIG["train_scenario"],
        "test_scenario": CONFIG["test_scenario"],
        "test_b0_rollout_nrmse": b0,
        "test_b3_rollout_nrmse": b3,
        "test_b1_rollout_nrmse": b1,
        "test_b3_vs_b0_improvement_percent": 100.0 * (b0 - b3) / b0 if b0 else float("nan"),
        "test_b3_vs_b1_gap_nrmse": b3 - b1,
    }
    stride = int(CONFIG["prediction_sample_stride"])
    for idx in range(0, len(test["zone"]), stride):
        row: dict[str, Any] = {
            "split": "test",
            "index": idx,
            "zone_c": float(test["zone"][idx]),
            "tout_c": float(test["tout"][idx]),
            "hvac_proxy": float(test["hvac"][idx]),
            "solar": float(test["solar"][idx]),
            "wind": float(test["wind"][idx]),
        }
        for model in ["B0_fixed_envelope", "B3_weather_adaptive_envelope_slot"]:
            row[f"{model}_rollout_c"] = float(rollouts[model][idx])
        prediction_rows.append(row)
    write_csv(RESULT_DIR / "metrics_by_split.csv", metric_rows)
    write_csv(RESULT_DIR / "coefficients.csv", coefficient_rows)
    write_csv(RESULT_DIR / "prediction_sample.csv", prediction_rows)
    write_rollout_svg(
        FIGURE_DIR / "test_rollout.svg",
        test,
        {"B0_fixed_envelope": rollouts["B0_fixed_envelope"], "B3_weather_adaptive_envelope_slot": rollouts["B3_weather_adaptive_envelope_slot"]},
        zh=False,
    )
    write_rollout_svg(
        FIGURE_DIR / "test_rollout_zh.svg",
        test,
        {"B0_fixed_envelope": rollouts["B0_fixed_envelope"], "B3_weather_adaptive_envelope_slot": rollouts["B3_weather_adaptive_envelope_slot"]},
        zh=True,
    )
    raw_files = collect_raw_files()
    metrics_payload = {
        "experiment_id": CONFIG["experiment_id"],
        "summary": summary,
        "metrics_by_split": metric_rows,
        "coefficients": coefficient_rows,
        "scaling": scale,
        "raw_files": raw_files,
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    provenance = {
        **CONFIG,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "pandas": package_version("pandas"),
        },
        "outputs": {
            "metrics_json": str((RESULT_DIR / "metrics.json").relative_to(ROOT)),
            "metrics_by_split_csv": str((RESULT_DIR / "metrics_by_split.csv").relative_to(ROOT)),
            "coefficients_csv": str((RESULT_DIR / "coefficients.csv").relative_to(ROOT)),
            "prediction_sample_csv": str((RESULT_DIR / "prediction_sample.csv").relative_to(ROOT)),
        },
        "raw_files": raw_files,
    }
    (PROVENANCE_DIR / "e9_ornl_building_thermal_slot_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
