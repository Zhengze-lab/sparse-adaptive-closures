#!/usr/bin/env python3
"""Run E8: PMSM temperature speed-adaptive thermal coefficient-slot experiment."""

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
RAW_ROOT = ROOT / "data" / "raw" / "external" / "pmsm_temperature"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "e8_pmsm_temperature_slot"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG: dict[str, Any] = {
    "experiment_id": "E8_pmsm_temperature_slot",
    "description": "Real PMSM thermal RC test: fixed cooling coefficient vs speed-adaptive cooling coefficient slot.",
    "source_dataset": "Kaggle Electric Motor Temperature",
    "source_url": "https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature",
    "download_command": "kaggle datasets download -d wkirgsn/electric-motor-temperature",
    "license": "CC-BY-SA-4.0",
    "citation_note": "Dataset owner requests citation of the associated PMSM temperature-estimation papers.",
    "sample_time_s": 0.5,
    "target_temperature": "pm",
    "coolant_temperature": "coolant",
    "train_profile_count": 10,
    "test_profile_count": 6,
    "min_profile_length": 12000,
    "savgol_window": 501,
    "savgol_polyorder": 3,
    "edge_trim": 300,
    "prediction_sample_stride": 50,
    "models": {
        "B0_fixed_cooling": "dT/dt = q_loss(i,torque,speed) - h0*(T-Tcoolant)",
        "B3_speed_adaptive_cooling_slot": "dT/dt = q_loss(i,torque,speed) - (h0+h1*|speed|)*(T-Tcoolant)",
        "B3_operating_adaptive_cooling_slot": "dT/dt = q_loss(i,torque,speed) - h(speed,current,power)*(T-Tcoolant)",
        "B1_full_linear_features": "wider linear feature model over thermal and operating variables",
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


def load_data() -> pd.DataFrame:
    path = RAW_ROOT / "measures_v2.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing PMSM data file: {path}")
    return pd.read_csv(path)


def select_profiles(df: pd.DataFrame) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    counts = df.groupby("profile_id").size().sort_values(ascending=False)
    eligible = [int(pid) for pid, count in counts.items() if int(count) >= int(CONFIG["min_profile_length"])]
    train_count = int(CONFIG["train_profile_count"])
    test_count = int(CONFIG["test_profile_count"])
    train = eligible[:train_count]
    test = eligible[train_count : train_count + test_count]
    rows = [
        {
            "profile_id": int(pid),
            "n_samples": int(counts.loc[pid]),
            "role": "train" if int(pid) in train else "test" if int(pid) in test else "unused",
        }
        for pid in counts.index
    ]
    return train, test, rows


def prepared_profile(df: pd.DataFrame, profile_id: int) -> dict[str, np.ndarray]:
    g = df[df["profile_id"] == profile_id].reset_index(drop=True)
    target = g[str(CONFIG["target_temperature"])].to_numpy(dtype=float)
    n = len(target)
    window = min(int(CONFIG["savgol_window"]), n // 2 * 2 - 1)
    if window < 11:
        raise ValueError(f"profile {profile_id} too short for derivative estimation")
    if window % 2 == 0:
        window -= 1
    deriv = savgol_filter(
        target,
        window_length=window,
        polyorder=int(CONFIG["savgol_polyorder"]),
        deriv=1,
        delta=float(CONFIG["sample_time_s"]),
        mode="interp",
    )
    edge = min(int(CONFIG["edge_trim"]), (n - 3) // 2)
    sl = slice(edge, n - edge)
    coolant = g["coolant"].to_numpy(dtype=float)
    ambient = g["ambient"].to_numpy(dtype=float)
    speed = g["motor_speed"].to_numpy(dtype=float)
    torque = g["torque"].to_numpy(dtype=float)
    id_current = g["i_d"].to_numpy(dtype=float)
    iq_current = g["i_q"].to_numpy(dtype=float)
    ud = g["u_d"].to_numpy(dtype=float)
    uq = g["u_q"].to_numpy(dtype=float)
    i2 = id_current**2 + iq_current**2
    elec_power = np.abs(ud * id_current + uq * iq_current)
    mech = np.abs(torque * speed)
    return {
        "profile_id": np.full(len(target[sl]), int(profile_id), dtype=int),
        "T": target[sl],
        "dT": deriv[sl],
        "coolant": coolant[sl],
        "ambient": ambient[sl],
        "speed": speed[sl],
        "speed_abs": np.abs(speed[sl]),
        "torque": torque[sl],
        "i2": i2[sl],
        "elec_power": elec_power[sl],
        "mech_power": mech[sl],
    }


def stack_profiles(profiles: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = profiles[0].keys()
    return {key: np.concatenate([profile[key] for profile in profiles]) for key in keys}


def scaling(train: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "i2": float(np.std(train["i2"])) or 1.0,
        "elec_power": float(np.std(train["elec_power"])) or 1.0,
        "mech_power": float(np.std(train["mech_power"])) or 1.0,
        "speed_abs": float(np.std(train["speed_abs"])) or 1.0,
        "torque": float(np.std(train["torque"])) or 1.0,
        "ambient": float(np.std(train["ambient"])) or 1.0,
    }


def design(model: str, data: dict[str, np.ndarray], scale: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    delta = data["T"] - data["coolant"]
    i2 = data["i2"] / scale["i2"]
    elec = data["elec_power"] / scale["elec_power"]
    mech = data["mech_power"] / scale["mech_power"]
    speed = data["speed_abs"] / scale["speed_abs"]
    ambient_delta = data["T"] - data["ambient"]
    if model == "B0_fixed_cooling":
        names = ["1", "i2", "elec_power", "mech_power", "-delta_coolant"]
        theta = np.column_stack([np.ones_like(delta), i2, elec, mech, -delta])
    elif model == "B3_speed_adaptive_cooling_slot":
        names = ["1", "i2", "elec_power", "mech_power", "-delta_coolant", "-delta_coolant*abs(speed)"]
        theta = np.column_stack([np.ones_like(delta), i2, elec, mech, -delta, -delta * speed])
    elif model == "B3_operating_adaptive_cooling_slot":
        names = [
            "1",
            "i2",
            "elec_power",
            "mech_power",
            "-delta_coolant",
            "-delta_coolant*abs(speed)",
            "-delta_coolant*i2",
            "-delta_coolant*mech_power",
        ]
        theta = np.column_stack([np.ones_like(delta), i2, elec, mech, -delta, -delta * speed, -delta * i2, -delta * mech])
    elif model == "B1_full_linear_features":
        names = [
            "1",
            "i2",
            "elec_power",
            "mech_power",
            "-delta_coolant",
            "-delta_coolant*abs(speed)",
            "-delta_ambient",
            "abs(speed)",
            "torque",
        ]
        theta = np.column_stack(
            [
                np.ones_like(delta),
                i2,
                elec,
                mech,
                -delta,
                -delta * speed,
                -ambient_delta,
                speed,
                data["torque"] / scale["torque"],
            ]
        )
    else:
        raise ValueError(model)
    return theta, names


def fit_model(model: str, train: dict[str, np.ndarray], scale: dict[str, float]) -> dict[str, Any]:
    theta, names = design(model, train, scale)
    coeffs, *_ = np.linalg.lstsq(theta, train["dT"], rcond=None)
    return {"model": model, "names": names, "coefficients": coeffs}


def predict_derivative(fit: dict[str, Any], data: dict[str, np.ndarray], scale: dict[str, float]) -> np.ndarray:
    theta, _ = design(fit["model"], data, scale)
    return theta @ fit["coefficients"]


def derivative_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - true
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.std(true)) or 1.0
    return {
        "derivative_rmse_c_per_s": rmse,
        "derivative_nrmse": rmse / denom,
        "derivative_mae_c_per_s": float(np.mean(np.abs(err))),
        "derivative_bias_c_per_s": float(np.mean(err)),
    }


def derivative_from_state(fit: dict[str, Any], row: dict[str, float], temp: float, scale: dict[str, float]) -> float:
    c = fit["coefficients"]
    delta = temp - row["coolant"]
    i2 = row["i2"] / scale["i2"]
    elec = row["elec_power"] / scale["elec_power"]
    mech = row["mech_power"] / scale["mech_power"]
    speed = abs(row["speed"]) / scale["speed_abs"]
    if fit["model"] == "B0_fixed_cooling":
        return float(c[0] + c[1] * i2 + c[2] * elec + c[3] * mech - c[4] * delta)
    if fit["model"] == "B3_speed_adaptive_cooling_slot":
        return float(c[0] + c[1] * i2 + c[2] * elec + c[3] * mech - c[4] * delta - c[5] * delta * speed)
    if fit["model"] == "B3_operating_adaptive_cooling_slot":
        return float(
            c[0]
            + c[1] * i2
            + c[2] * elec
            + c[3] * mech
            - c[4] * delta
            - c[5] * delta * speed
            - c[6] * delta * i2
            - c[7] * delta * mech
        )
    if fit["model"] == "B1_full_linear_features":
        ambient_delta = temp - row["ambient"]
        torque = row["torque"] / scale["torque"]
        return float(
            c[0]
            + c[1] * i2
            + c[2] * elec
            + c[3] * mech
            - c[4] * delta
            - c[5] * delta * speed
            - c[6] * ambient_delta
            + c[7] * speed
            + c[8] * torque
        )
    raise ValueError(fit["model"])


def rollout_profile(fit: dict[str, Any], profile: dict[str, np.ndarray], scale: dict[str, float]) -> np.ndarray:
    dt = float(CONFIG["sample_time_s"])
    pred = np.empty_like(profile["T"], dtype=float)
    pred[0] = float(profile["T"][0])
    for idx in range(len(pred) - 1):
        row = {
            "coolant": float(profile["coolant"][idx]),
            "ambient": float(profile["ambient"][idx]),
            "speed": float(profile["speed"][idx]),
            "torque": float(profile["torque"][idx]),
            "i2": float(profile["i2"][idx]),
            "elec_power": float(profile["elec_power"][idx]),
            "mech_power": float(profile["mech_power"][idx]),
        }
        k1 = derivative_from_state(fit, row, float(pred[idx]), scale)
        mid_temp = float(pred[idx] + 0.5 * dt * k1)
        k2 = derivative_from_state(fit, row, mid_temp, scale)
        next_temp = pred[idx] + dt * k2
        if not np.isfinite(next_temp):
            next_temp = pred[idx]
        pred[idx + 1] = float(np.clip(next_temp, -100.0, 250.0))
    return pred


def rollout_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - true
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.std(true)) or 1.0
    return {
        "rollout_rmse_c": rmse,
        "rollout_nrmse": rmse / denom,
        "rollout_mae_c": float(np.mean(np.abs(err))),
        "rollout_bias_c": float(np.mean(err)),
        "rollout_max_abs_c": float(np.max(np.abs(err))),
    }


def write_profile_svg(path: Path, profile: dict[str, np.ndarray], preds: dict[str, np.ndarray], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 920, 460
    left, right, top, bottom = 64, 20, 45, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    y_all = np.concatenate([profile["T"], *preds.values()])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad
    colors = {"Measured": "#000000", "B0_fixed_cooling": "#D55E00", "B3_operating_adaptive_cooling_slot": "#0072B2"}
    dashes = {"Measured": "", "B0_fixed_cooling": "8 4", "B3_operating_adaptive_cooling_slot": ""}
    labels = {
        "Measured": "实测 PM 温度" if zh else "Measured PM temperature",
        "B0_fixed_cooling": "B0 固定冷却系数" if zh else "B0 fixed cooling",
        "B3_operating_adaptive_cooling_slot": "B3 工况自适应冷却槽" if zh else "B3 operating-adaptive cooling slot",
    }
    title = "E8 PMSM 温度 rollout" if zh else "E8 PMSM temperature rollout"

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
        f'<polyline points="{poly(profile["T"])}" fill="none" stroke="{colors["Measured"]}" stroke-width="1.6"/>',
    ]
    for model in ["B0_fixed_cooling", "B3_operating_adaptive_cooling_slot"]:
        parts.append(f'<polyline points="{poly(preds[model])}" fill="none" stroke="{colors[model]}" stroke-width="2" stroke-dasharray="{dashes[model]}"/>')
    lx, ly = left, height - 24
    for i, key in enumerate(["Measured", "B0_fixed_cooling", "B3_operating_adaptive_cooling_slot"]):
        x = lx + i * 260
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
    df = load_data()
    train_ids, test_ids, profile_rows = select_profiles(df)
    train_profiles = [prepared_profile(df, pid) for pid in train_ids]
    test_profiles = [prepared_profile(df, pid) for pid in test_ids]
    train = stack_profiles(train_profiles)
    test = stack_profiles(test_profiles)
    scale = scaling(train)
    models = [
        "B0_fixed_cooling",
        "B3_speed_adaptive_cooling_slot",
        "B3_operating_adaptive_cooling_slot",
        "B1_full_linear_features",
    ]
    fits = {model: fit_model(model, train, scale) for model in models}

    coefficient_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    profile_metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    split_data = {"train": train, "test": test}
    split_profiles = {"train": train_profiles, "test": test_profiles}
    rollout_predictions: dict[tuple[str, str, int], np.ndarray] = {}

    for model, fit in fits.items():
        for name, coeff in zip(fit["names"], fit["coefficients"]):
            coefficient_rows.append({"model": model, "feature": name, "coefficient": float(coeff)})
        for split, data in split_data.items():
            d_pred = predict_derivative(fit, data, scale)
            d_met = derivative_metrics(data["dT"], d_pred)
            rollout_true_parts = []
            rollout_pred_parts = []
            for profile in split_profiles[split]:
                pred = rollout_profile(fit, profile, scale)
                pid = int(profile["profile_id"][0])
                rollout_predictions[(model, split, pid)] = pred
                r_met = rollout_metrics(profile["T"], pred)
                profile_metric_rows.append({"model": model, "split": split, "profile_id": pid, "n_samples": int(len(pred)), **r_met})
                rollout_true_parts.append(profile["T"])
                rollout_pred_parts.append(pred)
            roll_met = rollout_metrics(np.concatenate(rollout_true_parts), np.concatenate(rollout_pred_parts))
            metric_rows.append({"model": model, "split": split, "n_samples": int(len(data["T"])), **d_met, **roll_met})

    metric_by_key = {(row["model"], row["split"]): row for row in metric_rows}
    b0 = float(metric_by_key[("B0_fixed_cooling", "test")]["rollout_nrmse"])
    b3_speed = float(metric_by_key[("B3_speed_adaptive_cooling_slot", "test")]["rollout_nrmse"])
    b3 = float(metric_by_key[("B3_operating_adaptive_cooling_slot", "test")]["rollout_nrmse"])
    b1 = float(metric_by_key[("B1_full_linear_features", "test")]["rollout_nrmse"])
    summary = {
        "train_profiles": train_ids,
        "test_profiles": test_ids,
        "test_b0_rollout_nrmse": b0,
        "test_b3_speed_rollout_nrmse": b3_speed,
        "test_b3_operating_rollout_nrmse": b3,
        "test_b1_rollout_nrmse": b1,
        "test_b3_operating_vs_b0_improvement_percent": 100.0 * (b0 - b3) / b0 if b0 else float("nan"),
        "test_b3_operating_vs_b1_gap_nrmse": b3 - b1,
    }

    stride = int(CONFIG["prediction_sample_stride"])
    for split, profiles in split_profiles.items():
        for profile in profiles:
            pid = int(profile["profile_id"][0])
            for idx in range(0, len(profile["T"]), stride):
                row: dict[str, Any] = {
                    "split": split,
                    "profile_id": pid,
                    "index": idx,
                    "pm_temperature_c": float(profile["T"][idx]),
                    "coolant_c": float(profile["coolant"][idx]),
                    "speed": float(profile["speed"][idx]),
                    "torque": float(profile["torque"][idx]),
                    "i2": float(profile["i2"][idx]),
                }
                for model in ["B0_fixed_cooling", "B3_speed_adaptive_cooling_slot", "B3_operating_adaptive_cooling_slot"]:
                    row[f"{model}_rollout_c"] = float(rollout_predictions[(model, split, pid)][idx])
                prediction_rows.append(row)

    if test_profiles:
        example = test_profiles[0]
        pid = int(example["profile_id"][0])
        preds = {
            "B0_fixed_cooling": rollout_predictions[("B0_fixed_cooling", "test", pid)],
            "B3_operating_adaptive_cooling_slot": rollout_predictions[("B3_operating_adaptive_cooling_slot", "test", pid)],
        }
        write_profile_svg(FIGURE_DIR / "test_profile_rollout.svg", example, preds, zh=False)
        write_profile_svg(FIGURE_DIR / "test_profile_rollout_zh.svg", example, preds, zh=True)

    raw_files = collect_raw_files()
    metrics_payload = {
        "experiment_id": CONFIG["experiment_id"],
        "summary": summary,
        "metrics_by_split": metric_rows,
        "metrics_by_profile": profile_metric_rows,
        "coefficients": coefficient_rows,
        "profile_rows": profile_rows,
        "scaling": scale,
        "raw_files": raw_files,
    }
    write_csv(RESULT_DIR / "profile_summary.csv", profile_rows)
    write_csv(RESULT_DIR / "metrics_by_split.csv", metric_rows)
    write_csv(RESULT_DIR / "metrics_by_profile.csv", profile_metric_rows)
    write_csv(RESULT_DIR / "coefficients.csv", coefficient_rows)
    write_csv(RESULT_DIR / "prediction_sample.csv", prediction_rows)
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
            "kaggle": package_version("kaggle"),
        },
        "outputs": {
            "metrics_json": str((RESULT_DIR / "metrics.json").relative_to(ROOT)),
            "metrics_by_split_csv": str((RESULT_DIR / "metrics_by_split.csv").relative_to(ROOT)),
            "metrics_by_profile_csv": str((RESULT_DIR / "metrics_by_profile.csv").relative_to(ROOT)),
            "coefficients_csv": str((RESULT_DIR / "coefficients.csv").relative_to(ROOT)),
            "prediction_sample_csv": str((RESULT_DIR / "prediction_sample.csv").relative_to(ROOT)),
        },
        "raw_files": raw_files,
    }
    (PROVENANCE_DIR / "e8_pmsm_temperature_slot_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
