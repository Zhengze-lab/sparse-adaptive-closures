#!/usr/bin/env python3
"""Run E7: EMPS real-system friction-slot inverse-dynamics experiment."""

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
from scipy.io import loadmat
from scipy.signal import butter, decimate, filtfilt


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "external" / "emps"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "e7_emps_friction_slot"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG: dict[str, Any] = {
    "experiment_id": "E7_emps_friction_slot",
    "description": "Real EMPS inverse-dynamics test: symmetric friction baseline vs asymmetric velocity-dependent friction slot.",
    "source_dataset": "Electro-Mechanical Positioning System benchmark",
    "source_url": "https://www.nonlinearbenchmark.org/benchmarks/emps",
    "dataloader": "nonlinear_benchmarks.EMPS",
    "citation": (
        "A. Janot, M. Gautier, and M. Brunot. Data Set and Reference Models of EMPS. "
        "Workshop on Nonlinear System Identification Benchmarks, Eindhoven, 2019."
    ),
    "filter_cutoff_hz": 100.0,
    "filter_order": 4,
    "decimation_factor": 10,
    "edge_trim_samples": 50,
    "prediction_sample_stride": 10,
    "models": {
        "B0_viscous": "tau = M*a + Fv*v + offset",
        "B0_symmetric_coulomb": "tau = M*a + Fv*v + Fc*sign(v) + offset",
        "B3_asymmetric_friction_slot": "tau = M*a + Fv+*v+ + Fc+*I(v>0) + Fv-*v- + Fc-*sign(v<0) + offset",
        "B1_full_polynomial_inverse": "tau = sparse-free LS over a wider inverse-dynamics feature library",
    },
    "test_usage_rule": "The DATA_EMPS_PULSES record is used only for final cross-test force prediction.",
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


def ensure_data() -> list[str]:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    locations = nb.EMPS(data_file_locations=True, dir_placement=str(RAW_ROOT), force_download=False)
    return [str(loc) for loc in locations]


def central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    out[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    out[0] = (x[1] - x[0]) / dt
    out[-1] = (x[-1] - x[-2]) / dt
    return out


def load_record(path: Path, split: str) -> dict[str, Any]:
    mat = loadmat(path, squeeze_me=True)
    q = np.asarray(mat["qm"], dtype=float).reshape(-1)
    t = np.asarray(mat["t"], dtype=float).reshape(-1)
    vir = np.asarray(mat["vir"], dtype=float).reshape(-1)
    gtau = float(np.asarray(mat["gtau"]).reshape(-1)[0])
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    cutoff = float(CONFIG["filter_cutoff_hz"])
    order = int(CONFIG["filter_order"])
    b, a = butter(order, cutoff / (0.5 * fs))
    qf = filtfilt(b, a, q)
    v = central_diff(qf, dt)
    acc = central_diff(v, dt)
    force = gtau * vir
    trim = int(CONFIG["edge_trim_samples"])
    if trim > 0:
        qf = qf[trim:]
        v = v[trim:]
        acc = acc[trim:]
        force = force[trim:]
        t = t[trim:]
    decim = int(CONFIG["decimation_factor"])
    if decim > 1:
        qf = decimate(qf, decim, ftype="iir", zero_phase=True)
        v = decimate(v, decim, ftype="iir", zero_phase=True)
        acc = decimate(acc, decim, ftype="iir", zero_phase=True)
        force = decimate(force, decim, ftype="iir", zero_phase=True)
        t = decimate(t, decim, ftype="iir", zero_phase=True)
    return {
        "split": split,
        "path": str(path.relative_to(ROOT)),
        "dt": dt * decim,
        "t": t,
        "q": qf,
        "v": v,
        "acc": acc,
        "force": force,
        "gtau": gtau,
        "n": int(len(force)),
    }


def design_matrix(model: str, v: np.ndarray, acc: np.ndarray) -> tuple[np.ndarray, list[str]]:
    sign = np.sign(v)
    pos = (v > 0.0).astype(float)
    neg = (v < 0.0).astype(float)
    v_pos = np.where(v > 0.0, v, 0.0)
    v_neg = np.where(v < 0.0, v, 0.0)
    if model == "B0_viscous":
        names = ["acc", "v", "1"]
        theta = np.column_stack([acc, v, np.ones_like(v)])
    elif model == "B0_symmetric_coulomb":
        names = ["acc", "v", "sign(v)", "1"]
        theta = np.column_stack([acc, v, sign, np.ones_like(v)])
    elif model == "B3_asymmetric_friction_slot":
        names = ["acc", "v_pos", "I_pos", "v_neg", "sign_neg", "1"]
        theta = np.column_stack([acc, v_pos, pos, v_neg, -neg, np.ones_like(v)])
    elif model == "B1_full_polynomial_inverse":
        names = ["acc", "v", "sign(v)", "abs(v)", "v*abs(v)", "v^2", "acc*v", "1"]
        theta = np.column_stack([acc, v, sign, np.abs(v), v * np.abs(v), v**2, acc * v, np.ones_like(v)])
    else:
        raise ValueError(f"unknown model {model}")
    return theta, names


def fit_ls(model: str, record: dict[str, Any]) -> dict[str, Any]:
    theta, names = design_matrix(model, record["v"], record["acc"])
    y = record["force"]
    coeffs, *_ = np.linalg.lstsq(theta, y, rcond=None)
    return {"model": model, "feature_names": names, "coefficients": coeffs}


def predict(model_fit: dict[str, Any], record: dict[str, Any]) -> np.ndarray:
    theta, _ = design_matrix(model_fit["model"], record["v"], record["acc"])
    return theta @ model_fit["coefficients"]


def metrics(force: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - force
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.std(force)) or 1.0
    nrmse = rmse / denom
    rel_norm_percent = 100.0 * float(np.linalg.norm(err) / (np.linalg.norm(force) or 1.0))
    return {
        "rmse_n": rmse,
        "mae_n": mae,
        "nrmse": nrmse,
        "fit_percent": 100.0 * (1.0 - nrmse),
        "relative_error_percent": rel_norm_percent,
        "bias_n": float(np.mean(err)),
        "max_abs_error_n": float(np.max(np.abs(err))),
    }


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


def write_force_svg(path: Path, records: dict[str, dict[str, Any]], predictions: dict[tuple[str, str], np.ndarray], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 520
    left, right, top, bottom, gap = 70, 20, 45, 58, 40
    panel_w = (width - left - right - gap) / 2.0
    panel_h = height - top - bottom
    all_force = np.concatenate([records[key]["force"] for key in ["train", "test"]])
    y_min, y_max = float(np.min(all_force)), float(np.max(all_force))
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad
    colors = {
        "measured": "#000000",
        "B0_viscous": "#D55E00",
        "B0_symmetric_coulomb": "#E69F00",
        "B3_asymmetric_friction_slot": "#0072B2",
    }
    dashes = {
        "measured": "",
        "B0_viscous": "8 4",
        "B0_symmetric_coulomb": "3 3",
        "B3_asymmetric_friction_slot": "",
    }
    labels_en = {
        "measured": "Measured force",
        "B0_viscous": "B0 viscous friction",
        "B0_symmetric_coulomb": "B0 symmetric Coulomb",
        "B3_asymmetric_friction_slot": "B3 asymmetric friction slot",
    }
    labels_zh = {
        "measured": "实测力",
        "B0_viscous": "B0 粘滞摩擦",
        "B0_symmetric_coulomb": "B0 对称库仑摩擦",
        "B3_asymmetric_friction_slot": "B3 非对称摩擦槽",
    }
    labels = labels_zh if zh else labels_en
    title = "E7 EMPS 逆动力学力预测" if zh else "E7 EMPS inverse-dynamics force prediction"

    def sx(i: int, n: int, panel: int) -> float:
        return left + panel * (panel_w + gap) + panel_w * i / max(n - 1, 1)

    def sy(v: float) -> float:
        return top + panel_h * (1.0 - (v - y_min) / (y_max - y_min))

    def poly(values: np.ndarray, panel: int) -> str:
        step = max(1, len(values) // 600)
        return " ".join(f"{sx(i, len(values), panel):.2f},{sy(float(values[i])):.2f}" for i in range(0, len(values), step))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">{title}</text>',
    ]
    for panel, split in enumerate(["train", "test"]):
        x0 = left + panel * (panel_w + gap)
        panel_title = "训练集" if (zh and split == "train") else "测试集" if zh else split.capitalize()
        parts.append(f'<text x="{x0 + panel_w/2:.1f}" y="{top - 12}" text-anchor="middle" font-family="Arial" font-size="13">{panel_title}</text>')
        parts.append(f'<rect x="{x0:.2f}" y="{top:.2f}" width="{panel_w:.2f}" height="{panel_h:.2f}" fill="none" stroke="#666"/>')
        parts.append(f'<polyline points="{poly(records[split]["force"], panel)}" fill="none" stroke="{colors["measured"]}" stroke-width="1.5"/>')
        for model in ["B0_viscous", "B0_symmetric_coulomb", "B3_asymmetric_friction_slot"]:
            parts.append(f'<polyline points="{poly(predictions[(model, split)], panel)}" fill="none" stroke="{colors[model]}" stroke-width="1.9" stroke-dasharray="{dashes[model]}"/>')
    lx, ly = left, height - 23
    for i, key in enumerate(["measured", "B0_viscous", "B0_symmetric_coulomb", "B3_asymmetric_friction_slot"]):
        x = lx + i * 225
        parts.append(f'<line x1="{x}" y1="{ly}" x2="{x+30}" y2="{ly}" stroke="{colors[key]}" stroke-width="2" stroke-dasharray="{dashes[key]}"/>')
        parts.append(f'<text x="{x+36}" y="{ly+4}" font-family="Arial" font-size="11">{labels[key]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_data()
    train = load_record(RAW_ROOT / "EMPS" / "DATA_EMPS.mat", "train")
    test = load_record(RAW_ROOT / "EMPS" / "DATA_EMPS_PULSES.mat", "test")
    records = {"train": train, "test": test}
    models = ["B0_viscous", "B0_symmetric_coulomb", "B3_asymmetric_friction_slot", "B1_full_polynomial_inverse"]
    fits = {model: fit_ls(model, train) for model in models}

    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for model, fit in fits.items():
        for name, value in zip(fit["feature_names"], fit["coefficients"]):
            coefficient_rows.append({"model": model, "feature": name, "coefficient": float(value)})
        for split, record in records.items():
            pred = predict(fit, record)
            predictions[(model, split)] = pred
            metric_rows.append(
                {
                    "model": model,
                    "split": split,
                    "n_samples": int(record["n"]),
                    "dt_s": float(record["dt"]),
                    **metrics(record["force"], pred),
                }
            )

    metric_by_key = {(row["model"], row["split"]): row for row in metric_rows}
    b0 = float(metric_by_key[("B0_symmetric_coulomb", "test")]["nrmse"])
    b3 = float(metric_by_key[("B3_asymmetric_friction_slot", "test")]["nrmse"])
    b1 = float(metric_by_key[("B1_full_polynomial_inverse", "test")]["nrmse"])
    summary = {
        "test_b0_symmetric_nrmse": b0,
        "test_b3_asymmetric_slot_nrmse": b3,
        "test_b1_full_polynomial_nrmse": b1,
        "test_b3_vs_b0_improvement_percent": 100.0 * (b0 - b3) / b0 if b0 else float("nan"),
        "test_b3_vs_b1_gap_nrmse": b3 - b1,
    }

    sample_rows: list[dict[str, Any]] = []
    stride = int(CONFIG["prediction_sample_stride"])
    for split, record in records.items():
        for idx in range(0, len(record["force"]), stride):
            row: dict[str, Any] = {
                "split": split,
                "index": idx,
                "time_s": float(record["t"][idx]),
                "position_m": float(record["q"][idx]),
                "velocity_m_s": float(record["v"][idx]),
                "acceleration_m_s2": float(record["acc"][idx]),
                "force_n": float(record["force"][idx]),
            }
            for model in models:
                row[f"{model}_prediction_n"] = float(predictions[(model, split)][idx])
            sample_rows.append(row)

    raw_files = collect_raw_files(RAW_ROOT)
    metrics_payload = {
        "experiment_id": CONFIG["experiment_id"],
        "summary": summary,
        "metric_rows": metric_rows,
        "coefficients": coefficient_rows,
        "raw_files": raw_files,
    }
    write_csv(RESULT_DIR / "metrics_by_split.csv", metric_rows)
    write_csv(RESULT_DIR / "coefficients.csv", coefficient_rows)
    write_csv(RESULT_DIR / "prediction_sample.csv", sample_rows)
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_force_svg(FIGURE_DIR / "force_prediction.svg", records, predictions, zh=False)
    write_force_svg(FIGURE_DIR / "force_prediction_zh.svg", records, predictions, zh=True)

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
            "coefficients_csv": str((RESULT_DIR / "coefficients.csv").relative_to(ROOT)),
            "prediction_sample_csv": str((RESULT_DIR / "prediction_sample.csv").relative_to(ROOT)),
            "force_prediction_svg": str((FIGURE_DIR / "force_prediction.svg").relative_to(ROOT)),
            "force_prediction_zh_svg": str((FIGURE_DIR / "force_prediction_zh.svg").relative_to(ROOT)),
        },
        "raw_files": raw_files,
    }
    (PROVENANCE_DIR / "e7_emps_friction_slot_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
