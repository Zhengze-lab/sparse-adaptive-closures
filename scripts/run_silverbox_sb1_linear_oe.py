#!/usr/bin/env python3
"""Run SB-1: linear second-order output-error baseline for Silverbox."""

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
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "external" / "silverbox"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "silverbox_sb1_linear_oe"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG: dict[str, Any] = {
    "experiment_id": "SB1_silverbox_linear_output_error",
    "description": "Linear second-order output-error baseline for the Silverbox input-output benchmark.",
    "source_dataset": "silverbox_data_audit",
    "dataloader": "nonlinear_benchmarks.Silverbox",
    "train_fit_fraction": 0.8,
    "state_initialization_window": 50,
    "fit_residual_stride": 5,
    "fit_max_nfev": 100,
    "init_max_nfev": 60,
    "savgol_window": 101,
    "savgol_polyorder": 3,
    "parameterization": {
        "c": "positive damping, c=exp(log_c)",
        "k": "positive stiffness, k=exp(log_k)",
        "b": "signed input gain, b=b_scale*b_scaled",
        "x20": "signed initial latent velocity, x20=x2_scale*x20_scaled",
    },
    "test_usage_rule": "Official test records are used only after selecting the baseline by internal train_val validation.",
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


def discretize_second_order(dt: float, c: float, k: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.array(
        [
            [0.0, 1.0, 0.0],
            [-k, -c, b],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    exp_matrix = expm(matrix * dt)
    return exp_matrix[:2, :2], exp_matrix[:2, 2]


def simulate_linear_oe(
    u: np.ndarray,
    y0: float,
    x20: float,
    dt: float,
    c: float,
    k: float,
    b: float,
    u_offset: float,
    y_offset: float,
) -> np.ndarray:
    f_matrix, g_vector = discretize_second_order(dt, c, k, b)
    output = np.empty(len(u), dtype=float)
    state = np.array([y0 - y_offset, x20], dtype=float)
    u_centered = u - u_offset
    output[0] = state[0] + y_offset
    for idx in range(len(u) - 1):
        state = f_matrix @ state + g_vector * u_centered[idx]
        output[idx + 1] = state[0] + y_offset
    return output


def rmse_mv(y_true: np.ndarray, y_pred: np.ndarray, n_init: int) -> float:
    residual = y_pred[n_init:] - y_true[n_init:]
    return 1000.0 * float(np.sqrt(np.mean(residual * residual)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, n_init: int) -> float:
    residual = y_pred[n_init:] - y_true[n_init:]
    denom = float(np.std(y_true[n_init:])) or 1.0
    return float(np.sqrt(np.mean(residual * residual))) / denom


def derivative_initial_guess(u: np.ndarray, y: np.ndarray, dt: float, u_offset: float, y_offset: float) -> tuple[float, float, float]:
    y_centered = y - y_offset
    u_centered = u - u_offset
    window = min(int(CONFIG["savgol_window"]), len(y_centered) // 2 * 2 - 1)
    if window < 7:
        return 1.0, 100.0, 1.0
    dy = savgol_filter(y_centered, window, int(CONFIG["savgol_polyorder"]), deriv=1, delta=dt, mode="interp")
    ddy = savgol_filter(y_centered, window, int(CONFIG["savgol_polyorder"]), deriv=2, delta=dt, mode="interp")
    edge = window
    if len(y_centered) <= 2 * edge + 10:
        edge = window // 2
    design = np.column_stack([-dy[edge:-edge], -y_centered[edge:-edge], u_centered[edge:-edge]])
    target = ddy[edge:-edge]
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    c0 = max(float(coeffs[0]), 1e-3)
    k0 = max(float(coeffs[1]), 1e-3)
    b0 = float(coeffs[2])
    return c0, k0, b0


def make_starts(c0: float, k0: float, b0: float, b_scale: float) -> list[tuple[str, np.ndarray]]:
    starts = [
        ("derivative_ls", np.array([np.log(c0), np.log(k0), b0 / b_scale, 0.0], dtype=float)),
        ("low_frequency", np.array([np.log(1.0), np.log(100.0), np.sign(b0) or 1.0, 0.0], dtype=float)),
        ("medium_frequency", np.array([np.log(20.0), np.log(1000.0), np.sign(b0) or 1.0, 0.0], dtype=float)),
        ("high_frequency", np.array([np.log(100.0), np.log(10000.0), np.sign(b0) or 1.0, 0.0], dtype=float)),
    ]
    return starts


def unpack_params(params: np.ndarray, b_scale: float, x2_scale: float) -> tuple[float, float, float, float]:
    c = float(np.exp(params[0]))
    k = float(np.exp(params[1]))
    b = float(params[2] * b_scale)
    x20 = float(params[3] * x2_scale)
    return c, k, b, x20


def fit_candidate(
    start_name: str,
    start: np.ndarray,
    u_fit: np.ndarray,
    y_fit: np.ndarray,
    dt: float,
    u_offset: float,
    y_offset: float,
    y_scale: float,
    b_scale: float,
    x2_scale: float,
) -> dict[str, Any]:
    n_init = int(CONFIG["state_initialization_window"])
    stride = int(CONFIG["fit_residual_stride"])

    def residual(params: np.ndarray) -> np.ndarray:
        c, k, b, x20 = unpack_params(params, b_scale, x2_scale)
        prediction = simulate_linear_oe(u_fit, y_fit[0], x20, dt, c, k, b, u_offset, y_offset)
        return (prediction[n_init::stride] - y_fit[n_init::stride]) / y_scale

    lower = np.array([np.log(1e-3), np.log(1e-3), -100.0, -100.0], dtype=float)
    upper = np.array([np.log(1e5), np.log(1e8), 100.0, 100.0], dtype=float)
    result = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        max_nfev=int(CONFIG["fit_max_nfev"]),
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
    )
    c, k, b, x20 = unpack_params(result.x, b_scale, x2_scale)
    prediction = simulate_linear_oe(u_fit, y_fit[0], x20, dt, c, k, b, u_offset, y_offset)
    return {
        "start": start_name,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "c": c,
        "k": k,
        "b": b,
        "x20_fit": x20,
        "params_scaled": [float(v) for v in result.x],
        "fit_rmse_mv": rmse_mv(y_fit, prediction, n_init),
        "fit_nrmse": nrmse(y_fit, prediction, n_init),
    }


def estimate_initial_velocity(
    u: np.ndarray,
    y: np.ndarray,
    dt: float,
    c: float,
    k: float,
    b: float,
    u_offset: float,
    y_offset: float,
    y_scale: float,
    x2_scale: float,
    n_init: int,
) -> float:
    n_init = min(n_init, len(y))

    def residual(x2_scaled: np.ndarray) -> np.ndarray:
        prediction = simulate_linear_oe(
            u[:n_init],
            y[0],
            float(x2_scaled[0] * x2_scale),
            dt,
            c,
            k,
            b,
            u_offset,
            y_offset,
        )
        return (prediction - y[:n_init]) / y_scale

    result = least_squares(
        residual,
        np.array([0.0], dtype=float),
        bounds=(np.array([-100.0]), np.array([100.0])),
        max_nfev=int(CONFIG["init_max_nfev"]),
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    return float(result.x[0] * x2_scale)


def evaluate_record(
    name: str,
    split: str,
    u: np.ndarray,
    y: np.ndarray,
    dt: float,
    c: float,
    k: float,
    b: float,
    u_offset: float,
    y_offset: float,
    y_scale: float,
    x2_scale: float,
    n_init: int,
) -> tuple[dict[str, Any], np.ndarray]:
    x20 = estimate_initial_velocity(u, y, dt, c, k, b, u_offset, y_offset, y_scale, x2_scale, n_init)
    prediction = simulate_linear_oe(u, y[0], x20, dt, c, k, b, u_offset, y_offset)
    row = {
        "record": name,
        "split": split,
        "sample_count": int(len(y)),
        "sampling_time_s": dt,
        "n_init": int(n_init),
        "x20_initialized": x20,
        "rmse_mv": rmse_mv(y, prediction, n_init),
        "nrmse": nrmse(y, prediction, n_init),
        "y_std_after_init": float(np.std(y[n_init:])),
    }
    return row, prediction


def save_prediction_sample(path: Path, t: np.ndarray, u: np.ndarray, y: np.ndarray, y_pred: np.ndarray, max_rows: int = 5000) -> None:
    limit = min(max_rows, len(y))
    rows = [
        {
            "t": f"{float(t[idx]):.16g}",
            "u": f"{float(u[idx]):.16g}",
            "y_true": f"{float(y[idx]):.16g}",
            "y_pred": f"{float(y_pred[idx]):.16g}",
            "residual": f"{float(y_pred[idx] - y[idx]):.16g}",
        }
        for idx in range(limit)
    ]
    write_csv(path, rows)


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
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0
    xp = left + (x - x_min) / (x_max - x_min) * width
    yp = top + height - (y - y_min) / (y_max - y_min) * height
    return " ".join(f"{float(xx):.2f},{float(yy):.2f}" for xx, yy in zip(xp, yp))


def write_rollout_svg(
    path: Path,
    t: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    subtitle: str,
    labels: dict[str, str],
    max_points: int = 3000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    limit = min(max_points, len(y))
    idx = np.linspace(0, len(y) - 1, limit).astype(int)
    tt = t[idx]
    yy = y[idx]
    pp = y_pred[idx]
    y_min = float(min(np.min(yy), np.min(pp)))
    y_max = float(max(np.max(yy), np.max(pp)))
    pad = 0.05 * max(y_max - y_min, 1e-9)
    y_min -= pad
    y_max += pad
    left, top, width, height = 80.0, 70.0, 780.0, 320.0
    x_min, x_max = float(tt[0]), float(tt[-1])
    true_points = svg_polyline(tt, yy, x_min, x_max, y_min, y_max, left, top, width, height)
    pred_points = svg_polyline(tt, pp, x_min, x_max, y_min, y_max, left, top, width, height)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="940" height="470" viewBox="0 0 940 470">
  <rect width="940" height="470" fill="#ffffff"/>
  <text x="80" y="34" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  <text x="80" y="56" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#4b5563">{subtitle}</text>
  <rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#6b7280"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#6b7280"/>
  <polyline points="{true_points}" fill="none" stroke="#0f766e" stroke-width="2.2" opacity="0.95"/>
  <polyline points="{pred_points}" fill="none" stroke="#b91c1c" stroke-width="2.0" stroke-dasharray="8 5" opacity="0.95"/>
  <text x="{left + width / 2}" y="440" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{labels["time"]}</text>
  <text x="24" y="{top + height / 2}" transform="rotate(-90 24 {top + height / 2})" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{labels["output"]}</text>
  <line x1="670" y1="32" x2="710" y2="32" stroke="#0f766e" stroke-width="2.2"/>
  <text x="718" y="37" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#111827">{labels["measured"]}</text>
  <line x1="670" y1="52" x2="710" y2="52" stroke="#b91c1c" stroke-width="2.0" stroke-dasharray="8 5"/>
  <text x="718" y="57" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#111827">{labels["predicted"]}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

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

    u_offset = float(np.mean(u_fit))
    y_offset = float(np.mean(y_fit))
    y_scale = float(np.std(y_fit)) or 1.0
    x2_scale = y_scale / dt
    c0, k0, b0 = derivative_initial_guess(u_fit, y_fit, dt, u_offset, y_offset)
    b_scale = max(abs(b0), 1.0)

    candidate_rows = []
    candidates = []
    for start_name, start in make_starts(c0, k0, b0, b_scale):
        fit_result = fit_candidate(start_name, start, u_fit, y_fit, dt, u_offset, y_offset, y_scale, b_scale, x2_scale)
        val_row, _ = evaluate_record(
            "validation",
            "internal_validation",
            u_val,
            y_val,
            dt,
            float(fit_result["c"]),
            float(fit_result["k"]),
            float(fit_result["b"]),
            u_offset,
            y_offset,
            y_scale,
            x2_scale,
            int(CONFIG["state_initialization_window"]),
        )
        row = {
            **{key: fit_result[key] for key in ["start", "success", "status", "nfev", "cost", "c", "k", "b", "x20_fit", "fit_rmse_mv", "fit_nrmse"]},
            "validation_rmse_mv": val_row["rmse_mv"],
            "validation_nrmse": val_row["nrmse"],
        }
        candidate_rows.append(row)
        candidates.append((float(val_row["rmse_mv"]), fit_result))

    _, best = min(candidates, key=lambda item: item[0])
    c = float(best["c"])
    k = float(best["k"])
    b = float(best["b"])

    metrics_rows = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    eval_inputs = {
        "train_fit": ("internal_fit", u_fit, y_fit),
        "validation": ("internal_validation", u_val, y_val),
        "test_multisine": ("official_test", np.asarray(records["test_multisine"].u, dtype=float), np.asarray(records["test_multisine"].y, dtype=float)),
        "test_arrow_full": ("official_test", np.asarray(records["test_arrow_full"].u, dtype=float), np.asarray(records["test_arrow_full"].y, dtype=float)),
        "test_arrow_no_extrapolation": (
            "official_test",
            np.asarray(records["test_arrow_no_extrapolation"].u, dtype=float),
            np.asarray(records["test_arrow_no_extrapolation"].y, dtype=float),
        ),
    }

    for name, (split, u_record, y_record) in eval_inputs.items():
        row, prediction = evaluate_record(
            name,
            split,
            u_record,
            y_record,
            dt,
            c,
            k,
            b,
            u_offset,
            y_offset,
            y_scale,
            x2_scale,
            int(CONFIG["state_initialization_window"]),
        )
        metrics_rows.append(row)
        t = np.arange(len(y_record), dtype=float) * dt
        predictions[name] = (t, u_record, y_record, prediction)
        save_prediction_sample(RESULT_DIR / f"{name}_prediction_sample.csv", t, u_record, y_record, prediction)

    write_csv(RESULT_DIR / "candidate_starts.csv", candidate_rows)
    write_csv(RESULT_DIR / "metrics_by_record.csv", metrics_rows)

    for name in ["validation", "test_multisine", "test_arrow_full", "test_arrow_no_extrapolation"]:
        t, _, y_record, prediction = predictions[name]
        metric = next(row for row in metrics_rows if row["record"] == name)
        subtitle = f"RMSE={float(metric['rmse_mv']):.3f} mV, NRMSE={float(metric['nrmse']):.4f}, n_init={metric['n_init']}"
        write_rollout_svg(
            FIGURE_DIR / f"{name}_rollout.svg",
            t,
            y_record,
            prediction,
            f"Silverbox SB-1 Linear Output-Error Baseline: {name}",
            subtitle,
            {"time": "Time (s)", "output": "Output V2 (V)", "measured": "Measured output", "predicted": "Linear OE prediction"},
        )
        write_rollout_svg(
            FIGURE_DIR / f"{name}_rollout_zh.svg",
            t,
            y_record,
            prediction,
            f"Silverbox SB-1 线性输出误差基线：{name}",
            subtitle,
            {"time": "时间 (s)", "output": "输出 V2 (V)", "measured": "实测输出", "predicted": "线性 OE 预测"},
        )

    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "selected_start": best["start"],
        "model": {
            "equation": "x1_dot=x2; x2_dot=-c*x2-k*x1+b*(u-u_offset); y=x1+y_offset",
            "c": c,
            "k": k,
            "b": b,
            "u_offset": u_offset,
            "y_offset": y_offset,
            "y_scale": y_scale,
            "x2_scale": x2_scale,
        },
        "candidate_starts": candidate_rows,
        "metrics_by_record": metrics_rows,
    }
    metrics_path = RESULT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    provenance_path = PROVENANCE_DIR / "silverbox_sb1_linear_oe_provenance.json"
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
            "raw_audit_metrics": "results/silverbox_data_audit/metrics.json",
        },
        "outputs": {
            "candidate_starts_csv": "results/silverbox_sb1_linear_oe/candidate_starts.csv",
            "metrics_by_record_csv": "results/silverbox_sb1_linear_oe/metrics_by_record.csv",
            "metrics_json": "results/silverbox_sb1_linear_oe/metrics.json",
            "hashes_json": "results/silverbox_sb1_linear_oe/hashes.json",
            "figures": "results/silverbox_sb1_linear_oe/figures/",
        },
        "test_usage_rule": CONFIG["test_usage_rule"],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    hash_targets = [
        RESULT_DIR / "candidate_starts.csv",
        RESULT_DIR / "metrics_by_record.csv",
        metrics_path,
        provenance_path,
    ]
    hashes = {f"{path.relative_to(ROOT)}_sha256": sha256_file(path) for path in hash_targets}
    hashes_path = RESULT_DIR / "hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
