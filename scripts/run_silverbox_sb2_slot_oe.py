#!/usr/bin/env python3
"""Run SB-2: Duffing-like stiffness coefficient-slot output-error model for Silverbox."""

from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "external" / "silverbox"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "silverbox_sb2_slot_oe"
FIGURE_DIR = RESULT_DIR / "figures"
SB1_METRICS_PATH = ROOT / "results" / "silverbox_sb1_linear_oe" / "metrics.json"


CONFIG: dict[str, Any] = {
    "experiment_id": "SB2_silverbox_stiffness_slot_output_error",
    "description": "Pilot Duffing-like stiffness coefficient-slot output-error model for the Silverbox input-output benchmark.",
    "source_dataset": "silverbox_data_audit",
    "baseline_experiment": "SB1_silverbox_linear_output_error",
    "dataloader": "nonlinear_benchmarks.Silverbox",
    "train_fit_fraction": 0.8,
    "state_initialization_window": 50,
    "fit_residual_stride": 25,
    "fit_max_nfev": 35,
    "init_max_nfev": 60,
    "max_abs_state": 5.0,
    "candidate_beta2_scaled": [-1.0, 0.0, 1.0],
    "parameterization": {
        "c": "positive damping, c=exp(log_c)",
        "beta0": "positive linear stiffness component, beta0=exp(log_beta0)",
        "beta2": "signed cubic stiffness component, beta2=beta2_scale*beta2_scaled",
        "b": "signed input gain, b=b_scale*b_scaled",
        "x20": "signed initial latent velocity, x20=x2_scale*x20_scaled",
    },
    "model_equation": "x1_dot=x2; x2_dot=-c*x2-x1*(beta0+beta2*x1^2)+b*(u-u_offset); y=x1+y_offset",
    "test_usage_rule": "Official test records are used only after selecting the model by internal train_val validation.",
}


def load_sb1_model() -> dict[str, Any]:
    return json.loads(SB1_METRICS_PATH.read_text(encoding="utf-8"))["model"]


def load_sb1_record_metrics() -> dict[str, dict[str, Any]]:
    metrics = json.loads(SB1_METRICS_PATH.read_text(encoding="utf-8"))
    return {str(row["record"]): row for row in metrics["metrics_by_record"]}


def simulate_slot_oe(
    u: np.ndarray,
    y0: float,
    x20: float,
    dt: float,
    c: float,
    beta0: float,
    beta2: float,
    b: float,
    u_offset: float,
    y_offset: float,
    max_abs_state: float,
) -> tuple[np.ndarray, bool]:
    output = np.empty(len(u), dtype=float)
    x1 = float(y0 - y_offset)
    x2 = float(x20)
    u_centered = u - u_offset
    output[0] = x1 + y_offset

    def rhs(xx1: float, xx2: float, uu: float) -> tuple[float, float]:
        return xx2, -c * xx2 - beta0 * xx1 - beta2 * xx1**3 + b * uu

    stable = True
    for idx in range(len(u) - 1):
        ui = float(u_centered[idx])
        k1_x1, k1_x2 = rhs(x1, x2, ui)
        k2_x1, k2_x2 = rhs(x1 + 0.5 * dt * k1_x1, x2 + 0.5 * dt * k1_x2, ui)
        k3_x1, k3_x2 = rhs(x1 + 0.5 * dt * k2_x1, x2 + 0.5 * dt * k2_x2, ui)
        k4_x1, k4_x2 = rhs(x1 + dt * k3_x1, x2 + dt * k3_x2, ui)
        x1 += dt * (k1_x1 + 2.0 * k2_x1 + 2.0 * k3_x1 + k4_x1) / 6.0
        x2 += dt * (k1_x2 + 2.0 * k2_x2 + 2.0 * k3_x2 + k4_x2) / 6.0
        if not np.isfinite(x1) or not np.isfinite(x2) or abs(x1) > max_abs_state or abs(x2) > max_abs_state / dt:
            stable = False
            output[idx + 1 :] = np.nan
            break
        output[idx + 1] = x1 + y_offset
    return output, stable


def unpack_params(params: np.ndarray, beta2_scale: float, b_scale: float, x2_scale: float) -> tuple[float, float, float, float, float]:
    c = float(np.exp(params[0]))
    beta0 = float(np.exp(params[1]))
    beta2 = float(params[2] * beta2_scale)
    b = float(params[3] * b_scale)
    x20 = float(params[4] * x2_scale)
    return c, beta0, beta2, b, x20


def make_starts(sb1_model: dict[str, Any], beta2_scale: float, b_scale: float) -> list[tuple[str, np.ndarray]]:
    c0 = float(sb1_model["c"])
    beta00 = float(sb1_model["k"])
    b0 = float(sb1_model["b"])
    starts = []
    for beta2_scaled in CONFIG["candidate_beta2_scaled"]:
        starts.append(
            (
                f"sb1_beta2_scaled_{beta2_scaled:g}",
                np.array([np.log(c0), np.log(beta00), float(beta2_scaled), b0 / b_scale, 0.0], dtype=float),
            )
        )
    return starts


def fit_candidate(
    start_name: str,
    start: np.ndarray,
    u_fit: np.ndarray,
    y_fit: np.ndarray,
    dt: float,
    u_offset: float,
    y_offset: float,
    y_scale: float,
    beta2_scale: float,
    b_scale: float,
    x2_scale: float,
) -> dict[str, Any]:
    n_init = int(CONFIG["state_initialization_window"])
    stride = int(CONFIG["fit_residual_stride"])
    residual_length = len(y_fit[n_init::stride])

    def residual(params: np.ndarray) -> np.ndarray:
        c, beta0, beta2, b, x20 = unpack_params(params, beta2_scale, b_scale, x2_scale)
        prediction, stable = simulate_slot_oe(
            u_fit,
            y_fit[0],
            x20,
            dt,
            c,
            beta0,
            beta2,
            b,
            u_offset,
            y_offset,
            float(CONFIG["max_abs_state"]),
        )
        if not stable or not np.all(np.isfinite(prediction[n_init::stride])):
            return np.full(residual_length, 1e6, dtype=float)
        return (prediction[n_init::stride] - y_fit[n_init::stride]) / y_scale

    lower = np.array([np.log(1e-3), np.log(1e-3), -100.0, -100.0, -100.0], dtype=float)
    upper = np.array([np.log(1e5), np.log(1e8), 100.0, 100.0, 100.0], dtype=float)
    result = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        max_nfev=int(CONFIG["fit_max_nfev"]),
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
    )
    c, beta0, beta2, b, x20 = unpack_params(result.x, beta2_scale, b_scale, x2_scale)
    prediction, stable = simulate_slot_oe(
        u_fit,
        y_fit[0],
        x20,
        dt,
        c,
        beta0,
        beta2,
        b,
        u_offset,
        y_offset,
        float(CONFIG["max_abs_state"]),
    )
    fit_rmse = float("nan") if not stable else sb1.rmse_mv(y_fit, prediction, n_init)
    fit_nrmse = float("nan") if not stable else sb1.nrmse(y_fit, prediction, n_init)
    return {
        "start": start_name,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "stable_fit": bool(stable),
        "c": c,
        "beta0": beta0,
        "beta2": beta2,
        "b": b,
        "x20_fit": x20,
        "beta2_scaled": beta2 / beta2_scale,
        "params_scaled": [float(v) for v in result.x],
        "fit_rmse_mv": fit_rmse,
        "fit_nrmse": fit_nrmse,
    }


def estimate_initial_velocity(
    u: np.ndarray,
    y: np.ndarray,
    dt: float,
    c: float,
    beta0: float,
    beta2: float,
    b: float,
    u_offset: float,
    y_offset: float,
    y_scale: float,
    x2_scale: float,
    n_init: int,
) -> float:
    n_init = min(n_init, len(y))

    def residual(x2_scaled: np.ndarray) -> np.ndarray:
        prediction, stable = simulate_slot_oe(
            u[:n_init],
            y[0],
            float(x2_scaled[0] * x2_scale),
            dt,
            c,
            beta0,
            beta2,
            b,
            u_offset,
            y_offset,
            float(CONFIG["max_abs_state"]),
        )
        if not stable or not np.all(np.isfinite(prediction)):
            return np.full(n_init, 1e6, dtype=float)
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
    beta0: float,
    beta2: float,
    b: float,
    u_offset: float,
    y_offset: float,
    y_scale: float,
    x2_scale: float,
    n_init: int,
    sb1_metrics_by_record: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    x20 = estimate_initial_velocity(u, y, dt, c, beta0, beta2, b, u_offset, y_offset, y_scale, x2_scale, n_init)
    prediction, stable = simulate_slot_oe(
        u,
        y[0],
        x20,
        dt,
        c,
        beta0,
        beta2,
        b,
        u_offset,
        y_offset,
        float(CONFIG["max_abs_state"]),
    )
    rmse = float("nan") if not stable else sb1.rmse_mv(y, prediction, n_init)
    norm_rmse = float("nan") if not stable else sb1.nrmse(y, prediction, n_init)
    sb1_row = sb1_metrics_by_record.get(name)
    sb1_rmse = float(sb1_row["rmse_mv"]) if sb1_row else float("nan")
    improvement = 100.0 * (sb1_rmse - rmse) / sb1_rmse if np.isfinite(sb1_rmse) and np.isfinite(rmse) and sb1_rmse else float("nan")
    row = {
        "record": name,
        "split": split,
        "sample_count": int(len(y)),
        "sampling_time_s": dt,
        "n_init": int(n_init),
        "x20_initialized": x20,
        "stable": bool(stable),
        "rmse_mv": rmse,
        "nrmse": norm_rmse,
        "sb1_rmse_mv": sb1_rmse,
        "improvement_vs_sb1_percent": improvement,
        "y_std_after_init": float(np.std(y[n_init:])),
    }
    return row, prediction


def write_coefficient_svg(
    path: Path,
    beta0: float,
    beta2: float,
    y_min: float,
    y_max: float,
    title: str,
    labels: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(y_min, y_max, 401)
    coeff = beta0 + beta2 * grid**2
    left, top, width, height = 80.0, 70.0, 780.0, 320.0
    y_pad = 0.05 * max(float(np.max(coeff) - np.min(coeff)), 1e-9)
    c_min = float(np.min(coeff) - y_pad)
    c_max = float(np.max(coeff) + y_pad)
    points = sb1.svg_polyline(grid, coeff, float(y_min), float(y_max), c_min, c_max, left, top, width, height)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="940" height="470" viewBox="0 0 940 470">
  <rect width="940" height="470" fill="#ffffff"/>
  <text x="80" y="36" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  <text x="80" y="58" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#4b5563">p_k(y)=beta0+beta2*y^2, beta0={beta0:.4g}, beta2={beta2:.4g}</text>
  <rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#6b7280"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#6b7280"/>
  <polyline points="{points}" fill="none" stroke="#7c3aed" stroke-width="2.4"/>
  <text x="{left + width / 2}" y="440" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{labels["x"]}</text>
  <text x="24" y="{top + height / 2}" transform="rotate(-90 24 {top + height / 2})" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{labels["y"]}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    sb1_model = load_sb1_model()
    sb1_metrics_by_record = load_sb1_record_metrics()
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

    u_offset = float(sb1_model["u_offset"])
    y_offset = float(sb1_model["y_offset"])
    y_scale = float(sb1_model["y_scale"])
    x2_scale = float(sb1_model["x2_scale"])
    b_scale = max(abs(float(sb1_model["b"])), 1.0)
    beta2_scale = max(float(sb1_model["k"]) / max(y_scale * y_scale, 1e-12), 1.0)

    candidate_rows = []
    candidates = []
    for start_name, start in make_starts(sb1_model, beta2_scale, b_scale):
        fit_result = fit_candidate(
            start_name,
            start,
            u_fit,
            y_fit,
            dt,
            u_offset,
            y_offset,
            y_scale,
            beta2_scale,
            b_scale,
            x2_scale,
        )
        val_row, _ = evaluate_record(
            "validation",
            "internal_validation",
            u_val,
            y_val,
            dt,
            float(fit_result["c"]),
            float(fit_result["beta0"]),
            float(fit_result["beta2"]),
            float(fit_result["b"]),
            u_offset,
            y_offset,
            y_scale,
            x2_scale,
            int(CONFIG["state_initialization_window"]),
            sb1_metrics_by_record,
        )
        row = {
            **{
                key: fit_result[key]
                for key in [
                    "start",
                    "success",
                    "status",
                    "nfev",
                    "cost",
                    "stable_fit",
                    "c",
                    "beta0",
                    "beta2",
                    "beta2_scaled",
                    "b",
                    "x20_fit",
                    "fit_rmse_mv",
                    "fit_nrmse",
                ]
            },
            "validation_rmse_mv": val_row["rmse_mv"],
            "validation_nrmse": val_row["nrmse"],
            "validation_improvement_vs_sb1_percent": val_row["improvement_vs_sb1_percent"],
        }
        candidate_rows.append(row)
        candidates.append((float(val_row["rmse_mv"]), fit_result))

    _, best = min(candidates, key=lambda item: item[0])
    c = float(best["c"])
    beta0 = float(best["beta0"])
    beta2 = float(best["beta2"])
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
            beta0,
            beta2,
            b,
            u_offset,
            y_offset,
            y_scale,
            x2_scale,
            int(CONFIG["state_initialization_window"]),
            sb1_metrics_by_record,
        )
        metrics_rows.append(row)
        t = np.arange(len(y_record), dtype=float) * dt
        predictions[name] = (t, u_record, y_record, prediction)
        sb1.save_prediction_sample(RESULT_DIR / f"{name}_prediction_sample.csv", t, u_record, y_record, prediction)

    sb1.write_csv(RESULT_DIR / "candidate_starts.csv", candidate_rows)
    sb1.write_csv(RESULT_DIR / "metrics_by_record.csv", metrics_rows)

    for name in ["validation", "test_multisine", "test_arrow_full", "test_arrow_no_extrapolation"]:
        t, _, y_record, prediction = predictions[name]
        metric = next(row for row in metrics_rows if row["record"] == name)
        subtitle = (
            f"RMSE={float(metric['rmse_mv']):.3f} mV, "
            f"improvement vs SB-1={float(metric['improvement_vs_sb1_percent']):.2f}%, n_init={metric['n_init']}"
        )
        sb1.write_rollout_svg(
            FIGURE_DIR / f"{name}_rollout.svg",
            t,
            y_record,
            prediction,
            f"Silverbox SB-2 Stiffness-Slot Output-Error Model: {name}",
            subtitle,
            {"time": "Time (s)", "output": "Output V2 (V)", "measured": "Measured output", "predicted": "Slot OE prediction"},
        )
        sb1.write_rollout_svg(
            FIGURE_DIR / f"{name}_rollout_zh.svg",
            t,
            y_record,
            prediction,
            f"Silverbox SB-2 刚度系数槽输出误差模型：{name}",
            subtitle,
            {"time": "时间 (s)", "output": "输出 V2 (V)", "measured": "实测输出", "predicted": "系数槽 OE 预测"},
        )

    y_min = float(np.min(y_all - y_offset))
    y_max = float(np.max(y_all - y_offset))
    write_coefficient_svg(
        FIGURE_DIR / "stiffness_coefficient.svg",
        beta0,
        beta2,
        y_min,
        y_max,
        "Silverbox SB-2 Recovered Stiffness Coefficient",
        {"x": "Centered output y-y0 (V)", "y": "p_k(y)"},
    )
    write_coefficient_svg(
        FIGURE_DIR / "stiffness_coefficient_zh.svg",
        beta0,
        beta2,
        y_min,
        y_max,
        "Silverbox SB-2 恢复的刚度系数函数",
        {"x": "中心化输出 y-y0 (V)", "y": "p_k(y)"},
    )

    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "selected_start": best["start"],
        "model": {
            "equation": CONFIG["model_equation"],
            "c": c,
            "beta0": beta0,
            "beta2": beta2,
            "b": b,
            "u_offset": u_offset,
            "y_offset": y_offset,
            "y_scale": y_scale,
            "x2_scale": x2_scale,
            "beta2_scale": beta2_scale,
        },
        "candidate_starts": candidate_rows,
        "metrics_by_record": metrics_rows,
    }
    metrics_path = RESULT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    provenance_path = PROVENANCE_DIR / "silverbox_sb2_slot_oe_provenance.json"
    provenance = {
        **CONFIG,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "nonlinear-benchmarks": sb1.package_version("nonlinear-benchmarks"),
            "numpy": sb1.package_version("numpy"),
            "scipy": scipy.__version__,
        },
        "source_files": {
            "raw_audit_provenance": "data/provenance/silverbox_data_audit_provenance.json",
            "sb1_metrics": "results/silverbox_sb1_linear_oe/metrics.json",
        },
        "outputs": {
            "candidate_starts_csv": "results/silverbox_sb2_slot_oe/candidate_starts.csv",
            "metrics_by_record_csv": "results/silverbox_sb2_slot_oe/metrics_by_record.csv",
            "metrics_json": "results/silverbox_sb2_slot_oe/metrics.json",
            "hashes_json": "results/silverbox_sb2_slot_oe/hashes.json",
            "figures": "results/silverbox_sb2_slot_oe/figures/",
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
    hashes = {f"{path.relative_to(ROOT)}_sha256": sb1.sha256_file(path) for path in hash_targets}
    hashes_path = RESULT_DIR / "hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
