#!/usr/bin/env python3
"""Run E4: local linear drag to velocity-adaptive damping."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pysindy as ps
import scipy


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "generated" / "e4_velocity_drag"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "e4_velocity_drag"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG = {
    "experiment_id": "E4_velocity_dependent_drag",
    "description": "Velocity-adaptive damping from a locally linear drag model.",
    "source": {
        "name": "Synthetic low-speed linear drag to quadratic aerodynamic drag benchmark",
        "urls": ["synthetic truth generated in script"],
        "access_date": "2026-06-13",
    },
    "c0": 0.08,
    "c1": 0.55,
    "dt": 0.02,
    "t_end": 24.0,
    "stlsq_threshold": 1e-8,
    "stlsq_max_iter": 30,
    "slot_terms": ["1", "abs(v)"],
    "gain_abs_v_edges": [0.0, 0.18, 0.38, 0.7, 1.1],
    "train_low_inputs": [
        {"amplitude": 0.04, "phase": 0.0, "v0": 0.00},
        {"amplitude": 0.07, "phase": 0.9, "v0": 0.03},
        {"amplitude": 0.10, "phase": 1.7, "v0": -0.04},
    ],
    "train_medium_inputs": [
        {"amplitude": 0.18, "phase": 0.2, "v0": 0.00},
        {"amplitude": 0.28, "phase": 1.1, "v0": 0.08},
        {"amplitude": 0.42, "phase": 2.0, "v0": -0.08},
        {"amplitude": 0.55, "phase": 2.8, "v0": 0.05},
    ],
    "test_interp_inputs": [
        {"amplitude": 0.35, "phase": 0.55, "v0": 0.12},
        {"amplitude": 0.50, "phase": 1.55, "v0": -0.10},
        {"amplitude": 0.62, "phase": 2.55, "v0": 0.00},
    ],
    "test_extrap_inputs": [
        {"amplitude": 0.90, "phase": 0.35, "v0": 0.00},
        {"amplitude": 1.10, "phase": 1.35, "v0": -0.18},
        {"amplitude": 1.30, "phase": 2.35, "v0": 0.18},
    ],
    "noise": None,
}


def input_signal(t: float, amplitude: float, phase: float) -> float:
    return float(
        amplitude
        * (
            math.sin(0.72 * t + phase)
            + 0.35 * math.sin(1.65 * t + 0.4 * phase)
            + 0.18 * math.sin(0.19 * t + 1.3 * phase)
        )
    )


def true_drag_p(v: np.ndarray | float, c0: float, c1: float) -> np.ndarray | float:
    return c0 + c1 * np.abs(v)


def true_rhs(t: float, x: np.ndarray, amplitude: float, phase: float, c0: float, c1: float) -> np.ndarray:
    v = float(x[0])
    u = input_signal(t, amplitude, phase)
    return np.array([u - float(true_drag_p(v, c0, c1)) * v], dtype=float)


def b0_rhs(t: float, x: np.ndarray, amplitude: float, phase: float, c: float) -> np.ndarray:
    v = float(x[0])
    return np.array([input_signal(t, amplitude, phase) - c * v], dtype=float)


def lookup_gain(value: float, edges: np.ndarray, gains: np.ndarray) -> float:
    idx = int(np.searchsorted(edges[1:], abs(value), side="left"))
    idx = min(max(idx, 0), len(gains) - 1)
    return float(gains[idx])


def b0_gain_rhs(
    t: float,
    x: np.ndarray,
    amplitude: float,
    phase: float,
    edges: np.ndarray,
    gains: np.ndarray,
) -> np.ndarray:
    v = float(x[0])
    return np.array([input_signal(t, amplitude, phase) - lookup_gain(v, edges, gains) * v], dtype=float)


def full_library(v: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, list[str]]:
    abs_v = np.abs(v)
    features = [
        np.ones_like(v),
        v,
        abs_v,
        v * abs_v,
        u,
        v * u,
        abs_v * u,
        v**2,
        v**3,
    ]
    names = ["1", "v", "abs(v)", "v*abs(v)", "u", "v*u", "abs(v)*u", "v^2", "v^3"]
    return np.column_stack(features), names


def slot_library(v: np.ndarray) -> np.ndarray:
    return np.column_stack([-v, -v * np.abs(v)])


def slot_p(v: np.ndarray | float, coeffs: np.ndarray) -> np.ndarray | float:
    return coeffs[0] + coeffs[1] * np.abs(v)


def b1_rhs(t: float, x: np.ndarray, amplitude: float, phase: float, coeffs: np.ndarray) -> np.ndarray:
    v = np.array([float(x[0])])
    u = np.array([input_signal(t, amplitude, phase)])
    theta, _ = full_library(v, u)
    return np.array([float((theta @ coeffs).reshape(-1)[0])], dtype=float)


def b2_rhs(
    t: float,
    x: np.ndarray,
    amplitude: float,
    phase: float,
    c: float,
    coeffs: np.ndarray,
) -> np.ndarray:
    return b0_rhs(t, x, amplitude, phase, c) + b1_rhs(t, x, amplitude, phase, coeffs)


def b3_rhs(t: float, x: np.ndarray, amplitude: float, phase: float, coeffs: np.ndarray) -> np.ndarray:
    v = float(x[0])
    return np.array([input_signal(t, amplitude, phase) - float(slot_p(v, coeffs)) * v], dtype=float)


def simulate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    t_end: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(0.0, t_end + 0.5 * dt, dt)
    x = np.zeros((len(t_eval), len(x0)), dtype=float)
    x[0] = x0
    max_abs_state = 25.0
    for idx in range(len(t_eval) - 1):
        tt = float(t_eval[idx])
        yy = x[idx]
        k1 = rhs(tt, yy)
        k2 = rhs(tt + 0.5 * dt, yy + 0.5 * dt * k1)
        k3 = rhs(tt + 0.5 * dt, yy + 0.5 * dt * k2)
        k4 = rhs(tt + dt, yy + dt * k3)
        next_y = yy + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(next_y)) or np.any(np.abs(next_y) > max_abs_state):
            clipped = np.nan_to_num(next_y, nan=max_abs_state, posinf=max_abs_state, neginf=-max_abs_state)
            clipped = np.clip(clipped, -max_abs_state, max_abs_state)
            x[idx + 1 :] = clipped
            break
        x[idx + 1] = next_y
    return t_eval, x


def pysindy_stlsq(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    optimizer = ps.STLSQ(
        threshold=CONFIG["stlsq_threshold"],
        alpha=0.0,
        max_iter=CONFIG["stlsq_max_iter"],
        normalize_columns=False,
    )
    optimizer.fit(a, y.reshape(-1, 1))
    return optimizer.coef_.reshape(-1)


def fit_local_c(v: np.ndarray, dv: np.ndarray, u: np.ndarray) -> float:
    target = dv - u
    return float(-np.dot(v, target) / np.dot(v, v))


def fit_gain_schedule(v: np.ndarray, dv: np.ndarray, u: np.ndarray, edges: list[float], fallback: float) -> np.ndarray:
    target = dv - u
    abs_v = np.abs(v)
    edge_array = np.array(edges, dtype=float)
    gains: list[float] = []
    for idx in range(len(edge_array) - 1):
        lo = edge_array[idx]
        hi = edge_array[idx + 1]
        mask = (abs_v >= lo) & (abs_v <= hi) if idx == 0 else (abs_v > lo) & (abs_v <= hi)
        denom = float(np.dot(v[mask], v[mask]))
        if np.any(mask) and denom > 1e-14:
            gains.append(float(-np.dot(v[mask], target[mask]) / denom))
        else:
            gains.append(fallback)
    return np.array(gains, dtype=float)


def nrmse(true: np.ndarray, pred: np.ndarray) -> float:
    err = pred.reshape(-1) - true.reshape(-1)
    return float(np.sqrt(np.mean(err * err)) / (np.std(true) or 1.0))


def coefficient_metrics(coeffs: np.ndarray) -> dict[str, object]:
    grid = np.linspace(-1.55, 1.55, 801)
    p_true = true_drag_p(grid, CONFIG["c0"], CONFIG["c1"])
    p_pred = slot_p(grid, coeffs)
    err = p_pred - p_true
    active = {name for name, coeff in zip(CONFIG["slot_terms"], coeffs) if abs(coeff) >= 1e-6}
    expected = set(CONFIG["slot_terms"])
    true_positive = len(active & expected)
    return {
        "grid_rmse": float(np.sqrt(np.mean(err * err))),
        "grid_nrmse": float(np.sqrt(np.mean(err * err)) / (np.std(p_true) or 1.0)),
        "max_abs_error": float(np.max(np.abs(err))),
        "grid_min_pred": float(np.min(p_pred)),
        "grid_max_pred": float(np.max(p_pred)),
        "active_terms": sorted(active),
        "expected_terms": sorted(expected),
        "support_precision": true_positive / len(active) if active else 0.0,
        "support_recall": true_positive / len(expected),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def polyline(points: np.ndarray) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_svg_series(path: Path, title: str, x_values: np.ndarray, series: list[tuple], x_label: str, y_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1220, 450
    left, right, top, bottom = 76, 34, 42, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_all = np.concatenate([item[1] for item in series])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    pad = 0.08 * (y_max - y_min if y_max > y_min else 1.0)
    y_min -= pad
    y_max += pad

    def sx(x: np.ndarray) -> np.ndarray:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: np.ndarray) -> np.ndarray:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="26" text-anchor="middle" font-family="Arial, Noto Sans CJK SC, Microsoft YaHei, sans-serif" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<text x="{left + plot_w / 2:.0f}" y="{height - 16}" text-anchor="middle" font-family="Arial, Noto Sans CJK SC, Microsoft YaHei, sans-serif" font-size="13">{x_label}</text>',
        f'<text transform="translate(18,{top + plot_h / 2:.0f}) rotate(-90)" text-anchor="middle" font-family="Arial, Noto Sans CJK SC, Microsoft YaHei, sans-serif" font-size="13">{y_label}</text>',
    ]
    for tick in np.linspace(x_min, x_max, 6):
        x = sx(np.array([tick]))[0]
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#333"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 21}" text-anchor="middle" font-family="Arial" font-size="11">{tick:.1f}</text>')
    for tick in np.linspace(y_min, y_max, 5):
        y = sy(np.array([tick]))[0]
        lines.append(f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#333"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{tick:.2f}</text>')
    step = max(1, int(math.ceil(len(x_values) / 1200)))
    x_plot = sx(x_values[::step])
    for item in series:
        label, values, color = item[:3]
        dash = item[3] if len(item) > 3 else ""
        width_s = item[4] if len(item) > 4 else 2.0
        opacity = item[5] if len(item) > 5 else 1.0
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        pts = np.column_stack([x_plot, sy(values[::step])])
        lines.append(
            f'<polyline points="{polyline(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="{width_s}" stroke-opacity="{opacity}"{dash_attr}/>'
        )
    legend_x = left + 12
    legend_y = top + 12
    for idx, item in enumerate(series):
        label, _, color = item[:3]
        dash = item[3] if len(item) > 3 else ""
        width_s = item[4] if len(item) > 4 else 2.0
        opacity = item[5] if len(item) > 5 else 1.0
        y = legend_y + idx * 18
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" '
            f'stroke="{color}" stroke-width="{width_s}" stroke-opacity="{opacity}"{dash_attr}/>'
        )
        lines.append(
            f'<text x="{legend_x + 28}" y="{y + 4}" font-family="Arial, Noto Sans CJK SC, Microsoft YaHei, sans-serif" font-size="12">{label}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_trajectory(spec: dict[str, float], split: str, trajectory_id: str) -> dict[str, object]:
    amp = float(spec["amplitude"])
    phase = float(spec["phase"])
    v0 = float(spec["v0"])
    t, x = simulate(
        lambda tt, xx: true_rhs(tt, xx, amp, phase, CONFIG["c0"], CONFIG["c1"]),
        np.array([v0], dtype=float),
        CONFIG["t_end"],
        CONFIG["dt"],
    )
    u = np.array([input_signal(tt, amp, phase) for tt in t])
    dx = np.array([true_rhs(tt, xx, amp, phase, CONFIG["c0"], CONFIG["c1"])[0] for tt, xx in zip(t, x)])
    return {
        "trajectory_id": trajectory_id,
        "split": split,
        "amplitude": amp,
        "phase": phase,
        "v0": v0,
        "t": t,
        "x": x,
        "u": u,
        "dx": dx,
    }


def stack_data(trajectories: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.concatenate([traj["x"][:, 0] for traj in trajectories])
    dx = np.concatenate([traj["dx"] for traj in trajectories])
    u = np.concatenate([traj["u"] for traj in trajectories])
    return v, dx, u


def build_trajectories() -> list[dict[str, object]]:
    trajectories: list[dict[str, object]] = []
    for split, key in [
        ("train_low", "train_low_inputs"),
        ("train_medium", "train_medium_inputs"),
        ("test_interp", "test_interp_inputs"),
        ("test_extrap", "test_extrap_inputs"),
    ]:
        for idx, spec in enumerate(CONFIG[key]):
            trajectories.append(make_trajectory(spec, split, f"{split}_{idx}"))
    return trajectories


def model_rollout(
    model_name: str,
    traj: dict[str, object],
    c_local: float,
    gain_edges: np.ndarray,
    gain_values: np.ndarray,
    b1_coeffs: np.ndarray,
    b2_coeffs: np.ndarray,
    b3_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    amp = float(traj["amplitude"])
    phase = float(traj["phase"])
    x0 = np.array([float(traj["v0"])], dtype=float)
    if model_name == "B0_local_linear_drag":
        return simulate(lambda tt, xx: b0_rhs(tt, xx, amp, phase, c_local), x0, CONFIG["t_end"], CONFIG["dt"])
    if model_name == "B0_gain_scheduled_lookup":
        return simulate(lambda tt, xx: b0_gain_rhs(tt, xx, amp, phase, gain_edges, gain_values), x0, CONFIG["t_end"], CONFIG["dt"])
    if model_name == "B1_full_sindy":
        return simulate(lambda tt, xx: b1_rhs(tt, xx, amp, phase, b1_coeffs), x0, CONFIG["t_end"], CONFIG["dt"])
    if model_name == "B2_free_residual_sindy":
        return simulate(lambda tt, xx: b2_rhs(tt, xx, amp, phase, c_local, b2_coeffs), x0, CONFIG["t_end"], CONFIG["dt"])
    if model_name == "B3_slot_constrained":
        return simulate(lambda tt, xx: b3_rhs(tt, xx, amp, phase, b3_coeffs), x0, CONFIG["t_end"], CONFIG["dt"])
    if model_name == "B4_oracle_reference":
        return simulate(
            lambda tt, xx: true_rhs(tt, xx, amp, phase, CONFIG["c0"], CONFIG["c1"]),
            x0,
            CONFIG["t_end"],
            CONFIG["dt"],
        )
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    for directory in (DATA_DIR, PROVENANCE_DIR, RESULT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    trajectories = build_trajectories()
    train_low = [traj for traj in trajectories if traj["split"] == "train_low"]
    train_fit = [traj for traj in trajectories if traj["split"] in {"train_low", "train_medium"}]
    v_low, dx_low, u_low = stack_data(train_low)
    v_train, dx_train, u_train = stack_data(train_fit)

    c_local = fit_local_c(v_low, dx_low, u_low)
    gain_edges = np.array(CONFIG["gain_abs_v_edges"], dtype=float)
    gain_values = fit_gain_schedule(v_train, dx_train, u_train, CONFIG["gain_abs_v_edges"], c_local)

    theta, term_names = full_library(v_train, u_train)
    b1_coeffs = pysindy_stlsq(theta, dx_train)
    b0_train = u_train - c_local * v_train
    b2_coeffs = pysindy_stlsq(theta, dx_train - b0_train)
    b3_coeffs = pysindy_stlsq(slot_library(v_train), dx_train - u_train)

    trajectory_rows = []
    for traj in trajectories:
        for idx, tt in enumerate(traj["t"]):
            trajectory_rows.append(
                {
                    "trajectory_id": traj["trajectory_id"],
                    "split": traj["split"],
                    "amplitude": f"{traj['amplitude']:.16g}",
                    "phase": f"{traj['phase']:.16g}",
                    "v0": f"{traj['v0']:.16g}",
                    "t": f"{tt:.10g}",
                    "u": f"{traj['u'][idx]:.16g}",
                    "v": f"{traj['x'][idx, 0]:.16g}",
                    "dv": f"{traj['dx'][idx]:.16g}",
                }
            )
    trajectory_path = DATA_DIR / "e4_velocity_drag_trajectories.csv"
    write_csv(trajectory_path, trajectory_rows)

    models = [
        "B0_local_linear_drag",
        "B0_gain_scheduled_lookup",
        "B1_full_sindy",
        "B2_free_residual_sindy",
        "B3_slot_constrained",
        "B4_oracle_reference",
    ]
    rollout_summary_rows = []
    rollout_sample_rows = []
    split_values: dict[str, dict[str, list[float]]] = {}
    sample_id = "test_extrap_1"
    for traj in trajectories:
        split = str(traj["split"])
        split_values.setdefault(split, {model: [] for model in models})
        true_t = traj["t"]
        true_x = traj["x"]
        for model in models:
            pred_t, pred_x = model_rollout(model, traj, c_local, gain_edges, gain_values, b1_coeffs, b2_coeffs, b3_coeffs)
            if len(pred_t) != len(true_t):
                pred = np.interp(true_t, pred_t, pred_x[:, 0]).reshape(-1, 1)
            else:
                pred = pred_x
            metric = nrmse(true_x, pred)
            split_values[split][model].append(metric)
            rollout_summary_rows.append(
                {
                    "trajectory_id": traj["trajectory_id"],
                    "split": split,
                    "model": model,
                    "nrmse_all": f"{metric:.16g}",
                }
            )
            if traj["trajectory_id"] == sample_id:
                for idx, tt in enumerate(true_t):
                    rollout_sample_rows.append(
                        {
                            "trajectory_id": traj["trajectory_id"],
                            "split": split,
                            "model": model,
                            "t": f"{tt:.10g}",
                            "u": f"{traj['u'][idx]:.16g}",
                            "true_v": f"{true_x[idx, 0]:.16g}",
                            "pred_v": f"{pred[idx, 0]:.16g}",
                        }
                    )

    split_metrics = {
        split: {
            model: {
                "mean_nrmse_all": float(np.mean(values)),
                "max_nrmse_all": float(np.max(values)),
            }
            for model, values in model_values.items()
        }
        for split, model_values in split_values.items()
    }

    rollout_summary_path = RESULT_DIR / "rollout_summary.csv"
    rollout_samples_path = RESULT_DIR / "rollout_samples.csv"
    write_csv(rollout_summary_path, rollout_summary_rows)
    write_csv(rollout_samples_path, rollout_sample_rows)

    coeff_rows = [
        {
            "term": term,
            "b3_coefficient": f"{coeff:.16g}",
            "reference_coefficient": f"{ref:.16g}",
            "abs_error": f"{abs(coeff - ref):.16g}",
            "active": str(abs(coeff) >= 1e-6),
        }
        for term, coeff, ref in zip(CONFIG["slot_terms"], b3_coeffs, [CONFIG["c0"], CONFIG["c1"]])
    ]
    coefficients_path = RESULT_DIR / "coefficients.csv"
    write_csv(coefficients_path, coeff_rows)

    gain_path = RESULT_DIR / "gain_schedule.csv"
    write_csv(
        gain_path,
        [
            {
                "bin": idx,
                "abs_v_low": f"{gain_edges[idx]:.16g}",
                "abs_v_high": f"{gain_edges[idx + 1]:.16g}",
                "gain_value": f"{value:.16g}",
            }
            for idx, value in enumerate(gain_values)
        ],
    )

    model_coeff_rows = []
    for term, coeff in zip(term_names, b1_coeffs):
        model_coeff_rows.append({"model": "B1_full_sindy", "term": term, "dv_coefficient": f"{coeff:.16g}", "active": str(abs(coeff) >= 1e-6)})
    for term, coeff in zip(term_names, b2_coeffs):
        model_coeff_rows.append({"model": "B2_free_residual_sindy", "term": term, "dv_coefficient": f"{coeff:.16g}", "active": str(abs(coeff) >= 1e-6)})
    for term, coeff in zip(CONFIG["slot_terms"], b3_coeffs):
        model_coeff_rows.append({"model": "B3_slot_constrained_pv", "term": term, "dv_coefficient": f"{coeff:.16g}", "active": str(abs(coeff) >= 1e-6)})
    model_coefficients_path = RESULT_DIR / "model_coefficients.csv"
    write_csv(model_coefficients_path, model_coeff_rows)

    coeff_metrics = coefficient_metrics(b3_coeffs)
    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "b0_local_linear_c": c_local,
        "true_coefficients": {"1": CONFIG["c0"], "abs(v)": CONFIG["c1"]},
        "b0_gain_schedule": {"abs_v_edges": list(CONFIG["gain_abs_v_edges"]), "values": [float(v) for v in gain_values]},
        "b3_optimizer": "PySINDy STLSQ on coefficient-slot design matrix",
        "b3_coefficients": {term: float(coeff) for term, coeff in zip(CONFIG["slot_terms"], b3_coeffs)},
        "active_term_counts": {
            "B0_gain_scheduled_lookup": len(gain_values),
            "B1_full_sindy": int(np.sum(np.abs(b1_coeffs) >= 1e-6)),
            "B2_free_residual_sindy": int(np.sum(np.abs(b2_coeffs) >= 1e-6)),
            "B3_slot_constrained": int(np.sum(np.abs(b3_coeffs) >= 1e-6)),
        },
        "rollout_nrmse_by_split": split_metrics,
        "coefficient_function": coeff_metrics,
    }
    metrics_path = RESULT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sample_rows = [row for row in rollout_sample_rows if row["trajectory_id"] == sample_id]
    sample_t = np.array([float(row["t"]) for row in sample_rows if row["model"] == "B4_oracle_reference"])
    true_v = np.array([float(row["true_v"]) for row in sample_rows if row["model"] == "B4_oracle_reference"])
    styles = {
        "B0_local_linear_drag": ("Local linear drag model", "局部线性阻尼模型", "#E69F00", "", 2.4, 0.95),
        "B0_gain_scheduled_lookup": ("Piecewise gain-scheduled damping model", "分段查表阻尼模型", "#56B4E9", "5 4", 2.4, 0.95),
        "B1_full_sindy": ("Full SINDy vector-field model", "完整向量场 SINDy 模型", "#0072B2", "2 4", 2.3, 0.95),
        "B2_free_residual_sindy": ("Free residual SINDy model", "自由残差 SINDy 模型", "#CC79A7", "10 3 2 3", 2.3, 0.95),
        "B3_slot_constrained": ("Slot-constrained coefficient SINDy model", "系数槽约束 SINDy 模型", "#009E73", "12 5", 2.8, 1.0),
    }
    series = [("Reference velocity-dependent drag", true_v, "#4D4D4D", "", 3.2, 0.55)]
    series_zh = [("速度相关阻力参考轨迹", true_v, "#4D4D4D", "", 3.2, 0.55)]
    for model, (label_en, label_zh, color, dash, width, opacity) in styles.items():
        values = np.array([float(row["pred_v"]) for row in sample_rows if row["model"] == model])
        series.append((label_en, values, color, dash, width, opacity))
        series_zh.append((label_zh, values, color, dash, width, opacity))
    write_svg_series(FIGURE_DIR / "rollout_velocity_extrap.svg", "E4 high-amplitude rollout: velocity", sample_t, series, "t", "v")
    write_svg_series(FIGURE_DIR / "rollout_velocity_extrap_zh.svg", "E4 高输入幅值 rollout：速度 v", sample_t, series_zh, "时间 t", "速度 v")

    grid = np.linspace(-1.55, 1.55, 801)
    p_true = true_drag_p(grid, CONFIG["c0"], CONFIG["c1"])
    p_pred = slot_p(grid, b3_coeffs)
    p_gain = np.array([lookup_gain(value, gain_edges, gain_values) for value in grid])
    write_svg_series(
        FIGURE_DIR / "coefficient_function.svg",
        "E4 recovered velocity-adaptive damping p(v)",
        grid,
        [
            ("Reference p(v)=c0+c1*abs(v)", p_true, "#4D4D4D", "", 3.2, 0.55),
            ("Piecewise gain-scheduled damping model", p_gain, "#56B4E9", "5 4", 2.4, 0.95),
            ("Slot-constrained coefficient SINDy model", p_pred, "#009E73", "12 5", 2.8, 1.0),
        ],
        "v",
        "p(v)",
    )
    write_svg_series(
        FIGURE_DIR / "coefficient_function_zh.svg",
        "E4 恢复的速度自适应阻尼 p(v)",
        grid,
        [
            ("参考 p(v)=c0+c1*abs(v)", p_true, "#4D4D4D", "", 3.2, 0.55),
            ("分段查表阻尼模型", p_gain, "#56B4E9", "5 4", 2.4, 0.95),
            ("系数槽约束 SINDy 模型", p_pred, "#009E73", "12 5", 2.8, 1.0),
        ],
        "速度 v",
        "阻尼系数 p(v)",
    )

    provenance = {
        "dataset_id": "e4_velocity_drag_trajectories",
        "experiment_id": CONFIG["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": CONFIG["source"]["name"],
        "source_urls": CONFIG["source"]["urls"],
        "source_access_date": CONFIG["source"]["access_date"],
        "equation_reference": "First-order forced velocity dynamics with linear-plus-quadratic drag.",
        "parameters": {"c0": CONFIG["c0"], "c1": CONFIG["c1"]},
        "input_specs": {
            "train_low": CONFIG["train_low_inputs"],
            "train_medium": CONFIG["train_medium_inputs"],
            "test_interp": CONFIG["test_interp_inputs"],
            "test_extrap": CONFIG["test_extrap_inputs"],
        },
        "time_grid": {"t_end": CONFIG["t_end"], "dt": CONFIG["dt"]},
        "noise_model": CONFIG["noise"],
        "candidate_slots": [{"name": "drag", "g": "-v", "coefficient_function": "p(v)=c0+c1*abs(v)"}],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pysindy": ps.__version__,
        },
        "script": str(Path(__file__).relative_to(ROOT)),
        "outputs": {
            "trajectory_csv": str(trajectory_path.relative_to(ROOT)),
            "rollout_summary_csv": str(rollout_summary_path.relative_to(ROOT)),
            "rollout_samples_csv": str(rollout_samples_path.relative_to(ROOT)),
            "coefficients_csv": str(coefficients_path.relative_to(ROOT)),
            "gain_schedule_csv": str(gain_path.relative_to(ROOT)),
            "model_coefficients_csv": str(model_coefficients_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
        },
    }
    provenance_path = PROVENANCE_DIR / "e4_velocity_drag_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    hashes = {
        "trajectory_csv_sha256": sha256_file(trajectory_path),
        "rollout_summary_csv_sha256": sha256_file(rollout_summary_path),
        "rollout_samples_csv_sha256": sha256_file(rollout_samples_path),
        "coefficients_csv_sha256": sha256_file(coefficients_path),
        "gain_schedule_csv_sha256": sha256_file(gain_path),
        "model_coefficients_csv_sha256": sha256_file(model_coefficients_path),
        "metrics_json_sha256": sha256_file(metrics_path),
        "provenance_json_sha256": sha256_file(provenance_path),
    }
    (RESULT_DIR / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")

    summary = {
        "experiment_id": CONFIG["experiment_id"],
        "b0_local_linear_c": c_local,
        "b3_coefficients": metrics["b3_coefficients"],
        "coefficient_grid_nrmse": metrics["coefficient_function"]["grid_nrmse"],
        "test_extrap_B0_mean_nrmse": split_metrics["test_extrap"]["B0_local_linear_drag"]["mean_nrmse_all"],
        "test_extrap_B0_gain_mean_nrmse": split_metrics["test_extrap"]["B0_gain_scheduled_lookup"]["mean_nrmse_all"],
        "test_extrap_B3_mean_nrmse": split_metrics["test_extrap"]["B3_slot_constrained"]["mean_nrmse_all"],
        "result_dir": str(RESULT_DIR.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
