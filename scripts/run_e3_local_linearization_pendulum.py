#!/usr/bin/env python3
"""Run E3: local-linearization pendulum coefficient-slot experiment."""

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
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "generated" / "e3_pendulum"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "e3_local_linearization_pendulum"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG = {
    "experiment_id": "E3_local_linearization_pendulum",
    "description": "Local-linearization pendulum state-adaptive stiffness experiment.",
    "source": {
        "name": "Classical nonlinear pendulum and small-angle local-linearization benchmark",
        "urls": ["synthetic truth generated in script"],
        "access_date": "2026-06-13",
    },
    "g_over_l": 1.0,
    "damping_c": 0.05,
    "dt": 0.01,
    "t_end": 20.0,
    "b3_degrees": [0, 2, 4, 6],
    "state_library_max_degree": 7,
    "stlsq_threshold": 1e-8,
    "stlsq_max_iter": 20,
    "gain_abs_theta_edges": [0.0, 0.35, 0.7, 1.05],
    "train_small_initial_conditions": [
        [0.05, 0.0],
        [-0.10, 0.05],
        [0.20, -0.10],
        [-0.30, 0.10],
        [0.35, 0.0],
        [-0.35, 0.0],
    ],
    "train_medium_initial_conditions": [
        [0.45, 0.0],
        [-0.45, 0.0],
        [0.70, 0.0],
        [-0.80, 0.20],
        [1.00, 0.0],
        [-1.00, 0.0],
        [0.60, -0.50],
        [-0.60, 0.50],
    ],
    "test_interp_initial_conditions": [
        [1.10, 0.0],
        [-1.10, 0.0],
        [0.90, 0.70],
        [-0.90, -0.70],
        [0.40, 1.00],
    ],
    "test_extrap_initial_conditions": [
        [1.50, 0.0],
        [-1.60, 0.0],
        [2.00, 0.0],
        [-2.20, 0.0],
        [2.40, -0.20],
        [-2.40, 0.20],
    ],
    "noise": None,
}


def pendulum_rhs(_: float, x: np.ndarray, g_over_l: float, c: float) -> np.ndarray:
    theta, omega = x
    return np.array([omega, -c * omega - g_over_l * math.sin(theta)], dtype=float)


def b0_rhs(_: float, x: np.ndarray, c: float, k0: float) -> np.ndarray:
    theta, omega = x
    return np.array([omega, -c * omega - k0 * theta], dtype=float)


def b0_gain_rhs(_: float, x: np.ndarray, c: float, edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    theta, omega = x
    p = lookup_gain(theta, edges, values)
    return np.array([omega, -c * omega - p * theta], dtype=float)


def b1_rhs(_: float, x: np.ndarray, coeffs: np.ndarray, terms: list[tuple[int, int]]) -> np.ndarray:
    theta = state_polynomial_library(x.reshape(1, 2), terms)[0]
    return theta @ coeffs


def b2_rhs(
    _: float,
    x: np.ndarray,
    c: float,
    k0: float,
    coeffs: np.ndarray,
    terms: list[tuple[int, int]],
) -> np.ndarray:
    theta = state_polynomial_library(x.reshape(1, 2), terms)[0]
    return b0_rhs(0.0, x, c, k0) + theta @ coeffs


def b3_rhs(_: float, x: np.ndarray, c: float, coeffs: np.ndarray, degrees: list[int]) -> np.ndarray:
    theta, omega = x
    p = sum(coeff * (theta**degree) for coeff, degree in zip(coeffs, degrees))
    return np.array([omega, -c * omega - p * theta], dtype=float)


def simulate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    t_end: float,
    dt: float,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(0.0, t_end + 0.5 * dt, dt)
    sol = solve_ivp(rhs, (0.0, t_end), x0, t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return sol.t, sol.y.T


def state_polynomial_terms(max_degree: int) -> list[tuple[int, int]]:
    terms: list[tuple[int, int]] = []
    for total_degree in range(max_degree + 1):
        for theta_power in range(total_degree, -1, -1):
            omega_power = total_degree - theta_power
            terms.append((theta_power, omega_power))
    return terms


def power_name(base: str, power: int) -> str | None:
    if power == 0:
        return None
    if power == 1:
        return base
    return f"{base}^{power}"


def state_term_name(term: tuple[int, int]) -> str:
    parts = [name for name in (power_name("theta", term[0]), power_name("omega", term[1])) if name]
    return "*".join(parts) if parts else "1"


def state_polynomial_library(x: np.ndarray, terms: list[tuple[int, int]]) -> np.ndarray:
    theta = x[:, 0]
    omega = x[:, 1]
    return np.column_stack([(theta**theta_power) * (omega**omega_power) for theta_power, omega_power in terms])


def b3_library(theta: np.ndarray, degrees: list[int]) -> np.ndarray:
    return np.column_stack([theta**degree for degree in degrees])


def stlsq(a: np.ndarray, y: np.ndarray, threshold: float, max_iter: int) -> np.ndarray:
    active = np.ones(a.shape[1], dtype=bool)
    coeffs = np.zeros(a.shape[1], dtype=float)
    for _ in range(max_iter):
        if not np.any(active):
            break
        local, *_ = np.linalg.lstsq(a[:, active], y, rcond=None)
        next_coeffs = np.zeros(a.shape[1], dtype=float)
        next_coeffs[active] = local
        next_active = np.abs(next_coeffs) >= threshold
        if np.array_equal(active, next_active):
            coeffs = next_coeffs
            break
        active = next_active
        coeffs = next_coeffs
    if np.any(active):
        local, *_ = np.linalg.lstsq(a[:, active], y, rcond=None)
        coeffs = np.zeros(a.shape[1], dtype=float)
        coeffs[active] = local
    return coeffs


def vector_stlsq(a: np.ndarray, y: np.ndarray, threshold: float, max_iter: int) -> np.ndarray:
    return np.column_stack([stlsq(a, y[:, idx], threshold, max_iter) for idx in range(y.shape[1])])


def pysindy_stlsq_coeffs(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    optimizer = ps.STLSQ(
        threshold=CONFIG["stlsq_threshold"],
        alpha=0.0,
        max_iter=CONFIG["stlsq_max_iter"],
        normalize_columns=False,
    )
    optimizer.fit(a, y.reshape(-1, 1))
    return optimizer.coef_.reshape(-1)


def true_pk(theta: np.ndarray, g_over_l: float) -> np.ndarray:
    out = np.empty_like(theta, dtype=float)
    np.divide(np.sin(theta), theta, out=out, where=np.abs(theta) > 1e-12)
    out[np.abs(theta) <= 1e-12] = 1.0
    return g_over_l * out


def fit_local_k0(x: np.ndarray, dx: np.ndarray, c: float) -> float:
    theta = x[:, 0]
    target = dx[:, 1] + c * x[:, 1]
    return float(-np.dot(theta, target) / np.dot(theta, theta))


def fit_gain_schedule(x: np.ndarray, dx: np.ndarray, c: float, edges: list[float]) -> np.ndarray:
    theta = x[:, 0]
    target = dx[:, 1] + c * x[:, 1]
    abs_theta = np.abs(theta)
    edge_array = np.array(edges, dtype=float)
    values = []
    global_value = fit_local_k0(x, dx, c)
    for idx in range(len(edge_array) - 1):
        lo = edge_array[idx]
        hi = edge_array[idx + 1]
        if idx == 0:
            mask = (abs_theta >= lo) & (abs_theta <= hi)
        else:
            mask = (abs_theta > lo) & (abs_theta <= hi)
        denom = float(np.dot(theta[mask], theta[mask]))
        if np.any(mask) and denom > 1e-14:
            values.append(float(-np.dot(theta[mask], target[mask]) / denom))
        else:
            values.append(global_value)
    return np.array(values, dtype=float)


def lookup_gain(theta: float, edges: np.ndarray, values: np.ndarray) -> float:
    abs_theta = abs(theta)
    idx = int(np.searchsorted(edges[1:], abs_theta, side="left"))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def count_active(values: np.ndarray, threshold: float = 1e-6) -> int:
    return int(np.sum(np.abs(values) >= threshold))


def nrmse(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - true
    rmse_state = np.sqrt(np.mean(err * err, axis=0))
    denom_state = np.std(true, axis=0)
    denom_state = np.where(denom_state > 0, denom_state, 1.0)
    all_rmse = float(np.sqrt(np.mean(err * err)))
    all_denom = float(np.std(true)) or 1.0
    return {
        "all": all_rmse / all_denom,
        "theta": float(rmse_state[0] / denom_state[0]),
        "omega": float(rmse_state[1] / denom_state[1]),
    }


def coefficient_metrics(coeffs: np.ndarray, degrees: list[int], g_over_l: float) -> dict[str, object]:
    theta_grid = np.linspace(-2.4, 2.4, 801)
    p_true = true_pk(theta_grid, g_over_l)
    p_pred = b3_library(theta_grid, degrees) @ coeffs
    err = p_pred - p_true
    active = {degree for coeff, degree in zip(coeffs, degrees) if abs(coeff) >= 1e-6}
    expected = set(degrees)
    true_positive = len(active & expected)
    return {
        "grid_rmse": float(np.sqrt(np.mean(err * err))),
        "grid_nrmse": float(np.sqrt(np.mean(err * err)) / (np.std(p_true) or 1.0)),
        "max_abs_error": float(np.max(np.abs(err))),
        "grid_min_pred": float(np.min(p_pred)),
        "grid_max_pred": float(np.max(p_pred)),
        "active_degrees": sorted(active),
        "expected_degrees": sorted(expected),
        "support_precision": true_positive / len(active) if active else 0.0,
        "support_recall": true_positive / len(expected),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def polyline(points: np.ndarray) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_svg_series(
    path: Path,
    title: str,
    x_values: np.ndarray,
    series: list[tuple],
    x_label: str,
    y_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1220, 450
    left, right, top, bottom = 74, 34, 40, 60
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
        f'<text x="{width / 2:.0f}" y="25" text-anchor="middle" font-family="Arial, Noto Sans CJK SC, Microsoft YaHei, sans-serif" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<text x="{left + plot_w / 2:.0f}" y="{height - 15}" text-anchor="middle" font-family="Arial, Noto Sans CJK SC, Microsoft YaHei, sans-serif" font-size="13">{x_label}</text>',
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
        stroke_width = item[4] if len(item) > 4 else 1.8
        opacity = item[5] if len(item) > 5 else 1.0
        pts = np.column_stack([x_plot, sy(values[::step])])
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<polyline points="{polyline(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_width}" stroke-opacity="{opacity}"{dash_attr}/>'
        )
    legend_x = left + 12
    legend_y = top + 10
    for idx, item in enumerate(series):
        label, _, color = item[:3]
        dash = item[3] if len(item) > 3 else ""
        stroke_width = item[4] if len(item) > 4 else 2.0
        opacity = item[5] if len(item) > 5 else 1.0
        y = legend_y + idx * 18
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-opacity="{opacity}"{dash_attr}/>'
        )
        lines.append(
            f'<text x="{legend_x + 28}" y="{y + 4}" font-family="Arial, Noto Sans CJK SC, Microsoft YaHei, sans-serif" font-size="12">{label}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_trajectory(x0: list[float], split: str, trajectory_id: str) -> dict[str, object]:
    g_over_l = CONFIG["g_over_l"]
    c = CONFIG["damping_c"]
    t, x = simulate(
        lambda tt, xx: pendulum_rhs(tt, xx, g_over_l, c),
        np.array(x0, dtype=float),
        CONFIG["t_end"],
        CONFIG["dt"],
    )
    dx = np.array([pendulum_rhs(tt, xx, g_over_l, c) for tt, xx in zip(t, x)])
    return {"trajectory_id": trajectory_id, "split": split, "x0": x0, "t": t, "x": x, "dx": dx}


def stack_data(trajectories: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack([traj["x"] for traj in trajectories]), np.vstack([traj["dx"] for traj in trajectories])


def build_trajectories() -> list[dict[str, object]]:
    trajectories: list[dict[str, object]] = []
    for split, key in [
        ("train_small", "train_small_initial_conditions"),
        ("train_medium", "train_medium_initial_conditions"),
        ("test_interp", "test_interp_initial_conditions"),
        ("test_extrap", "test_extrap_initial_conditions"),
    ]:
        for idx, x0 in enumerate(CONFIG[key]):
            trajectories.append(make_trajectory(x0, split, f"{split}_ic{idx}"))
    return trajectories


def model_rollout(
    model_name: str,
    x0: list[float],
    c: float,
    k0: float,
    gain_edges: np.ndarray,
    gain_values: np.ndarray,
    b1_coeffs: np.ndarray,
    b2_coeffs: np.ndarray,
    b3_coeffs: np.ndarray,
    terms: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    x0_array = np.array(x0, dtype=float)
    if model_name == "B0_local_linear":
        return simulate(lambda tt, xx: b0_rhs(tt, xx, c, k0), x0_array, CONFIG["t_end"], CONFIG["dt"])
    if model_name == "B0_gain_scheduled_lookup":
        return simulate(lambda tt, xx: b0_gain_rhs(tt, xx, c, gain_edges, gain_values), x0_array, CONFIG["t_end"], CONFIG["dt"])
    if model_name == "B1_full_sindy":
        return simulate(lambda tt, xx: b1_rhs(tt, xx, b1_coeffs, terms), x0_array, CONFIG["t_end"], CONFIG["dt"])
    if model_name == "B2_free_residual_sindy":
        return simulate(lambda tt, xx: b2_rhs(tt, xx, c, k0, b2_coeffs, terms), x0_array, CONFIG["t_end"], CONFIG["dt"])
    if model_name == "B3_slot_constrained":
        return simulate(lambda tt, xx: b3_rhs(tt, xx, c, b3_coeffs, CONFIG["b3_degrees"]), x0_array, CONFIG["t_end"], CONFIG["dt"])
    if model_name == "B4_oracle_reference":
        return simulate(lambda tt, xx: pendulum_rhs(tt, xx, CONFIG["g_over_l"], c), x0_array, CONFIG["t_end"], CONFIG["dt"])
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    for directory in (DATA_DIR, PROVENANCE_DIR, RESULT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    g_over_l = CONFIG["g_over_l"]
    c = CONFIG["damping_c"]
    degrees = list(CONFIG["b3_degrees"])
    terms = state_polynomial_terms(CONFIG["state_library_max_degree"])
    term_names = [state_term_name(term) for term in terms]

    trajectories = build_trajectories()
    train_small = [traj for traj in trajectories if traj["split"] == "train_small"]
    train_medium = [traj for traj in trajectories if traj["split"] == "train_medium"]
    x_small, dx_small = stack_data(train_small)
    x_train, dx_train = stack_data(train_medium)

    k0 = fit_local_k0(x_small, dx_small, c)
    gain_edges = np.array(CONFIG["gain_abs_theta_edges"], dtype=float)
    gain_values = fit_gain_schedule(x_train, dx_train, c, CONFIG["gain_abs_theta_edges"])

    theta_library = state_polynomial_library(x_train, terms)
    b1_coeffs = vector_stlsq(theta_library, dx_train, CONFIG["stlsq_threshold"], CONFIG["stlsq_max_iter"])
    b0_train_rhs = np.array([b0_rhs(0.0, xx, c, k0) for xx in x_train])
    b2_coeffs = vector_stlsq(theta_library, dx_train - b0_train_rhs, CONFIG["stlsq_threshold"], CONFIG["stlsq_max_iter"])

    theta = x_train[:, 0]
    slot_library = -theta[:, None] * b3_library(theta, degrees)
    slot_target = dx_train[:, 1] + c * x_train[:, 1]
    b3_coeffs = pysindy_stlsq_coeffs(slot_library, slot_target)

    trajectory_rows = []
    for traj in trajectories:
        for idx, tt in enumerate(traj["t"]):
            trajectory_rows.append(
                {
                    "trajectory_id": traj["trajectory_id"],
                    "split": traj["split"],
                    "theta0": f"{traj['x0'][0]:.16g}",
                    "omega0": f"{traj['x0'][1]:.16g}",
                    "t": f"{tt:.10g}",
                    "theta": f"{traj['x'][idx, 0]:.16g}",
                    "omega": f"{traj['x'][idx, 1]:.16g}",
                    "dtheta": f"{traj['dx'][idx, 0]:.16g}",
                    "domega": f"{traj['dx'][idx, 1]:.16g}",
                }
            )
    trajectory_path = DATA_DIR / "e3_local_linearization_pendulum_trajectories.csv"
    write_csv(
        trajectory_path,
        trajectory_rows,
        ["trajectory_id", "split", "theta0", "omega0", "t", "theta", "omega", "dtheta", "domega"],
    )

    models = [
        "B0_local_linear",
        "B0_gain_scheduled_lookup",
        "B1_full_sindy",
        "B2_free_residual_sindy",
        "B3_slot_constrained",
        "B4_oracle_reference",
    ]
    rollout_summary_rows = []
    rollout_sample_rows = []
    metrics_by_split: dict[str, dict[str, dict[str, list[float]]]] = {}
    sample_trajectory_id = "test_extrap_ic2"
    for traj in trajectories:
        true_t = traj["t"]
        true_x = traj["x"]
        split = traj["split"]
        metrics_by_split.setdefault(split, {model: {"all": [], "theta": [], "omega": []} for model in models})
        for model_name in models:
            pred_t, pred_x = model_rollout(
                model_name,
                traj["x0"],
                c,
                k0,
                gain_edges,
                gain_values,
                b1_coeffs,
                b2_coeffs,
                b3_coeffs,
                terms,
            )
            if len(pred_t) != len(true_t):
                interp_pred = np.column_stack([np.interp(true_t, pred_t, pred_x[:, dim]) for dim in range(2)])
            else:
                interp_pred = pred_x
            model_nrmse = nrmse(true_x, interp_pred)
            for key, value in model_nrmse.items():
                metrics_by_split[split][model_name][key].append(value)
            rollout_summary_rows.append(
                {
                    "trajectory_id": traj["trajectory_id"],
                    "split": split,
                    "x0": json.dumps(traj["x0"]),
                    "model": model_name,
                    "nrmse_all": f"{model_nrmse['all']:.16g}",
                    "nrmse_theta": f"{model_nrmse['theta']:.16g}",
                    "nrmse_omega": f"{model_nrmse['omega']:.16g}",
                }
            )
            if traj["trajectory_id"] == sample_trajectory_id:
                for idx, tt in enumerate(true_t):
                    rollout_sample_rows.append(
                        {
                            "trajectory_id": traj["trajectory_id"],
                            "split": split,
                            "model": model_name,
                            "t": f"{tt:.10g}",
                            "true_theta": f"{true_x[idx, 0]:.16g}",
                            "true_omega": f"{true_x[idx, 1]:.16g}",
                            "pred_theta": f"{interp_pred[idx, 0]:.16g}",
                            "pred_omega": f"{interp_pred[idx, 1]:.16g}",
                        }
                    )

    rollout_summary_path = RESULT_DIR / "rollout_summary.csv"
    rollout_samples_path = RESULT_DIR / "rollout_samples.csv"
    write_csv(
        rollout_summary_path,
        rollout_summary_rows,
        ["trajectory_id", "split", "x0", "model", "nrmse_all", "nrmse_theta", "nrmse_omega"],
    )
    write_csv(
        rollout_samples_path,
        rollout_sample_rows,
        ["trajectory_id", "split", "model", "t", "true_theta", "true_omega", "pred_theta", "pred_omega"],
    )

    split_metrics = {}
    for split, model_values in metrics_by_split.items():
        split_metrics[split] = {}
        for model_name, values in model_values.items():
            split_metrics[split][model_name] = {
                f"mean_nrmse_{key}": float(np.mean(value_list)) for key, value_list in values.items()
            } | {
                f"max_nrmse_{key}": float(np.max(value_list)) for key, value_list in values.items()
            }

    taylor = {0: g_over_l, 2: -g_over_l / 6.0, 4: g_over_l / 120.0, 6: -g_over_l / 5040.0}
    coeff_rows = []
    for degree, coeff in zip(degrees, b3_coeffs):
        coeff_rows.append(
            {
                "term": f"theta^{degree}",
                "degree": degree,
                "b3_coefficient": f"{coeff:.16g}",
                "taylor_reference_coefficient": f"{taylor.get(degree, 0.0):.16g}",
                "abs_error_vs_taylor": f"{abs(coeff - taylor.get(degree, 0.0)):.16g}",
                "active": str(abs(coeff) >= 1e-6),
            }
        )
    coefficients_path = RESULT_DIR / "coefficients.csv"
    write_csv(
        coefficients_path,
        coeff_rows,
        ["term", "degree", "b3_coefficient", "taylor_reference_coefficient", "abs_error_vs_taylor", "active"],
    )

    gain_rows = []
    for idx, value in enumerate(gain_values):
        gain_rows.append(
            {
                "bin": idx,
                "abs_theta_low": f"{gain_edges[idx]:.16g}",
                "abs_theta_high": f"{gain_edges[idx + 1]:.16g}",
                "gain_value": f"{value:.16g}",
            }
        )
    gain_path = RESULT_DIR / "gain_schedule.csv"
    write_csv(gain_path, gain_rows, ["bin", "abs_theta_low", "abs_theta_high", "gain_value"])

    model_coeff_rows = []
    for row_idx, term_name in enumerate(term_names):
        model_coeff_rows.append(
            {
                "model": "B1_full_sindy",
                "term": term_name,
                "dtheta_coefficient": f"{b1_coeffs[row_idx, 0]:.16g}",
                "domega_coefficient": f"{b1_coeffs[row_idx, 1]:.16g}",
                "active": str(np.any(np.abs(b1_coeffs[row_idx]) >= 1e-6)),
            }
        )
        model_coeff_rows.append(
            {
                "model": "B2_free_residual_sindy",
                "term": term_name,
                "dtheta_coefficient": f"{b2_coeffs[row_idx, 0]:.16g}",
                "domega_coefficient": f"{b2_coeffs[row_idx, 1]:.16g}",
                "active": str(np.any(np.abs(b2_coeffs[row_idx]) >= 1e-6)),
            }
        )
    for degree, coeff in zip(degrees, b3_coeffs):
        model_coeff_rows.append(
            {
                "model": "B3_slot_constrained_pk",
                "term": f"theta^{degree}",
                "dtheta_coefficient": "",
                "domega_coefficient": f"{coeff:.16g}",
                "active": str(abs(coeff) >= 1e-6),
            }
        )
    model_coefficients_path = RESULT_DIR / "model_coefficients.csv"
    write_csv(
        model_coefficients_path,
        model_coeff_rows,
        ["model", "term", "dtheta_coefficient", "domega_coefficient", "active"],
    )

    coeff_metrics = coefficient_metrics(b3_coeffs, degrees, g_over_l)
    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "b0_local_linear_k0": k0,
        "b0_gain_schedule": {
            "abs_theta_edges": list(CONFIG["gain_abs_theta_edges"]),
            "values": [float(v) for v in gain_values],
        },
        "b3_optimizer": "PySINDy STLSQ",
        "b3_coefficients": {f"theta^{degree}": float(coeff) for degree, coeff in zip(degrees, b3_coeffs)},
        "taylor_reference_coefficients": {f"theta^{degree}": float(taylor[degree]) for degree in degrees},
        "active_term_counts": {
            "B0_gain_scheduled_lookup": len(gain_values),
            "B1_full_sindy": count_active(b1_coeffs),
            "B2_free_residual_sindy": count_active(b2_coeffs),
            "B3_slot_constrained": count_active(b3_coeffs),
        },
        "rollout_nrmse_by_split": split_metrics,
        "coefficient_function": coeff_metrics,
    }
    metrics_path = RESULT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    sample_rows = [row for row in rollout_sample_rows if row["trajectory_id"] == sample_trajectory_id]
    sample_t = np.array([float(row["t"]) for row in sample_rows if row["model"] == "B4_oracle_reference"])
    true_theta = np.array([float(row["true_theta"]) for row in sample_rows if row["model"] == "B4_oracle_reference"])
    series_theta = [("Reference nonlinear pendulum", true_theta, "#4D4D4D", "", 3.2, 0.55)]
    series_theta_zh = [("完整非线性摆参考轨迹", true_theta, "#4D4D4D", "", 3.2, 0.55)]
    styles = {
        "B0_local_linear": ("Small-angle local linear model", "小角度局部线性模型", "#E69F00", "", 2.4, 0.95),
        "B0_gain_scheduled_lookup": ("Piecewise gain-scheduled stiffness model", "分段查表刚度模型", "#56B4E9", "5 4", 2.4, 0.95),
        "B1_full_sindy": ("Full SINDy vector-field model", "完整向量场 SINDy 模型", "#0072B2", "2 4", 2.3, 0.95),
        "B2_free_residual_sindy": ("Free residual SINDy model", "自由残差 SINDy 模型", "#CC79A7", "10 3 2 3", 2.3, 0.95),
        "B3_slot_constrained": ("Slot-constrained coefficient SINDy model", "系数槽约束 SINDy 模型", "#009E73", "12 5", 2.8, 1.0),
    }
    for model_name, (label_en, label_zh, color, dash, width, opacity) in styles.items():
        values = np.array([float(row["pred_theta"]) for row in sample_rows if row["model"] == model_name])
        series_theta.append((label_en, values, color, dash, width, opacity))
        series_theta_zh.append((label_zh, values, color, dash, width, opacity))
    write_svg_series(
        FIGURE_DIR / "rollout_theta_extrap.svg",
        "E3 large-angle extrapolation rollout: theta",
        sample_t,
        series_theta,
        "t",
        "theta",
    )
    write_svg_series(
        FIGURE_DIR / "rollout_theta_extrap_zh.svg",
        "E3 大角度外推 rollout：角度 theta",
        sample_t,
        series_theta_zh,
        "时间 t",
        "角度 theta",
    )

    theta_grid = np.linspace(-2.4, 2.4, 801)
    p_true = true_pk(theta_grid, g_over_l)
    p_pred = b3_library(theta_grid, degrees) @ b3_coeffs
    p_gain = np.array([lookup_gain(value, gain_edges, gain_values) for value in theta_grid])
    write_svg_series(
        FIGURE_DIR / "coefficient_function.svg",
        "E3 recovered state-adaptive stiffness p_k(theta)",
        theta_grid,
        [
            ("Reference equivalent stiffness sin(theta)/theta", p_true, "#4D4D4D", "", 3.2, 0.55),
            ("Piecewise gain-scheduled stiffness model", p_gain, "#56B4E9", "5 4", 2.4, 0.95),
            ("Slot-constrained coefficient SINDy model", p_pred, "#009E73", "12 5", 2.8, 1.0),
        ],
        "theta",
        "p_k(theta)",
    )
    write_svg_series(
        FIGURE_DIR / "coefficient_function_zh.svg",
        "E3 恢复的状态自适应刚度 p_k(theta)",
        theta_grid,
        [
            ("等效刚度参考 sin(theta)/theta", p_true, "#4D4D4D", "", 3.2, 0.55),
            ("分段查表刚度模型", p_gain, "#56B4E9", "5 4", 2.4, 0.95),
            ("系数槽约束 SINDy 模型", p_pred, "#009E73", "12 5", 2.8, 1.0),
        ],
        "角度 theta",
        "刚度系数 p_k(theta)",
    )

    provenance = {
        "dataset_id": "e3_local_linearization_pendulum_trajectories",
        "experiment_id": CONFIG["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": CONFIG["source"]["name"],
        "source_urls": CONFIG["source"]["urls"],
        "source_access_date": CONFIG["source"]["access_date"],
        "equation_reference": "Full nonlinear pendulum with small-angle local-linearization baseline.",
        "parameters": {"g_over_l": g_over_l, "damping_c": c},
        "initial_conditions": {
            "train_small": CONFIG["train_small_initial_conditions"],
            "train_medium": CONFIG["train_medium_initial_conditions"],
            "test_interp": CONFIG["test_interp_initial_conditions"],
            "test_extrap": CONFIG["test_extrap_initial_conditions"],
        },
        "time_grid": {"t_end": CONFIG["t_end"], "dt": CONFIG["dt"]},
        "noise_model": CONFIG["noise"],
        "candidate_slots": [{"name": "stiffness", "g": "-theta", "coefficient_function": "p_k(theta)"}],
        "expected_coefficients": metrics["taylor_reference_coefficients"],
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
    provenance_path = PROVENANCE_DIR / "e3_local_linearization_pendulum_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

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
    (RESULT_DIR / "hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    summary = {
        "experiment_id": CONFIG["experiment_id"],
        "b0_local_linear_k0": k0,
        "b0_gain_values": [float(v) for v in gain_values],
        "b3_coefficients": metrics["b3_coefficients"],
        "coefficient_grid_nrmse": metrics["coefficient_function"]["grid_nrmse"],
        "active_term_counts": metrics["active_term_counts"],
        "test_interp_B0_mean_nrmse": split_metrics["test_interp"]["B0_local_linear"]["mean_nrmse_all"],
        "test_interp_B3_mean_nrmse": split_metrics["test_interp"]["B3_slot_constrained"]["mean_nrmse_all"],
        "test_extrap_B0_mean_nrmse": split_metrics["test_extrap"]["B0_local_linear"]["mean_nrmse_all"],
        "test_extrap_B0_gain_mean_nrmse": split_metrics["test_extrap"]["B0_gain_scheduled_lookup"]["mean_nrmse_all"],
        "test_extrap_B3_mean_nrmse": split_metrics["test_extrap"]["B3_slot_constrained"]["mean_nrmse_all"],
        "result_dir": str(RESULT_DIR.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
