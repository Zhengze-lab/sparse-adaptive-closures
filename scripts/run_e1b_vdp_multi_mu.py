#!/usr/bin/env python3
"""Run E1b: multi-mu Van der Pol coefficient-slot SINDy experiment."""

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
import scipy
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "generated" / "e1_vdp"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "e1b_vdp_multi_mu"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG = {
    "experiment_id": "E1b_vdp_multi_mu",
    "description": "Multi-mu Van der Pol damping coefficient-function recovery.",
    "source": {
        "name": "Designed from DSINDy/WSINDy Van der Pol equation",
        "urls": [
            "https://www.osti.gov/servlets/purl/2417947",
            "https://github.com/MathBioCU/WSINDy_ODE",
        ],
        "access_date": "2026-06-10",
    },
    "train_mu": [0.5, 1.0, 2.0, 4.0],
    "interpolation_mu": [1.5, 3.0],
    "extrapolation_mu": [5.0],
    "train_initial_conditions": [[0.0, 1.0], [1.0, 0.0], [-2.0, 1.0], [2.0, 0.0]],
    "test_initial_conditions": [[0.5, -1.0], [-1.5, 0.5]],
    "dt": 0.01,
    "t_end": 30.0,
    "stlsq_threshold": 1e-8,
    "stlsq_max_iter": 12,
    "param_state_library_max_degree": 4,
    "noise": None,
}


B3_FEATURE_NAMES = ["1", "x1", "x1^2", "mu", "mu*x1", "mu*x1^2"]


def vdp_rhs(_: float, x: np.ndarray, mu: float) -> np.ndarray:
    x1, x2 = x
    return np.array([x2, -x1 + mu * x2 - mu * x1 * x1 * x2], dtype=float)


def b0_rhs(_: float, x: np.ndarray, c0: float) -> np.ndarray:
    x1, x2 = x
    return np.array([x2, -x1 + c0 * x2], dtype=float)


def b1_rhs(
    _: float,
    x: np.ndarray,
    mu: float,
    coeffs: np.ndarray,
    terms: list[tuple[int, int, int]],
) -> np.ndarray:
    theta = param_state_library(x.reshape(1, 2), np.array([mu]), terms)[0]
    return theta @ coeffs


def b2_rhs(
    _: float,
    x: np.ndarray,
    mu: float,
    c0: float,
    coeffs: np.ndarray,
    terms: list[tuple[int, int, int]],
) -> np.ndarray:
    theta = param_state_library(x.reshape(1, 2), np.array([mu]), terms)[0]
    return b0_rhs(0.0, x, c0) + theta @ coeffs


def b3_rhs(_: float, x: np.ndarray, mu: float, coeffs: np.ndarray) -> np.ndarray:
    x1, x2 = x
    p = b3_library(np.array([x1]), np.array([mu]))[0] @ coeffs
    return np.array([x2, -x1 + x2 * p], dtype=float)


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


def b3_library(x1: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.ones_like(x1),
            x1,
            x1 * x1,
            mu,
            mu * x1,
            mu * x1 * x1,
        ]
    )


def param_state_terms(max_degree: int) -> list[tuple[int, int, int]]:
    terms: list[tuple[int, int, int]] = []
    for total_degree in range(max_degree + 1):
        for x1_power in range(total_degree, -1, -1):
            for x2_power in range(total_degree - x1_power, -1, -1):
                mu_power = total_degree - x1_power - x2_power
                terms.append((x1_power, x2_power, mu_power))
    return terms


def power_name(base: str, power: int) -> str | None:
    if power == 0:
        return None
    if power == 1:
        return base
    return f"{base}^{power}"


def param_state_term_name(term: tuple[int, int, int]) -> str:
    parts = [
        name
        for name in (
            power_name("x1", term[0]),
            power_name("x2", term[1]),
            power_name("mu", term[2]),
        )
        if name is not None
    ]
    return "*".join(parts) if parts else "1"


def param_state_library(x: np.ndarray, mu: np.ndarray, terms: list[tuple[int, int, int]]) -> np.ndarray:
    x1 = x[:, 0]
    x2 = x[:, 1]
    return np.column_stack(
        [(x1**x1_power) * (x2**x2_power) * (mu**mu_power) for x1_power, x2_power, mu_power in terms]
    )


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
        "x1": float(rmse_state[0] / denom_state[0]),
        "x2": float(rmse_state[1] / denom_state[1]),
    }


def coefficient_metrics(coeffs: np.ndarray, mu_values: list[float]) -> dict[str, object]:
    x_grid = np.linspace(-3.0, 3.0, 601)
    rows = []
    squared_errors = []
    true_values = []
    for mu in mu_values:
        mu_grid = np.full_like(x_grid, mu)
        p_true = mu - mu * x_grid * x_grid
        p_pred = b3_library(x_grid, mu_grid) @ coeffs
        err = p_pred - p_true
        rows.append(
            {
                "mu": mu,
                "grid_rmse": float(np.sqrt(np.mean(err * err))),
                "grid_nrmse": float(np.sqrt(np.mean(err * err)) / (np.std(p_true) or 1.0)),
            }
        )
        squared_errors.append(err * err)
        true_values.append(p_true)
    all_err = np.concatenate(squared_errors)
    all_true = np.concatenate(true_values)
    active = {name for name, coeff in zip(B3_FEATURE_NAMES, coeffs) if abs(coeff) >= 1e-6}
    expected = {"mu", "mu*x1^2"}
    true_positive = len(active & expected)
    precision = true_positive / len(active) if active else 0.0
    recall = true_positive / len(expected)
    return {
        "overall_grid_rmse": float(np.sqrt(np.mean(all_err))),
        "overall_grid_nrmse": float(np.sqrt(np.mean(all_err)) / (np.std(all_true) or 1.0)),
        "by_mu": rows,
        "active_features": sorted(active),
        "expected_features": sorted(expected),
        "support_precision": precision,
        "support_recall": recall,
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
    width, height = 1200, 430
    left, right, top, bottom = 70, 30, 38, 58
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
        f'<text x="{width / 2:.0f}" y="24" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<text x="{left + plot_w / 2:.0f}" y="{height - 14}" text-anchor="middle" font-family="Arial" font-size="13">{x_label}</text>',
        f'<text transform="translate(18,{top + plot_h / 2:.0f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{y_label}</text>',
    ]
    for tick in np.linspace(x_min, x_max, 6):
        x = sx(np.array([tick]))[0]
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#333"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{tick:.1f}</text>')
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
        lines.append(f'<text x="{legend_x + 28}" y="{y + 4}" font-family="Arial" font-size="12">{label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_trajectory(mu: float, x0: list[float], split: str, trajectory_id: str) -> dict[str, object]:
    t, x = simulate(lambda tt, xx: vdp_rhs(tt, xx, mu), np.array(x0, dtype=float), CONFIG["t_end"], CONFIG["dt"])
    dx = np.array([vdp_rhs(tt, xx, mu) for tt, xx in zip(t, x)])
    return {"trajectory_id": trajectory_id, "split": split, "mu": mu, "x0": x0, "t": t, "x": x, "dx": dx}


def stack_training_data(trajectories: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.vstack([traj["x"] for traj in trajectories])
    dx = np.vstack([traj["dx"] for traj in trajectories])
    mu = np.concatenate([np.full(len(traj["t"]), traj["mu"]) for traj in trajectories])
    return x, dx, mu


def model_rollout(
    model_name: str,
    mu: float,
    x0: list[float],
    c0: float,
    b1_coeffs: np.ndarray,
    b2_coeffs: np.ndarray,
    b3_coeffs: np.ndarray,
    terms: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "B0_constant_coefficient":
        return simulate(lambda tt, xx: b0_rhs(tt, xx, c0), np.array(x0, dtype=float), CONFIG["t_end"], CONFIG["dt"])
    if model_name == "B1_full_sindy":
        return simulate(lambda tt, xx: b1_rhs(tt, xx, mu, b1_coeffs, terms), np.array(x0, dtype=float), CONFIG["t_end"], CONFIG["dt"])
    if model_name == "B2_free_residual_sindy":
        return simulate(lambda tt, xx: b2_rhs(tt, xx, mu, c0, b2_coeffs, terms), np.array(x0, dtype=float), CONFIG["t_end"], CONFIG["dt"])
    if model_name == "B3_slot_constrained":
        return simulate(lambda tt, xx: b3_rhs(tt, xx, mu, b3_coeffs), np.array(x0, dtype=float), CONFIG["t_end"], CONFIG["dt"])
    if model_name == "B4_oracle_reference":
        return simulate(lambda tt, xx: vdp_rhs(tt, xx, mu), np.array(x0, dtype=float), CONFIG["t_end"], CONFIG["dt"])
    raise ValueError(f"Unknown model: {model_name}")


def model_rhs_on_states(
    model_name: str,
    x: np.ndarray,
    mu: np.ndarray,
    c0: float,
    b1_coeffs: np.ndarray,
    b2_coeffs: np.ndarray,
    b3_coeffs: np.ndarray,
    terms: list[tuple[int, int, int]],
) -> np.ndarray:
    if model_name == "B0_constant_coefficient":
        return np.array([b0_rhs(0.0, xx, c0) for xx in x])
    if model_name == "B1_full_sindy":
        return param_state_library(x, mu, terms) @ b1_coeffs
    if model_name == "B2_free_residual_sindy":
        b0_values = np.array([b0_rhs(0.0, xx, c0) for xx in x])
        return b0_values + param_state_library(x, mu, terms) @ b2_coeffs
    if model_name == "B3_slot_constrained":
        return np.array([b3_rhs(0.0, xx, mm, b3_coeffs) for xx, mm in zip(x, mu)])
    if model_name == "B4_oracle_reference":
        return np.array([vdp_rhs(0.0, xx, mm) for xx, mm in zip(x, mu)])
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    for directory in (DATA_DIR, PROVENANCE_DIR, RESULT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    trajectories: list[dict[str, object]] = []
    for mu in CONFIG["train_mu"]:
        for idx, x0 in enumerate(CONFIG["train_initial_conditions"]):
            trajectories.append(make_trajectory(mu, x0, "train", f"train_mu{mu:g}_ic{idx}"))
    for mu in CONFIG["train_mu"]:
        for idx, x0 in enumerate(CONFIG["test_initial_conditions"]):
            trajectories.append(make_trajectory(mu, x0, "seen_mu_unseen_ic", f"seen_mu{mu:g}_ic{idx}"))
    for mu in CONFIG["interpolation_mu"]:
        for idx, x0 in enumerate(CONFIG["test_initial_conditions"]):
            trajectories.append(make_trajectory(mu, x0, "interpolation_mu", f"interp_mu{mu:g}_ic{idx}"))
    for mu in CONFIG["extrapolation_mu"]:
        for idx, x0 in enumerate(CONFIG["test_initial_conditions"]):
            trajectories.append(make_trajectory(mu, x0, "extrapolation_mu", f"extrap_mu{mu:g}_ic{idx}"))

    train_trajectories = [traj for traj in trajectories if traj["split"] == "train"]
    x_train, dx_train, mu_train = stack_training_data(train_trajectories)
    residual_target = dx_train[:, 1] + x_train[:, 0]
    x2 = x_train[:, 1]
    c0 = float(np.dot(x2, residual_target) / np.dot(x2, x2))

    terms = param_state_terms(CONFIG["param_state_library_max_degree"])
    term_names = [param_state_term_name(term) for term in terms]
    theta = param_state_library(x_train, mu_train, terms)
    b1_coeffs = vector_stlsq(theta, dx_train, CONFIG["stlsq_threshold"], CONFIG["stlsq_max_iter"])
    b0_train_rhs = np.array([b0_rhs(0.0, xx, c0) for xx in x_train])
    b2_coeffs = vector_stlsq(theta, dx_train - b0_train_rhs, CONFIG["stlsq_threshold"], CONFIG["stlsq_max_iter"])

    slot_library = x2[:, None] * b3_library(x_train[:, 0], mu_train)
    b3_coeffs = stlsq(slot_library, residual_target, CONFIG["stlsq_threshold"], CONFIG["stlsq_max_iter"])

    trajectory_rows = []
    for traj in trajectories:
        for idx, tt in enumerate(traj["t"]):
            trajectory_rows.append(
                {
                    "trajectory_id": traj["trajectory_id"],
                    "split": traj["split"],
                    "mu": f"{traj['mu']:.16g}",
                    "x0_1": f"{traj['x0'][0]:.16g}",
                    "x0_2": f"{traj['x0'][1]:.16g}",
                    "t": f"{tt:.10g}",
                    "x1": f"{traj['x'][idx, 0]:.16g}",
                    "x2": f"{traj['x'][idx, 1]:.16g}",
                    "dx1": f"{traj['dx'][idx, 0]:.16g}",
                    "dx2": f"{traj['dx'][idx, 1]:.16g}",
                }
            )
    trajectory_path = DATA_DIR / "e1b_vdp_multi_mu_trajectories.csv"
    write_csv(
        trajectory_path,
        trajectory_rows,
        ["trajectory_id", "split", "mu", "x0_1", "x0_2", "t", "x1", "x2", "dx1", "dx2"],
    )

    rollout_models = [
        "B0_constant_coefficient",
        "B3_slot_constrained",
        "B4_oracle_reference",
    ]
    vector_field_models = [
        "B0_constant_coefficient",
        "B1_full_sindy",
        "B2_free_residual_sindy",
        "B3_slot_constrained",
        "B4_oracle_reference",
    ]
    rollout_summary_rows = []
    rollout_sample_rows = []
    rollout_metrics_by_split: dict[str, dict[str, dict[str, float]]] = {}
    for traj in trajectories:
        true_x = traj["x"]
        split = traj["split"]
        rollout_metrics_by_split.setdefault(split, {model: {"all": [], "x1": [], "x2": []} for model in rollout_models})
        for model_name in rollout_models:
            _, pred_x = model_rollout(
                model_name,
                traj["mu"],
                traj["x0"],
                c0,
                b1_coeffs,
                b2_coeffs,
                b3_coeffs,
                terms,
            )
            model_nrmse = nrmse(true_x, pred_x)
            for key, value in model_nrmse.items():
                rollout_metrics_by_split[split][model_name][key].append(value)
            rollout_summary_rows.append(
                {
                    "trajectory_id": traj["trajectory_id"],
                    "split": split,
                    "mu": f"{traj['mu']:.16g}",
                    "x0": json.dumps(traj["x0"]),
                    "model": model_name,
                    "nrmse_all": f"{model_nrmse['all']:.16g}",
                    "nrmse_x1": f"{model_nrmse['x1']:.16g}",
                    "nrmse_x2": f"{model_nrmse['x2']:.16g}",
                }
            )
            if traj["trajectory_id"] in {"interp_mu1.5_ic0", "extrap_mu5_ic0"}:
                for idx, tt in enumerate(traj["t"]):
                    rollout_sample_rows.append(
                        {
                            "trajectory_id": traj["trajectory_id"],
                            "split": split,
                            "mu": f"{traj['mu']:.16g}",
                            "model": model_name,
                            "t": f"{tt:.10g}",
                            "true_x1": f"{true_x[idx, 0]:.16g}",
                            "true_x2": f"{true_x[idx, 1]:.16g}",
                            "pred_x1": f"{pred_x[idx, 0]:.16g}",
                            "pred_x2": f"{pred_x[idx, 1]:.16g}",
                        }
                    )

    vector_field_metrics_by_split: dict[str, dict[str, dict[str, float]]] = {}
    for split in sorted({traj["split"] for traj in trajectories}):
        split_trajectories = [traj for traj in trajectories if traj["split"] == split]
        x_split = np.vstack([traj["x"] for traj in split_trajectories])
        dx_split = np.vstack([traj["dx"] for traj in split_trajectories])
        mu_split = np.concatenate([np.full(len(traj["t"]), traj["mu"]) for traj in split_trajectories])
        vector_field_metrics_by_split[split] = {}
        for model_name in vector_field_models:
            pred_dx = model_rhs_on_states(model_name, x_split, mu_split, c0, b1_coeffs, b2_coeffs, b3_coeffs, terms)
            vector_field_metrics_by_split[split][model_name] = nrmse(dx_split, pred_dx)

    rollout_summary_path = RESULT_DIR / "rollout_summary.csv"
    rollout_samples_path = RESULT_DIR / "rollout_samples.csv"
    write_csv(
        rollout_summary_path,
        rollout_summary_rows,
        ["trajectory_id", "split", "mu", "x0", "model", "nrmse_all", "nrmse_x1", "nrmse_x2"],
    )
    write_csv(
        rollout_samples_path,
        rollout_sample_rows,
        ["trajectory_id", "split", "mu", "model", "t", "true_x1", "true_x2", "pred_x1", "pred_x2"],
    )

    rollout_split_metrics = {}
    for split, model_values in rollout_metrics_by_split.items():
        rollout_split_metrics[split] = {}
        for model_name, values in model_values.items():
            rollout_split_metrics[split][model_name] = {
                f"mean_nrmse_{key}": float(np.mean(value_list)) for key, value_list in values.items()
            } | {
                f"max_nrmse_{key}": float(np.max(value_list)) for key, value_list in values.items()
            }

    expected_b3 = {"1": 0.0, "x1": 0.0, "x1^2": 0.0, "mu": 1.0, "mu*x1": 0.0, "mu*x1^2": -1.0}
    b3_coeff_rows = []
    for name, coeff in zip(B3_FEATURE_NAMES, b3_coeffs):
        b3_coeff_rows.append(
            {
                "feature": name,
                "coefficient": f"{coeff:.16g}",
                "expected_coefficient": f"{expected_b3[name]:.16g}",
                "abs_error": f"{abs(coeff - expected_b3[name]):.16g}",
                "active": str(abs(coeff) >= 1e-6),
            }
        )
    coefficients_path = RESULT_DIR / "coefficients.csv"
    write_csv(coefficients_path, b3_coeff_rows, ["feature", "coefficient", "expected_coefficient", "abs_error", "active"])

    model_coeff_rows = []
    for row_idx, term_name in enumerate(term_names):
        model_coeff_rows.append(
            {
                "model": "B1_full_sindy",
                "term": term_name,
                "dx1_coefficient": f"{b1_coeffs[row_idx, 0]:.16g}",
                "dx2_coefficient": f"{b1_coeffs[row_idx, 1]:.16g}",
                "active": str(np.any(np.abs(b1_coeffs[row_idx]) >= 1e-6)),
            }
        )
        model_coeff_rows.append(
            {
                "model": "B2_free_residual_sindy",
                "term": term_name,
                "dx1_coefficient": f"{b2_coeffs[row_idx, 0]:.16g}",
                "dx2_coefficient": f"{b2_coeffs[row_idx, 1]:.16g}",
                "active": str(np.any(np.abs(b2_coeffs[row_idx]) >= 1e-6)),
            }
        )
    for name, coeff in zip(B3_FEATURE_NAMES, b3_coeffs):
        model_coeff_rows.append(
            {
                "model": "B3_slot_constrained_pc",
                "term": name,
                "dx1_coefficient": "",
                "dx2_coefficient": f"{coeff:.16g}",
                "active": str(abs(coeff) >= 1e-6),
            }
        )
    model_coefficients_path = RESULT_DIR / "model_coefficients.csv"
    write_csv(
        model_coefficients_path,
        model_coeff_rows,
        ["model", "term", "dx1_coefficient", "dx2_coefficient", "active"],
    )

    coefficient_mu_values = CONFIG["train_mu"] + CONFIG["interpolation_mu"] + CONFIG["extrapolation_mu"]
    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "b0_constant_damping_c0": c0,
        "b3_coefficients": {name: float(coeff) for name, coeff in zip(B3_FEATURE_NAMES, b3_coeffs)},
        "expected_b3_coefficients": expected_b3,
        "active_term_counts": {
            "B1_full_sindy": count_active(b1_coeffs),
            "B2_free_residual_sindy": count_active(b2_coeffs),
            "B3_slot_constrained": count_active(b3_coeffs),
        },
        "rollout_nrmse_by_split": rollout_split_metrics,
        "vector_field_nrmse_by_split": vector_field_metrics_by_split,
        "coefficient_function": coefficient_metrics(b3_coeffs, coefficient_mu_values),
    }
    metrics_path = RESULT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    x_grid = np.linspace(-3.0, 3.0, 601)
    series = []
    series_zh = []
    colors = ["#0072B2", "#D55E00", "#009E73"]
    dash_patterns = ["2 4", "10 3 2 3", "12 5"]
    for mu, color, dash in zip([1.5, 3.0, 5.0], colors, dash_patterns):
        mu_grid = np.full_like(x_grid, mu)
        p_true = mu - mu * x_grid * x_grid
        p_pred = b3_library(x_grid, mu_grid) @ b3_coeffs
        series.append((f"Reference coefficient from full equation, mu={mu:g}", p_true, color, "", 3.2, 0.38))
        series.append((f"Slot-constrained coefficient SINDy model, mu={mu:g}", p_pred, color, dash, 2.8, 1.0))
        series_zh.append((f"完整方程参考系数，mu={mu:g}", p_true, color, "", 3.2, 0.38))
        series_zh.append((f"系数槽约束 SINDy 模型，mu={mu:g}", p_pred, color, dash, 2.8, 1.0))
    write_svg_series(
        FIGURE_DIR / "coefficient_function_slices.svg",
        "E1b recovered p_c(x1, mu) slices",
        x_grid,
        series,
        "x1",
        "p_c(x1, mu)",
    )
    write_svg_series(
        FIGURE_DIR / "coefficient_function_slices_zh.svg",
        "E1b 恢复的系数函数 p_c(x1, mu) 切片",
        x_grid,
        series_zh,
        "状态 x1",
        "阻尼系数 p_c(x1, mu)",
    )

    provenance = {
        "dataset_id": "e1b_vdp_multi_mu_trajectories",
        "experiment_id": CONFIG["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": CONFIG["source"]["name"],
        "source_urls": CONFIG["source"]["urls"],
        "source_access_date": CONFIG["source"]["access_date"],
        "equation_reference": "Van der Pol equation with designed multi-mu train/test split.",
        "parameters": {
            "train_mu": CONFIG["train_mu"],
            "interpolation_mu": CONFIG["interpolation_mu"],
            "extrapolation_mu": CONFIG["extrapolation_mu"],
        },
        "initial_conditions": {
            "train": CONFIG["train_initial_conditions"],
            "test": CONFIG["test_initial_conditions"],
        },
        "time_grid": {"start": 0.0, "end": CONFIG["t_end"], "dt": CONFIG["dt"]},
        "noise_model": CONFIG["noise"],
        "candidate_slots": [{"name": "damping", "g": "x2", "coefficient_function": "p_c(x1, mu)"}],
        "expected_coefficients": expected_b3,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "script": str(Path(__file__).relative_to(ROOT)),
        "outputs": {
            "trajectory_csv": str(trajectory_path.relative_to(ROOT)),
            "rollout_summary_csv": str(rollout_summary_path.relative_to(ROOT)),
            "rollout_samples_csv": str(rollout_samples_path.relative_to(ROOT)),
            "coefficients_csv": str(coefficients_path.relative_to(ROOT)),
            "model_coefficients_csv": str(model_coefficients_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
        },
    }
    provenance_path = PROVENANCE_DIR / "e1b_vdp_multi_mu_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    hashes = {
        "trajectory_csv_sha256": sha256_file(trajectory_path),
        "rollout_summary_csv_sha256": sha256_file(rollout_summary_path),
        "rollout_samples_csv_sha256": sha256_file(rollout_samples_path),
        "coefficients_csv_sha256": sha256_file(coefficients_path),
        "model_coefficients_csv_sha256": sha256_file(model_coefficients_path),
        "metrics_json_sha256": sha256_file(metrics_path),
        "provenance_json_sha256": sha256_file(provenance_path),
    }
    (RESULT_DIR / "hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    summary = {
        "experiment_id": CONFIG["experiment_id"],
        "b3_coefficients": metrics["b3_coefficients"],
        "coefficient_grid_nrmse": metrics["coefficient_function"]["overall_grid_nrmse"],
        "active_term_counts": metrics["active_term_counts"],
        "interpolation_B3_rollout_mean_nrmse": rollout_split_metrics["interpolation_mu"]["B3_slot_constrained"]["mean_nrmse_all"],
        "extrapolation_B3_rollout_mean_nrmse": rollout_split_metrics["extrapolation_mu"]["B3_slot_constrained"]["mean_nrmse_all"],
        "interpolation_B2_vector_field_nrmse": vector_field_metrics_by_split["interpolation_mu"]["B2_free_residual_sindy"]["all"],
        "extrapolation_B2_vector_field_nrmse": vector_field_metrics_by_split["extrapolation_mu"]["B2_free_residual_sindy"]["all"],
        "result_dir": str(RESULT_DIR.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
