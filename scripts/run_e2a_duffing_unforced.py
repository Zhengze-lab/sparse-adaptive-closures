#!/usr/bin/env python3
"""Run E2a: unforced Duffing stiffness coefficient-function recovery."""

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
DATA_DIR = ROOT / "data" / "generated" / "e2_duffing"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "e2a_duffing_unforced"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG = {
    "experiment_id": "E2a_duffing_unforced",
    "description": "Unforced Duffing stiffness coefficient-function recovery.",
    "source": {
        "name": "DSINDy PS2 / WSINDy default Duffing parameters",
        "urls": [
            "https://www.osti.gov/servlets/purl/2417947",
            "https://github.com/MathBioCU/WSINDy_ODE",
        ],
        "access_date": "2026-06-10",
    },
    "eta": 0.2,
    "kappa": 0.2,
    "epsilon": 1.0,
    "train_initial_conditions": [[0.0, 1.0], [0.0, 2.0], [-2.0, -2.0], [1.0, 0.0]],
    "test_initial_conditions": [[-1.0, 1.0], [2.0, -1.0], [0.5, -0.5]],
    "dt": 0.01,
    "train_t_end": 10.0,
    "rollout_t_end": 25.0,
    "poly_degrees": [0, 1, 2, 3, 4],
    "state_library_max_degree": 3,
    "stlsq_threshold": 1e-8,
    "stlsq_max_iter": 12,
    "noise": None,
}


def duffing_rhs(_: float, x: np.ndarray, eta: float, kappa: float, epsilon: float) -> np.ndarray:
    x1, x2 = x
    return np.array([x2, -eta * x2 - kappa * x1 - epsilon * x1**3], dtype=float)


def b0_rhs(_: float, x: np.ndarray, eta: float, k0: float) -> np.ndarray:
    x1, x2 = x
    return np.array([x2, -eta * x2 - k0 * x1], dtype=float)


def b1_rhs(_: float, x: np.ndarray, coeffs: np.ndarray, terms: list[tuple[int, int]]) -> np.ndarray:
    theta = state_polynomial_library(x.reshape(1, 2), terms)[0]
    return theta @ coeffs


def b2_rhs(
    _: float,
    x: np.ndarray,
    eta: float,
    k0: float,
    coeffs: np.ndarray,
    terms: list[tuple[int, int]],
) -> np.ndarray:
    theta = state_polynomial_library(x.reshape(1, 2), terms)[0]
    return b0_rhs(0.0, x, eta, k0) + theta @ coeffs


def b3_rhs(_: float, x: np.ndarray, eta: float, coeffs: np.ndarray, degrees: list[int]) -> np.ndarray:
    x1, x2 = x
    p = sum(coeff * (x1**degree) for coeff, degree in zip(coeffs, degrees))
    return np.array([x2, -eta * x2 - x1 * p], dtype=float)


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


def polynomial_library(x1: np.ndarray, degrees: list[int]) -> np.ndarray:
    return np.column_stack([x1**degree for degree in degrees])


def state_polynomial_terms(max_degree: int) -> list[tuple[int, int]]:
    terms: list[tuple[int, int]] = []
    for total_degree in range(max_degree + 1):
        for x1_power in range(total_degree, -1, -1):
            x2_power = total_degree - x1_power
            terms.append((x1_power, x2_power))
    return terms


def state_term_name(term: tuple[int, int]) -> str:
    x1_power, x2_power = term
    if x1_power == 0 and x2_power == 0:
        return "1"
    parts = []
    if x1_power == 1:
        parts.append("x1")
    elif x1_power > 1:
        parts.append(f"x1^{x1_power}")
    if x2_power == 1:
        parts.append("x2")
    elif x2_power > 1:
        parts.append(f"x2^{x2_power}")
    return "*".join(parts)


def state_polynomial_library(x: np.ndarray, terms: list[tuple[int, int]]) -> np.ndarray:
    x1 = x[:, 0]
    x2 = x[:, 1]
    return np.column_stack([(x1**x1_power) * (x2**x2_power) for x1_power, x2_power in terms])


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


def coefficient_metrics(coeffs: np.ndarray, degrees: list[int], kappa: float, epsilon: float) -> dict[str, object]:
    x_grid = np.linspace(-3.0, 3.0, 601)
    p_true = kappa + epsilon * x_grid * x_grid
    p_pred = sum(coeff * (x_grid**degree) for coeff, degree in zip(coeffs, degrees))
    rmse = float(np.sqrt(np.mean((p_pred - p_true) ** 2)))
    denom = float(np.std(p_true)) or 1.0
    active = {degree for coeff, degree in zip(coeffs, degrees) if abs(coeff) >= 1e-6}
    expected = {0, 2}
    true_positive = len(active & expected)
    return {
        "grid_rmse": rmse,
        "grid_nrmse": rmse / denom,
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


def write_svg_series(path: Path, title: str, x_values: np.ndarray, series: list[tuple], x_label: str, y_label: str) -> None:
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


def make_trajectory(x0: list[float], split: str, trajectory_id: str, t_end: float) -> dict[str, object]:
    eta, kappa, epsilon = CONFIG["eta"], CONFIG["kappa"], CONFIG["epsilon"]
    t, x = simulate(lambda tt, xx: duffing_rhs(tt, xx, eta, kappa, epsilon), np.array(x0, dtype=float), t_end, CONFIG["dt"])
    dx = np.array([duffing_rhs(tt, xx, eta, kappa, epsilon) for tt, xx in zip(t, x)])
    return {"trajectory_id": trajectory_id, "split": split, "x0": x0, "t": t, "x": x, "dx": dx}


def stack_data(trajectories: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack([traj["x"] for traj in trajectories]), np.vstack([traj["dx"] for traj in trajectories])


def model_rollout(
    model_name: str,
    x0: list[float],
    eta: float,
    k0: float,
    b1_coeffs: np.ndarray,
    b2_coeffs: np.ndarray,
    b3_coeffs: np.ndarray,
    terms: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "B0_constant_stiffness":
        return simulate(lambda tt, xx: b0_rhs(tt, xx, eta, k0), np.array(x0, dtype=float), CONFIG["rollout_t_end"], CONFIG["dt"])
    if model_name == "B1_full_sindy":
        return simulate(lambda tt, xx: b1_rhs(tt, xx, b1_coeffs, terms), np.array(x0, dtype=float), CONFIG["rollout_t_end"], CONFIG["dt"])
    if model_name == "B2_free_residual_sindy":
        return simulate(lambda tt, xx: b2_rhs(tt, xx, eta, k0, b2_coeffs, terms), np.array(x0, dtype=float), CONFIG["rollout_t_end"], CONFIG["dt"])
    if model_name == "B3_slot_constrained":
        return simulate(lambda tt, xx: b3_rhs(tt, xx, eta, b3_coeffs, CONFIG["poly_degrees"]), np.array(x0, dtype=float), CONFIG["rollout_t_end"], CONFIG["dt"])
    if model_name == "B4_oracle_reference":
        return simulate(
            lambda tt, xx: duffing_rhs(tt, xx, eta, CONFIG["kappa"], CONFIG["epsilon"]),
            np.array(x0, dtype=float),
            CONFIG["rollout_t_end"],
            CONFIG["dt"],
        )
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    for directory in (DATA_DIR, PROVENANCE_DIR, RESULT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    eta, kappa, epsilon = CONFIG["eta"], CONFIG["kappa"], CONFIG["epsilon"]
    degrees = list(CONFIG["poly_degrees"])
    terms = state_polynomial_terms(CONFIG["state_library_max_degree"])
    term_names = [state_term_name(term) for term in terms]

    train_trajectories = [
        make_trajectory(x0, "train", f"train_ic{idx}", CONFIG["train_t_end"])
        for idx, x0 in enumerate(CONFIG["train_initial_conditions"])
    ]
    x_train, dx_train = stack_data(train_trajectories)
    stiffness_target = -(dx_train[:, 1] + eta * x_train[:, 1])
    x1 = x_train[:, 0]
    k0 = float(np.dot(x1, stiffness_target) / np.dot(x1, x1))

    theta = state_polynomial_library(x_train, terms)
    b1_coeffs = vector_stlsq(theta, dx_train, CONFIG["stlsq_threshold"], CONFIG["stlsq_max_iter"])
    b0_train_rhs = np.array([b0_rhs(0.0, xx, eta, k0) for xx in x_train])
    b2_coeffs = vector_stlsq(theta, dx_train - b0_train_rhs, CONFIG["stlsq_threshold"], CONFIG["stlsq_max_iter"])

    slot_library = -x1[:, None] * polynomial_library(x1, degrees)
    b3_coeffs = stlsq(slot_library, dx_train[:, 1] + eta * x_train[:, 1], CONFIG["stlsq_threshold"], CONFIG["stlsq_max_iter"])

    all_trajectories = train_trajectories + [
        make_trajectory(x0, "unseen_ic", f"test_ic{idx}", CONFIG["rollout_t_end"])
        for idx, x0 in enumerate(CONFIG["test_initial_conditions"])
    ]

    trajectory_rows = []
    for traj in all_trajectories:
        for idx, tt in enumerate(traj["t"]):
            trajectory_rows.append(
                {
                    "trajectory_id": traj["trajectory_id"],
                    "split": traj["split"],
                    "x0_1": f"{traj['x0'][0]:.16g}",
                    "x0_2": f"{traj['x0'][1]:.16g}",
                    "t": f"{tt:.10g}",
                    "x1": f"{traj['x'][idx, 0]:.16g}",
                    "x2": f"{traj['x'][idx, 1]:.16g}",
                    "dx1": f"{traj['dx'][idx, 0]:.16g}",
                    "dx2": f"{traj['dx'][idx, 1]:.16g}",
                }
            )
    trajectory_path = DATA_DIR / "e2a_duffing_unforced_trajectories.csv"
    write_csv(
        trajectory_path,
        trajectory_rows,
        ["trajectory_id", "split", "x0_1", "x0_2", "t", "x1", "x2", "dx1", "dx2"],
    )

    models = [
        "B0_constant_stiffness",
        "B1_full_sindy",
        "B2_free_residual_sindy",
        "B3_slot_constrained",
        "B4_oracle_reference",
    ]
    rollout_summary_rows = []
    rollout_sample_rows = []
    metrics_by_split: dict[str, dict[str, dict[str, list[float]]]] = {}
    for traj in all_trajectories:
        true_t = traj["t"]
        true_x = traj["x"]
        split = traj["split"]
        metrics_by_split.setdefault(split, {model: {"all": [], "x1": [], "x2": []} for model in models})
        for model_name in models:
            pred_t, pred_x = model_rollout(model_name, traj["x0"], eta, k0, b1_coeffs, b2_coeffs, b3_coeffs, terms)
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
                    "nrmse_x1": f"{model_nrmse['x1']:.16g}",
                    "nrmse_x2": f"{model_nrmse['x2']:.16g}",
                }
            )
            if traj["trajectory_id"] == "test_ic0":
                for idx, tt in enumerate(true_t):
                    rollout_sample_rows.append(
                        {
                            "trajectory_id": traj["trajectory_id"],
                            "split": split,
                            "model": model_name,
                            "t": f"{tt:.10g}",
                            "true_x1": f"{true_x[idx, 0]:.16g}",
                            "true_x2": f"{true_x[idx, 1]:.16g}",
                            "pred_x1": f"{interp_pred[idx, 0]:.16g}",
                            "pred_x2": f"{interp_pred[idx, 1]:.16g}",
                        }
                    )

    rollout_summary_path = RESULT_DIR / "rollout_summary.csv"
    rollout_samples_path = RESULT_DIR / "rollout_samples.csv"
    write_csv(rollout_summary_path, rollout_summary_rows, ["trajectory_id", "split", "x0", "model", "nrmse_all", "nrmse_x1", "nrmse_x2"])
    write_csv(rollout_samples_path, rollout_sample_rows, ["trajectory_id", "split", "model", "t", "true_x1", "true_x2", "pred_x1", "pred_x2"])

    split_metrics = {}
    for split, model_values in metrics_by_split.items():
        split_metrics[split] = {}
        for model_name, values in model_values.items():
            split_metrics[split][model_name] = {
                f"mean_nrmse_{key}": float(np.mean(value_list)) for key, value_list in values.items()
            } | {
                f"max_nrmse_{key}": float(np.max(value_list)) for key, value_list in values.items()
            }

    expected = {0: kappa, 2: epsilon}
    coeff_rows = []
    for degree, coeff in zip(degrees, b3_coeffs):
        coeff_rows.append(
            {
                "term": f"x1^{degree}",
                "degree": degree,
                "b3_coefficient": f"{coeff:.16g}",
                "expected_coefficient": f"{expected.get(degree, 0.0):.16g}",
                "abs_error": f"{abs(coeff - expected.get(degree, 0.0)):.16g}",
                "active": str(abs(coeff) >= 1e-6),
            }
        )
    coefficients_path = RESULT_DIR / "coefficients.csv"
    write_csv(coefficients_path, coeff_rows, ["term", "degree", "b3_coefficient", "expected_coefficient", "abs_error", "active"])

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
    for degree, coeff in zip(degrees, b3_coeffs):
        model_coeff_rows.append(
            {
                "model": "B3_slot_constrained_pk",
                "term": f"x1^{degree}",
                "dx1_coefficient": "",
                "dx2_coefficient": f"{coeff:.16g}",
                "active": str(abs(coeff) >= 1e-6),
            }
        )
    model_coefficients_path = RESULT_DIR / "model_coefficients.csv"
    write_csv(model_coefficients_path, model_coeff_rows, ["model", "term", "dx1_coefficient", "dx2_coefficient", "active"])

    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "b0_constant_stiffness_k0": k0,
        "b3_coefficients": {f"x1^{degree}": float(coeff) for degree, coeff in zip(degrees, b3_coeffs)},
        "expected_coefficients": {"x1^0": kappa, "x1^2": epsilon},
        "active_term_counts": {
            "B1_full_sindy": count_active(b1_coeffs),
            "B2_free_residual_sindy": count_active(b2_coeffs),
            "B3_slot_constrained": count_active(b3_coeffs),
        },
        "rollout_nrmse_by_split": split_metrics,
        "coefficient_function": coefficient_metrics(b3_coeffs, degrees, kappa, epsilon),
    }
    metrics_path = RESULT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    sample_rows = [row for row in rollout_sample_rows if row["trajectory_id"] == "test_ic0"]
    sample_t = np.array([float(row["t"]) for row in sample_rows if row["model"] == "B4_oracle_reference"])
    true_x1 = np.array([float(row["true_x1"]) for row in sample_rows if row["model"] == "B4_oracle_reference"])
    series_x1 = [
        ("Reference trajectory from full equation", true_x1, "#4D4D4D", "", 3.2, 0.55),
    ]
    series_x1_zh = [
        ("完整方程参考轨迹", true_x1, "#4D4D4D", "", 3.2, 0.55),
    ]
    styles = {
        "B0_constant_stiffness": ("Simplified constant-stiffness model", "常刚度简化模型", "#E69F00", "", 2.6, 0.95),
        "B1_full_sindy": ("Full SINDy vector-field model", "完整向量场 SINDy 模型", "#0072B2", "2 4", 2.4, 0.95),
        "B2_free_residual_sindy": ("Free residual SINDy model", "自由残差 SINDy 模型", "#CC79A7", "10 3 2 3", 2.4, 0.95),
        "B3_slot_constrained": ("Slot-constrained coefficient SINDy model", "系数槽约束 SINDy 模型", "#009E73", "12 5", 2.8, 1.0),
    }
    for model_name, (label_en, label_zh, color, dash, width, opacity) in styles.items():
        values = np.array([float(row["pred_x1"]) for row in sample_rows if row["model"] == model_name])
        series_x1.append((label_en, values, color, dash, width, opacity))
        series_x1_zh.append((label_zh, values, color, dash, width, opacity))
    write_svg_series(FIGURE_DIR / "rollout_x1.svg", "E2a Duffing rollout: x1", sample_t, series_x1, "t", "x1")
    write_svg_series(FIGURE_DIR / "rollout_x1_zh.svg", "E2a Duffing rollout：状态 x1", sample_t, series_x1_zh, "时间 t", "状态 x1")

    x_grid = np.linspace(-3.0, 3.0, 601)
    p_true = kappa + epsilon * x_grid * x_grid
    p_pred = sum(coeff * (x_grid**degree) for coeff, degree in zip(b3_coeffs, degrees))
    write_svg_series(
        FIGURE_DIR / "coefficient_function.svg",
        "E2a recovered stiffness coefficient p_k(x1)",
        x_grid,
        [
            ("Reference stiffness coefficient from full equation", p_true, "#4D4D4D", "", 3.2, 0.55),
            ("Slot-constrained coefficient SINDy model", p_pred, "#009E73", "12 5", 2.8, 1.0),
        ],
        "x1",
        "p_k(x1)",
    )
    write_svg_series(
        FIGURE_DIR / "coefficient_function_zh.svg",
        "E2a 恢复的刚度系数函数 p_k(x1)",
        x_grid,
        [
            ("完整方程参考刚度系数", p_true, "#4D4D4D", "", 3.2, 0.55),
            ("系数槽约束 SINDy 模型", p_pred, "#009E73", "12 5", 2.8, 1.0),
        ],
        "状态 x1",
        "刚度系数 p_k(x1)",
    )

    provenance = {
        "dataset_id": "e2a_duffing_unforced_trajectories",
        "experiment_id": CONFIG["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": CONFIG["source"]["name"],
        "source_urls": CONFIG["source"]["urls"],
        "source_access_date": CONFIG["source"]["access_date"],
        "equation_reference": "Unforced Duffing equation with DSINDy PS2 / WSINDy default parameters.",
        "parameters": {"eta": eta, "kappa": kappa, "epsilon": epsilon},
        "initial_conditions": {"train": CONFIG["train_initial_conditions"], "test": CONFIG["test_initial_conditions"]},
        "time_grid": {"train_end": CONFIG["train_t_end"], "rollout_end": CONFIG["rollout_t_end"], "dt": CONFIG["dt"]},
        "noise_model": CONFIG["noise"],
        "candidate_slots": [{"name": "stiffness", "g": "-x1", "coefficient_function": "p_k(x1)"}],
        "expected_coefficients": metrics["expected_coefficients"],
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
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
    provenance_path = PROVENANCE_DIR / "e2a_duffing_unforced_provenance.json"
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
        "k0": k0,
        "b3_coefficients": metrics["b3_coefficients"],
        "coefficient_grid_nrmse": metrics["coefficient_function"]["grid_nrmse"],
        "active_term_counts": metrics["active_term_counts"],
        "unseen_ic_B0_mean_nrmse": split_metrics["unseen_ic"]["B0_constant_stiffness"]["mean_nrmse_all"],
        "unseen_ic_B3_mean_nrmse": split_metrics["unseen_ic"]["B3_slot_constrained"]["mean_nrmse_all"],
        "result_dir": str(RESULT_DIR.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
