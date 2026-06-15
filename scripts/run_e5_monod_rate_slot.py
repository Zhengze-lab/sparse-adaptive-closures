#!/usr/bin/env python3
"""Run E5: Monod/Michaelis-Menten rate coefficient-slot experiment."""

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
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "generated" / "e5_monod_rate"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "e5_monod_rate_slot"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG = {
    "experiment_id": "E5_monod_michaelis_menten_rate_slot",
    "description": "Monod/Michaelis-Menten growth-rate coefficient slot from a constant-rate local model.",
    "source": {
        "name": "Standard Monod/Michaelis-Menten saturating reaction-rate benchmark",
        "urls": [
            "synthetic truth generated in script",
            "https://www.nature.com/articles/s41598-020-61174-0",
        ],
        "access_date": "2026-06-13",
    },
    "mu_max": 1.2,
    "k_s": 0.35,
    "yield_y": 0.55,
    "dt": 0.01,
    "t_end": 5.5,
    "stlsq_threshold": 1e-8,
    "stlsq_max_iter": 30,
    "slot_poly_terms": ["1", "S", "S^2", "S^3", "S^4"],
    "gain_s_edges": [0.0, 0.2, 0.5, 0.95, 1.55, 2.0],
    "train_local_initial_conditions": [
        [0.045, 1.10],
        [0.060, 1.30],
        [0.055, 1.55],
    ],
    "train_wide_initial_conditions": [
        [0.050, 0.25],
        [0.065, 0.45],
        [0.040, 0.75],
        [0.055, 1.00],
        [0.050, 1.65],
    ],
    "test_interp_initial_conditions": [
        [0.052, 0.60],
        [0.070, 1.20],
        [0.045, 1.45],
    ],
    "test_extrap_initial_conditions": [
        [0.050, 0.08],
        [0.055, 2.25],
        [0.080, 2.60],
    ],
    "noise": None,
}


def monod_mu(s: np.ndarray | float, mu_max: float, k_s: float) -> np.ndarray | float:
    s_eff = np.maximum(s, 0.0)
    return mu_max * s_eff / (k_s + s_eff + 1e-15)


def true_rhs(_: float, x: np.ndarray, mu_max: float, k_s: float, yield_y: float) -> np.ndarray:
    biomass, substrate = float(x[0]), float(x[1])
    mu = float(monod_mu(substrate, mu_max, k_s))
    growth = mu * biomass
    return np.array([growth, -(1.0 / yield_y) * growth], dtype=float)


def b0_rhs(_: float, x: np.ndarray, mu0: float, yield_y: float) -> np.ndarray:
    growth = mu0 * float(x[0])
    return np.array([growth, -(1.0 / yield_y) * growth], dtype=float)


def lookup_gain(s: float, edges: np.ndarray, gains: np.ndarray) -> float:
    idx = int(np.searchsorted(edges[1:], max(s, 0.0), side="left"))
    idx = min(max(idx, 0), len(gains) - 1)
    return float(gains[idx])


def b0_gain_rhs(_: float, x: np.ndarray, edges: np.ndarray, gains: np.ndarray, yield_y: float) -> np.ndarray:
    mu = lookup_gain(float(x[1]), edges, gains)
    growth = mu * float(x[0])
    return np.array([growth, -(1.0 / yield_y) * growth], dtype=float)


def full_library(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    biomass = x[:, 0]
    substrate = x[:, 1]
    features = [
        np.ones_like(biomass),
        biomass,
        substrate,
        biomass * substrate,
        biomass * substrate**2,
        biomass * substrate**3,
        biomass * substrate**4,
        substrate**2,
        substrate**3,
        substrate**4,
        biomass**2,
        biomass**2 * substrate,
    ]
    names = ["1", "X", "S", "X*S", "X*S^2", "X*S^3", "X*S^4", "S^2", "S^3", "S^4", "X^2", "X^2*S"]
    return np.column_stack(features), names


def poly_slot_library(substrate: np.ndarray) -> np.ndarray:
    return np.column_stack([substrate**degree for degree in range(len(CONFIG["slot_poly_terms"]))])


def poly_mu(substrate: np.ndarray | float, coeffs: np.ndarray) -> np.ndarray | float:
    s = np.asarray(substrate)
    values = np.zeros_like(s, dtype=float)
    for degree, coeff in enumerate(coeffs):
        values = values + coeff * s**degree
    if np.isscalar(substrate):
        return float(values)
    return values


def rational_mu(substrate: np.ndarray | float, params: np.ndarray) -> np.ndarray | float:
    a, b = float(params[0]), float(params[1])
    s = np.maximum(np.asarray(substrate, dtype=float), 0.0)
    values = a * s / (1.0 + b * s + 1e-15)
    if np.isscalar(substrate):
        return float(values)
    return values


def b1_rhs(_: float, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    theta, _ = full_library(x.reshape(1, 2))
    return (theta @ coeffs).reshape(-1)


def b2_rhs(_: float, x: np.ndarray, mu0: float, coeffs: np.ndarray, yield_y: float) -> np.ndarray:
    return b0_rhs(0.0, x, mu0, yield_y) + b1_rhs(0.0, x, coeffs)


def b3_poly_rhs(_: float, x: np.ndarray, coeffs: np.ndarray, yield_y: float) -> np.ndarray:
    growth = float(poly_mu(float(x[1]), coeffs)) * float(x[0])
    return np.array([growth, -(1.0 / yield_y) * growth], dtype=float)


def b5_rational_rhs(_: float, x: np.ndarray, params: np.ndarray, yield_y: float) -> np.ndarray:
    growth = float(rational_mu(float(x[1]), params)) * float(x[0])
    return np.array([growth, -(1.0 / yield_y) * growth], dtype=float)


def b1r_rational_full_rhs(_: float, x: np.ndarray, params: np.ndarray) -> np.ndarray:
    ax, a_s, b = float(params[0]), float(params[1]), float(params[2])
    biomass = float(x[0])
    substrate = max(float(x[1]), 0.0)
    shared = biomass * substrate / (1.0 + b * substrate + 1e-15)
    return np.array([ax * shared, -a_s * shared], dtype=float)


def simulate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    t_end: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(0.0, t_end + 0.5 * dt, dt)
    x = np.zeros((len(t_eval), len(x0)), dtype=float)
    x[0] = x0
    max_abs_state = 50.0
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


def pysindy_stlsq_matrix(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    optimizer = ps.STLSQ(
        threshold=CONFIG["stlsq_threshold"],
        alpha=0.0,
        max_iter=CONFIG["stlsq_max_iter"],
        normalize_columns=False,
    )
    optimizer.fit(a, y)
    return optimizer.coef_.T


def pysindy_stlsq_vector(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    optimizer = ps.STLSQ(
        threshold=CONFIG["stlsq_threshold"],
        alpha=0.0,
        max_iter=CONFIG["stlsq_max_iter"],
        normalize_columns=False,
    )
    optimizer.fit(a, y.reshape(-1, 1))
    return optimizer.coef_.reshape(-1)


def slot_design(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    biomass = x[:, 0]
    substrate = x[:, 1]
    phi = poly_slot_library(substrate)
    a_x = biomass[:, None] * phi
    a_s = -(1.0 / CONFIG["yield_y"]) * biomass[:, None] * phi
    return np.vstack([a_x, a_s]), np.concatenate([biomass, substrate])


def fit_constant_mu(x: np.ndarray, dx: np.ndarray) -> float:
    biomass = x[:, 0]
    a = np.concatenate([biomass, -(1.0 / CONFIG["yield_y"]) * biomass])
    y = np.concatenate([dx[:, 0], dx[:, 1]])
    return float(np.dot(a, y) / np.dot(a, a))


def fit_gain_schedule(x: np.ndarray, dx: np.ndarray, edges: list[float], fallback: float) -> np.ndarray:
    biomass = x[:, 0]
    substrate = x[:, 1]
    edge_array = np.array(edges, dtype=float)
    gains: list[float] = []
    for idx in range(len(edge_array) - 1):
        lo = edge_array[idx]
        hi = edge_array[idx + 1]
        mask = (substrate >= lo) & (substrate <= hi) if idx == 0 else (substrate > lo) & (substrate <= hi)
        a = np.concatenate([biomass[mask], -(1.0 / CONFIG["yield_y"]) * biomass[mask]])
        y = np.concatenate([dx[mask, 0], dx[mask, 1]])
        denom = float(np.dot(a, a))
        if np.any(mask) and denom > 1e-14:
            gains.append(float(np.dot(a, y) / denom))
        else:
            gains.append(fallback)
    return np.array(gains, dtype=float)


def fit_poly_slot(x: np.ndarray, dx: np.ndarray) -> np.ndarray:
    biomass = x[:, 0]
    substrate = x[:, 1]
    phi = poly_slot_library(substrate)
    a = np.vstack([biomass[:, None] * phi, -(1.0 / CONFIG["yield_y"]) * biomass[:, None] * phi])
    y = np.concatenate([dx[:, 0], dx[:, 1]])
    return pysindy_stlsq_vector(a, y)


def fit_rational_slot(x: np.ndarray, dx: np.ndarray) -> np.ndarray:
    biomass = x[:, 0]
    substrate = x[:, 1]
    y = np.concatenate([dx[:, 0], dx[:, 1]])

    def residual(params: np.ndarray) -> np.ndarray:
        mu = rational_mu(substrate, params)
        pred = np.concatenate([biomass * mu, -(1.0 / CONFIG["yield_y"]) * biomass * mu])
        return pred - y

    true_a = CONFIG["mu_max"] / CONFIG["k_s"]
    true_b = 1.0 / CONFIG["k_s"]
    result = least_squares(
        residual,
        x0=np.array([0.8 * true_a, 1.2 * true_b]),
        bounds=([0.0, 0.0], [20.0, 20.0]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=2000,
    )
    if not result.success:
        raise RuntimeError(f"Rational slot optimization failed: {result.message}")
    return result.x


def fit_rational_full_vector_field(x: np.ndarray, dx: np.ndarray) -> np.ndarray:
    biomass = x[:, 0]
    substrate = np.maximum(x[:, 1], 0.0)

    def residual(params: np.ndarray) -> np.ndarray:
        ax, a_s, b = params
        shared = biomass * substrate / (1.0 + b * substrate + 1e-15)
        pred = np.column_stack([ax * shared, -a_s * shared])
        return (pred - dx).reshape(-1)

    result = least_squares(
        residual,
        x0=np.array([1.0, 2.0, 1.0]),
        bounds=([0.0, 0.0, 0.0], [20.0, 40.0, 20.0]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=4000,
    )
    if not result.success:
        raise RuntimeError(f"Rational full-vector-field optimization failed: {result.message}")
    return result.x


def nrmse(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - true
    rmse_state = np.sqrt(np.mean(err * err, axis=0))
    denom_state = np.std(true, axis=0)
    denom_state = np.where(denom_state > 0, denom_state, 1.0)
    all_rmse = float(np.sqrt(np.mean(err * err)))
    all_denom = float(np.std(true)) or 1.0
    return {
        "all": all_rmse / all_denom,
        "biomass": float(rmse_state[0] / denom_state[0]),
        "substrate": float(rmse_state[1] / denom_state[1]),
    }


def coefficient_metrics(poly_coeffs: np.ndarray, rational_params: np.ndarray) -> dict[str, object]:
    grid = np.linspace(0.0, 2.65, 801)
    p_true = monod_mu(grid, CONFIG["mu_max"], CONFIG["k_s"])
    p_poly = poly_mu(grid, poly_coeffs)
    p_rat = rational_mu(grid, rational_params)

    def block(pred: np.ndarray) -> dict[str, float]:
        err = pred - p_true
        return {
            "grid_rmse": float(np.sqrt(np.mean(err * err))),
            "grid_nrmse": float(np.sqrt(np.mean(err * err)) / (np.std(p_true) or 1.0)),
            "max_abs_error": float(np.max(np.abs(err))),
            "grid_min_pred": float(np.min(pred)),
            "grid_max_pred": float(np.max(pred)),
        }

    active_poly = [term for term, coeff in zip(CONFIG["slot_poly_terms"], poly_coeffs) if abs(coeff) >= 1e-6]
    rational_active = [name for name, value in zip(["numerator_S", "denominator_S"], rational_params) if abs(value) >= 1e-6]
    rational_block = block(p_rat)
    return {
        "grid_nrmse": rational_block["grid_nrmse"],
        "support_precision": 1.0 if len(rational_active) == 2 else 0.0,
        "support_recall": 1.0 if len(rational_active) == 2 else 0.0,
        "B3_slot_polynomial": block(p_poly) | {"active_terms": active_poly},
        "B5_sindy_pi_rational_slot": rational_block | {
            "active_terms": rational_active,
            "support_precision": 1.0 if len(rational_active) == 2 else 0.0,
            "support_recall": 1.0 if len(rational_active) == 2 else 0.0,
        },
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


def make_trajectory(x0: list[float], split: str, trajectory_id: str) -> dict[str, object]:
    t, x = simulate(
        lambda tt, xx: true_rhs(tt, xx, CONFIG["mu_max"], CONFIG["k_s"], CONFIG["yield_y"]),
        np.array(x0, dtype=float),
        CONFIG["t_end"],
        CONFIG["dt"],
    )
    dx = np.array([true_rhs(tt, xx, CONFIG["mu_max"], CONFIG["k_s"], CONFIG["yield_y"]) for tt, xx in zip(t, x)])
    return {"trajectory_id": trajectory_id, "split": split, "x0": x0, "t": t, "x": x, "dx": dx}


def stack_data(trajectories: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack([traj["x"] for traj in trajectories]), np.vstack([traj["dx"] for traj in trajectories])


def build_trajectories() -> list[dict[str, object]]:
    trajectories: list[dict[str, object]] = []
    for split, key in [
        ("train_local", "train_local_initial_conditions"),
        ("train_wide", "train_wide_initial_conditions"),
        ("test_interp", "test_interp_initial_conditions"),
        ("test_extrap", "test_extrap_initial_conditions"),
    ]:
        for idx, x0 in enumerate(CONFIG[key]):
            trajectories.append(make_trajectory(x0, split, f"{split}_{idx}"))
    return trajectories


def model_rollout(
    model: str,
    x0: list[float],
    mu0: float,
    gain_edges: np.ndarray,
    gain_values: np.ndarray,
    b1_coeffs: np.ndarray,
    b2_coeffs: np.ndarray,
    b1r_params: np.ndarray,
    b3_coeffs: np.ndarray,
    b5_params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x0_array = np.array(x0, dtype=float)
    if model == "B0_constant_rate":
        return simulate(lambda tt, xx: b0_rhs(tt, xx, mu0, CONFIG["yield_y"]), x0_array, CONFIG["t_end"], CONFIG["dt"])
    if model == "B0_gain_scheduled_lookup":
        return simulate(lambda tt, xx: b0_gain_rhs(tt, xx, gain_edges, gain_values, CONFIG["yield_y"]), x0_array, CONFIG["t_end"], CONFIG["dt"])
    if model == "B1_full_sindy":
        return simulate(lambda tt, xx: b1_rhs(tt, xx, b1_coeffs), x0_array, CONFIG["t_end"], CONFIG["dt"])
    if model == "B2_free_residual_sindy":
        return simulate(lambda tt, xx: b2_rhs(tt, xx, mu0, b2_coeffs, CONFIG["yield_y"]), x0_array, CONFIG["t_end"], CONFIG["dt"])
    if model == "B1R_rational_full_vector_field":
        return simulate(lambda tt, xx: b1r_rational_full_rhs(tt, xx, b1r_params), x0_array, CONFIG["t_end"], CONFIG["dt"])
    if model == "B3_slot_polynomial":
        return simulate(lambda tt, xx: b3_poly_rhs(tt, xx, b3_coeffs, CONFIG["yield_y"]), x0_array, CONFIG["t_end"], CONFIG["dt"])
    if model == "B5_sindy_pi_rational_slot":
        return simulate(lambda tt, xx: b5_rational_rhs(tt, xx, b5_params, CONFIG["yield_y"]), x0_array, CONFIG["t_end"], CONFIG["dt"])
    if model == "B4_oracle_reference":
        return simulate(
            lambda tt, xx: true_rhs(tt, xx, CONFIG["mu_max"], CONFIG["k_s"], CONFIG["yield_y"]),
            x0_array,
            CONFIG["t_end"],
            CONFIG["dt"],
        )
    raise ValueError(f"Unknown model: {model}")


def main() -> None:
    for directory in (DATA_DIR, PROVENANCE_DIR, RESULT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    trajectories = build_trajectories()
    train_local = [traj for traj in trajectories if traj["split"] == "train_local"]
    train_fit = [traj for traj in trajectories if traj["split"] in {"train_local", "train_wide"}]
    x_local, dx_local = stack_data(train_local)
    x_train, dx_train = stack_data(train_fit)

    mu0 = fit_constant_mu(x_local, dx_local)
    gain_edges = np.array(CONFIG["gain_s_edges"], dtype=float)
    gain_values = fit_gain_schedule(x_train, dx_train, CONFIG["gain_s_edges"], mu0)
    theta, term_names = full_library(x_train)
    b1_coeffs = pysindy_stlsq_matrix(theta, dx_train)
    b0_train = np.array([b0_rhs(0.0, xx, mu0, CONFIG["yield_y"]) for xx in x_train])
    b2_coeffs = pysindy_stlsq_matrix(theta, dx_train - b0_train)
    b1r_params = fit_rational_full_vector_field(x_train, dx_train)
    b3_coeffs = fit_poly_slot(x_train, dx_train)
    b5_params = fit_rational_slot(x_train, dx_train)

    trajectory_rows = []
    for traj in trajectories:
        for idx, tt in enumerate(traj["t"]):
            trajectory_rows.append(
                {
                    "trajectory_id": traj["trajectory_id"],
                    "split": traj["split"],
                    "X0": f"{traj['x0'][0]:.16g}",
                    "S0": f"{traj['x0'][1]:.16g}",
                    "t": f"{tt:.10g}",
                    "X": f"{traj['x'][idx, 0]:.16g}",
                    "S": f"{traj['x'][idx, 1]:.16g}",
                    "dX": f"{traj['dx'][idx, 0]:.16g}",
                    "dS": f"{traj['dx'][idx, 1]:.16g}",
                }
            )
    trajectory_path = DATA_DIR / "e5_monod_rate_trajectories.csv"
    write_csv(trajectory_path, trajectory_rows)

    models = [
        "B0_constant_rate",
        "B0_gain_scheduled_lookup",
        "B1_full_sindy",
        "B2_free_residual_sindy",
        "B1R_rational_full_vector_field",
        "B3_slot_polynomial",
        "B5_sindy_pi_rational_slot",
        "B4_oracle_reference",
    ]
    rollout_summary_rows = []
    rollout_sample_rows = []
    split_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    sample_id = "test_extrap_1"
    for traj in trajectories:
        split = str(traj["split"])
        split_values.setdefault(split, {model: {"all": [], "biomass": [], "substrate": []} for model in models})
        true_t = traj["t"]
        true_x = traj["x"]
        for model in models:
            pred_t, pred_x = model_rollout(
                model,
                traj["x0"],
                mu0,
                gain_edges,
                gain_values,
                b1_coeffs,
                b2_coeffs,
                b1r_params,
                b3_coeffs,
                b5_params,
            )
            if len(pred_t) != len(true_t):
                pred = np.column_stack([np.interp(true_t, pred_t, pred_x[:, dim]) for dim in range(2)])
            else:
                pred = pred_x
            metric = nrmse(true_x, pred)
            for key, value in metric.items():
                split_values[split][model][key].append(value)
            rollout_summary_rows.append(
                {
                    "trajectory_id": traj["trajectory_id"],
                    "split": split,
                    "x0": json.dumps(traj["x0"]),
                    "model": model,
                    "nrmse_all": f"{metric['all']:.16g}",
                    "nrmse_biomass": f"{metric['biomass']:.16g}",
                    "nrmse_substrate": f"{metric['substrate']:.16g}",
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
                            "true_X": f"{true_x[idx, 0]:.16g}",
                            "true_S": f"{true_x[idx, 1]:.16g}",
                            "pred_X": f"{pred[idx, 0]:.16g}",
                            "pred_S": f"{pred[idx, 1]:.16g}",
                        }
                    )

    split_metrics = {
        split: {
            model: {
                f"mean_nrmse_{key}": float(np.mean(values))
                for key, values in metric_lists.items()
            }
            | {f"max_nrmse_{key}": float(np.max(values)) for key, values in metric_lists.items()}
            for model, metric_lists in model_values.items()
        }
        for split, model_values in split_values.items()
    }
    rollout_summary_path = RESULT_DIR / "rollout_summary.csv"
    rollout_samples_path = RESULT_DIR / "rollout_samples.csv"
    write_csv(rollout_summary_path, rollout_summary_rows)
    write_csv(rollout_samples_path, rollout_sample_rows)

    coefficients_path = RESULT_DIR / "coefficients.csv"
    coefficient_rows = [
        {
            "model": "B3_slot_polynomial",
            "term": term,
            "coefficient": f"{coeff:.16g}",
            "reference": "",
            "active": str(abs(coeff) >= 1e-6),
        }
        for term, coeff in zip(CONFIG["slot_poly_terms"], b3_coeffs)
    ]
    true_a = CONFIG["mu_max"] / CONFIG["k_s"]
    true_b = 1.0 / CONFIG["k_s"]
    coefficient_rows.extend(
        [
            {
                "model": "B1R_rational_full_vector_field",
                "term": "dX_numerator_XS",
                "coefficient": f"{b1r_params[0]:.16g}",
                "reference": f"{true_a:.16g}",
                "active": str(abs(b1r_params[0]) >= 1e-6),
            },
            {
                "model": "B1R_rational_full_vector_field",
                "term": "dS_numerator_XS",
                "coefficient": f"{b1r_params[1]:.16g}",
                "reference": f"{(1.0 / CONFIG['yield_y']) * true_a:.16g}",
                "active": str(abs(b1r_params[1]) >= 1e-6),
            },
            {
                "model": "B1R_rational_full_vector_field",
                "term": "shared_denominator_S",
                "coefficient": f"{b1r_params[2]:.16g}",
                "reference": f"{true_b:.16g}",
                "active": str(abs(b1r_params[2]) >= 1e-6),
            },
            {
                "model": "B5_sindy_pi_rational_slot",
                "term": "numerator_S",
                "coefficient": f"{b5_params[0]:.16g}",
                "reference": f"{true_a:.16g}",
                "active": str(abs(b5_params[0]) >= 1e-6),
            },
            {
                "model": "B5_sindy_pi_rational_slot",
                "term": "denominator_S",
                "coefficient": f"{b5_params[1]:.16g}",
                "reference": f"{true_b:.16g}",
                "active": str(abs(b5_params[1]) >= 1e-6),
            },
        ]
    )
    write_csv(coefficients_path, coefficient_rows)

    gain_path = RESULT_DIR / "gain_schedule.csv"
    write_csv(
        gain_path,
        [
            {
                "bin": idx,
                "S_low": f"{gain_edges[idx]:.16g}",
                "S_high": f"{gain_edges[idx + 1]:.16g}",
                "gain_value": f"{value:.16g}",
            }
            for idx, value in enumerate(gain_values)
        ],
    )

    model_coeff_rows = []
    for row_idx, term in enumerate(term_names):
        model_coeff_rows.append(
            {
                "model": "B1_full_sindy",
                "term": term,
                "dX_coefficient": f"{b1_coeffs[row_idx, 0]:.16g}",
                "dS_coefficient": f"{b1_coeffs[row_idx, 1]:.16g}",
                "active": str(np.any(np.abs(b1_coeffs[row_idx]) >= 1e-6)),
            }
        )
        model_coeff_rows.append(
            {
                "model": "B2_free_residual_sindy",
                "term": term,
                "dX_coefficient": f"{b2_coeffs[row_idx, 0]:.16g}",
                "dS_coefficient": f"{b2_coeffs[row_idx, 1]:.16g}",
                "active": str(np.any(np.abs(b2_coeffs[row_idx]) >= 1e-6)),
            }
        )
    model_coefficients_path = RESULT_DIR / "model_coefficients.csv"
    write_csv(model_coefficients_path, model_coeff_rows)

    coeff_metrics = coefficient_metrics(b3_coeffs, b5_params)
    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "b0_constant_mu": mu0,
        "b0_gain_schedule": {"S_edges": list(CONFIG["gain_s_edges"]), "values": [float(v) for v in gain_values]},
        "b3_optimizer": "PySINDy STLSQ on polynomial coefficient-slot design matrix",
        "b1r_optimizer": "SciPy least_squares rational full-vector-field loss with independent dX/dS numerators and shared denominator",
        "b5_optimizer": "SciPy least_squares rational coefficient-slot loss; SINDy-PI-style rational/implicit comparison",
        "pysindy_sindy_pi_available": hasattr(ps, "SINDyPI"),
        "true_rational_parameters": {"numerator_S": true_a, "denominator_S": true_b},
        "true_rational_full_vector_field_parameters": {
            "dX_numerator_XS": true_a,
            "dS_numerator_XS": (1.0 / CONFIG["yield_y"]) * true_a,
            "shared_denominator_S": true_b,
        },
        "b3_coefficients": {term: float(coeff) for term, coeff in zip(CONFIG["slot_poly_terms"], b3_coeffs)},
        "b1r_rational_full_vector_field_parameters": {
            "dX_numerator_XS": float(b1r_params[0]),
            "dS_numerator_XS": float(b1r_params[1]),
            "shared_denominator_S": float(b1r_params[2]),
        },
        "b5_rational_parameters": {"numerator_S": float(b5_params[0]), "denominator_S": float(b5_params[1])},
        "active_term_counts": {
            "B0_gain_scheduled_lookup": len(gain_values),
            "B1_full_sindy": int(np.sum(np.abs(b1_coeffs) >= 1e-6)),
            "B2_free_residual_sindy": int(np.sum(np.abs(b2_coeffs) >= 1e-6)),
            "B1R_rational_full_vector_field": int(np.sum(np.abs(b1r_params) >= 1e-6)),
            "B3_slot_polynomial": int(np.sum(np.abs(b3_coeffs) >= 1e-6)),
            "B5_sindy_pi_rational_slot": int(np.sum(np.abs(b5_params) >= 1e-6)),
        },
        "rollout_nrmse_by_split": split_metrics,
        "coefficient_function": coeff_metrics,
    }
    metrics_path = RESULT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sample_rows = [row for row in rollout_sample_rows if row["trajectory_id"] == sample_id]
    sample_t = np.array([float(row["t"]) for row in sample_rows if row["model"] == "B4_oracle_reference"])
    true_x = np.array([float(row["true_X"]) for row in sample_rows if row["model"] == "B4_oracle_reference"])
    styles = {
        "B0_constant_rate": ("Local constant-rate model", "局部常速率模型", "#E69F00", "", 2.4, 0.95),
        "B0_gain_scheduled_lookup": ("Piecewise gain-scheduled rate model", "分段查表速率模型", "#56B4E9", "5 4", 2.4, 0.95),
        "B1_full_sindy": ("Full SINDy vector-field model", "完整向量场 SINDy 模型", "#0072B2", "2 4", 2.3, 0.95),
        "B2_free_residual_sindy": ("Free residual SINDy model", "自由残差 SINDy 模型", "#CC79A7", "10 3 2 3", 2.3, 0.95),
        "B1R_rational_full_vector_field": ("Rational full vector-field model", "有理完整向量场模型", "#6A51A3", "6 3", 2.4, 0.95),
        "B3_slot_polynomial": ("Polynomial coefficient-slot model", "多项式系数槽模型", "#009E73", "12 5", 2.5, 1.0),
        "B5_sindy_pi_rational_slot": ("Rational SINDy-PI-style coefficient-slot model", "有理/SINDy-PI 风格系数槽模型", "#D55E00", "8 3", 2.9, 1.0),
    }
    series = [("Reference Monod model", true_x, "#4D4D4D", "", 3.2, 0.55)]
    series_zh = [("Monod 参考模型", true_x, "#4D4D4D", "", 3.2, 0.55)]
    for model, (label_en, label_zh, color, dash, width, opacity) in styles.items():
        values = np.array([float(row["pred_X"]) for row in sample_rows if row["model"] == model])
        series.append((label_en, values, color, dash, width, opacity))
        series_zh.append((label_zh, values, color, dash, width, opacity))
    write_svg_series(FIGURE_DIR / "rollout_biomass_extrap.svg", "E5 substrate-extrapolation rollout: biomass X", sample_t, series, "t", "X")
    write_svg_series(FIGURE_DIR / "rollout_biomass_extrap_zh.svg", "E5 底物外推 rollout：生物量 X", sample_t, series_zh, "时间 t", "生物量 X")

    grid = np.linspace(0.0, 2.65, 801)
    p_true = monod_mu(grid, CONFIG["mu_max"], CONFIG["k_s"])
    p_gain = np.array([lookup_gain(value, gain_edges, gain_values) for value in grid])
    p_poly = poly_mu(grid, b3_coeffs)
    p_rat = rational_mu(grid, b5_params)
    write_svg_series(
        FIGURE_DIR / "coefficient_function.svg",
        "E5 recovered substrate-dependent growth rate mu(S)",
        grid,
        [
            ("Reference Monod rate", p_true, "#4D4D4D", "", 3.2, 0.55),
            ("Piecewise gain-scheduled rate model", p_gain, "#56B4E9", "5 4", 2.4, 0.95),
            ("Polynomial coefficient-slot model", p_poly, "#009E73", "12 5", 2.5, 1.0),
            ("Rational SINDy-PI-style coefficient-slot model", p_rat, "#D55E00", "8 3", 2.9, 1.0),
        ],
        "S",
        "mu(S)",
    )
    write_svg_series(
        FIGURE_DIR / "coefficient_function_zh.svg",
        "E5 恢复的底物依赖生长速率 mu(S)",
        grid,
        [
            ("Monod 速率参考", p_true, "#4D4D4D", "", 3.2, 0.55),
            ("分段查表速率模型", p_gain, "#56B4E9", "5 4", 2.4, 0.95),
            ("多项式系数槽模型", p_poly, "#009E73", "12 5", 2.5, 1.0),
            ("有理/SINDy-PI 风格系数槽模型", p_rat, "#D55E00", "8 3", 2.9, 1.0),
        ],
        "底物 S",
        "生长速率 mu(S)",
    )

    provenance = {
        "dataset_id": "e5_monod_rate_trajectories",
        "experiment_id": CONFIG["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": CONFIG["source"]["name"],
        "source_urls": CONFIG["source"]["urls"],
        "source_access_date": CONFIG["source"]["access_date"],
        "equation_reference": "Batch biomass-substrate balance with Monod/Michaelis-Menten saturating growth rate.",
        "parameters": {"mu_max": CONFIG["mu_max"], "k_s": CONFIG["k_s"], "yield_y": CONFIG["yield_y"]},
        "initial_conditions": {
            "train_local": CONFIG["train_local_initial_conditions"],
            "train_wide": CONFIG["train_wide_initial_conditions"],
            "test_interp": CONFIG["test_interp_initial_conditions"],
            "test_extrap": CONFIG["test_extrap_initial_conditions"],
        },
        "time_grid": {"t_end": CONFIG["t_end"], "dt": CONFIG["dt"]},
        "noise_model": CONFIG["noise"],
        "candidate_slots": [{"name": "growth_rate", "g": "X", "coefficient_function": "mu(S)"}],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pysindy": ps.__version__,
            "pysindy_sindy_pi_available": hasattr(ps, "SINDyPI"),
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
    provenance_path = PROVENANCE_DIR / "e5_monod_rate_slot_provenance.json"
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
        "b0_constant_mu": mu0,
        "b3_coefficients": metrics["b3_coefficients"],
        "b1r_rational_full_vector_field_parameters": metrics["b1r_rational_full_vector_field_parameters"],
        "b5_rational_parameters": metrics["b5_rational_parameters"],
        "true_rational_parameters": metrics["true_rational_parameters"],
        "b5_coefficient_grid_nrmse": metrics["coefficient_function"]["B5_sindy_pi_rational_slot"]["grid_nrmse"],
        "b3_coefficient_grid_nrmse": metrics["coefficient_function"]["B3_slot_polynomial"]["grid_nrmse"],
        "test_extrap_B0_mean_nrmse": split_metrics["test_extrap"]["B0_constant_rate"]["mean_nrmse_all"],
        "test_extrap_B1R_mean_nrmse": split_metrics["test_extrap"]["B1R_rational_full_vector_field"]["mean_nrmse_all"],
        "test_extrap_B3_mean_nrmse": split_metrics["test_extrap"]["B3_slot_polynomial"]["mean_nrmse_all"],
        "test_extrap_B5_mean_nrmse": split_metrics["test_extrap"]["B5_sindy_pi_rational_slot"]["mean_nrmse_all"],
        "result_dir": str(RESULT_DIR.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
