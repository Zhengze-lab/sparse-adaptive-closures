#!/usr/bin/env python3
"""Run E1a: fixed-mu Van der Pol coefficient-slot SINDy experiment."""

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
RESULT_DIR = ROOT / "results" / "e1a_vdp_fixed_mu"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG = {
    "experiment_id": "E1a_vdp_fixed_mu",
    "description": "Fixed-mu Van der Pol damping coefficient-function recovery.",
    "source": {
        "name": "DSINDy Table 2",
        "url": "https://www.osti.gov/servlets/purl/2417947",
        "access_date": "2026-06-10",
    },
    "mu": 2.0,
    "x0": [0.0, 1.0],
    "dt": 0.01,
    "train_t_end": 10.0,
    "rollout_t_end": 20.0,
    "poly_degrees": [0, 1, 2, 3, 4],
    "state_library_max_degree": 3,
    "stlsq_threshold": 1e-8,
    "stlsq_max_iter": 10,
    "noise": None,
}


def vdp_rhs(_: float, x: np.ndarray, mu: float) -> np.ndarray:
    x1, x2 = x
    return np.array([x2, -x1 + mu * x2 - mu * x1 * x1 * x2], dtype=float)


def b0_rhs(_: float, x: np.ndarray, c0: float) -> np.ndarray:
    x1, x2 = x
    return np.array([x2, -x1 + c0 * x2], dtype=float)


def b3_rhs(_: float, x: np.ndarray, coeffs: np.ndarray, degrees: list[int]) -> np.ndarray:
    x1, x2 = x
    p = sum(coeff * (x1**degree) for coeff, degree in zip(coeffs, degrees))
    return np.array([x2, -x1 + x2 * p], dtype=float)


def b1_rhs(_: float, x: np.ndarray, coeffs: np.ndarray, terms: list[tuple[int, int]]) -> np.ndarray:
    theta = state_polynomial_library(x.reshape(1, 2), terms)[0]
    return theta @ coeffs


def b2_rhs(
    _: float,
    x: np.ndarray,
    c0: float,
    coeffs: np.ndarray,
    terms: list[tuple[int, int]],
) -> np.ndarray:
    theta = state_polynomial_library(x.reshape(1, 2), terms)[0]
    return b0_rhs(0.0, x, c0) + theta @ coeffs


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
    columns = [stlsq(a, y[:, idx], threshold, max_iter) for idx in range(y.shape[1])]
    return np.column_stack(columns)


def coefficient_matrix_to_dict(
    coeffs: np.ndarray,
    term_names: list[str],
    output_names: list[str],
) -> dict[str, dict[str, float]]:
    return {
        term_name: {
            output_name: float(coeffs[row_idx, col_idx])
            for col_idx, output_name in enumerate(output_names)
        }
        for row_idx, term_name in enumerate(term_names)
    }


def count_active(values: np.ndarray, threshold: float = 1e-6) -> int:
    return int(np.sum(np.abs(values) >= threshold))


def nrmse(true: np.ndarray, pred: np.ndarray) -> dict[str, float | list[float]]:
    err = pred - true
    rmse_state = np.sqrt(np.mean(err * err, axis=0))
    denom_state = np.std(true, axis=0)
    denom_state = np.where(denom_state > 0, denom_state, 1.0)
    per_state = rmse_state / denom_state
    all_rmse = float(np.sqrt(np.mean(err * err)))
    all_denom = float(np.std(true))
    if all_denom <= 0:
        all_denom = 1.0
    return {
        "all": all_rmse / all_denom,
        "x1": float(per_state[0]),
        "x2": float(per_state[1]),
    }


def coefficient_metrics(coeffs: np.ndarray, degrees: list[int], mu: float) -> dict[str, object]:
    x_grid = np.linspace(-3.0, 3.0, 601)
    p_true = mu - mu * x_grid * x_grid
    p_pred = sum(coeff * (x_grid**degree) for coeff, degree in zip(coeffs, degrees))
    rmse = float(np.sqrt(np.mean((p_pred - p_true) ** 2)))
    denom = float(np.std(p_true)) or 1.0
    active = {degree for coeff, degree in zip(coeffs, degrees) if abs(coeff) >= 1e-6}
    expected = {0, 2}
    true_positive = len(active & expected)
    precision = true_positive / len(active) if active else 0.0
    recall = true_positive / len(expected)
    return {
        "grid_rmse": rmse,
        "grid_nrmse": rmse / denom,
        "active_degrees": sorted(active),
        "expected_degrees": sorted(expected),
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


def main() -> None:
    for directory in (DATA_DIR, PROVENANCE_DIR, RESULT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    mu = CONFIG["mu"]
    x0 = np.array(CONFIG["x0"], dtype=float)
    dt = CONFIG["dt"]
    degrees = list(CONFIG["poly_degrees"])
    state_terms = state_polynomial_terms(CONFIG["state_library_max_degree"])
    state_term_names = [state_term_name(term) for term in state_terms]

    t, x_true = simulate(lambda tt, xx: vdp_rhs(tt, xx, mu), x0, CONFIG["rollout_t_end"], dt)
    dx_true = np.array([vdp_rhs(tt, xx, mu) for tt, xx in zip(t, x_true)])
    train_mask = t <= CONFIG["train_t_end"] + 1e-12

    x_train = x_true[train_mask]
    dx_train = dx_true[train_mask]
    residual_target = dx_train[:, 1] + x_train[:, 0]

    x2 = x_train[:, 1]
    c0 = float(np.dot(x2, residual_target) / np.dot(x2, x2))

    state_theta = state_polynomial_library(x_train, state_terms)
    b1_coeffs = vector_stlsq(
        state_theta,
        dx_train,
        CONFIG["stlsq_threshold"],
        CONFIG["stlsq_max_iter"],
    )
    b0_train_rhs = np.array([b0_rhs(0.0, xx, c0) for xx in x_train])
    b2_coeffs = vector_stlsq(
        state_theta,
        dx_train - b0_train_rhs,
        CONFIG["stlsq_threshold"],
        CONFIG["stlsq_max_iter"],
    )

    phi = polynomial_library(x_train[:, 0], degrees)
    slot_library = x2[:, None] * phi
    b3_coeffs = stlsq(
        slot_library,
        residual_target,
        CONFIG["stlsq_threshold"],
        CONFIG["stlsq_max_iter"],
    )

    _, x_b0 = simulate(lambda tt, xx: b0_rhs(tt, xx, c0), x0, CONFIG["rollout_t_end"], dt)
    _, x_b1 = simulate(lambda tt, xx: b1_rhs(tt, xx, b1_coeffs, state_terms), x0, CONFIG["rollout_t_end"], dt)
    _, x_b2 = simulate(lambda tt, xx: b2_rhs(tt, xx, c0, b2_coeffs, state_terms), x0, CONFIG["rollout_t_end"], dt)
    _, x_b3 = simulate(lambda tt, xx: b3_rhs(tt, xx, b3_coeffs, degrees), x0, CONFIG["rollout_t_end"], dt)
    _, x_b4 = simulate(lambda tt, xx: vdp_rhs(tt, xx, mu), x0, CONFIG["rollout_t_end"], dt)

    segments = {
        "train_interval": t <= CONFIG["train_t_end"] + 1e-12,
        "future_interval": t > CONFIG["train_t_end"] + 1e-12,
        "full_rollout": np.ones_like(t, dtype=bool),
    }

    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "b0_constant_damping_c0": c0,
        "b1_full_sindy_coefficients": coefficient_matrix_to_dict(b1_coeffs, state_term_names, ["dx1", "dx2"]),
        "b2_residual_sindy_coefficients": coefficient_matrix_to_dict(b2_coeffs, state_term_names, ["r1", "r2"]),
        "b3_coefficients": {f"x1^{degree}": float(coeff) for degree, coeff in zip(degrees, b3_coeffs)},
        "expected_coefficients": {"x1^0": mu, "x1^2": -mu},
        "active_term_counts": {
            "B1_full_sindy": count_active(b1_coeffs),
            "B2_free_residual_sindy": count_active(b2_coeffs),
            "B3_slot_constrained": count_active(b3_coeffs),
        },
        "rollout_nrmse": {},
        "coefficient_function": coefficient_metrics(b3_coeffs, degrees, mu),
    }
    for segment_name, mask in segments.items():
        metrics["rollout_nrmse"][segment_name] = {
            "B0_constant_coefficient": nrmse(x_true[mask], x_b0[mask]),
            "B1_full_sindy": nrmse(x_true[mask], x_b1[mask]),
            "B2_free_residual_sindy": nrmse(x_true[mask], x_b2[mask]),
            "B3_slot_constrained": nrmse(x_true[mask], x_b3[mask]),
            "B4_oracle_reference": nrmse(x_true[mask], x_b4[mask]),
        }

    trajectory_rows = []
    rollout_rows = []
    for idx, tt in enumerate(t):
        split = "train" if train_mask[idx] else "future"
        trajectory_rows.append(
            {
                "t": f"{tt:.10g}",
                "x1": f"{x_true[idx, 0]:.16g}",
                "x2": f"{x_true[idx, 1]:.16g}",
                "dx1": f"{dx_true[idx, 0]:.16g}",
                "dx2": f"{dx_true[idx, 1]:.16g}",
                "split": split,
            }
        )
        rollout_rows.append(
            {
                "t": f"{tt:.10g}",
                "true_x1": f"{x_true[idx, 0]:.16g}",
                "true_x2": f"{x_true[idx, 1]:.16g}",
                "b0_x1": f"{x_b0[idx, 0]:.16g}",
                "b0_x2": f"{x_b0[idx, 1]:.16g}",
                "b1_x1": f"{x_b1[idx, 0]:.16g}",
                "b1_x2": f"{x_b1[idx, 1]:.16g}",
                "b2_x1": f"{x_b2[idx, 0]:.16g}",
                "b2_x2": f"{x_b2[idx, 1]:.16g}",
                "b3_x1": f"{x_b3[idx, 0]:.16g}",
                "b3_x2": f"{x_b3[idx, 1]:.16g}",
                "b4_x1": f"{x_b4[idx, 0]:.16g}",
                "b4_x2": f"{x_b4[idx, 1]:.16g}",
            }
        )

    trajectory_path = DATA_DIR / "e1a_vdp_fixed_mu_trajectory.csv"
    rollout_path = RESULT_DIR / "rollout.csv"
    write_csv(trajectory_path, trajectory_rows, ["t", "x1", "x2", "dx1", "dx2", "split"])
    write_csv(
        rollout_path,
        rollout_rows,
        [
            "t",
            "true_x1",
            "true_x2",
            "b0_x1",
            "b0_x2",
            "b1_x1",
            "b1_x2",
            "b2_x1",
            "b2_x2",
            "b3_x1",
            "b3_x2",
            "b4_x1",
            "b4_x2",
        ],
    )

    coeff_rows = []
    expected = {0: mu, 2: -mu}
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
    write_csv(
        RESULT_DIR / "coefficients.csv",
        coeff_rows,
        ["term", "degree", "b3_coefficient", "expected_coefficient", "abs_error", "active"],
    )

    model_coeff_rows = []
    for row_idx, term_name in enumerate(state_term_names):
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
                "model": "B3_slot_constrained_pc",
                "term": f"x1^{degree}",
                "dx1_coefficient": "",
                "dx2_coefficient": f"{coeff:.16g}",
                "active": str(abs(coeff) >= 1e-6),
            }
        )
    write_csv(
        RESULT_DIR / "model_coefficients.csv",
        model_coeff_rows,
        ["model", "term", "dx1_coefficient", "dx2_coefficient", "active"],
    )

    metrics_path = RESULT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    provenance = {
        "dataset_id": "e1a_vdp_fixed_mu_trajectory",
        "experiment_id": CONFIG["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": CONFIG["source"]["name"],
        "source_url": CONFIG["source"]["url"],
        "source_access_date": CONFIG["source"]["access_date"],
        "equation_reference": "Van der Pol equation from DSINDy Table 2.",
        "parameters": {"mu": mu},
        "initial_conditions": [CONFIG["x0"]],
        "time_grid": {"start": 0.0, "end": CONFIG["rollout_t_end"], "dt": dt},
        "train_test_split": {"train": [0.0, CONFIG["train_t_end"]], "future_rollout": [CONFIG["train_t_end"], CONFIG["rollout_t_end"]]},
        "noise_model": CONFIG["noise"],
        "candidate_slots": [{"name": "damping", "g": "x2", "coefficient_function": "p_c(x1)"}],
        "expected_coefficients": metrics["expected_coefficients"],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "script": str(Path(__file__).relative_to(ROOT)),
        "outputs": {
            "trajectory_csv": str(trajectory_path.relative_to(ROOT)),
            "rollout_csv": str(rollout_path.relative_to(ROOT)),
            "coefficients_csv": str((RESULT_DIR / "coefficients.csv").relative_to(ROOT)),
            "model_coefficients_csv": str((RESULT_DIR / "model_coefficients.csv").relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
        },
    }

    provenance_path = PROVENANCE_DIR / "e1a_vdp_fixed_mu_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    hashes = {
        "trajectory_csv_sha256": sha256_file(trajectory_path),
        "rollout_csv_sha256": sha256_file(rollout_path),
        "coefficients_csv_sha256": sha256_file(RESULT_DIR / "coefficients.csv"),
        "model_coefficients_csv_sha256": sha256_file(RESULT_DIR / "model_coefficients.csv"),
        "metrics_json_sha256": sha256_file(metrics_path),
        "provenance_json_sha256": sha256_file(provenance_path),
    }
    (RESULT_DIR / "hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    write_svg_series(
        FIGURE_DIR / "rollout_x1.svg",
        "E1a Van der Pol rollout: x1",
        t,
        [
            ("Reference trajectory from full equation", x_true[:, 0], "#4D4D4D", "", 3.2, 0.55),
            ("Simplified constant-coefficient model", x_b0[:, 0], "#E69F00", "", 2.6, 0.95),
            ("Full SINDy vector-field model", x_b1[:, 0], "#0072B2", "2 4", 2.4, 0.95),
            ("Free residual SINDy model", x_b2[:, 0], "#CC79A7", "10 3 2 3", 2.4, 0.95),
            ("Slot-constrained coefficient SINDy model", x_b3[:, 0], "#009E73", "12 5", 2.8, 1.0),
        ],
        "t",
        "x1",
    )
    write_svg_series(
        FIGURE_DIR / "rollout_x1_zh.svg",
        "E1a Van der Pol rollout：状态 x1",
        t,
        [
            ("完整方程参考轨迹", x_true[:, 0], "#4D4D4D", "", 3.2, 0.55),
            ("常系数简化模型", x_b0[:, 0], "#E69F00", "", 2.6, 0.95),
            ("完整向量场 SINDy 模型", x_b1[:, 0], "#0072B2", "2 4", 2.4, 0.95),
            ("自由残差 SINDy 模型", x_b2[:, 0], "#CC79A7", "10 3 2 3", 2.4, 0.95),
            ("系数槽约束 SINDy 模型", x_b3[:, 0], "#009E73", "12 5", 2.8, 1.0),
        ],
        "时间 t",
        "状态 x1",
    )
    write_svg_series(
        FIGURE_DIR / "rollout_x2.svg",
        "E1a Van der Pol rollout: x2",
        t,
        [
            ("Reference trajectory from full equation", x_true[:, 1], "#4D4D4D", "", 3.2, 0.55),
            ("Simplified constant-coefficient model", x_b0[:, 1], "#E69F00", "", 2.6, 0.95),
            ("Full SINDy vector-field model", x_b1[:, 1], "#0072B2", "2 4", 2.4, 0.95),
            ("Free residual SINDy model", x_b2[:, 1], "#CC79A7", "10 3 2 3", 2.4, 0.95),
            ("Slot-constrained coefficient SINDy model", x_b3[:, 1], "#009E73", "12 5", 2.8, 1.0),
        ],
        "t",
        "x2",
    )
    write_svg_series(
        FIGURE_DIR / "rollout_x2_zh.svg",
        "E1a Van der Pol rollout：状态 x2",
        t,
        [
            ("完整方程参考轨迹", x_true[:, 1], "#4D4D4D", "", 3.2, 0.55),
            ("常系数简化模型", x_b0[:, 1], "#E69F00", "", 2.6, 0.95),
            ("完整向量场 SINDy 模型", x_b1[:, 1], "#0072B2", "2 4", 2.4, 0.95),
            ("自由残差 SINDy 模型", x_b2[:, 1], "#CC79A7", "10 3 2 3", 2.4, 0.95),
            ("系数槽约束 SINDy 模型", x_b3[:, 1], "#009E73", "12 5", 2.8, 1.0),
        ],
        "时间 t",
        "状态 x2",
    )
    x_grid = np.linspace(-3.0, 3.0, 601)
    p_true = mu - mu * x_grid * x_grid
    p_pred = sum(coeff * (x_grid**degree) for coeff, degree in zip(b3_coeffs, degrees))
    write_svg_series(
        FIGURE_DIR / "coefficient_function.svg",
        "E1a recovered damping coefficient p_c(x1)",
        x_grid,
        [
            ("Reference damping coefficient from full equation", p_true, "#4D4D4D", "", 3.2, 0.55),
            ("Slot-constrained coefficient SINDy model", p_pred, "#009E73", "12 5", 2.8, 1.0),
        ],
        "x1",
        "p_c(x1)",
    )
    write_svg_series(
        FIGURE_DIR / "coefficient_function_zh.svg",
        "E1a 恢复的阻尼系数函数 p_c(x1)",
        x_grid,
        [
            ("完整方程参考阻尼系数", p_true, "#4D4D4D", "", 3.2, 0.55),
            ("系数槽约束 SINDy 模型", p_pred, "#009E73", "12 5", 2.8, 1.0),
        ],
        "状态 x1",
        "阻尼系数 p_c(x1)",
    )

    summary = {
        "experiment_id": CONFIG["experiment_id"],
        "c0": c0,
        "b3_coefficients": metrics["b3_coefficients"],
        "future_nrmse_B0": metrics["rollout_nrmse"]["future_interval"]["B0_constant_coefficient"]["all"],
        "future_nrmse_B1": metrics["rollout_nrmse"]["future_interval"]["B1_full_sindy"]["all"],
        "future_nrmse_B2": metrics["rollout_nrmse"]["future_interval"]["B2_free_residual_sindy"]["all"],
        "future_nrmse_B3": metrics["rollout_nrmse"]["future_interval"]["B3_slot_constrained"]["all"],
        "coefficient_grid_nrmse": metrics["coefficient_function"]["grid_nrmse"],
        "active_term_counts": metrics["active_term_counts"],
        "result_dir": str(RESULT_DIR.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
