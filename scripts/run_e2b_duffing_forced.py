#!/usr/bin/env python3
"""Run E2b: forced Duffing stiffness coefficient-function recovery."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from run_e2a_duffing_unforced import (
    count_active,
    nrmse,
    polynomial_library,
    sha256_file,
    stlsq,
    vector_stlsq,
    write_csv,
    write_svg_series,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "generated" / "e2_duffing"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "e2b_duffing_forced"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG = {
    "experiment_id": "E2b_duffing_forced",
    "description": "Forced Duffing stiffness coefficient-function recovery with known forcing.",
    "source": {
        "name": "DORA forced Duffing generator",
        "urls": [
            "https://github.com/maneesh51/Benchmark-Tasks",
            "https://raw.githubusercontent.com/maneesh51/Benchmark-Tasks/main/DORA_generator.py",
        ],
        "commit": "a39f9fd06035c6b8693199c5705acb4b4c9a164f",
        "access_date": "2026-06-10",
        "provenance_note": "Synthetic forced Duffing parameters are taken from the DORA generator; literature and license provenance remain weaker than DSINDy/WSINDy.",
    },
    "eta": 0.3,
    "kappa": -1.0,
    "epsilon": 1.0,
    "omega": 1.5,
    "phi": 0.0,
    "x0": [0.05, 0.05],
    "train_amplitudes": [0.46, 0.49],
    "test_amplitudes": [0.2, 0.35, 0.48, 0.58, 0.75],
    "dt": 0.05,
    "train_t_end": 60.0,
    "rollout_t_end": 80.0,
    "poly_degrees": [0, 1, 2, 3, 4],
    "input_library_max_degree": 3,
    "stlsq_threshold": 1e-8,
    "stlsq_max_iter": 12,
    "noise": None,
}


def forcing(t: float | np.ndarray, amplitude: float, omega: float, phi: float) -> float | np.ndarray:
    return amplitude * np.cos(omega * t + phi)


def duffing_rhs(
    t: float,
    x: np.ndarray,
    eta: float,
    kappa: float,
    epsilon: float,
    amplitude: float,
    omega: float,
    phi: float,
) -> np.ndarray:
    x1, x2 = x
    u = forcing(t, amplitude, omega, phi)
    return np.array([x2, -eta * x2 - kappa * x1 - epsilon * x1**3 + u], dtype=float)


def b0_rhs(t: float, x: np.ndarray, eta: float, k0: float, amplitude: float, omega: float, phi: float) -> np.ndarray:
    x1, x2 = x
    u = forcing(t, amplitude, omega, phi)
    return np.array([x2, -eta * x2 - k0 * x1 + u], dtype=float)


def b1_rhs(t: float, x: np.ndarray, coeffs: np.ndarray, terms: list[tuple[int, int, int]], amplitude: float) -> np.ndarray:
    u = forcing(t, amplitude, CONFIG["omega"], CONFIG["phi"])
    theta = input_polynomial_library(x.reshape(1, 2), np.array([u], dtype=float), terms)[0]
    return theta @ coeffs


def b2_rhs(
    t: float,
    x: np.ndarray,
    eta: float,
    k0: float,
    coeffs: np.ndarray,
    terms: list[tuple[int, int, int]],
    amplitude: float,
) -> np.ndarray:
    u = forcing(t, amplitude, CONFIG["omega"], CONFIG["phi"])
    theta = input_polynomial_library(x.reshape(1, 2), np.array([u], dtype=float), terms)[0]
    return b0_rhs(t, x, eta, k0, amplitude, CONFIG["omega"], CONFIG["phi"]) + theta @ coeffs


def b3_rhs(t: float, x: np.ndarray, eta: float, coeffs: np.ndarray, degrees: list[int], amplitude: float) -> np.ndarray:
    x1, x2 = x
    u = forcing(t, amplitude, CONFIG["omega"], CONFIG["phi"])
    p = sum(coeff * (x1**degree) for coeff, degree in zip(coeffs, degrees))
    return np.array([x2, -eta * x2 - x1 * p + u], dtype=float)


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


def input_polynomial_terms(max_degree: int) -> list[tuple[int, int, int]]:
    terms: list[tuple[int, int, int]] = []
    for total_degree in range(max_degree + 1):
        for x1_power in range(total_degree, -1, -1):
            for x2_power in range(total_degree - x1_power, -1, -1):
                u_power = total_degree - x1_power - x2_power
                terms.append((x1_power, x2_power, u_power))
    return terms


def input_term_name(term: tuple[int, int, int]) -> str:
    names = []
    for label, power in zip(("x1", "x2", "u"), term):
        if power == 1:
            names.append(label)
        elif power > 1:
            names.append(f"{label}^{power}")
    return "*".join(names) if names else "1"


def input_polynomial_library(x: np.ndarray, u: np.ndarray, terms: list[tuple[int, int, int]]) -> np.ndarray:
    x1 = x[:, 0]
    x2 = x[:, 1]
    return np.column_stack([(x1**a) * (x2**b) * (u**c) for a, b, c in terms])


def coefficient_metrics(coeffs: np.ndarray, degrees: list[int], kappa: float, epsilon: float) -> dict[str, object]:
    x_grid = np.linspace(-2.5, 2.5, 601)
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


def vector_field_nrmse(true_dx: np.ndarray, pred_dx: np.ndarray) -> float:
    err = pred_dx - true_dx
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.std(true_dx)) or 1.0
    return rmse / denom


def make_trajectory(amplitude: float, split: str, trajectory_id: str, t_end: float) -> dict[str, object]:
    eta, kappa, epsilon = CONFIG["eta"], CONFIG["kappa"], CONFIG["epsilon"]
    omega, phi = CONFIG["omega"], CONFIG["phi"]
    x0 = np.array(CONFIG["x0"], dtype=float)
    t, x = simulate(
        lambda tt, xx: duffing_rhs(tt, xx, eta, kappa, epsilon, amplitude, omega, phi),
        x0,
        t_end,
        CONFIG["dt"],
    )
    u = forcing(t, amplitude, omega, phi)
    dx = np.array([duffing_rhs(tt, xx, eta, kappa, epsilon, amplitude, omega, phi) for tt, xx in zip(t, x)])
    return {"trajectory_id": trajectory_id, "split": split, "amplitude": amplitude, "t": t, "u": u, "x": x, "dx": dx}


def stack_data(trajectories: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.vstack([traj["x"] for traj in trajectories]),
        np.concatenate([traj["u"] for traj in trajectories]),
        np.vstack([traj["dx"] for traj in trajectories]),
    )


def model_rollout(
    model_name: str,
    amplitude: float,
    eta: float,
    k0: float,
    b1_coeffs: np.ndarray,
    b2_coeffs: np.ndarray,
    b3_coeffs: np.ndarray,
    terms: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.array(CONFIG["x0"], dtype=float)
    omega, phi = CONFIG["omega"], CONFIG["phi"]
    if model_name == "B0_known_forcing_constant_stiffness":
        return simulate(lambda tt, xx: b0_rhs(tt, xx, eta, k0, amplitude, omega, phi), x0, CONFIG["rollout_t_end"], CONFIG["dt"])
    if model_name == "B1_full_sindy_with_input":
        return simulate(lambda tt, xx: b1_rhs(tt, xx, b1_coeffs, terms, amplitude), x0, CONFIG["rollout_t_end"], CONFIG["dt"])
    if model_name == "B2_free_residual_sindy_with_input":
        return simulate(lambda tt, xx: b2_rhs(tt, xx, eta, k0, b2_coeffs, terms, amplitude), x0, CONFIG["rollout_t_end"], CONFIG["dt"])
    if model_name == "B3_slot_constrained_known_forcing":
        return simulate(lambda tt, xx: b3_rhs(tt, xx, eta, b3_coeffs, CONFIG["poly_degrees"], amplitude), x0, CONFIG["rollout_t_end"], CONFIG["dt"])
    if model_name == "B4_oracle_reference":
        return simulate(
            lambda tt, xx: duffing_rhs(tt, xx, eta, CONFIG["kappa"], CONFIG["epsilon"], amplitude, omega, phi),
            x0,
            CONFIG["rollout_t_end"],
            CONFIG["dt"],
        )
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_rhs_on_data(
    model_name: str,
    x: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    amplitude: float,
    eta: float,
    k0: float,
    b1_coeffs: np.ndarray,
    b2_coeffs: np.ndarray,
    b3_coeffs: np.ndarray,
    terms: list[tuple[int, int, int]],
) -> np.ndarray:
    if model_name == "B0_known_forcing_constant_stiffness":
        return np.array([b0_rhs(tt, xx, eta, k0, amplitude, CONFIG["omega"], CONFIG["phi"]) for tt, xx in zip(t, x)])
    if model_name == "B1_full_sindy_with_input":
        theta = input_polynomial_library(x, u, terms)
        return theta @ b1_coeffs
    if model_name == "B2_free_residual_sindy_with_input":
        theta = input_polynomial_library(x, u, terms)
        b0_values = np.array([b0_rhs(tt, xx, eta, k0, amplitude, CONFIG["omega"], CONFIG["phi"]) for tt, xx in zip(t, x)])
        return b0_values + theta @ b2_coeffs
    if model_name == "B3_slot_constrained_known_forcing":
        return np.array([b3_rhs(tt, xx, eta, b3_coeffs, CONFIG["poly_degrees"], amplitude) for tt, xx in zip(t, x)])
    if model_name == "B4_oracle_reference":
        return np.array(
            [
                duffing_rhs(tt, xx, eta, CONFIG["kappa"], CONFIG["epsilon"], amplitude, CONFIG["omega"], CONFIG["phi"])
                for tt, xx in zip(t, x)
            ]
        )
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    for directory in (DATA_DIR, PROVENANCE_DIR, RESULT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    eta, kappa, epsilon = CONFIG["eta"], CONFIG["kappa"], CONFIG["epsilon"]
    omega, phi = CONFIG["omega"], CONFIG["phi"]
    degrees = list(CONFIG["poly_degrees"])
    terms = input_polynomial_terms(CONFIG["input_library_max_degree"])
    term_names = [input_term_name(term) for term in terms]

    train_trajectories = [
        make_trajectory(amplitude, "train", f"train_A{amplitude:.2f}", CONFIG["train_t_end"])
        for amplitude in CONFIG["train_amplitudes"]
    ]
    x_train, u_train, dx_train = stack_data(train_trajectories)

    x1 = x_train[:, 0]
    stiffness_target = -(dx_train[:, 1] + eta * x_train[:, 1] - u_train)
    k0 = float(np.dot(x1, stiffness_target) / np.dot(x1, x1))

    theta = input_polynomial_library(x_train, u_train, terms)
    b1_coeffs = vector_stlsq(theta, dx_train, CONFIG["stlsq_threshold"], CONFIG["stlsq_max_iter"])
    b0_train_rhs = np.array(
        [b0_rhs(0.0, xx, eta, k0, 0.0, omega, phi) + np.array([0.0, uu]) for xx, uu in zip(x_train, u_train)]
    )
    b2_coeffs = vector_stlsq(theta, dx_train - b0_train_rhs, CONFIG["stlsq_threshold"], CONFIG["stlsq_max_iter"])

    slot_library = -x1[:, None] * polynomial_library(x1, degrees)
    b3_coeffs = stlsq(slot_library, dx_train[:, 1] + eta * x_train[:, 1] - u_train, CONFIG["stlsq_threshold"], CONFIG["stlsq_max_iter"])

    test_trajectories = [
        make_trajectory(amplitude, "test_amplitude", f"test_A{amplitude:.2f}", CONFIG["rollout_t_end"])
        for amplitude in CONFIG["test_amplitudes"]
    ]
    train_rollout_trajectories = [
        make_trajectory(amplitude, "train_amplitude_extended", f"train_rollout_A{amplitude:.2f}", CONFIG["rollout_t_end"])
        for amplitude in CONFIG["train_amplitudes"]
    ]
    all_trajectories = train_trajectories + train_rollout_trajectories + test_trajectories

    trajectory_rows = []
    for traj in all_trajectories:
        for idx, tt in enumerate(traj["t"]):
            trajectory_rows.append(
                {
                    "trajectory_id": traj["trajectory_id"],
                    "split": traj["split"],
                    "amplitude": f"{traj['amplitude']:.16g}",
                    "x0_1": f"{CONFIG['x0'][0]:.16g}",
                    "x0_2": f"{CONFIG['x0'][1]:.16g}",
                    "t": f"{tt:.10g}",
                    "u": f"{traj['u'][idx]:.16g}",
                    "x1": f"{traj['x'][idx, 0]:.16g}",
                    "x2": f"{traj['x'][idx, 1]:.16g}",
                    "dx1": f"{traj['dx'][idx, 0]:.16g}",
                    "dx2": f"{traj['dx'][idx, 1]:.16g}",
                }
            )
    trajectory_path = DATA_DIR / "e2b_duffing_forced_trajectories.csv"
    write_csv(
        trajectory_path,
        trajectory_rows,
        ["trajectory_id", "split", "amplitude", "x0_1", "x0_2", "t", "u", "x1", "x2", "dx1", "dx2"],
    )

    models = [
        "B0_known_forcing_constant_stiffness",
        "B1_full_sindy_with_input",
        "B2_free_residual_sindy_with_input",
        "B3_slot_constrained_known_forcing",
        "B4_oracle_reference",
    ]
    rollout_summary_rows = []
    rollout_sample_rows = []
    vector_field_rows = []
    metrics_by_split: dict[str, dict[str, dict[str, list[float]]]] = {}
    vector_field_by_split: dict[str, dict[str, list[float]]] = {}
    sample_id = "test_A0.75"

    for traj in train_rollout_trajectories + test_trajectories:
        true_t = traj["t"]
        true_x = traj["x"]
        true_dx = traj["dx"]
        true_u = traj["u"]
        amplitude = float(traj["amplitude"])
        split = traj["split"]
        metrics_by_split.setdefault(split, {model: {"all": [], "x1": [], "x2": []} for model in models})
        vector_field_by_split.setdefault(split, {model: [] for model in models})

        for model_name in models:
            pred_t, pred_x = model_rollout(model_name, amplitude, eta, k0, b1_coeffs, b2_coeffs, b3_coeffs, terms)
            if len(pred_t) != len(true_t):
                interp_pred = np.column_stack([np.interp(true_t, pred_t, pred_x[:, dim]) for dim in range(2)])
            else:
                interp_pred = pred_x
            model_nrmse = nrmse(true_x, interp_pred)
            for key, value in model_nrmse.items():
                metrics_by_split[split][model_name][key].append(value)
            pred_dx = evaluate_rhs_on_data(
                model_name, true_x, true_u, true_t, amplitude, eta, k0, b1_coeffs, b2_coeffs, b3_coeffs, terms
            )
            vf_nrmse = vector_field_nrmse(true_dx, pred_dx)
            vector_field_by_split[split][model_name].append(vf_nrmse)
            vector_field_rows.append(
                {
                    "trajectory_id": traj["trajectory_id"],
                    "split": split,
                    "amplitude": f"{amplitude:.16g}",
                    "model": model_name,
                    "vector_field_nrmse": f"{vf_nrmse:.16g}",
                }
            )
            rollout_summary_rows.append(
                {
                    "trajectory_id": traj["trajectory_id"],
                    "split": split,
                    "amplitude": f"{amplitude:.16g}",
                    "model": model_name,
                    "nrmse_all": f"{model_nrmse['all']:.16g}",
                    "nrmse_x1": f"{model_nrmse['x1']:.16g}",
                    "nrmse_x2": f"{model_nrmse['x2']:.16g}",
                }
            )
            if traj["trajectory_id"] == sample_id:
                for idx, tt in enumerate(true_t):
                    rollout_sample_rows.append(
                        {
                            "trajectory_id": traj["trajectory_id"],
                            "split": split,
                            "amplitude": f"{amplitude:.16g}",
                            "model": model_name,
                            "t": f"{tt:.10g}",
                            "u": f"{true_u[idx]:.16g}",
                            "true_x1": f"{true_x[idx, 0]:.16g}",
                            "true_x2": f"{true_x[idx, 1]:.16g}",
                            "pred_x1": f"{interp_pred[idx, 0]:.16g}",
                            "pred_x2": f"{interp_pred[idx, 1]:.16g}",
                        }
                    )

    rollout_summary_path = RESULT_DIR / "rollout_summary.csv"
    rollout_samples_path = RESULT_DIR / "rollout_samples.csv"
    vector_field_path = RESULT_DIR / "vector_field_summary.csv"
    write_csv(
        rollout_summary_path,
        rollout_summary_rows,
        ["trajectory_id", "split", "amplitude", "model", "nrmse_all", "nrmse_x1", "nrmse_x2"],
    )
    write_csv(
        rollout_samples_path,
        rollout_sample_rows,
        ["trajectory_id", "split", "amplitude", "model", "t", "u", "true_x1", "true_x2", "pred_x1", "pred_x2"],
    )
    write_csv(vector_field_path, vector_field_rows, ["trajectory_id", "split", "amplitude", "model", "vector_field_nrmse"])

    split_metrics = {}
    for split, model_values in metrics_by_split.items():
        split_metrics[split] = {}
        for model_name, values in model_values.items():
            split_metrics[split][model_name] = {
                f"mean_nrmse_{key}": float(np.mean(value_list)) for key, value_list in values.items()
            } | {
                f"max_nrmse_{key}": float(np.max(value_list)) for key, value_list in values.items()
            }

    vector_field_metrics = {}
    for split, model_values in vector_field_by_split.items():
        vector_field_metrics[split] = {
            model_name: {
                "mean_vector_field_nrmse": float(np.mean(values)),
                "max_vector_field_nrmse": float(np.max(values)),
            }
            for model_name, values in model_values.items()
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
                "model": "B1_full_sindy_with_input",
                "term": term_name,
                "dx1_coefficient": f"{b1_coeffs[row_idx, 0]:.16g}",
                "dx2_coefficient": f"{b1_coeffs[row_idx, 1]:.16g}",
                "active": str(np.any(np.abs(b1_coeffs[row_idx]) >= 1e-6)),
            }
        )
        model_coeff_rows.append(
            {
                "model": "B2_free_residual_sindy_with_input",
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
            "B1_full_sindy_with_input": count_active(b1_coeffs),
            "B2_free_residual_sindy_with_input": count_active(b2_coeffs),
            "B3_slot_constrained_known_forcing": count_active(b3_coeffs),
        },
        "rollout_nrmse_by_split": split_metrics,
        "vector_field_nrmse_by_split": vector_field_metrics,
        "coefficient_function": coefficient_metrics(b3_coeffs, degrees, kappa, epsilon),
        "forcing": {
            "known_in_fixed_model": True,
            "used_inside_coefficient_function": False,
            "omega": omega,
            "phi": phi,
            "train_amplitudes": CONFIG["train_amplitudes"],
            "test_amplitudes": CONFIG["test_amplitudes"],
        },
    }
    metrics_path = RESULT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    sample_rows = [row for row in rollout_sample_rows if row["trajectory_id"] == sample_id]
    sample_t = np.array([float(row["t"]) for row in sample_rows if row["model"] == "B4_oracle_reference"])
    true_x1 = np.array([float(row["true_x1"]) for row in sample_rows if row["model"] == "B4_oracle_reference"])
    series_x1 = [
        ("Reference trajectory from full forced equation", true_x1, "#4D4D4D", "", 3.2, 0.55),
    ]
    series_x1_zh = [
        ("完整受迫方程参考轨迹", true_x1, "#4D4D4D", "", 3.2, 0.55),
    ]
    styles = {
        "B0_known_forcing_constant_stiffness": ("Known-forcing constant-stiffness model", "已知输入的常刚度模型", "#E69F00", "", 2.6, 0.95),
        "B1_full_sindy_with_input": ("Full SINDy vector-field model with input", "含输入的完整向量场 SINDy 模型", "#0072B2", "2 4", 2.4, 0.95),
        "B2_free_residual_sindy_with_input": ("Free residual SINDy model with input", "含输入的自由残差 SINDy 模型", "#CC79A7", "10 3 2 3", 2.4, 0.95),
        "B3_slot_constrained_known_forcing": ("Slot-constrained coefficient SINDy model with known forcing", "已知输入的系数槽约束 SINDy 模型", "#009E73", "12 5", 2.8, 1.0),
    }
    for model_name, (label_en, label_zh, color, dash, width, opacity) in styles.items():
        values = np.array([float(row["pred_x1"]) for row in sample_rows if row["model"] == model_name])
        series_x1.append((label_en, values, color, dash, width, opacity))
        series_x1_zh.append((label_zh, values, color, dash, width, opacity))
    write_svg_series(FIGURE_DIR / "rollout_x1.svg", "E2b forced Duffing rollout: x1, A=0.75", sample_t, series_x1, "t", "x1")
    write_svg_series(FIGURE_DIR / "rollout_x1_zh.svg", "E2b 受迫 Duffing rollout：状态 x1，A=0.75", sample_t, series_x1_zh, "时间 t", "状态 x1")

    x_grid = np.linspace(-2.5, 2.5, 601)
    p_true = kappa + epsilon * x_grid * x_grid
    p_pred = sum(coeff * (x_grid**degree) for coeff, degree in zip(b3_coeffs, degrees))
    write_svg_series(
        FIGURE_DIR / "coefficient_function.svg",
        "E2b recovered stiffness coefficient p_k(x1)",
        x_grid,
        [
            ("Reference stiffness coefficient from full forced equation", p_true, "#4D4D4D", "", 3.2, 0.55),
            ("Slot-constrained coefficient SINDy model with known forcing", p_pred, "#009E73", "12 5", 2.8, 1.0),
        ],
        "x1",
        "p_k(x1)",
    )
    write_svg_series(
        FIGURE_DIR / "coefficient_function_zh.svg",
        "E2b 恢复的刚度系数函数 p_k(x1)",
        x_grid,
        [
            ("完整受迫方程参考刚度系数", p_true, "#4D4D4D", "", 3.2, 0.55),
            ("已知输入的系数槽约束 SINDy 模型", p_pred, "#009E73", "12 5", 2.8, 1.0),
        ],
        "状态 x1",
        "刚度系数 p_k(x1)",
    )

    provenance = {
        "dataset_id": "e2b_duffing_forced_trajectories",
        "experiment_id": CONFIG["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": CONFIG["source"]["name"],
        "source_urls": CONFIG["source"]["urls"],
        "source_commit": CONFIG["source"]["commit"],
        "source_access_date": CONFIG["source"]["access_date"],
        "source_provenance_note": CONFIG["source"]["provenance_note"],
        "equation_reference": "Forced Duffing equation from the DORA generator: q1_dot=q2, q2_dot=-c*q2-k*q1-beta*q1^3+A*cos(omega*t+phi).",
        "parameters": {"eta": eta, "kappa": kappa, "epsilon": epsilon, "omega": omega, "phi": phi},
        "initial_conditions": {"shared_x0": CONFIG["x0"]},
        "forcing": {
            "form": "u(t)=A*cos(omega*t+phi)",
            "train_amplitudes": CONFIG["train_amplitudes"],
            "test_amplitudes": CONFIG["test_amplitudes"],
            "known_in_fixed_model": True,
            "used_inside_coefficient_function": False,
        },
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
            "vector_field_summary_csv": str(vector_field_path.relative_to(ROOT)),
            "coefficients_csv": str(coefficients_path.relative_to(ROOT)),
            "model_coefficients_csv": str(model_coefficients_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
        },
    }
    provenance_path = PROVENANCE_DIR / "e2b_duffing_forced_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    hashes = {
        "trajectory_csv_sha256": sha256_file(trajectory_path),
        "rollout_summary_csv_sha256": sha256_file(rollout_summary_path),
        "rollout_samples_csv_sha256": sha256_file(rollout_samples_path),
        "vector_field_summary_csv_sha256": sha256_file(vector_field_path),
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
        "test_amplitude_B0_mean_nrmse": split_metrics["test_amplitude"]["B0_known_forcing_constant_stiffness"]["mean_nrmse_all"],
        "test_amplitude_B3_mean_nrmse": split_metrics["test_amplitude"]["B3_slot_constrained_known_forcing"]["mean_nrmse_all"],
        "test_amplitude_B1_vector_field_nrmse": vector_field_metrics["test_amplitude"]["B1_full_sindy_with_input"]["mean_vector_field_nrmse"],
        "result_dir": str(RESULT_DIR.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
