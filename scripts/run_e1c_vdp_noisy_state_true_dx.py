#!/usr/bin/env python3
"""Run E1c: noisy-state Van der Pol coefficient-slot robustness experiment."""

from __future__ import annotations

import json
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from run_e1b_vdp_multi_mu import (
    B3_FEATURE_NAMES,
    b0_rhs,
    b1_rhs,
    b2_rhs,
    b3_library,
    b3_rhs,
    coefficient_metrics,
    count_active,
    nrmse,
    param_state_library,
    param_state_term_name,
    param_state_terms,
    sha256_file,
    simulate,
    stlsq,
    vdp_rhs,
    vector_stlsq,
    write_csv,
    write_svg_series,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "generated" / "e1_vdp"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "e1c_vdp_noisy_state_true_dx"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG = {
    "experiment_id": "E1c_vdp_noisy_state_true_dx",
    "description": "Noisy-state, clean-derivative robustness test for multi-mu Van der Pol coefficient-slot SINDy.",
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
    "dt": 0.02,
    "t_end": 20.0,
    "noise_levels_relative_state_std": [0.0, 0.005, 0.01, 0.03, 0.05],
    "seeds": list(range(5)),
    "rollout_trajectory_suffix": "_ic0",
    "stlsq_threshold": 1e-3,
    "stlsq_max_iter": 12,
    "param_state_library_max_degree": 4,
}


def make_trajectory(mu: float, x0: list[float], split: str, trajectory_id: str) -> dict[str, object]:
    t, x = simulate(lambda tt, xx: vdp_rhs(tt, xx, mu), np.array(x0, dtype=float), CONFIG["t_end"], CONFIG["dt"])
    dx = np.array([vdp_rhs(tt, xx, mu) for tt, xx in zip(t, x)])
    return {"trajectory_id": trajectory_id, "split": split, "mu": mu, "x0": x0, "t": t, "x": x, "dx": dx}


def build_trajectories() -> list[dict[str, object]]:
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
    return trajectories


def stack_data(trajectories: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.vstack([traj["x"] for traj in trajectories])
    dx = np.vstack([traj["dx"] for traj in trajectories])
    mu = np.concatenate([np.full(len(traj["t"]), traj["mu"]) for traj in trajectories])
    return x, dx, mu


def add_state_noise(x: np.ndarray, noise_level: float, seed: int, state_std: np.ndarray) -> np.ndarray:
    if noise_level == 0.0:
        return x.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_level * state_std, size=x.shape)
    return x + noise


def fit_models(
    x_obs: np.ndarray,
    dx_clean: np.ndarray,
    mu_train: np.ndarray,
    terms: list[tuple[int, int, int]],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    residual_target = dx_clean[:, 1] + x_obs[:, 0]
    x2 = x_obs[:, 1]
    c0 = float(np.dot(x2, residual_target) / np.dot(x2, x2))

    theta = param_state_library(x_obs, mu_train, terms)
    b1_coeffs = vector_stlsq(theta, dx_clean, CONFIG["stlsq_threshold"], CONFIG["stlsq_max_iter"])
    b0_train_rhs = np.array([b0_rhs(0.0, xx, c0) for xx in x_obs])
    b2_coeffs = vector_stlsq(theta, dx_clean - b0_train_rhs, CONFIG["stlsq_threshold"], CONFIG["stlsq_max_iter"])

    slot_library = x2[:, None] * b3_library(x_obs[:, 0], mu_train)
    b3_coeffs = stlsq(slot_library, residual_target, CONFIG["stlsq_threshold"], CONFIG["stlsq_max_iter"])
    return c0, b1_coeffs, b2_coeffs, b3_coeffs


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


def vector_field_nrmse(true_dx: np.ndarray, pred_dx: np.ndarray) -> float:
    err = pred_dx - true_dx
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.std(true_dx)) or 1.0
    return rmse / denom


def rollout_nrmse_for_split(
    trajectories: list[dict[str, object]],
    split: str,
    model_name: str,
    c0: float,
    b3_coeffs: np.ndarray,
) -> tuple[float, int]:
    values = []
    failures = 0
    for traj in trajectories:
        if traj["split"] != split:
            continue
        if not str(traj["trajectory_id"]).endswith(CONFIG["rollout_trajectory_suffix"]):
            continue
        try:
            if model_name == "B0_constant_coefficient":
                _, pred_x = simulate(
                    lambda tt, xx: b0_rhs(tt, xx, c0),
                    np.array(traj["x0"], dtype=float),
                    CONFIG["t_end"],
                    CONFIG["dt"],
                    rtol=1e-7,
                    atol=1e-9,
                )
            elif model_name == "B3_slot_constrained":
                _, pred_x = simulate(
                    lambda tt, xx: b3_rhs(tt, xx, traj["mu"], b3_coeffs),
                    np.array(traj["x0"], dtype=float),
                    CONFIG["t_end"],
                    CONFIG["dt"],
                    rtol=1e-7,
                    atol=1e-9,
                )
            else:
                raise ValueError(f"Unsupported rollout model: {model_name}")
            values.append(nrmse(traj["x"], pred_x)["all"])
        except (RuntimeError, ValueError, FloatingPointError, OverflowError):
            failures += 1
    return (float(np.mean(values)) if values else float("nan")), failures


def write_clean_trajectory_csv(trajectories: list[dict[str, object]]) -> Path:
    rows = []
    for traj in trajectories:
        for idx, tt in enumerate(traj["t"]):
            rows.append(
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
    path = DATA_DIR / "e1c_vdp_clean_reference_trajectories.csv"
    write_csv(path, rows, ["trajectory_id", "split", "mu", "x0_1", "x0_2", "t", "x1", "x2", "dx1", "dx2"])
    return path


def write_seed0_noisy_sample_csv(
    train_trajectories: list[dict[str, object]],
    state_std: np.ndarray,
) -> Path:
    rows = []
    x_clean, _, _ = stack_data(train_trajectories)
    row_offsets = []
    start = 0
    for traj in train_trajectories:
        end = start + len(traj["t"])
        row_offsets.append((traj, start, end))
        start = end
    for noise_level in CONFIG["noise_levels_relative_state_std"]:
        x_obs = add_state_noise(x_clean, noise_level, 0, state_std)
        for traj, start, end in row_offsets:
            local_x_obs = x_obs[start:end]
            for idx, tt in enumerate(traj["t"]):
                rows.append(
                    {
                        "noise_level": f"{noise_level:.16g}",
                        "seed": 0,
                        "trajectory_id": traj["trajectory_id"],
                        "mu": f"{traj['mu']:.16g}",
                        "t": f"{tt:.10g}",
                        "clean_x1": f"{traj['x'][idx, 0]:.16g}",
                        "clean_x2": f"{traj['x'][idx, 1]:.16g}",
                        "observed_x1": f"{local_x_obs[idx, 0]:.16g}",
                        "observed_x2": f"{local_x_obs[idx, 1]:.16g}",
                        "clean_dx1": f"{traj['dx'][idx, 0]:.16g}",
                        "clean_dx2": f"{traj['dx'][idx, 1]:.16g}",
                    }
                )
    path = DATA_DIR / "e1c_vdp_noisy_state_seed0_training_sample.csv"
    write_csv(
        path,
        rows,
        ["noise_level", "seed", "trajectory_id", "mu", "t", "clean_x1", "clean_x2", "observed_x1", "observed_x2", "clean_dx1", "clean_dx2"],
    )
    return path


def aggregate_seed_metrics(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        grouped[float(row["noise_level"])].append(row)

    summary_rows = []
    metric_names = [
        "b3_coefficient_grid_nrmse",
        "b3_support_precision",
        "b3_support_recall",
        "b1_active_terms",
        "b2_active_terms",
        "b3_active_terms",
        "interp_b1_vector_field_nrmse",
        "interp_b2_vector_field_nrmse",
        "interp_b3_vector_field_nrmse",
        "extrap_b1_vector_field_nrmse",
        "extrap_b2_vector_field_nrmse",
        "extrap_b3_vector_field_nrmse",
        "interp_b3_rollout_nrmse",
        "extrap_b3_rollout_nrmse",
    ]
    for noise_level, rows in sorted(grouped.items()):
        summary = {"noise_level": f"{noise_level:.16g}", "n_seeds": len(rows)}
        for metric_name in metric_names:
            values = np.array([float(row[metric_name]) for row in rows], dtype=float)
            summary[f"{metric_name}_mean"] = f"{np.nanmean(values):.16g}"
            summary[f"{metric_name}_std"] = f"{np.nanstd(values):.16g}"
        summary["total_rollout_failures"] = int(sum(int(row["rollout_failures"]) for row in rows))
        summary_rows.append(summary)
    return summary_rows


def main() -> None:
    for directory in (DATA_DIR, PROVENANCE_DIR, RESULT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    trajectories = build_trajectories()
    train_trajectories = [traj for traj in trajectories if traj["split"] == "train"]
    eval_trajectories = [traj for traj in trajectories if traj["split"] != "train"]
    x_train_clean, dx_train_clean, mu_train = stack_data(train_trajectories)
    state_std = np.std(x_train_clean, axis=0)
    terms = param_state_terms(CONFIG["param_state_library_max_degree"])
    term_names = [param_state_term_name(term) for term in terms]

    clean_trajectory_path = write_clean_trajectory_csv(trajectories)
    noisy_sample_path = write_seed0_noisy_sample_csv(train_trajectories, state_std)

    seed_rows = []
    coeff_rows = []
    model_coeff_rows = []
    eval_x_by_split: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in sorted({traj["split"] for traj in eval_trajectories}):
        split_traj = [traj for traj in eval_trajectories if traj["split"] == split]
        eval_x_by_split[split] = stack_data(split_traj)

    for noise_level in CONFIG["noise_levels_relative_state_std"]:
        for seed in CONFIG["seeds"]:
            x_obs = add_state_noise(x_train_clean, noise_level, seed, state_std)
            c0, b1_coeffs, b2_coeffs, b3_coeffs = fit_models(x_obs, dx_train_clean, mu_train, terms)
            coeff_metrics = coefficient_metrics(
                b3_coeffs,
                CONFIG["train_mu"] + CONFIG["interpolation_mu"] + CONFIG["extrapolation_mu"],
            )

            vf_metrics = {}
            for split, (x_eval, dx_eval, mu_eval) in eval_x_by_split.items():
                for model_name, coeff_label in [
                    ("B1_full_sindy", "b1"),
                    ("B2_free_residual_sindy", "b2"),
                    ("B3_slot_constrained", "b3"),
                ]:
                    pred_dx = model_rhs_on_states(model_name, x_eval, mu_eval, c0, b1_coeffs, b2_coeffs, b3_coeffs, terms)
                    vf_metrics[f"{split}_{coeff_label}_vector_field_nrmse"] = vector_field_nrmse(dx_eval, pred_dx)

            interp_b0_rollout, interp_b0_failures = rollout_nrmse_for_split(
                trajectories, "interpolation_mu", "B0_constant_coefficient", c0, b3_coeffs
            )
            interp_b3_rollout, interp_b3_failures = rollout_nrmse_for_split(
                trajectories, "interpolation_mu", "B3_slot_constrained", c0, b3_coeffs
            )
            extrap_b0_rollout, extrap_b0_failures = rollout_nrmse_for_split(
                trajectories, "extrapolation_mu", "B0_constant_coefficient", c0, b3_coeffs
            )
            extrap_b3_rollout, extrap_b3_failures = rollout_nrmse_for_split(
                trajectories, "extrapolation_mu", "B3_slot_constrained", c0, b3_coeffs
            )
            rollout_failures = interp_b0_failures + interp_b3_failures + extrap_b0_failures + extrap_b3_failures

            seed_row = {
                "noise_level": f"{noise_level:.16g}",
                "seed": seed,
                "c0": f"{c0:.16g}",
                "b3_coefficient_grid_nrmse": f"{coeff_metrics['overall_grid_nrmse']:.16g}",
                "b3_support_precision": f"{coeff_metrics['support_precision']:.16g}",
                "b3_support_recall": f"{coeff_metrics['support_recall']:.16g}",
                "b1_active_terms": count_active(b1_coeffs),
                "b2_active_terms": count_active(b2_coeffs),
                "b3_active_terms": count_active(b3_coeffs),
                "interp_b1_vector_field_nrmse": f"{vf_metrics['interpolation_mu_b1_vector_field_nrmse']:.16g}",
                "interp_b2_vector_field_nrmse": f"{vf_metrics['interpolation_mu_b2_vector_field_nrmse']:.16g}",
                "interp_b3_vector_field_nrmse": f"{vf_metrics['interpolation_mu_b3_vector_field_nrmse']:.16g}",
                "extrap_b1_vector_field_nrmse": f"{vf_metrics['extrapolation_mu_b1_vector_field_nrmse']:.16g}",
                "extrap_b2_vector_field_nrmse": f"{vf_metrics['extrapolation_mu_b2_vector_field_nrmse']:.16g}",
                "extrap_b3_vector_field_nrmse": f"{vf_metrics['extrapolation_mu_b3_vector_field_nrmse']:.16g}",
                "interp_b0_rollout_nrmse": f"{interp_b0_rollout:.16g}",
                "interp_b3_rollout_nrmse": f"{interp_b3_rollout:.16g}",
                "extrap_b0_rollout_nrmse": f"{extrap_b0_rollout:.16g}",
                "extrap_b3_rollout_nrmse": f"{extrap_b3_rollout:.16g}",
                "rollout_failures": rollout_failures,
            }
            seed_rows.append(seed_row)

            if seed == 0:
                expected = {"1": 0.0, "x1": 0.0, "x1^2": 0.0, "mu": 1.0, "mu*x1": 0.0, "mu*x1^2": -1.0}
                for feature_name, coeff in zip(B3_FEATURE_NAMES, b3_coeffs):
                    coeff_rows.append(
                        {
                            "noise_level": f"{noise_level:.16g}",
                            "seed": seed,
                            "feature": feature_name,
                            "b3_coefficient": f"{coeff:.16g}",
                            "expected_coefficient": f"{expected[feature_name]:.16g}",
                            "abs_error": f"{abs(coeff - expected[feature_name]):.16g}",
                            "active": str(abs(coeff) >= 1e-6),
                        }
                    )
                for row_idx, term_name in enumerate(term_names):
                    if np.any(np.abs(b1_coeffs[row_idx]) >= 1e-6) or np.any(np.abs(b2_coeffs[row_idx]) >= 1e-6):
                        model_coeff_rows.append(
                            {
                                "noise_level": f"{noise_level:.16g}",
                                "seed": seed,
                                "model": "B1_full_sindy",
                                "term": term_name,
                                "dx1_coefficient": f"{b1_coeffs[row_idx, 0]:.16g}",
                                "dx2_coefficient": f"{b1_coeffs[row_idx, 1]:.16g}",
                            }
                        )
                        model_coeff_rows.append(
                            {
                                "noise_level": f"{noise_level:.16g}",
                                "seed": seed,
                                "model": "B2_free_residual_sindy",
                                "term": term_name,
                                "dx1_coefficient": f"{b2_coeffs[row_idx, 0]:.16g}",
                                "dx2_coefficient": f"{b2_coeffs[row_idx, 1]:.16g}",
                            }
                        )

    summary_rows = aggregate_seed_metrics(seed_rows)
    seed_metrics_path = RESULT_DIR / "seed_metrics.csv"
    summary_path = RESULT_DIR / "summary_by_noise.csv"
    coeff_path = RESULT_DIR / "coefficients_seed0.csv"
    model_coeff_path = RESULT_DIR / "model_coefficients_seed0_active.csv"
    seed_fieldnames = list(seed_rows[0].keys())
    summary_fieldnames = list(summary_rows[0].keys())
    write_csv(seed_metrics_path, seed_rows, seed_fieldnames)
    write_csv(summary_path, summary_rows, summary_fieldnames)
    write_csv(coeff_path, coeff_rows, ["noise_level", "seed", "feature", "b3_coefficient", "expected_coefficient", "abs_error", "active"])
    write_csv(model_coeff_path, model_coeff_rows, ["noise_level", "seed", "model", "term", "dx1_coefficient", "dx2_coefficient"])

    noise_x = np.array([float(row["noise_level"]) for row in summary_rows])
    coeff_nrmse = np.array([float(row["b3_coefficient_grid_nrmse_mean"]) for row in summary_rows])
    support_precision = np.array([float(row["b3_support_precision_mean"]) for row in summary_rows])
    support_recall = np.array([float(row["b3_support_recall_mean"]) for row in summary_rows])
    interp_b3_rollout = np.array([float(row["interp_b3_rollout_nrmse_mean"]) for row in summary_rows])
    write_svg_series(
        FIGURE_DIR / "noise_robustness.svg",
        "E1c noisy-state robustness",
        noise_x,
        [
            ("B3 coefficient-function grid NRMSE", coeff_nrmse, "#009E73", "12 5", 2.8, 1.0),
            ("B3 interpolation rollout NRMSE", interp_b3_rollout, "#0072B2", "2 4", 2.4, 0.95),
            ("B3 support precision", support_precision, "#E69F00", "", 2.4, 0.95),
            ("B3 support recall", support_recall, "#CC79A7", "10 3 2 3", 2.4, 0.95),
        ],
        "relative state-noise level",
        "metric value",
    )
    write_svg_series(
        FIGURE_DIR / "noise_robustness_zh.svg",
        "E1c 带噪状态鲁棒性",
        noise_x,
        [
            ("B3 系数函数网格 NRMSE", coeff_nrmse, "#009E73", "12 5", 2.8, 1.0),
            ("B3 插值 rollout NRMSE", interp_b3_rollout, "#0072B2", "2 4", 2.4, 0.95),
            ("B3 支持集 precision", support_precision, "#E69F00", "", 2.4, 0.95),
            ("B3 支持集 recall", support_recall, "#CC79A7", "10 3 2 3", 2.4, 0.95),
        ],
        "相对状态噪声水平",
        "指标值",
    )

    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "noise_model": {
            "type": "Gaussian state-observation noise on training library inputs",
            "relative_to": "per-state standard deviation over clean training trajectories",
            "levels": CONFIG["noise_levels_relative_state_std"],
            "seeds": CONFIG["seeds"],
            "derivative_target": "clean true derivatives",
        },
        "state_std": {"x1": float(state_std[0]), "x2": float(state_std[1])},
        "stlsq_threshold": CONFIG["stlsq_threshold"],
        "summary_by_noise": summary_rows,
    }
    metrics_path = RESULT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    provenance = {
        "dataset_id": "e1c_vdp_noisy_state_true_dx",
        "experiment_id": CONFIG["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": CONFIG["source"]["name"],
        "source_urls": CONFIG["source"]["urls"],
        "source_access_date": CONFIG["source"]["access_date"],
        "equation_reference": "Van der Pol equation with parameterized damping coefficient p_c(x1, mu)=mu-mu*x1^2.",
        "parameters": {
            "train_mu": CONFIG["train_mu"],
            "interpolation_mu": CONFIG["interpolation_mu"],
            "extrapolation_mu": CONFIG["extrapolation_mu"],
        },
        "initial_conditions": {
            "train": CONFIG["train_initial_conditions"],
            "test": CONFIG["test_initial_conditions"],
        },
        "time_grid": {"t_end": CONFIG["t_end"], "dt": CONFIG["dt"]},
        "noise_model": metrics["noise_model"],
        "candidate_slots": [{"name": "damping", "g": "x2", "coefficient_function": "p_c(x1, mu)"}],
        "expected_coefficients": {"mu": 1.0, "mu*x1^2": -1.0},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "script": str(Path(__file__).relative_to(ROOT)),
        "outputs": {
            "clean_trajectory_csv": str(clean_trajectory_path.relative_to(ROOT)),
            "noisy_sample_csv": str(noisy_sample_path.relative_to(ROOT)),
            "seed_metrics_csv": str(seed_metrics_path.relative_to(ROOT)),
            "summary_by_noise_csv": str(summary_path.relative_to(ROOT)),
            "coefficients_seed0_csv": str(coeff_path.relative_to(ROOT)),
            "model_coefficients_seed0_active_csv": str(model_coeff_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
        },
    }
    provenance_path = PROVENANCE_DIR / "e1c_vdp_noisy_state_true_dx_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    hashes = {
        "clean_trajectory_csv_sha256": sha256_file(clean_trajectory_path),
        "noisy_sample_csv_sha256": sha256_file(noisy_sample_path),
        "seed_metrics_csv_sha256": sha256_file(seed_metrics_path),
        "summary_by_noise_csv_sha256": sha256_file(summary_path),
        "coefficients_seed0_csv_sha256": sha256_file(coeff_path),
        "model_coefficients_seed0_active_csv_sha256": sha256_file(model_coeff_path),
        "metrics_json_sha256": sha256_file(metrics_path),
        "provenance_json_sha256": sha256_file(provenance_path),
    }
    (RESULT_DIR / "hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "experiment_id": CONFIG["experiment_id"],
                "noise_levels": CONFIG["noise_levels_relative_state_std"],
                "n_seeds": len(CONFIG["seeds"]),
                "summary_by_noise_csv": str(summary_path.relative_to(ROOT)),
                "result_dir": str(RESULT_DIR.relative_to(ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
