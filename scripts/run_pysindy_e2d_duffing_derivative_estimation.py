#!/usr/bin/env python3
"""Run E2d: Duffing stiffness-slot discovery with estimated derivatives."""

from __future__ import annotations

import json
import platform
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pysindy as ps
import scipy
import sklearn

from run_e2a_duffing_unforced import (
    CONFIG as E2A_CONFIG,
    coefficient_metrics,
    count_active,
    polynomial_library,
    sha256_file,
    write_csv,
)
from run_pysindy_e1c_regularization_probe import write_metric_svg
from run_pysindy_e2c_duffing_noisy_state import (
    EXPECTED_COEFFICIENTS,
    add_state_noise,
    b3_vector_field_nrmse,
    build_trajectories,
    rollout_nrmse,
    stack_data,
    vector_field_nrmse,
)


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "pysindy_e2d_duffing_derivative_estimation"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG = {
    "experiment_id": "E2d_duffing_derivative_estimation_pysindy",
    "description": "Duffing stiffness-slot discovery from noisy states with PySINDy derivative estimation.",
    "source_experiment": "E2c_duffing_noisy_state_pysindy",
    "noise_levels_relative_state_std": [0.0, 0.005, 0.01, 0.03, 0.05],
    "seeds": list(range(5)),
    "active_threshold": 1e-6,
    "pysindy_stlsq_max_iter": 20,
    "pysindy_sr3_max_iter": 1000,
    "pysindy_sr3_relax_coeff_nu": 1.0,
    "smoother_kws": {"window_length": 5, "polyorder": 3, "mode": "interp"},
}


def derivative_specs() -> list[dict[str, Any]]:
    return [
        {"name": "clean_true_derivative", "family": "clean"},
        {"name": "finite_difference", "family": "finite_difference", "order": 2},
        {
            "name": "smoothed_finite_difference",
            "family": "smoothed_finite_difference",
            "order": 2,
            "smoother_kws": CONFIG["smoother_kws"],
        },
    ]


def optimizer_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "pysindy_stlsq_t1e-2",
            "family": "pysindy_stlsq",
            "threshold": 1e-2,
            "alpha": 0.0,
            "regularizer": "",
            "reg_weight_lam": "",
        },
        {
            "name": "pysindy_sr3_l0_lam1e-4",
            "family": "pysindy_sr3",
            "threshold": "",
            "alpha": "",
            "regularizer": "L0",
            "reg_weight_lam": 1e-4,
        },
    ]


def trajectory_offsets(trajectories: list[dict[str, object]]) -> list[tuple[dict[str, object], int, int]]:
    offsets = []
    start = 0
    for traj in trajectories:
        end = start + len(traj["t"])
        offsets.append((traj, start, end))
        start = end
    return offsets


def estimate_derivatives(
    spec: dict[str, Any],
    train_trajectories: list[dict[str, object]],
    x_obs: np.ndarray,
) -> np.ndarray:
    if spec["family"] == "clean":
        return np.vstack([traj["dx"] for traj in train_trajectories])
    if spec["family"] == "finite_difference":
        differentiator = ps.FiniteDifference(order=int(spec["order"]))
    elif spec["family"] == "smoothed_finite_difference":
        differentiator = ps.SmoothedFiniteDifference(order=int(spec["order"]), smoother_kws=dict(spec["smoother_kws"]))
    else:
        raise ValueError(f"Unknown derivative family: {spec['family']}")

    dx_est = np.zeros_like(x_obs)
    for traj, start, end in trajectory_offsets(train_trajectories):
        dx_est[start:end] = differentiator._differentiate(x_obs[start:end], traj["t"])
    return dx_est


def fit_coefficients(spec: dict[str, Any], library: np.ndarray, target: np.ndarray) -> np.ndarray:
    if spec["family"] == "pysindy_stlsq":
        optimizer = ps.STLSQ(
            threshold=float(spec["threshold"]),
            alpha=float(spec["alpha"]),
            max_iter=CONFIG["pysindy_stlsq_max_iter"],
            normalize_columns=False,
        )
    elif spec["family"] == "pysindy_sr3":
        optimizer = ps.SR3(
            reg_weight_lam=float(spec["reg_weight_lam"]),
            regularizer=str(spec["regularizer"]),
            relax_coeff_nu=CONFIG["pysindy_sr3_relax_coeff_nu"],
            max_iter=CONFIG["pysindy_sr3_max_iter"],
            normalize_columns=False,
        )
    else:
        raise ValueError(f"Unknown optimizer family: {spec['family']}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        optimizer.fit(library, target.reshape(-1, 1))
    return optimizer.coef_.reshape(-1)


def scalar_nrmse(true: np.ndarray, pred: np.ndarray) -> float:
    err = pred - true
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.std(true)) or 1.0
    return rmse / denom


def aggregate(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(float(row["noise_level"]), str(row["derivative_method"]), str(row["optimizer"]))].append(row)
    metric_names = [
        "derivative_vector_nrmse",
        "derivative_dx2_nrmse",
        "coefficient_grid_nrmse",
        "support_precision",
        "support_recall",
        "active_terms",
        "spurious_terms",
        "test_vector_field_nrmse",
        "test_rollout_nrmse",
    ]
    spec_by_name = {str(spec["name"]): spec for spec in optimizer_specs()}
    summary_rows = []
    for (noise_level, derivative_method, optimizer), rows in sorted(grouped.items()):
        spec = spec_by_name[optimizer]
        summary = {
            "noise_level": f"{noise_level:.16g}",
            "derivative_method": derivative_method,
            "optimizer": optimizer,
            "optimizer_family": spec["family"],
            "threshold": spec["threshold"],
            "regularizer": spec["regularizer"],
            "reg_weight_lam": spec["reg_weight_lam"],
            "n_seeds": len(rows),
        }
        for metric_name in metric_names:
            values = np.array([float(row[metric_name]) for row in rows], dtype=float)
            summary[f"{metric_name}_mean"] = f"{np.nanmean(values):.16g}"
            summary[f"{metric_name}_std"] = f"{np.nanstd(values):.16g}"
        summary["total_rollout_failures"] = int(sum(int(row["rollout_failures"]) for row in rows))
        summary_rows.append(summary)
    return summary_rows


def series_from_summary(
    summary_rows: list[dict[str, object]],
    metric_name: str,
    optimizer: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    noise_levels = sorted({float(row["noise_level"]) for row in summary_rows})
    methods = [spec["name"] for spec in derivative_specs()]
    lookup = {
        (float(row["noise_level"]), str(row["derivative_method"]), str(row["optimizer"])): float(row[f"{metric_name}_mean"])
        for row in summary_rows
    }
    values = {
        method: np.array([lookup[(noise_level, method, optimizer)] for noise_level in noise_levels], dtype=float)
        for method in methods
    }
    return np.array(noise_levels, dtype=float), values


def write_figures(summary_rows: list[dict[str, object]]) -> list[Path]:
    styles = {
        "clean_true_derivative": ("#0072B2", "", "circle"),
        "finite_difference": ("#D55E00", "8 3", "square"),
        "smoothed_finite_difference": ("#009E73", "6 2 2 2", "triangle"),
    }
    labels_en = {
        "clean_true_derivative": "Clean true derivative + PySINDy STLSQ, threshold=1e-2",
        "finite_difference": "Finite difference + PySINDy STLSQ, threshold=1e-2",
        "smoothed_finite_difference": "Smoothed finite difference + PySINDy STLSQ, threshold=1e-2",
    }
    labels_zh = {
        "clean_true_derivative": "干净真导数 + PySINDy STLSQ，阈值=1e-2",
        "finite_difference": "有限差分导数 + PySINDy STLSQ，阈值=1e-2",
        "smoothed_finite_difference": "平滑有限差分导数 + PySINDy STLSQ，阈值=1e-2",
    }
    created = []
    for metric, y_label_en, y_label_zh, filename in [
        ("derivative_dx2_nrmse", "dx2 derivative NRMSE", "dx2 导数 NRMSE", "derivative_dx2_nrmse"),
        ("coefficient_grid_nrmse", "stiffness-function grid NRMSE", "刚度函数网格 NRMSE", "coefficient_grid_nrmse"),
        ("support_precision", "support precision", "支持集 precision", "support_precision"),
        ("test_rollout_nrmse", "unseen-IC rollout NRMSE", "未见初值 rollout NRMSE", "test_rollout_nrmse"),
    ]:
        noise_x, values = series_from_summary(summary_rows, metric, "pysindy_stlsq_t1e-2")
        series_en = [(labels_en[name], values[name], *styles[name]) for name in labels_en]
        series_zh = [(labels_zh[name], values[name], *styles[name]) for name in labels_zh]
        en_path = FIGURE_DIR / f"{filename}.svg"
        zh_path = FIGURE_DIR / f"{filename}_zh.svg"
        write_metric_svg(en_path, f"E2d Duffing derivative estimation: {y_label_en}", noise_x, series_en, "relative state-noise level", y_label_en)
        write_metric_svg(zh_path, f"E2d Duffing 导数估计：{y_label_zh}", noise_x, series_zh, "相对状态噪声水平", y_label_zh)
        created.extend([en_path, zh_path])
    return created


def main() -> None:
    for directory in (PROVENANCE_DIR, RESULT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    trajectories = build_trajectories()
    train_trajectories = [traj for traj in trajectories if traj["split"] == "train"]
    test_trajectories = [traj for traj in trajectories if traj["split"] == "test_unseen_ic"]
    x_train_clean, dx_train_clean = stack_data(train_trajectories)
    x_test_clean, dx_test_clean = stack_data(test_trajectories)
    state_std = np.std(x_train_clean, axis=0)
    degrees = list(E2A_CONFIG["poly_degrees"])
    eta = E2A_CONFIG["eta"]
    kappa = E2A_CONFIG["kappa"]
    epsilon = E2A_CONFIG["epsilon"]
    specs = optimizer_specs()
    deriv_specs = derivative_specs()

    seed_rows = []
    coefficient_rows = []
    for noise_level in CONFIG["noise_levels_relative_state_std"]:
        for seed in CONFIG["seeds"]:
            x_obs = add_state_noise(x_train_clean, float(noise_level), int(seed), state_std)
            slot_library = -x_obs[:, 0, None] * polynomial_library(x_obs[:, 0], degrees)
            for deriv_spec in deriv_specs:
                dx_est = estimate_derivatives(deriv_spec, train_trajectories, x_obs)
                fit_target = dx_est[:, 1] + eta * x_obs[:, 1]
                finite_mask = np.isfinite(fit_target) & np.all(np.isfinite(slot_library), axis=1)
                for opt_spec in specs:
                    coeffs = fit_coefficients(opt_spec, slot_library[finite_mask], fit_target[finite_mask])
                    coeff_metrics = coefficient_metrics(coeffs, degrees, kappa, epsilon)
                    active_degrees = set(coeff_metrics["active_degrees"])
                    spurious_terms = len(active_degrees - {0, 2})
                    rollout_value, rollout_failures = rollout_nrmse(test_trajectories, coeffs)
                    seed_rows.append(
                        {
                            "noise_level": f"{float(noise_level):.16g}",
                            "seed": int(seed),
                            "derivative_method": deriv_spec["name"],
                            "optimizer": opt_spec["name"],
                            "fit_rows": int(np.sum(finite_mask)),
                            "derivative_vector_nrmse": f"{vector_field_nrmse(dx_train_clean, dx_est):.16g}",
                            "derivative_dx2_nrmse": f"{scalar_nrmse(dx_train_clean[:, 1], dx_est[:, 1]):.16g}",
                            "coefficient_grid_nrmse": f"{coeff_metrics['grid_nrmse']:.16g}",
                            "support_precision": f"{coeff_metrics['support_precision']:.16g}",
                            "support_recall": f"{coeff_metrics['support_recall']:.16g}",
                            "active_terms": count_active(coeffs, CONFIG["active_threshold"]),
                            "spurious_terms": spurious_terms,
                            "test_vector_field_nrmse": f"{b3_vector_field_nrmse(x_test_clean, dx_test_clean, coeffs):.16g}",
                            "test_rollout_nrmse": f"{rollout_value:.16g}",
                            "rollout_failures": rollout_failures,
                        }
                    )
                    if int(seed) == 0:
                        for degree, coeff in zip(degrees, coeffs):
                            expected = EXPECTED_COEFFICIENTS[degree]
                            coefficient_rows.append(
                                {
                                    "noise_level": f"{float(noise_level):.16g}",
                                    "seed": int(seed),
                                    "derivative_method": deriv_spec["name"],
                                    "optimizer": opt_spec["name"],
                                    "degree": degree,
                                    "feature": f"x1^{degree}" if degree > 1 else ("x1" if degree == 1 else "1"),
                                    "coefficient": f"{coeff:.16g}",
                                    "expected_coefficient": f"{expected:.16g}",
                                    "abs_error": f"{abs(coeff - expected):.16g}",
                                    "active": str(abs(coeff) >= CONFIG["active_threshold"]),
                                }
                            )

    summary_rows = aggregate(seed_rows)
    figure_paths = write_figures(summary_rows)

    seed_metrics_path = RESULT_DIR / "seed_metrics.csv"
    summary_path = RESULT_DIR / "summary_by_noise_derivative_optimizer.csv"
    coefficients_path = RESULT_DIR / "coefficients_seed0.csv"
    metrics_path = RESULT_DIR / "metrics.json"
    provenance_path = PROVENANCE_DIR / "pysindy_e2d_duffing_derivative_estimation_provenance.json"

    write_csv(seed_metrics_path, seed_rows, list(seed_rows[0].keys()))
    write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    write_csv(coefficients_path, coefficient_rows, list(coefficient_rows[0].keys()))

    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "source_experiment": CONFIG["source_experiment"],
        "noise_model": {
            "type": "Gaussian state-observation noise on training states used for libraries and derivative estimation",
            "relative_to": "per-state standard deviation over clean training trajectories",
            "levels": CONFIG["noise_levels_relative_state_std"],
            "seeds": CONFIG["seeds"],
        },
        "derivative_methods": deriv_specs,
        "optimizer_specs": specs,
        "state_std": {"x1": float(state_std[0]), "x2": float(state_std[1])},
        "summary_by_noise_derivative_optimizer": summary_rows,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pysindy": ps.__version__,
        },
        "known_dependency_note": "PySINDy 2.1.0 required numpy>=2.0; local qpth 0.0.18 declares numpy<2 but is not used by these experiments.",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    provenance = {
        "dataset_id": "pysindy_e2d_duffing_derivative_estimation",
        "experiment_id": CONFIG["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment": CONFIG["source_experiment"],
        "source_script": "scripts/run_pysindy_e2c_duffing_noisy_state.py",
        "script": str(Path(__file__).relative_to(ROOT)),
        "purpose": "Estimate derivatives from noisy Duffing states and test the stiffness coefficient slot without clean derivative targets.",
        "equation_reference": "Duffing stiffness slot p_k(x1)=kappa+epsilon*x1^2 in dx2=-eta*x2-x1*p_k.",
        "parameters": {"eta": eta, "kappa": kappa, "epsilon": epsilon},
        "candidate_slots": [{"name": "stiffness", "g": "-x1", "coefficient_function": "p_k(x1)"}],
        "expected_coefficients": EXPECTED_COEFFICIENTS,
        "noise_model": metrics["noise_model"],
        "derivative_methods": deriv_specs,
        "optimizer_specs": specs,
        "outputs": {
            "seed_metrics_csv": str(seed_metrics_path.relative_to(ROOT)),
            "summary_by_noise_derivative_optimizer_csv": str(summary_path.relative_to(ROOT)),
            "coefficients_seed0_csv": str(coefficients_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
            "figures": [str(path.relative_to(ROOT)) for path in figure_paths],
        },
        "software": metrics["software"],
        "dependency_note": metrics["known_dependency_note"],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    hashes = {
        "seed_metrics_csv_sha256": sha256_file(seed_metrics_path),
        "summary_by_noise_derivative_optimizer_csv_sha256": sha256_file(summary_path),
        "coefficients_seed0_csv_sha256": sha256_file(coefficients_path),
        "metrics_json_sha256": sha256_file(metrics_path),
        "provenance_json_sha256": sha256_file(provenance_path),
        **{f"figure_{path.stem}_sha256": sha256_file(path) for path in figure_paths},
    }
    (RESULT_DIR / "hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "experiment_id": CONFIG["experiment_id"],
                "pysindy_version": ps.__version__,
                "noise_levels": CONFIG["noise_levels_relative_state_std"],
                "n_seeds": len(CONFIG["seeds"]),
                "derivative_methods": [spec["name"] for spec in deriv_specs],
                "optimizers": [spec["name"] for spec in specs],
                "summary_by_noise_derivative_optimizer_csv": str(summary_path.relative_to(ROOT)),
                "result_dir": str(RESULT_DIR.relative_to(ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
