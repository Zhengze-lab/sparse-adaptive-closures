#!/usr/bin/env python3
"""Run E1e/E2e: Savitzky-Golay derivative-estimation parameter scans."""

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

import run_pysindy_e1d_vdp_derivative_estimation as e1d
import run_pysindy_e2d_duffing_derivative_estimation as e2d
from run_e1b_vdp_multi_mu import B3_FEATURE_NAMES, b3_library, count_active, sha256_file, write_csv
from run_e2a_duffing_unforced import CONFIG as E2A_CONFIG, polynomial_library
from run_pysindy_e1c_regularization_probe import write_metric_svg


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_ROOTS = {
    "E1e": ROOT / "results" / "pysindy_e1e_vdp_savgol_scan",
    "E2e": ROOT / "results" / "pysindy_e2e_duffing_savgol_scan",
}


CONFIG = {
    "experiment_id": "E1e_E2e_savgol_parameter_scan",
    "description": "Savitzky-Golay derivative-estimation parameter scan for coefficient-slot SINDy.",
    "source_experiments": ["E1d_vdp_derivative_estimation_pysindy", "E2d_duffing_derivative_estimation_pysindy"],
    "noise_levels_relative_state_std": [0.0, 0.005, 0.01, 0.03, 0.05],
    "seeds": list(range(5)),
    "window_lengths": [5, 7, 9, 11, 15, 21, 31],
    "polyorders": [2, 3, 4],
    "finite_difference_order": 2,
    "smoother_mode": "interp",
    "optimizer": {
        "name": "pysindy_stlsq_t1e-2",
        "family": "pysindy_stlsq",
        "threshold": 1e-2,
        "alpha": 0.0,
        "max_iter": 20,
    },
    "active_threshold": 1e-6,
}


EXPECTED_E1 = {"1": 0.0, "x1": 0.0, "x1^2": 0.0, "mu": 1.0, "mu*x1": 0.0, "mu*x1^2": -1.0}


def derivative_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "name": "clean_true_derivative",
            "family": "clean",
            "window_length": "",
            "polyorder": "",
        },
        {
            "name": "finite_difference",
            "family": "finite_difference",
            "order": CONFIG["finite_difference_order"],
            "window_length": "",
            "polyorder": "",
        },
    ]
    for window_length in CONFIG["window_lengths"]:
        for polyorder in CONFIG["polyorders"]:
            if polyorder >= window_length:
                continue
            specs.append(
                {
                    "name": f"savgol_w{window_length}_p{polyorder}",
                    "family": "smoothed_finite_difference",
                    "order": CONFIG["finite_difference_order"],
                    "window_length": window_length,
                    "polyorder": polyorder,
                    "smoother_kws": {
                        "window_length": window_length,
                        "polyorder": polyorder,
                        "mode": CONFIG["smoother_mode"],
                    },
                }
            )
    return specs


def optimizer_spec() -> dict[str, Any]:
    return {
        "name": CONFIG["optimizer"]["name"],
        "family": CONFIG["optimizer"]["family"],
        "threshold": CONFIG["optimizer"]["threshold"],
        "alpha": CONFIG["optimizer"]["alpha"],
    }


def fit_stlsq(library: np.ndarray, target: np.ndarray) -> np.ndarray:
    optimizer = ps.STLSQ(
        threshold=float(CONFIG["optimizer"]["threshold"]),
        alpha=float(CONFIG["optimizer"]["alpha"]),
        max_iter=int(CONFIG["optimizer"]["max_iter"]),
        normalize_columns=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        optimizer.fit(library, target.reshape(-1, 1))
    return optimizer.coef_.reshape(-1)


def scalar_nrmse(true: np.ndarray, pred: np.ndarray) -> float:
    err = pred - true
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.std(true)) or 1.0
    return rmse / denom


def aggregate(rows: list[dict[str, object]], metric_names: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["noise_level"]), str(row["derivative_method"]))].append(row)
    spec_lookup = {spec["name"]: spec for spec in derivative_specs()}
    summary_rows = []
    for (noise_level, derivative_method), group_rows in sorted(grouped.items()):
        spec = spec_lookup[derivative_method]
        summary = {
            "noise_level": f"{noise_level:.16g}",
            "derivative_method": derivative_method,
            "family": spec["family"],
            "window_length": spec.get("window_length", ""),
            "polyorder": spec.get("polyorder", ""),
            "n_seeds": len(group_rows),
        }
        for metric_name in metric_names:
            values = np.array([float(row[metric_name]) for row in group_rows], dtype=float)
            summary[f"{metric_name}_mean"] = f"{np.nanmean(values):.16g}"
            summary[f"{metric_name}_std"] = f"{np.nanstd(values):.16g}"
        summary["total_rollout_failures"] = int(sum(int(row["rollout_failures"]) for row in group_rows))
        summary_rows.append(summary)
    return summary_rows


def reference_or_best_rows(
    system_id: str,
    summary_rows: list[dict[str, object]],
    rollout_metric: str,
) -> list[dict[str, object]]:
    by_noise: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_noise[float(row["noise_level"])].append(row)
    selected = []
    for noise_level, rows in sorted(by_noise.items()):
        lookup = {str(row["derivative_method"]): row for row in rows}
        for method in ["clean_true_derivative", "finite_difference"]:
            row = lookup[method]
            selected.append(best_row_payload(system_id, noise_level, f"{method}_reference", row, rollout_metric))
        savgol_rows = [row for row in rows if str(row["family"]) == "smoothed_finite_difference"]
        for objective, metric_name in [
            ("best_savgol_by_rollout", rollout_metric),
            ("best_savgol_by_coefficient", "coefficient_grid_nrmse"),
            ("best_savgol_by_derivative", "derivative_dx2_nrmse"),
        ]:
            row = min(savgol_rows, key=lambda item: float(item[f"{metric_name}_mean"]))
            selected.append(best_row_payload(system_id, noise_level, objective, row, rollout_metric, metric_name))
    return selected


def best_row_payload(
    system_id: str,
    noise_level: float,
    objective: str,
    row: dict[str, object],
    rollout_metric: str,
    objective_metric: str | None = None,
) -> dict[str, object]:
    objective_metric = objective_metric or rollout_metric
    return {
        "system": system_id,
        "noise_level": f"{noise_level:.16g}",
        "objective": objective,
        "objective_metric": objective_metric,
        "derivative_method": row["derivative_method"],
        "family": row["family"],
        "window_length": row["window_length"],
        "polyorder": row["polyorder"],
        "objective_metric_value": row[f"{objective_metric}_mean"],
        "derivative_dx2_nrmse_mean": row["derivative_dx2_nrmse_mean"],
        "coefficient_grid_nrmse_mean": row["coefficient_grid_nrmse_mean"],
        "support_precision_mean": row["support_precision_mean"],
        "support_recall_mean": row["support_recall_mean"],
        f"{rollout_metric}_mean": row[f"{rollout_metric}_mean"],
    }


def summary_value(
    summary_rows: list[dict[str, object]],
    noise_level: float,
    method: str,
    metric: str,
) -> float:
    for row in summary_rows:
        if float(row["noise_level"]) == noise_level and str(row["derivative_method"]) == method:
            return float(row[f"{metric}_mean"])
    raise KeyError((noise_level, method, metric))


def write_high_noise_figures(
    system_label_en: str,
    system_label_zh: str,
    summary_rows: list[dict[str, object]],
    result_dir: Path,
    rollout_metric: str,
    rollout_label_en: str,
    rollout_label_zh: str,
) -> list[Path]:
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    high_noise = max(float(row["noise_level"]) for row in summary_rows)
    windows = np.array(CONFIG["window_lengths"], dtype=float)
    styles = {
        2: ("#0072B2", "", "circle"),
        3: ("#D55E00", "8 3", "square"),
        4: ("#009E73", "6 2 2 2", "triangle"),
        "finite": ("#6A3D9A", "4 3", "diamond"),
        "clean": ("#666666", "2 3", "cross"),
    }
    created: list[Path] = []
    for metric, y_label_en, y_label_zh, filename in [
        ("derivative_dx2_nrmse", "dx2 derivative NRMSE", "dx2 导数 NRMSE", "high_noise_derivative_dx2_nrmse"),
        ("coefficient_grid_nrmse", "coefficient-function grid NRMSE", "系数函数网格 NRMSE", "high_noise_coefficient_grid_nrmse"),
        (rollout_metric, rollout_label_en, rollout_label_zh, "high_noise_rollout_nrmse"),
    ]:
        series_en = []
        series_zh = []
        for polyorder in CONFIG["polyorders"]:
            values = np.array(
                [summary_value(summary_rows, high_noise, f"savgol_w{int(window)}_p{polyorder}", metric) for window in windows],
                dtype=float,
            )
            color, dash, marker = styles[polyorder]
            series_en.append((f"Savitzky-Golay polyorder={polyorder}", values, color, dash, marker))
            series_zh.append((f"Savitzky-Golay 多项式阶数={polyorder}", values, color, dash, marker))
        finite_value = summary_value(summary_rows, high_noise, "finite_difference", metric)
        clean_value = summary_value(summary_rows, high_noise, "clean_true_derivative", metric)
        color, dash, marker = styles["finite"]
        series_en.append(("Finite difference reference", np.full_like(windows, finite_value), color, dash, marker))
        series_zh.append(("有限差分参考", np.full_like(windows, finite_value), color, dash, marker))
        color, dash, marker = styles["clean"]
        series_en.append(("Clean derivative reference", np.full_like(windows, clean_value), color, dash, marker))
        series_zh.append(("干净真导数参考", np.full_like(windows, clean_value), color, dash, marker))
        en_path = figure_dir / f"{filename}.svg"
        zh_path = figure_dir / f"{filename}_zh.svg"
        write_metric_svg(
            en_path,
            f"{system_label_en} SG scan at noise={high_noise:g}: {y_label_en}",
            windows,
            series_en,
            "Savitzky-Golay window length",
            y_label_en,
        )
        write_metric_svg(
            zh_path,
            f"{system_label_zh} SG 扫描，噪声={high_noise:g}：{y_label_zh}",
            windows,
            series_zh,
            "Savitzky-Golay 窗口长度",
            y_label_zh,
        )
        created.extend([en_path, zh_path])
    return created


def write_outputs(
    system_id: str,
    result_dir: Path,
    seed_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    best_rows: list[dict[str, object]],
    coefficient_rows: list[dict[str, object]],
    figure_paths: list[Path],
    metadata: dict[str, object],
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    seed_path = result_dir / "seed_metrics.csv"
    summary_path = result_dir / "summary_by_noise_savgol.csv"
    best_path = result_dir / "best_by_noise.csv"
    coefficients_path = result_dir / "coefficients_seed0.csv"
    metrics_path = result_dir / "metrics.json"
    provenance_path = PROVENANCE_DIR / f"{metadata['dataset_id']}_provenance.json"
    write_csv(seed_path, seed_rows, list(seed_rows[0].keys()))
    write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    write_csv(best_path, best_rows, list(best_rows[0].keys()))
    write_csv(coefficients_path, coefficient_rows, list(coefficient_rows[0].keys()))
    metrics = {
        "experiment_id": metadata["experiment_id"],
        "system": system_id,
        "source_experiment": metadata["source_experiment"],
        "scan_config": CONFIG,
        "summary_by_noise_savgol": summary_rows,
        "best_by_noise": best_rows,
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
        "dataset_id": metadata["dataset_id"],
        "experiment_id": metadata["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": system_id,
        "source_experiment": metadata["source_experiment"],
        "source_script": metadata["source_script"],
        "script": str(Path(__file__).relative_to(ROOT)),
        "purpose": metadata["purpose"],
        "equation_reference": metadata["equation_reference"],
        "candidate_slots": metadata["candidate_slots"],
        "scan_config": CONFIG,
        "outputs": {
            "seed_metrics_csv": str(seed_path.relative_to(ROOT)),
            "summary_by_noise_savgol_csv": str(summary_path.relative_to(ROOT)),
            "best_by_noise_csv": str(best_path.relative_to(ROOT)),
            "coefficients_seed0_csv": str(coefficients_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
            "figures": [str(path.relative_to(ROOT)) for path in figure_paths],
        },
        "software": metrics["software"],
        "dependency_note": metrics["known_dependency_note"],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    hashes = {
        "seed_metrics_csv_sha256": sha256_file(seed_path),
        "summary_by_noise_savgol_csv_sha256": sha256_file(summary_path),
        "best_by_noise_csv_sha256": sha256_file(best_path),
        "coefficients_seed0_csv_sha256": sha256_file(coefficients_path),
        "metrics_json_sha256": sha256_file(metrics_path),
        "provenance_json_sha256": sha256_file(provenance_path),
        **{f"figure_{path.stem}_sha256": sha256_file(path) for path in figure_paths},
    }
    (result_dir / "hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def run_e1e() -> dict[str, object]:
    result_dir = RESULT_ROOTS["E1e"]
    (result_dir / "figures").mkdir(parents=True, exist_ok=True)
    trajectories = e1d.build_trajectories()
    train_trajectories = [traj for traj in trajectories if traj["split"] == "train"]
    x_train_clean, dx_train_clean, mu_train = e1d.stack_data(train_trajectories)
    state_std = np.std(x_train_clean, axis=0)
    eval_data = e1d.split_eval_data(trajectories)
    specs = derivative_specs()
    metric_names = [
        "derivative_vector_nrmse",
        "derivative_dx2_nrmse",
        "coefficient_grid_nrmse",
        "support_precision",
        "support_recall",
        "active_terms",
        "spurious_terms",
        "interp_vector_field_nrmse",
        "extrap_vector_field_nrmse",
        "interp_rollout_nrmse",
        "extrap_rollout_nrmse",
    ]
    seed_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for noise_level in CONFIG["noise_levels_relative_state_std"]:
        for seed in CONFIG["seeds"]:
            x_obs = e1d.add_state_noise(x_train_clean, float(noise_level), int(seed), state_std)
            slot_library = x_obs[:, 1, None] * b3_library(x_obs[:, 0], mu_train)
            for deriv_spec in specs:
                dx_est = e1d.estimate_derivatives(deriv_spec, train_trajectories, x_obs)
                fit_target = dx_est[:, 1] + x_obs[:, 0]
                finite_mask = np.isfinite(fit_target) & np.all(np.isfinite(slot_library), axis=1)
                coeffs = fit_stlsq(slot_library[finite_mask], fit_target[finite_mask])
                coeff_metrics = e1d.coefficient_metrics(
                    coeffs,
                    e1d.E1C_CONFIG["train_mu"] + e1d.E1C_CONFIG["interpolation_mu"] + e1d.E1C_CONFIG["extrapolation_mu"],
                )
                active_features = set(coeff_metrics["active_features"])
                spurious_terms = len(active_features - {"mu", "mu*x1^2"})
                interp_x, interp_dx, interp_mu = eval_data["interpolation_mu"]
                extrap_x, extrap_dx, extrap_mu = eval_data["extrapolation_mu"]
                interp_rollout, interp_failures = e1d.rollout_nrmse_for_split(trajectories, "interpolation_mu", coeffs)
                extrap_rollout, extrap_failures = e1d.rollout_nrmse_for_split(trajectories, "extrapolation_mu", coeffs)
                seed_rows.append(
                    {
                        "noise_level": f"{float(noise_level):.16g}",
                        "seed": int(seed),
                        "derivative_method": deriv_spec["name"],
                        "family": deriv_spec["family"],
                        "window_length": deriv_spec.get("window_length", ""),
                        "polyorder": deriv_spec.get("polyorder", ""),
                        "fit_rows": int(np.sum(finite_mask)),
                        "derivative_vector_nrmse": f"{e1d.vector_field_nrmse(dx_train_clean, dx_est):.16g}",
                        "derivative_dx2_nrmse": f"{scalar_nrmse(dx_train_clean[:, 1], dx_est[:, 1]):.16g}",
                        "coefficient_grid_nrmse": f"{coeff_metrics['overall_grid_nrmse']:.16g}",
                        "support_precision": f"{coeff_metrics['support_precision']:.16g}",
                        "support_recall": f"{coeff_metrics['support_recall']:.16g}",
                        "active_terms": count_active(coeffs, CONFIG["active_threshold"]),
                        "spurious_terms": spurious_terms,
                        "interp_vector_field_nrmse": f"{e1d.b3_vector_field_nrmse(interp_x, interp_dx, interp_mu, coeffs):.16g}",
                        "extrap_vector_field_nrmse": f"{e1d.b3_vector_field_nrmse(extrap_x, extrap_dx, extrap_mu, coeffs):.16g}",
                        "interp_rollout_nrmse": f"{interp_rollout:.16g}",
                        "extrap_rollout_nrmse": f"{extrap_rollout:.16g}",
                        "rollout_failures": interp_failures + extrap_failures,
                    }
                )
                if int(seed) == 0:
                    for feature_name, coeff in zip(B3_FEATURE_NAMES, coeffs):
                        expected = EXPECTED_E1[feature_name]
                        coefficient_rows.append(
                            {
                                "noise_level": f"{float(noise_level):.16g}",
                                "seed": int(seed),
                                "derivative_method": deriv_spec["name"],
                                "family": deriv_spec["family"],
                                "window_length": deriv_spec.get("window_length", ""),
                                "polyorder": deriv_spec.get("polyorder", ""),
                                "feature": feature_name,
                                "coefficient": f"{coeff:.16g}",
                                "expected_coefficient": f"{expected:.16g}",
                                "abs_error": f"{abs(coeff - expected):.16g}",
                                "active": str(abs(coeff) >= CONFIG["active_threshold"]),
                            }
                        )
    summary_rows = aggregate(seed_rows, metric_names)
    best_rows = reference_or_best_rows("E1e_vdp", summary_rows, "interp_rollout_nrmse")
    figure_paths = write_high_noise_figures(
        "E1e Van der Pol",
        "E1e Van der Pol",
        summary_rows,
        result_dir,
        "interp_rollout_nrmse",
        "interpolation rollout NRMSE",
        "插值 rollout NRMSE",
    )
    write_outputs(
        "E1e_vdp",
        result_dir,
        seed_rows,
        summary_rows,
        best_rows,
        coefficient_rows,
        figure_paths,
        {
            "dataset_id": "pysindy_e1e_vdp_savgol_scan",
            "experiment_id": "E1e_vdp_savgol_scan",
            "source_experiment": "E1d_vdp_derivative_estimation_pysindy",
            "source_script": "scripts/run_pysindy_e1d_vdp_derivative_estimation.py",
            "purpose": "Tune Savitzky-Golay derivative-estimation parameters for Van der Pol coefficient-slot discovery.",
            "equation_reference": "Van der Pol coefficient slot p_c(x1, mu)=mu-mu*x1^2 in dx2=-x1+x2*p_c.",
            "candidate_slots": [{"name": "damping", "g": "x2", "coefficient_function": "p_c(x1, mu)"}],
        },
    )
    return {"result_dir": str(result_dir.relative_to(ROOT)), "best_rows": best_rows}


def run_e2e() -> dict[str, object]:
    result_dir = RESULT_ROOTS["E2e"]
    (result_dir / "figures").mkdir(parents=True, exist_ok=True)
    trajectories = e2d.build_trajectories()
    train_trajectories = [traj for traj in trajectories if traj["split"] == "train"]
    test_trajectories = [traj for traj in trajectories if traj["split"] == "test_unseen_ic"]
    x_train_clean, dx_train_clean = e2d.stack_data(train_trajectories)
    x_test_clean, dx_test_clean = e2d.stack_data(test_trajectories)
    state_std = np.std(x_train_clean, axis=0)
    degrees = list(E2A_CONFIG["poly_degrees"])
    eta = E2A_CONFIG["eta"]
    kappa = E2A_CONFIG["kappa"]
    epsilon = E2A_CONFIG["epsilon"]
    specs = derivative_specs()
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
    seed_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for noise_level in CONFIG["noise_levels_relative_state_std"]:
        for seed in CONFIG["seeds"]:
            x_obs = e2d.add_state_noise(x_train_clean, float(noise_level), int(seed), state_std)
            slot_library = -x_obs[:, 0, None] * polynomial_library(x_obs[:, 0], degrees)
            for deriv_spec in specs:
                dx_est = e2d.estimate_derivatives(deriv_spec, train_trajectories, x_obs)
                fit_target = dx_est[:, 1] + eta * x_obs[:, 1]
                finite_mask = np.isfinite(fit_target) & np.all(np.isfinite(slot_library), axis=1)
                coeffs = fit_stlsq(slot_library[finite_mask], fit_target[finite_mask])
                coeff_metrics = e2d.coefficient_metrics(coeffs, degrees, kappa, epsilon)
                active_degrees = set(coeff_metrics["active_degrees"])
                spurious_terms = len(active_degrees - {0, 2})
                rollout_value, rollout_failures = e2d.rollout_nrmse(test_trajectories, coeffs)
                seed_rows.append(
                    {
                        "noise_level": f"{float(noise_level):.16g}",
                        "seed": int(seed),
                        "derivative_method": deriv_spec["name"],
                        "family": deriv_spec["family"],
                        "window_length": deriv_spec.get("window_length", ""),
                        "polyorder": deriv_spec.get("polyorder", ""),
                        "fit_rows": int(np.sum(finite_mask)),
                        "derivative_vector_nrmse": f"{e2d.vector_field_nrmse(dx_train_clean, dx_est):.16g}",
                        "derivative_dx2_nrmse": f"{scalar_nrmse(dx_train_clean[:, 1], dx_est[:, 1]):.16g}",
                        "coefficient_grid_nrmse": f"{coeff_metrics['grid_nrmse']:.16g}",
                        "support_precision": f"{coeff_metrics['support_precision']:.16g}",
                        "support_recall": f"{coeff_metrics['support_recall']:.16g}",
                        "active_terms": count_active(coeffs, CONFIG["active_threshold"]),
                        "spurious_terms": spurious_terms,
                        "test_vector_field_nrmse": f"{e2d.b3_vector_field_nrmse(x_test_clean, dx_test_clean, coeffs):.16g}",
                        "test_rollout_nrmse": f"{rollout_value:.16g}",
                        "rollout_failures": rollout_failures,
                    }
                )
                if int(seed) == 0:
                    for degree, coeff in zip(degrees, coeffs):
                        expected = e2d.EXPECTED_COEFFICIENTS[degree]
                        coefficient_rows.append(
                            {
                                "noise_level": f"{float(noise_level):.16g}",
                                "seed": int(seed),
                                "derivative_method": deriv_spec["name"],
                                "family": deriv_spec["family"],
                                "window_length": deriv_spec.get("window_length", ""),
                                "polyorder": deriv_spec.get("polyorder", ""),
                                "degree": degree,
                                "feature": f"x1^{degree}" if degree > 1 else ("x1" if degree == 1 else "1"),
                                "coefficient": f"{coeff:.16g}",
                                "expected_coefficient": f"{expected:.16g}",
                                "abs_error": f"{abs(coeff - expected):.16g}",
                                "active": str(abs(coeff) >= CONFIG["active_threshold"]),
                            }
                        )
    summary_rows = aggregate(seed_rows, metric_names)
    best_rows = reference_or_best_rows("E2e_duffing", summary_rows, "test_rollout_nrmse")
    figure_paths = write_high_noise_figures(
        "E2e Duffing",
        "E2e Duffing",
        summary_rows,
        result_dir,
        "test_rollout_nrmse",
        "unseen-IC rollout NRMSE",
        "未见初值 rollout NRMSE",
    )
    write_outputs(
        "E2e_duffing",
        result_dir,
        seed_rows,
        summary_rows,
        best_rows,
        coefficient_rows,
        figure_paths,
        {
            "dataset_id": "pysindy_e2e_duffing_savgol_scan",
            "experiment_id": "E2e_duffing_savgol_scan",
            "source_experiment": "E2d_duffing_derivative_estimation_pysindy",
            "source_script": "scripts/run_pysindy_e2d_duffing_derivative_estimation.py",
            "purpose": "Tune Savitzky-Golay derivative-estimation parameters for Duffing stiffness-slot discovery.",
            "equation_reference": "Duffing stiffness slot p_k(x1)=kappa+epsilon*x1^2 in dx2=-eta*x2-x1*p_k.",
            "candidate_slots": [{"name": "stiffness", "g": "-x1", "coefficient_function": "p_k(x1)"}],
        },
    )
    return {"result_dir": str(result_dir.relative_to(ROOT)), "best_rows": best_rows}


def main() -> None:
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    for result_dir in RESULT_ROOTS.values():
        result_dir.mkdir(parents=True, exist_ok=True)
    e1_result = run_e1e()
    e2_result = run_e2e()
    print(
        json.dumps(
            {
                "experiment_id": CONFIG["experiment_id"],
                "pysindy_version": ps.__version__,
                "noise_levels": CONFIG["noise_levels_relative_state_std"],
                "seeds": CONFIG["seeds"],
                "window_lengths": CONFIG["window_lengths"],
                "polyorders": CONFIG["polyorders"],
                "optimizer": CONFIG["optimizer"],
                "results": {"E1e": e1_result["result_dir"], "E2e": e2_result["result_dir"]},
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
