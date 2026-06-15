#!/usr/bin/env python3
"""Run E2c: noisy-state Duffing stiffness-slot optimizer comparison."""

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
    b3_rhs,
    coefficient_metrics,
    count_active,
    duffing_rhs,
    nrmse,
    polynomial_library,
    sha256_file,
    simulate,
    stlsq,
    write_csv,
)
from run_pysindy_e1c_regularization_probe import write_metric_svg


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "generated" / "e2_duffing"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "pysindy_e2c_duffing_noisy_state"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG = {
    "experiment_id": "E2c_duffing_noisy_state_pysindy",
    "description": "Noisy-state Duffing stiffness-slot comparison using PySINDy optimizers.",
    "source_experiment": E2A_CONFIG["experiment_id"],
    "noise_levels_relative_state_std": [0.0, 0.005, 0.01, 0.03, 0.05],
    "seeds": list(range(5)),
    "active_threshold": 1e-6,
    "custom_stlsq_max_iter": E2A_CONFIG["stlsq_max_iter"],
    "pysindy_stlsq_max_iter": 20,
    "pysindy_sr3_max_iter": 1000,
    "pysindy_sr3_relax_coeff_nu": 1.0,
}


EXPECTED_COEFFICIENTS = {0: E2A_CONFIG["kappa"], 1: 0.0, 2: E2A_CONFIG["epsilon"], 3: 0.0, 4: 0.0}


def make_trajectory(x0: list[float], split: str, trajectory_id: str, t_end: float) -> dict[str, object]:
    eta = E2A_CONFIG["eta"]
    kappa = E2A_CONFIG["kappa"]
    epsilon = E2A_CONFIG["epsilon"]
    t, x = simulate(
        lambda tt, xx: duffing_rhs(tt, xx, eta, kappa, epsilon),
        np.array(x0, dtype=float),
        t_end,
        E2A_CONFIG["dt"],
    )
    dx = np.array([duffing_rhs(tt, xx, eta, kappa, epsilon) for tt, xx in zip(t, x)])
    return {"trajectory_id": trajectory_id, "split": split, "x0": x0, "t": t, "x": x, "dx": dx}


def build_trajectories() -> list[dict[str, object]]:
    trajectories: list[dict[str, object]] = []
    for idx, x0 in enumerate(E2A_CONFIG["train_initial_conditions"]):
        trajectories.append(make_trajectory(x0, "train", f"train_ic{idx}", E2A_CONFIG["train_t_end"]))
    for idx, x0 in enumerate(E2A_CONFIG["test_initial_conditions"]):
        trajectories.append(make_trajectory(x0, "test_unseen_ic", f"test_ic{idx}", E2A_CONFIG["rollout_t_end"]))
    return trajectories


def stack_data(trajectories: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([traj["x"] for traj in trajectories])
    dx = np.vstack([traj["dx"] for traj in trajectories])
    return x, dx


def add_state_noise(x: np.ndarray, noise_level: float, seed: int, state_std: np.ndarray) -> np.ndarray:
    if noise_level == 0.0:
        return x.copy()
    rng = np.random.default_rng(seed)
    return x + rng.normal(loc=0.0, scale=noise_level * state_std, size=x.shape)


def optimizer_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "custom_stlsq_t1e-3",
            "family": "custom_stlsq",
            "threshold": 1e-3,
            "alpha": 0.0,
            "regularizer": "",
            "reg_weight_lam": "",
        },
        {
            "name": "pysindy_stlsq_t1e-3",
            "family": "pysindy_stlsq",
            "threshold": 1e-3,
            "alpha": 0.0,
            "regularizer": "",
            "reg_weight_lam": "",
        },
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


def fit_coefficients(spec: dict[str, Any], library: np.ndarray, target: np.ndarray) -> np.ndarray:
    if spec["family"] == "custom_stlsq":
        return stlsq(library, target, float(spec["threshold"]), CONFIG["custom_stlsq_max_iter"])
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


def vector_field_nrmse(true_dx: np.ndarray, pred_dx: np.ndarray) -> float:
    err = pred_dx - true_dx
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.std(true_dx)) or 1.0
    return rmse / denom


def b3_vector_field_nrmse(x_eval: np.ndarray, dx_eval: np.ndarray, coeffs: np.ndarray) -> float:
    degrees = E2A_CONFIG["poly_degrees"]
    eta = E2A_CONFIG["eta"]
    pred_dx = np.array([b3_rhs(0.0, xx, eta, coeffs, degrees) for xx in x_eval])
    return vector_field_nrmse(dx_eval, pred_dx)


def rollout_nrmse(trajectories: list[dict[str, object]], coeffs: np.ndarray) -> tuple[float, int]:
    values = []
    failures = 0
    eta = E2A_CONFIG["eta"]
    degrees = E2A_CONFIG["poly_degrees"]
    for traj in trajectories:
        try:
            _, pred_x = simulate(
                lambda tt, xx: b3_rhs(tt, xx, eta, coeffs, degrees),
                np.array(traj["x0"], dtype=float),
                E2A_CONFIG["rollout_t_end"],
                E2A_CONFIG["dt"],
                rtol=1e-7,
                atol=1e-9,
            )
            values.append(nrmse(traj["x"], pred_x)["all"])
        except (RuntimeError, ValueError, FloatingPointError, OverflowError):
            failures += 1
    return (float(np.mean(values)) if values else float("nan")), failures


def aggregate(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, str], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(float(row["noise_level"]), str(row["optimizer"]))].append(row)

    metric_names = [
        "coefficient_grid_nrmse",
        "support_precision",
        "support_recall",
        "active_terms",
        "spurious_terms",
        "train_vector_field_nrmse",
        "test_vector_field_nrmse",
        "test_rollout_nrmse",
    ]
    spec_by_name = {str(spec["name"]): spec for spec in optimizer_specs()}
    summary_rows = []
    for (noise_level, optimizer), rows in sorted(grouped.items()):
        spec = spec_by_name[optimizer]
        summary = {
            "noise_level": f"{noise_level:.16g}",
            "optimizer": optimizer,
            "family": spec["family"],
            "threshold": spec["threshold"],
            "alpha": spec["alpha"],
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


def series_from_summary(summary_rows: list[dict[str, object]], metric_name: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    noise_levels = sorted({float(row["noise_level"]) for row in summary_rows})
    optimizers = [spec["name"] for spec in optimizer_specs()]
    lookup = {
        (float(row["noise_level"]), str(row["optimizer"])): float(row[f"{metric_name}_mean"])
        for row in summary_rows
    }
    values = {
        optimizer: np.array([lookup[(noise_level, optimizer)] for noise_level in noise_levels], dtype=float)
        for optimizer in optimizers
    }
    return np.array(noise_levels, dtype=float), values


def write_clean_trajectory_csv(trajectories: list[dict[str, object]]) -> Path:
    rows = []
    for traj in trajectories:
        for idx, tt in enumerate(traj["t"]):
            rows.append(
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
    path = DATA_DIR / "e2c_duffing_clean_reference_trajectories.csv"
    write_csv(path, rows, ["trajectory_id", "split", "x0_1", "x0_2", "t", "x1", "x2", "dx1", "dx2"])
    return path


def write_seed0_noisy_sample_csv(train_trajectories: list[dict[str, object]], state_std: np.ndarray) -> Path:
    rows = []
    x_clean, dx_clean = stack_data(train_trajectories)
    offsets = []
    start = 0
    for traj in train_trajectories:
        end = start + len(traj["t"])
        offsets.append((traj, start, end))
        start = end
    for noise_level in CONFIG["noise_levels_relative_state_std"]:
        x_obs = add_state_noise(x_clean, float(noise_level), 0, state_std)
        for traj, start, end in offsets:
            local_x_obs = x_obs[start:end]
            local_dx = dx_clean[start:end]
            for idx, tt in enumerate(traj["t"]):
                rows.append(
                    {
                        "noise_level": f"{float(noise_level):.16g}",
                        "seed": 0,
                        "trajectory_id": traj["trajectory_id"],
                        "t": f"{tt:.10g}",
                        "clean_x1": f"{traj['x'][idx, 0]:.16g}",
                        "clean_x2": f"{traj['x'][idx, 1]:.16g}",
                        "observed_x1": f"{local_x_obs[idx, 0]:.16g}",
                        "observed_x2": f"{local_x_obs[idx, 1]:.16g}",
                        "clean_dx1": f"{local_dx[idx, 0]:.16g}",
                        "clean_dx2": f"{local_dx[idx, 1]:.16g}",
                    }
                )
    path = DATA_DIR / "e2c_duffing_noisy_state_seed0_training_sample.csv"
    write_csv(
        path,
        rows,
        ["noise_level", "seed", "trajectory_id", "t", "clean_x1", "clean_x2", "observed_x1", "observed_x2", "clean_dx1", "clean_dx2"],
    )
    return path


def write_figures(summary_rows: list[dict[str, object]]) -> list[Path]:
    styles = {
        "custom_stlsq_t1e-3": ("#0072B2", "", "circle"),
        "pysindy_stlsq_t1e-3": ("#D55E00", "8 3", "square"),
        "pysindy_stlsq_t1e-2": ("#009E73", "6 2 2 2", "triangle"),
        "pysindy_sr3_l0_lam1e-4": ("#CC79A7", "2 4", "circle"),
    }
    labels_en = {
        "custom_stlsq_t1e-3": "Custom STLSQ, threshold=1e-3",
        "pysindy_stlsq_t1e-3": "PySINDy STLSQ, threshold=1e-3",
        "pysindy_stlsq_t1e-2": "PySINDy STLSQ, threshold=1e-2",
        "pysindy_sr3_l0_lam1e-4": "PySINDy SR3 L0, lambda=1e-4",
    }
    labels_zh = {
        "custom_stlsq_t1e-3": "自写 STLSQ，阈值=1e-3",
        "pysindy_stlsq_t1e-3": "PySINDy STLSQ，阈值=1e-3",
        "pysindy_stlsq_t1e-2": "PySINDy STLSQ，阈值=1e-2",
        "pysindy_sr3_l0_lam1e-4": "PySINDy SR3 L0，lambda=1e-4",
    }
    created = []
    for metric, y_label_en, y_label_zh, filename in [
        ("support_precision", "support precision", "支持集 precision", "support_precision"),
        ("active_terms", "active terms", "活跃项数量", "active_terms"),
        ("coefficient_grid_nrmse", "stiffness-function grid NRMSE", "刚度函数网格 NRMSE", "coefficient_grid_nrmse"),
        ("test_rollout_nrmse", "unseen-IC rollout NRMSE", "未见初值 rollout NRMSE", "test_rollout_nrmse"),
    ]:
        noise_x, values = series_from_summary(summary_rows, metric)
        series_en = [(labels_en[name], values[name], *styles[name]) for name in labels_en]
        series_zh = [(labels_zh[name], values[name], *styles[name]) for name in labels_zh]
        en_path = FIGURE_DIR / f"{filename}.svg"
        zh_path = FIGURE_DIR / f"{filename}_zh.svg"
        write_metric_svg(
            en_path,
            f"E2c Duffing noisy-state: {y_label_en}",
            noise_x,
            series_en,
            "relative state-noise level",
            y_label_en,
        )
        write_metric_svg(
            zh_path,
            f"E2c Duffing 带噪状态：{y_label_zh}",
            noise_x,
            series_zh,
            "相对状态噪声水平",
            y_label_zh,
        )
        created.extend([en_path, zh_path])
    return created


def main() -> None:
    for directory in (DATA_DIR, PROVENANCE_DIR, RESULT_DIR, FIGURE_DIR):
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

    clean_trajectory_path = write_clean_trajectory_csv(trajectories)
    noisy_sample_path = write_seed0_noisy_sample_csv(train_trajectories, state_std)

    seed_rows = []
    coefficient_rows = []
    specs = optimizer_specs()
    for noise_level in CONFIG["noise_levels_relative_state_std"]:
        for seed in CONFIG["seeds"]:
            x_obs = add_state_noise(x_train_clean, float(noise_level), int(seed), state_std)
            target = dx_train_clean[:, 1] + eta * x_obs[:, 1]
            library = -x_obs[:, 0, None] * polynomial_library(x_obs[:, 0], degrees)
            for spec in specs:
                coeffs = fit_coefficients(spec, library, target)
                coeff_metrics = coefficient_metrics(coeffs, degrees, kappa, epsilon)
                active_degrees = set(coeff_metrics["active_degrees"])
                spurious_terms = len(active_degrees - {0, 2})
                rollout_value, rollout_failures = rollout_nrmse(test_trajectories, coeffs)
                seed_rows.append(
                    {
                        "noise_level": f"{float(noise_level):.16g}",
                        "seed": int(seed),
                        "optimizer": spec["name"],
                        "family": spec["family"],
                        "threshold": spec["threshold"],
                        "alpha": spec["alpha"],
                        "regularizer": spec["regularizer"],
                        "reg_weight_lam": spec["reg_weight_lam"],
                        "coefficient_grid_nrmse": f"{coeff_metrics['grid_nrmse']:.16g}",
                        "support_precision": f"{coeff_metrics['support_precision']:.16g}",
                        "support_recall": f"{coeff_metrics['support_recall']:.16g}",
                        "active_terms": count_active(coeffs, CONFIG["active_threshold"]),
                        "spurious_terms": spurious_terms,
                        "active_degree_0": str(abs(coeffs[degrees.index(0)]) >= CONFIG["active_threshold"]),
                        "active_degree_2": str(abs(coeffs[degrees.index(2)]) >= CONFIG["active_threshold"]),
                        "train_vector_field_nrmse": f"{b3_vector_field_nrmse(x_train_clean, dx_train_clean, coeffs):.16g}",
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
                                "optimizer": spec["name"],
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
    summary_path = RESULT_DIR / "summary_by_noise_optimizer.csv"
    coefficients_path = RESULT_DIR / "coefficients_seed0.csv"
    metrics_path = RESULT_DIR / "metrics.json"
    provenance_path = PROVENANCE_DIR / "pysindy_e2c_duffing_noisy_state_provenance.json"

    write_csv(seed_metrics_path, seed_rows, list(seed_rows[0].keys()))
    write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    write_csv(coefficients_path, coefficient_rows, list(coefficient_rows[0].keys()))

    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "source_experiment": CONFIG["source_experiment"],
        "noise_model": {
            "type": "Gaussian state-observation noise on training library inputs",
            "relative_to": "per-state standard deviation over clean training trajectories",
            "levels": CONFIG["noise_levels_relative_state_std"],
            "seeds": CONFIG["seeds"],
            "derivative_target": "clean true derivatives",
        },
        "equation": "x1_dot=x2; x2_dot=-eta*x2-x1*p_k(x1); p_k(x1)=kappa+epsilon*x1^2",
        "state_std": {"x1": float(state_std[0]), "x2": float(state_std[1])},
        "optimizer_specs": specs,
        "summary_by_noise_optimizer": summary_rows,
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
        "dataset_id": "pysindy_e2c_duffing_noisy_state",
        "experiment_id": CONFIG["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": E2A_CONFIG["source"]["name"],
        "source_urls": E2A_CONFIG["source"]["urls"],
        "source_access_date": E2A_CONFIG["source"]["access_date"],
        "source_experiment": CONFIG["source_experiment"],
        "source_script": "scripts/run_e2a_duffing_unforced.py",
        "script": str(Path(__file__).relative_to(ROOT)),
        "purpose": "Transfer the E1c validated PySINDy optimizer backends to the Duffing stiffness coefficient slot under state-observation noise.",
        "parameters": {"eta": eta, "kappa": kappa, "epsilon": epsilon},
        "time_grid": {"train_t_end": E2A_CONFIG["train_t_end"], "rollout_t_end": E2A_CONFIG["rollout_t_end"], "dt": E2A_CONFIG["dt"]},
        "initial_conditions": {"train": E2A_CONFIG["train_initial_conditions"], "test": E2A_CONFIG["test_initial_conditions"]},
        "candidate_slots": [{"name": "stiffness", "g": "-x1", "coefficient_function": "p_k(x1)"}],
        "expected_coefficients": EXPECTED_COEFFICIENTS,
        "noise_model": metrics["noise_model"],
        "optimizer_specs": specs,
        "outputs": {
            "clean_trajectory_csv": str(clean_trajectory_path.relative_to(ROOT)),
            "noisy_sample_csv": str(noisy_sample_path.relative_to(ROOT)),
            "seed_metrics_csv": str(seed_metrics_path.relative_to(ROOT)),
            "summary_by_noise_optimizer_csv": str(summary_path.relative_to(ROOT)),
            "coefficients_seed0_csv": str(coefficients_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
            "figures": [str(path.relative_to(ROOT)) for path in figure_paths],
        },
        "software": metrics["software"],
        "dependency_note": metrics["known_dependency_note"],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    hashes = {
        "clean_trajectory_csv_sha256": sha256_file(clean_trajectory_path),
        "noisy_sample_csv_sha256": sha256_file(noisy_sample_path),
        "seed_metrics_csv_sha256": sha256_file(seed_metrics_path),
        "summary_by_noise_optimizer_csv_sha256": sha256_file(summary_path),
        "coefficients_seed0_csv_sha256": sha256_file(coefficients_path),
        "metrics_json_sha256": sha256_file(metrics_path),
        "provenance_json_sha256": sha256_file(provenance_path),
        **{f"figure_{path.stem}_sha256": sha256_file(path) for path in figure_paths},
    }
    hashes_path = RESULT_DIR / "hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "experiment_id": CONFIG["experiment_id"],
                "pysindy_version": ps.__version__,
                "noise_levels": CONFIG["noise_levels_relative_state_std"],
                "n_seeds": len(CONFIG["seeds"]),
                "optimizers": [spec["name"] for spec in specs],
                "summary_by_noise_optimizer_csv": str(summary_path.relative_to(ROOT)),
                "result_dir": str(RESULT_DIR.relative_to(ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
