#!/usr/bin/env python3
"""Probe PySINDy on E1b by reproducing B2 and backing B3 optimizers."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pysindy as ps
import scipy
import sklearn

from run_e1b_vdp_multi_mu import (
    B3_FEATURE_NAMES,
    CONFIG as E1B_CONFIG,
    b0_rhs,
    b2_rhs,
    b3_library,
    b3_rhs,
    coefficient_metrics,
    make_trajectory,
    param_state_library,
    param_state_term_name,
    param_state_terms,
    sha256_file,
    stlsq,
    vector_stlsq,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "pysindy_e1b_b2_b3_probe"


CONFIG = {
    "experiment_id": "PYS2_PYS3_e1b_b2_b3_probe",
    "description": "PySINDy reproduction of E1b B2 residual SINDy and B3 coefficient-slot optimizer backend.",
    "source_experiment": "E1b_vdp_multi_mu",
    "library_degree": E1B_CONFIG["param_state_library_max_degree"],
    "pysindy_stlsq_threshold": 1e-8,
    "pysindy_stlsq_alpha": 0.0,
    "pysindy_stlsq_max_iter": 20,
    "pysindy_sr3_reg_weight_lam": 1e-8,
    "pysindy_sr3_relax_coeff_nu": 1.0,
    "pysindy_sr3_max_iter": 1000,
}


def stack_data(trajectories: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.vstack([traj["x"] for traj in trajectories])
    dx = np.vstack([traj["dx"] for traj in trajectories])
    mu = np.concatenate([np.full(len(traj["t"]), traj["mu"]) for traj in trajectories])
    return x, dx, mu


def build_trajectories() -> list[dict[str, object]]:
    trajectories: list[dict[str, object]] = []
    for mu in E1B_CONFIG["train_mu"]:
        for idx, x0 in enumerate(E1B_CONFIG["train_initial_conditions"]):
            trajectories.append(make_trajectory(mu, x0, "train", f"train_mu{mu:g}_ic{idx}"))
    for mu in E1B_CONFIG["train_mu"]:
        for idx, x0 in enumerate(E1B_CONFIG["test_initial_conditions"]):
            trajectories.append(make_trajectory(mu, x0, "seen_mu_unseen_ic", f"seen_mu{mu:g}_ic{idx}"))
    for mu in E1B_CONFIG["interpolation_mu"]:
        for idx, x0 in enumerate(E1B_CONFIG["test_initial_conditions"]):
            trajectories.append(make_trajectory(mu, x0, "interpolation_mu", f"interp_mu{mu:g}_ic{idx}"))
    for mu in E1B_CONFIG["extrapolation_mu"]:
        for idx, x0 in enumerate(E1B_CONFIG["test_initial_conditions"]):
            trajectories.append(make_trajectory(mu, x0, "extrapolation_mu", f"extrap_mu{mu:g}_ic{idx}"))
    return trajectories


def fit_constant_b0(x_train: np.ndarray, dx_train: np.ndarray) -> float:
    residual_target = dx_train[:, 1] + x_train[:, 0]
    x2 = x_train[:, 1]
    return float(np.dot(x2, residual_target) / np.dot(x2, x2))


def fit_custom_b2(
    x_train: np.ndarray,
    dx_train: np.ndarray,
    mu_train: np.ndarray,
    c0: float,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    terms = param_state_terms(E1B_CONFIG["param_state_library_max_degree"])
    theta = param_state_library(x_train, mu_train, terms)
    b0_train_rhs = np.array([b0_rhs(0.0, xx, c0) for xx in x_train])
    coeffs = vector_stlsq(theta, dx_train - b0_train_rhs, E1B_CONFIG["stlsq_threshold"], E1B_CONFIG["stlsq_max_iter"])
    return coeffs, terms


def fit_pysindy_b2(x_train: np.ndarray, dx_train: np.ndarray, mu_train: np.ndarray, c0: float) -> ps.SINDy:
    optimizer = ps.STLSQ(
        threshold=CONFIG["pysindy_stlsq_threshold"],
        alpha=CONFIG["pysindy_stlsq_alpha"],
        max_iter=CONFIG["pysindy_stlsq_max_iter"],
        normalize_columns=False,
    )
    library = ps.PolynomialLibrary(degree=CONFIG["library_degree"], include_bias=True)
    model = ps.SINDy(optimizer=optimizer, feature_library=library)
    b0_train_rhs = np.array([b0_rhs(0.0, xx, c0) for xx in x_train])
    model.fit(
        x_train,
        t=E1B_CONFIG["dt"],
        x_dot=dx_train - b0_train_rhs,
        u=mu_train.reshape(-1, 1),
        feature_names=["x1", "x2", "mu"],
    )
    return model


def fit_custom_b3(x_train: np.ndarray, dx_train: np.ndarray, mu_train: np.ndarray) -> np.ndarray:
    residual_target = dx_train[:, 1] + x_train[:, 0]
    slot_library = x_train[:, 1, None] * b3_library(x_train[:, 0], mu_train)
    return stlsq(slot_library, residual_target, E1B_CONFIG["stlsq_threshold"], E1B_CONFIG["stlsq_max_iter"])


def fit_optimizer_coeffs(optimizer: ps.optimizers.BaseOptimizer, library: np.ndarray, target: np.ndarray) -> np.ndarray:
    optimizer.fit(library, target.reshape(-1, 1))
    return optimizer.coef_.reshape(-1)


def b3_optimizer_configs() -> list[tuple[str, ps.optimizers.BaseOptimizer]]:
    return [
        (
            "pysindy_stlsq",
            ps.STLSQ(
                threshold=CONFIG["pysindy_stlsq_threshold"],
                alpha=CONFIG["pysindy_stlsq_alpha"],
                max_iter=CONFIG["pysindy_stlsq_max_iter"],
                normalize_columns=False,
            ),
        ),
        (
            "pysindy_stlsq_normalized",
            ps.STLSQ(
                threshold=CONFIG["pysindy_stlsq_threshold"],
                alpha=CONFIG["pysindy_stlsq_alpha"],
                max_iter=CONFIG["pysindy_stlsq_max_iter"],
                normalize_columns=True,
            ),
        ),
        (
            "pysindy_sr3_l0",
            ps.SR3(
                reg_weight_lam=CONFIG["pysindy_sr3_reg_weight_lam"],
                regularizer="L0",
                relax_coeff_nu=CONFIG["pysindy_sr3_relax_coeff_nu"],
                max_iter=CONFIG["pysindy_sr3_max_iter"],
                normalize_columns=False,
            ),
        ),
        (
            "pysindy_sr3_l1",
            ps.SR3(
                reg_weight_lam=CONFIG["pysindy_sr3_reg_weight_lam"],
                regularizer="L1",
                relax_coeff_nu=CONFIG["pysindy_sr3_relax_coeff_nu"],
                max_iter=CONFIG["pysindy_sr3_max_iter"],
                normalize_columns=False,
            ),
        ),
    ]


def vector_field_nrmse(true_dx: np.ndarray, pred_dx: np.ndarray) -> float:
    err = pred_dx - true_dx
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.std(true_dx)) or 1.0
    return rmse / denom


def count_active(coeffs: np.ndarray, threshold: float = 1e-6) -> int:
    return int(np.sum(np.abs(coeffs) >= threshold))


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    trajectories = build_trajectories()
    train_trajectories = [traj for traj in trajectories if traj["split"] == "train"]
    x_train, dx_train, mu_train = stack_data(train_trajectories)
    c0 = fit_constant_b0(x_train, dx_train)

    custom_b2_coeffs, terms = fit_custom_b2(x_train, dx_train, mu_train, c0)
    custom_b3_coeffs = fit_custom_b3(x_train, dx_train, mu_train)
    pysindy_b2_model = fit_pysindy_b2(x_train, dx_train, mu_train, c0)

    residual_target = dx_train[:, 1] + x_train[:, 0]
    slot_library = x_train[:, 1, None] * b3_library(x_train[:, 0], mu_train)
    b3_coefficients = {"custom_stlsq": custom_b3_coeffs}
    for name, optimizer in b3_optimizer_configs():
        b3_coefficients[name] = fit_optimizer_coeffs(optimizer, slot_library, residual_target)

    comparison_rows = []
    for split in ["train", "seen_mu_unseen_ic", "interpolation_mu", "extrapolation_mu"]:
        split_trajectories = [traj for traj in trajectories if traj["split"] == split]
        x_eval, dx_eval, mu_eval = stack_data(split_trajectories)
        b0_eval_rhs = np.array([b0_rhs(0.0, xx, c0) for xx in x_eval])
        custom_b2_pred = np.array([b2_rhs(0.0, xx, mm, c0, custom_b2_coeffs, terms) for xx, mm in zip(x_eval, mu_eval)])
        pysindy_b2_pred = b0_eval_rhs + pysindy_b2_model.predict(x_eval, u=mu_eval.reshape(-1, 1))
        custom_b3_pred = np.array([b3_rhs(0.0, xx, mm, custom_b3_coeffs) for xx, mm in zip(x_eval, mu_eval)])
        pysindy_stlsq_b3_pred = np.array([b3_rhs(0.0, xx, mm, b3_coefficients["pysindy_stlsq"]) for xx, mm in zip(x_eval, mu_eval)])
        comparison_rows.append(
            {
                "split": split,
                "custom_b2_vector_field_nrmse": f"{vector_field_nrmse(dx_eval, custom_b2_pred):.16g}",
                "pysindy_b2_vector_field_nrmse": f"{vector_field_nrmse(dx_eval, pysindy_b2_pred):.16g}",
                "pysindy_b2_vs_custom_b2_rmse": f"{float(np.sqrt(np.mean((pysindy_b2_pred - custom_b2_pred) ** 2))):.16g}",
                "custom_b3_vector_field_nrmse": f"{vector_field_nrmse(dx_eval, custom_b3_pred):.16g}",
                "pysindy_stlsq_b3_vector_field_nrmse": f"{vector_field_nrmse(dx_eval, pysindy_stlsq_b3_pred):.16g}",
                "pysindy_stlsq_b3_vs_custom_b3_rmse": f"{float(np.sqrt(np.mean((pysindy_stlsq_b3_pred - custom_b3_pred) ** 2))):.16g}",
            }
        )

    b3_optimizer_rows = []
    for name, coeffs in b3_coefficients.items():
        metrics = coefficient_metrics(coeffs, E1B_CONFIG["train_mu"] + E1B_CONFIG["interpolation_mu"] + E1B_CONFIG["extrapolation_mu"])
        b3_optimizer_rows.append(
            {
                "optimizer": name,
                "active_terms": count_active(coeffs),
                "coefficient_grid_nrmse": f"{metrics['overall_grid_nrmse']:.16g}",
                "support_precision": f"{metrics['support_precision']:.16g}",
                "support_recall": f"{metrics['support_recall']:.16g}",
            }
        )

    coefficient_rows = []
    b2_feature_names = [feature.replace(" ", "*") for feature in pysindy_b2_model.get_feature_names()]
    b2_coeffs = pysindy_b2_model.coefficients()
    custom_term_names = [param_state_term_name(term) for term in terms]
    for feature_idx, feature_name in enumerate(b2_feature_names):
        coefficient_rows.append(
            {
                "model": "pysindy_b2_residual",
                "feature": feature_name,
                "dx1_coefficient": f"{b2_coeffs[0, feature_idx]:.16g}",
                "dx2_coefficient": f"{b2_coeffs[1, feature_idx]:.16g}",
                "active": str(np.any(np.abs(b2_coeffs[:, feature_idx]) >= 1e-6)),
            }
        )
    for feature_idx, feature_name in enumerate(custom_term_names):
        coefficient_rows.append(
            {
                "model": "custom_b2_residual",
                "feature": feature_name,
                "dx1_coefficient": f"{custom_b2_coeffs[feature_idx, 0]:.16g}",
                "dx2_coefficient": f"{custom_b2_coeffs[feature_idx, 1]:.16g}",
                "active": str(np.any(np.abs(custom_b2_coeffs[feature_idx]) >= 1e-6)),
            }
        )
    for optimizer_name, coeffs in b3_coefficients.items():
        for feature_name, coeff in zip(B3_FEATURE_NAMES, coeffs):
            coefficient_rows.append(
                {
                    "model": f"b3_{optimizer_name}",
                    "feature": feature_name,
                    "dx1_coefficient": "",
                    "dx2_coefficient": f"{coeff:.16g}",
                    "active": str(abs(coeff) >= 1e-6),
                }
            )

    comparison_path = RESULT_DIR / "comparison_by_split.csv"
    b3_optimizer_path = RESULT_DIR / "b3_optimizer_comparison.csv"
    coefficients_path = RESULT_DIR / "model_coefficients.csv"
    metrics_path = RESULT_DIR / "metrics.json"
    write_csv(
        comparison_path,
        comparison_rows,
        [
            "split",
            "custom_b2_vector_field_nrmse",
            "pysindy_b2_vector_field_nrmse",
            "pysindy_b2_vs_custom_b2_rmse",
            "custom_b3_vector_field_nrmse",
            "pysindy_stlsq_b3_vector_field_nrmse",
            "pysindy_stlsq_b3_vs_custom_b3_rmse",
        ],
    )
    write_csv(
        b3_optimizer_path,
        b3_optimizer_rows,
        ["optimizer", "active_terms", "coefficient_grid_nrmse", "support_precision", "support_recall"],
    )
    write_csv(coefficients_path, coefficient_rows, ["model", "feature", "dx1_coefficient", "dx2_coefficient", "active"])

    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "source_experiment": CONFIG["source_experiment"],
        "constant_coefficient_c0": c0,
        "pysindy": {
            "version": ps.__version__,
            "b2_optimizer": {
                "name": "STLSQ",
                "threshold": CONFIG["pysindy_stlsq_threshold"],
                "alpha": CONFIG["pysindy_stlsq_alpha"],
                "max_iter": CONFIG["pysindy_stlsq_max_iter"],
            },
            "b3_optimizer_candidates": [row["optimizer"] for row in b3_optimizer_rows],
            "sr3_reg_weight_lam": CONFIG["pysindy_sr3_reg_weight_lam"],
            "sr3_relax_coeff_nu": CONFIG["pysindy_sr3_relax_coeff_nu"],
        },
        "active_term_counts": {
            "custom_b2": count_active(custom_b2_coeffs),
            "pysindy_b2": count_active(b2_coeffs),
            **{f"b3_{name}": count_active(coeffs) for name, coeffs in b3_coefficients.items()},
        },
        "comparison_by_split": comparison_rows,
        "b3_optimizer_comparison": b3_optimizer_rows,
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
        "experiment_id": CONFIG["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment": CONFIG["source_experiment"],
        "source_script": "scripts/run_e1b_vdp_multi_mu.py",
        "script": str(Path(__file__).relative_to(ROOT)),
        "purpose": "Validate PySINDy as a mature B2 residual baseline and B3 coefficient-slot optimizer backend on E1b.",
        "outputs": {
            "comparison_by_split_csv": str(comparison_path.relative_to(ROOT)),
            "b3_optimizer_comparison_csv": str(b3_optimizer_path.relative_to(ROOT)),
            "model_coefficients_csv": str(coefficients_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
        },
        "software": metrics["software"],
        "dependency_note": metrics["known_dependency_note"],
    }
    provenance_path = PROVENANCE_DIR / "pysindy_e1b_b2_b3_probe_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    hashes = {
        "comparison_by_split_csv_sha256": sha256_file(comparison_path),
        "b3_optimizer_comparison_csv_sha256": sha256_file(b3_optimizer_path),
        "model_coefficients_csv_sha256": sha256_file(coefficients_path),
        "metrics_json_sha256": sha256_file(metrics_path),
        "provenance_json_sha256": sha256_file(provenance_path),
    }
    (RESULT_DIR / "hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "experiment_id": CONFIG["experiment_id"],
                "pysindy_version": ps.__version__,
                "active_term_counts": metrics["active_term_counts"],
                "b3_optimizer_comparison": b3_optimizer_rows,
                "comparison_by_split": comparison_rows,
                "result_dir": str(RESULT_DIR.relative_to(ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
