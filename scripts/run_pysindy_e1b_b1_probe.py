#!/usr/bin/env python3
"""Probe PySINDy on E1b by reproducing the B1 full-vector-field baseline."""

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
    CONFIG as E1B_CONFIG,
    b1_rhs,
    make_trajectory,
    param_state_library,
    param_state_term_name,
    param_state_terms,
    sha256_file,
    vector_stlsq,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "pysindy_e1b_b1_probe"


CONFIG = {
    "experiment_id": "PYS1_e1b_b1_probe",
    "description": "PySINDy reproduction of E1b B1 full-vector-field baseline.",
    "source_experiment": "E1b_vdp_multi_mu",
    "pysindy_optimizer": "STLSQ",
    "pysindy_threshold": 1e-8,
    "pysindy_alpha": 0.0,
    "pysindy_max_iter": 20,
    "library_degree": E1B_CONFIG["param_state_library_max_degree"],
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


def fit_custom_b1(x_train: np.ndarray, dx_train: np.ndarray, mu_train: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    terms = param_state_terms(E1B_CONFIG["param_state_library_max_degree"])
    theta = param_state_library(x_train, mu_train, terms)
    coeffs = vector_stlsq(theta, dx_train, E1B_CONFIG["stlsq_threshold"], E1B_CONFIG["stlsq_max_iter"])
    return coeffs, terms


def fit_pysindy_b1(x_train: np.ndarray, dx_train: np.ndarray, mu_train: np.ndarray) -> ps.SINDy:
    optimizer = ps.STLSQ(
        threshold=CONFIG["pysindy_threshold"],
        alpha=CONFIG["pysindy_alpha"],
        max_iter=CONFIG["pysindy_max_iter"],
        normalize_columns=False,
    )
    library = ps.PolynomialLibrary(degree=CONFIG["library_degree"], include_bias=True)
    model = ps.SINDy(optimizer=optimizer, feature_library=library)
    model.fit(
        x_train,
        t=E1B_CONFIG["dt"],
        x_dot=dx_train,
        u=mu_train.reshape(-1, 1),
        feature_names=["x1", "x2", "mu"],
    )
    return model


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
    custom_coeffs, custom_terms = fit_custom_b1(x_train, dx_train, mu_train)
    pysindy_model = fit_pysindy_b1(x_train, dx_train, mu_train)

    pysindy_coeffs = pysindy_model.coefficients()
    pysindy_feature_names = pysindy_model.get_feature_names()
    custom_term_names = [param_state_term_name(term) for term in custom_terms]

    comparison_rows = []
    for split in ["train", "seen_mu_unseen_ic", "interpolation_mu", "extrapolation_mu"]:
        split_trajectories = [traj for traj in trajectories if traj["split"] == split]
        x_eval, dx_eval, mu_eval = stack_data(split_trajectories)
        custom_pred = np.array([b1_rhs(0.0, xx, mm, custom_coeffs, custom_terms) for xx, mm in zip(x_eval, mu_eval)])
        pysindy_pred = pysindy_model.predict(x_eval, u=mu_eval.reshape(-1, 1))
        diff = pysindy_pred - custom_pred
        comparison_rows.append(
            {
                "split": split,
                "custom_b1_vector_field_nrmse": f"{vector_field_nrmse(dx_eval, custom_pred):.16g}",
                "pysindy_b1_vector_field_nrmse": f"{vector_field_nrmse(dx_eval, pysindy_pred):.16g}",
                "pysindy_vs_custom_rmse": f"{float(np.sqrt(np.mean(diff * diff))):.16g}",
                "pysindy_vs_custom_max_abs": f"{float(np.max(np.abs(diff))):.16g}",
            }
        )

    coefficient_rows = []
    for feature_idx, feature_name in enumerate(pysindy_feature_names):
        coefficient_rows.append(
            {
                "implementation": "pysindy_stlsq",
                "feature": feature_name.replace(" ", "*"),
                "dx1_coefficient": f"{pysindy_coeffs[0, feature_idx]:.16g}",
                "dx2_coefficient": f"{pysindy_coeffs[1, feature_idx]:.16g}",
                "active": str(np.any(np.abs(pysindy_coeffs[:, feature_idx]) >= 1e-6)),
            }
        )
    for feature_idx, feature_name in enumerate(custom_term_names):
        coefficient_rows.append(
            {
                "implementation": "custom_stlsq",
                "feature": feature_name,
                "dx1_coefficient": f"{custom_coeffs[feature_idx, 0]:.16g}",
                "dx2_coefficient": f"{custom_coeffs[feature_idx, 1]:.16g}",
                "active": str(np.any(np.abs(custom_coeffs[feature_idx]) >= 1e-6)),
            }
        )

    comparison_path = RESULT_DIR / "comparison_by_split.csv"
    coefficients_path = RESULT_DIR / "model_coefficients.csv"
    metrics_path = RESULT_DIR / "metrics.json"
    write_csv(
        comparison_path,
        comparison_rows,
        ["split", "custom_b1_vector_field_nrmse", "pysindy_b1_vector_field_nrmse", "pysindy_vs_custom_rmse", "pysindy_vs_custom_max_abs"],
    )
    write_csv(coefficients_path, coefficient_rows, ["implementation", "feature", "dx1_coefficient", "dx2_coefficient", "active"])

    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "source_experiment": CONFIG["source_experiment"],
        "pysindy": {
            "version": ps.__version__,
            "optimizer": CONFIG["pysindy_optimizer"],
            "threshold": CONFIG["pysindy_threshold"],
            "alpha": CONFIG["pysindy_alpha"],
            "max_iter": CONFIG["pysindy_max_iter"],
            "library": "PolynomialLibrary",
            "library_degree": CONFIG["library_degree"],
            "include_bias": True,
            "used_x_dot": True,
            "used_u_for_mu": True,
        },
        "active_term_counts": {
            "pysindy_b1": count_active(pysindy_coeffs),
            "custom_b1": count_active(custom_coeffs),
        },
        "comparison_by_split": comparison_rows,
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
        "purpose": "Validate PySINDy as a mature B1 full-vector-field baseline on E1b.",
        "outputs": {
            "comparison_by_split_csv": str(comparison_path.relative_to(ROOT)),
            "model_coefficients_csv": str(coefficients_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
        },
        "software": metrics["software"],
        "dependency_note": metrics["known_dependency_note"],
    }
    provenance_path = PROVENANCE_DIR / "pysindy_e1b_b1_probe_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    hashes = {
        "comparison_by_split_csv_sha256": sha256_file(comparison_path),
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
                "comparison_by_split": comparison_rows,
                "result_dir": str(RESULT_DIR.relative_to(ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
