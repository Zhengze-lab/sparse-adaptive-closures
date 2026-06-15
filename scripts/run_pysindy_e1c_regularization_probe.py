#!/usr/bin/env python3
"""Compare PySINDy sparse optimizers on the E1c noisy-state coefficient slot."""

from __future__ import annotations

import html
import json
import math
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

from run_e1b_vdp_multi_mu import (
    B3_FEATURE_NAMES,
    b3_library,
    b3_rhs,
    coefficient_metrics,
    count_active,
    polyline,
    sha256_file,
    stlsq,
    write_csv,
)
from run_e1c_vdp_noisy_state_true_dx import (
    CONFIG as E1C_CONFIG,
    add_state_noise,
    build_trajectories,
    stack_data,
    vector_field_nrmse,
)


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_DIR = ROOT / "results" / "pysindy_e1c_regularization_probe"
FIGURE_DIR = RESULT_DIR / "figures"


CONFIG = {
    "experiment_id": "PYS4_e1c_regularization_probe",
    "description": "PySINDy optimizer comparison on the E1c noisy-state Van der Pol coefficient slot.",
    "source_experiment": E1C_CONFIG["experiment_id"],
    "noise_levels_relative_state_std": E1C_CONFIG["noise_levels_relative_state_std"],
    "seeds": E1C_CONFIG["seeds"],
    "active_threshold": 1e-6,
    "stlsq_max_iter": 20,
    "sr3_max_iter": 1000,
    "sr3_relax_coeff_nu": 1.0,
}


EXPECTED_COEFFICIENTS = {
    "1": 0.0,
    "x1": 0.0,
    "x1^2": 0.0,
    "mu": 1.0,
    "mu*x1": 0.0,
    "mu*x1^2": -1.0,
}


def optimizer_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "custom_stlsq_t1e-3",
            "family": "custom_stlsq",
            "threshold": 1e-3,
            "alpha": 0.0,
            "normalize_columns": False,
            "regularizer": "",
            "reg_weight_lam": "",
        },
        {
            "name": "pysindy_stlsq_t1e-3",
            "family": "pysindy_stlsq",
            "threshold": 1e-3,
            "alpha": 0.0,
            "normalize_columns": False,
            "regularizer": "",
            "reg_weight_lam": "",
        },
        {
            "name": "pysindy_stlsq_t1e-2",
            "family": "pysindy_stlsq",
            "threshold": 1e-2,
            "alpha": 0.0,
            "normalize_columns": False,
            "regularizer": "",
            "reg_weight_lam": "",
        },
        {
            "name": "pysindy_stlsq_norm_t1e-3",
            "family": "pysindy_stlsq",
            "threshold": 1e-3,
            "alpha": 0.0,
            "normalize_columns": True,
            "regularizer": "",
            "reg_weight_lam": "",
        },
        {
            "name": "pysindy_stlsq_norm_t1e-2",
            "family": "pysindy_stlsq",
            "threshold": 1e-2,
            "alpha": 0.0,
            "normalize_columns": True,
            "regularizer": "",
            "reg_weight_lam": "",
        },
        {
            "name": "pysindy_sr3_l0_lam1e-4",
            "family": "pysindy_sr3",
            "threshold": "",
            "alpha": "",
            "normalize_columns": False,
            "regularizer": "L0",
            "reg_weight_lam": 1e-4,
        },
        {
            "name": "pysindy_sr3_l1_lam1e-4",
            "family": "pysindy_sr3",
            "threshold": "",
            "alpha": "",
            "normalize_columns": False,
            "regularizer": "L1",
            "reg_weight_lam": 1e-4,
        },
    ]


def fit_coefficients(spec: dict[str, Any], library: np.ndarray, target: np.ndarray) -> np.ndarray:
    if spec["family"] == "custom_stlsq":
        return stlsq(library, target, float(spec["threshold"]), E1C_CONFIG["stlsq_max_iter"])

    if spec["family"] == "pysindy_stlsq":
        optimizer = ps.STLSQ(
            threshold=float(spec["threshold"]),
            alpha=float(spec["alpha"]),
            max_iter=CONFIG["stlsq_max_iter"],
            normalize_columns=bool(spec["normalize_columns"]),
        )
    elif spec["family"] == "pysindy_sr3":
        optimizer = ps.SR3(
            reg_weight_lam=float(spec["reg_weight_lam"]),
            regularizer=str(spec["regularizer"]),
            relax_coeff_nu=CONFIG["sr3_relax_coeff_nu"],
            max_iter=CONFIG["sr3_max_iter"],
            normalize_columns=bool(spec["normalize_columns"]),
        )
    else:
        raise ValueError(f"Unknown optimizer family: {spec['family']}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        optimizer.fit(library, target.reshape(-1, 1))
    return optimizer.coef_.reshape(-1)


def split_eval_data(trajectories: list[dict[str, object]]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    eval_data = {}
    for split in ["seen_mu_unseen_ic", "interpolation_mu", "extrapolation_mu"]:
        split_trajectories = [traj for traj in trajectories if traj["split"] == split]
        eval_data[split] = stack_data(split_trajectories)
    return eval_data


def b3_vector_field_nrmse(x_eval: np.ndarray, dx_eval: np.ndarray, mu_eval: np.ndarray, coeffs: np.ndarray) -> float:
    pred_dx = np.array([b3_rhs(0.0, xx, mm, coeffs) for xx, mm in zip(x_eval, mu_eval)])
    return vector_field_nrmse(dx_eval, pred_dx)


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
        "seen_vector_field_nrmse",
        "interp_vector_field_nrmse",
        "extrap_vector_field_nrmse",
    ]
    summary_rows = []
    spec_by_name = {str(spec["name"]): spec for spec in optimizer_specs()}
    for (noise_level, optimizer), rows in sorted(grouped.items()):
        spec = spec_by_name[optimizer]
        summary = {
            "noise_level": f"{noise_level:.16g}",
            "optimizer": optimizer,
            "family": spec["family"],
            "threshold": spec["threshold"],
            "alpha": spec["alpha"],
            "normalize_columns": str(spec["normalize_columns"]),
            "regularizer": spec["regularizer"],
            "reg_weight_lam": spec["reg_weight_lam"],
            "n_seeds": len(rows),
        }
        for metric_name in metric_names:
            values = np.array([float(row[metric_name]) for row in rows], dtype=float)
            summary[f"{metric_name}_mean"] = f"{np.nanmean(values):.16g}"
            summary[f"{metric_name}_std"] = f"{np.nanstd(values):.16g}"
        summary_rows.append(summary)
    return summary_rows


def series_from_summary(
    summary_rows: list[dict[str, object]],
    metric_name: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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


def write_metric_svg(
    path: Path,
    title: str,
    x_values: np.ndarray,
    series: list[tuple[str, np.ndarray, str, str, str]],
    x_label: str,
    y_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1240, 500
    left, right, top, bottom = 78, 30, 42, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    if math.isclose(x_min, x_max):
        x_min -= 0.01
        x_max += 0.01
    y_all = np.concatenate([item[1] for item in series])
    y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    if math.isclose(y_min, y_max):
        pad = 0.05 if math.isclose(y_min, 0.0) else abs(y_min) * 0.05
    else:
        pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    def sx(x: np.ndarray) -> np.ndarray:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: np.ndarray) -> np.ndarray:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="26" text-anchor="middle" font-family="Arial" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<text x="{left + plot_w / 2:.0f}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(x_label)}</text>',
        f'<text transform="translate(20,{top + plot_h / 2:.0f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(y_label)}</text>',
    ]
    for tick in x_values:
        x = sx(np.array([tick]))[0]
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#333"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="11">{tick:.3f}</text>')
    for tick in np.linspace(y_min, y_max, 5):
        y = sy(np.array([tick]))[0]
        label = f"{tick:.3g}"
        lines.append(f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#333"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{label}</text>')

    x_plot = sx(x_values)
    for label, values, color, dash, marker in series:
        pts = np.column_stack([x_plot, sy(values)])
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<polyline points="{polyline(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="2.6" stroke-opacity="0.96"{dash_attr}/>'
        )
        for x, y in pts:
            if marker == "circle":
                lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.0" fill="white" stroke="{color}" stroke-width="2"/>')
            elif marker == "square":
                lines.append(f'<rect x="{x - 3.7:.2f}" y="{y - 3.7:.2f}" width="7.4" height="7.4" fill="white" stroke="{color}" stroke-width="2"/>')
            else:
                lines.append(
                    f'<path d="M {x:.2f} {y - 4.2:.2f} L {x + 4.2:.2f} {y + 4.2:.2f} '
                    f'L {x - 4.2:.2f} {y + 4.2:.2f} Z" fill="white" stroke="{color}" stroke-width="2"/>'
                )

    legend_x = left + 12
    legend_y = top + 14
    for idx, (label, _, color, dash, marker) in enumerate(series):
        y = legend_y + idx * 20
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" '
            f'stroke="{color}" stroke-width="2.6"{dash_attr}/>'
        )
        if marker == "circle":
            lines.append(f'<circle cx="{legend_x + 12}" cy="{y}" r="3.8" fill="white" stroke="{color}" stroke-width="2"/>')
        elif marker == "square":
            lines.append(f'<rect x="{legend_x + 8.2}" y="{y - 3.8}" width="7.6" height="7.6" fill="white" stroke="{color}" stroke-width="2"/>')
        else:
            lines.append(
                f'<path d="M {legend_x + 12:.2f} {y - 4.0:.2f} L {legend_x + 16:.2f} {y + 4.0:.2f} '
                f'L {legend_x + 8:.2f} {y + 4.0:.2f} Z" fill="white" stroke="{color}" stroke-width="2"/>'
            )
        lines.append(f'<text x="{legend_x + 34}" y="{y + 4}" font-family="Arial" font-size="12">{html.escape(label)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figures(summary_rows: list[dict[str, object]]) -> list[Path]:
    styles = {
        "custom_stlsq_t1e-3": ("#0072B2", "", "circle"),
        "pysindy_stlsq_t1e-3": ("#D55E00", "8 3", "square"),
        "pysindy_stlsq_t1e-2": ("#009E73", "6 2 2 2", "triangle"),
        "pysindy_stlsq_norm_t1e-3": ("#CC79A7", "3 3", "circle"),
        "pysindy_stlsq_norm_t1e-2": ("#E69F00", "10 3 2 3", "square"),
        "pysindy_sr3_l0_lam1e-4": ("#56B4E9", "2 4", "triangle"),
        "pysindy_sr3_l1_lam1e-4": ("#000000", "12 4", "circle"),
    }
    labels_en = {
        "custom_stlsq_t1e-3": "Custom STLSQ, threshold=1e-3",
        "pysindy_stlsq_t1e-3": "PySINDy STLSQ, threshold=1e-3",
        "pysindy_stlsq_t1e-2": "PySINDy STLSQ, threshold=1e-2",
        "pysindy_stlsq_norm_t1e-3": "PySINDy normalized STLSQ, threshold=1e-3",
        "pysindy_stlsq_norm_t1e-2": "PySINDy normalized STLSQ, threshold=1e-2",
        "pysindy_sr3_l0_lam1e-4": "PySINDy SR3 L0, lambda=1e-4",
        "pysindy_sr3_l1_lam1e-4": "PySINDy SR3 L1, lambda=1e-4",
    }
    labels_zh = {
        "custom_stlsq_t1e-3": "自写 STLSQ，阈值=1e-3",
        "pysindy_stlsq_t1e-3": "PySINDy STLSQ，阈值=1e-3",
        "pysindy_stlsq_t1e-2": "PySINDy STLSQ，阈值=1e-2",
        "pysindy_stlsq_norm_t1e-3": "PySINDy 归一化 STLSQ，阈值=1e-3",
        "pysindy_stlsq_norm_t1e-2": "PySINDy 归一化 STLSQ，阈值=1e-2",
        "pysindy_sr3_l0_lam1e-4": "PySINDy SR3 L0，lambda=1e-4",
        "pysindy_sr3_l1_lam1e-4": "PySINDy SR3 L1，lambda=1e-4",
    }

    created = []
    for metric, y_label_en, y_label_zh, filename in [
        ("support_precision", "support precision", "支持集 precision", "support_precision"),
        ("active_terms", "active terms", "活跃项数量", "active_terms"),
        ("coefficient_grid_nrmse", "coefficient-function grid NRMSE", "系数函数网格 NRMSE", "coefficient_grid_nrmse"),
        ("interp_vector_field_nrmse", "interpolation vector-field NRMSE", "插值向量场 NRMSE", "interp_vector_field_nrmse"),
    ]:
        noise_x, values = series_from_summary(summary_rows, metric)
        series_en = [
            (labels_en[name], values[name], *styles[name])
            for name in labels_en
        ]
        series_zh = [
            (labels_zh[name], values[name], *styles[name])
            for name in labels_zh
        ]
        en_path = FIGURE_DIR / f"{filename}.svg"
        zh_path = FIGURE_DIR / f"{filename}_zh.svg"
        write_metric_svg(
            en_path,
            f"E1c PySINDy regularization: {y_label_en}",
            noise_x,
            series_en,
            "relative state-noise level",
            y_label_en,
        )
        write_metric_svg(
            zh_path,
            f"E1c PySINDy 正则化对照：{y_label_zh}",
            noise_x,
            series_zh,
            "相对状态噪声水平",
            y_label_zh,
        )
        created.extend([en_path, zh_path])
    return created


def main() -> None:
    for directory in (PROVENANCE_DIR, RESULT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    trajectories = build_trajectories()
    train_trajectories = [traj for traj in trajectories if traj["split"] == "train"]
    x_train_clean, dx_train_clean, mu_train = stack_data(train_trajectories)
    state_std = np.std(x_train_clean, axis=0)
    eval_data = split_eval_data(trajectories)
    specs = optimizer_specs()

    seed_rows = []
    coefficient_rows = []
    for noise_level in CONFIG["noise_levels_relative_state_std"]:
        for seed in CONFIG["seeds"]:
            x_obs = add_state_noise(x_train_clean, float(noise_level), int(seed), state_std)
            target = dx_train_clean[:, 1] + x_obs[:, 0]
            library = x_obs[:, 1, None] * b3_library(x_obs[:, 0], mu_train)
            for spec in specs:
                coeffs = fit_coefficients(spec, library, target)
                coeff_metrics = coefficient_metrics(
                    coeffs,
                    E1C_CONFIG["train_mu"] + E1C_CONFIG["interpolation_mu"] + E1C_CONFIG["extrapolation_mu"],
                )
                active_features = set(coeff_metrics["active_features"])
                spurious_terms = len(active_features - {"mu", "mu*x1^2"})
                seen_x, seen_dx, seen_mu = eval_data["seen_mu_unseen_ic"]
                interp_x, interp_dx, interp_mu = eval_data["interpolation_mu"]
                extrap_x, extrap_dx, extrap_mu = eval_data["extrapolation_mu"]
                seed_rows.append(
                    {
                        "noise_level": f"{float(noise_level):.16g}",
                        "seed": int(seed),
                        "optimizer": spec["name"],
                        "family": spec["family"],
                        "threshold": spec["threshold"],
                        "alpha": spec["alpha"],
                        "normalize_columns": str(spec["normalize_columns"]),
                        "regularizer": spec["regularizer"],
                        "reg_weight_lam": spec["reg_weight_lam"],
                        "coefficient_grid_nrmse": f"{coeff_metrics['overall_grid_nrmse']:.16g}",
                        "support_precision": f"{coeff_metrics['support_precision']:.16g}",
                        "support_recall": f"{coeff_metrics['support_recall']:.16g}",
                        "active_terms": count_active(coeffs, CONFIG["active_threshold"]),
                        "spurious_terms": spurious_terms,
                        "active_mu": str(abs(coeffs[B3_FEATURE_NAMES.index("mu")]) >= CONFIG["active_threshold"]),
                        "active_mu_x1_squared": str(abs(coeffs[B3_FEATURE_NAMES.index("mu*x1^2")]) >= CONFIG["active_threshold"]),
                        "seen_vector_field_nrmse": f"{b3_vector_field_nrmse(seen_x, seen_dx, seen_mu, coeffs):.16g}",
                        "interp_vector_field_nrmse": f"{b3_vector_field_nrmse(interp_x, interp_dx, interp_mu, coeffs):.16g}",
                        "extrap_vector_field_nrmse": f"{b3_vector_field_nrmse(extrap_x, extrap_dx, extrap_mu, coeffs):.16g}",
                    }
                )
                if int(seed) == 0:
                    for feature_name, coeff in zip(B3_FEATURE_NAMES, coeffs):
                        expected = EXPECTED_COEFFICIENTS[feature_name]
                        coefficient_rows.append(
                            {
                                "noise_level": f"{float(noise_level):.16g}",
                                "seed": int(seed),
                                "optimizer": spec["name"],
                                "feature": feature_name,
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
    provenance_path = PROVENANCE_DIR / "pysindy_e1c_regularization_probe_provenance.json"
    hashes_path = RESULT_DIR / "hashes.json"

    seed_fieldnames = list(seed_rows[0].keys())
    summary_fieldnames = list(summary_rows[0].keys())
    coefficient_fieldnames = list(coefficient_rows[0].keys())
    write_csv(seed_metrics_path, seed_rows, seed_fieldnames)
    write_csv(summary_path, summary_rows, summary_fieldnames)
    write_csv(coefficients_path, coefficient_rows, coefficient_fieldnames)

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
        "dataset_id": "pysindy_e1c_regularization_probe",
        "experiment_id": CONFIG["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment": CONFIG["source_experiment"],
        "source_script": "scripts/run_e1c_vdp_noisy_state_true_dx.py",
        "script": str(Path(__file__).relative_to(ROOT)),
        "purpose": "Compare custom STLSQ, PySINDy STLSQ, normalized STLSQ, and SR3 on the noisy E1c coefficient-slot design matrix.",
        "equation_reference": "Van der Pol coefficient slot p_c(x1, mu)=mu-mu*x1^2 in dx2=-x1+x2*p_c.",
        "candidate_slots": [{"name": "damping", "g": "x2", "coefficient_function": "p_c(x1, mu)"}],
        "expected_coefficients": EXPECTED_COEFFICIENTS,
        "noise_model": metrics["noise_model"],
        "optimizer_specs": specs,
        "outputs": {
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
        "seed_metrics_csv_sha256": sha256_file(seed_metrics_path),
        "summary_by_noise_optimizer_csv_sha256": sha256_file(summary_path),
        "coefficients_seed0_csv_sha256": sha256_file(coefficients_path),
        "metrics_json_sha256": sha256_file(metrics_path),
        "provenance_json_sha256": sha256_file(provenance_path),
        **{f"figure_{path.stem}_sha256": sha256_file(path) for path in figure_paths},
    }
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
