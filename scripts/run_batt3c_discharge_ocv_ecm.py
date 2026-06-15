#!/usr/bin/env python3
"""Run BATT-3c: discharge-OCV corrected ECM and current-sign audit."""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator

import run_batt3_constant_ecm as batt3


ROOT = batt3.ROOT
RESULT_DIR = ROOT / "results" / "battery_lfp_discharge_ocv_ecm"
FIGURE_DIR = RESULT_DIR / "figures"
PROVENANCE_DIR = ROOT / "data" / "provenance"


CONFIG: dict[str, Any] = {
    "experiment_id": "BATT3c_lfp_discharge_ocv_current_sign_ecm",
    "description": "Rerun ohmic and constant first-order ECM using discharge OCV and audit current sign convention.",
    "source_dataset": "battery_lfp_ocv_dyn",
    "source_url": "https://data.mendeley.com/datasets/p8kf893yv3/1",
    "source_doi": "10.17632/p8kf893yv3.1",
    "license": "CC BY 4.0",
    "raw_zip": str(batt3.RAW_ZIP.relative_to(ROOT)),
    "ocv_grid_csv": str(batt3.OCV_GRID_CSV.relative_to(ROOT)),
    "ocv_curve": "discharge",
    "dynamic_script": batt3.CONFIG["dyn_script"],
    "current_conventions": {
        "I_model=-I_raw": "same as BATT-3; batt3 current_a field",
        "I_model=I_raw": "opposite sign of BATT-3 current_a field",
    },
    "fit_stride": batt3.CONFIG["fit_stride"],
    "prediction_sample_stride": batt3.CONFIG["prediction_sample_stride"],
    "initialization_window_s": batt3.CONFIG["initialization_window_s"],
    "score_start_s": batt3.CONFIG["score_start_s"],
    "least_squares_loss": batt3.CONFIG["least_squares_loss"],
    "least_squares_f_scale_v": batt3.CONFIG["least_squares_f_scale_v"],
    "max_nfev_ohmic": batt3.CONFIG["max_nfev_ohmic"],
    "max_nfev_ecm": batt3.CONFIG["max_nfev_ecm"],
    "ocv_training_temperatures_c": batt3.CONFIG["ocv_training_temperatures_c"],
    "heldout_temperature_validation_c": batt3.CONFIG["heldout_temperature_validation_c"],
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    batt3.write_csv(path, rows)


def load_ocv_model(curve: str) -> batt3.OcvModel:
    rows: list[dict[str, str]] = []
    with batt3.OCV_GRID_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["curve"] == curve:
                rows.append(row)
    train_temps = np.array(CONFIG["ocv_training_temperatures_c"], dtype=float)
    soc_grid = np.array(sorted({float(row["soc"]) for row in rows}), dtype=float)
    lookup = {
        (int(row["temperature_c"]), float(row["soc"])): float(row["voltage"])
        for row in rows
        if int(row["temperature_c"]) in set(int(v) for v in train_temps)
    }
    matrix = np.empty((len(train_temps), len(soc_grid)), dtype=float)
    for i, temp in enumerate(train_temps):
        for j, soc in enumerate(soc_grid):
            matrix[i, j] = lookup[(int(temp), float(soc))]
    return batt3.OcvModel(
        interpolator=RegularGridInterpolator(
            (train_temps, soc_grid),
            matrix,
            method="linear",
            bounds_error=False,
            fill_value=None,
        ),
        soc_grid=soc_grid,
        temperatures_c=train_temps,
    )


def records_for_current_convention(records: list[batt3.DynRecord], convention: str) -> list[batt3.DynRecord]:
    if convention == "I_model=-I_raw":
        factor = 1.0
    elif convention == "I_model=I_raw":
        factor = -1.0
    else:
        raise ValueError(f"Unknown current convention: {convention}")
    return [replace(record, current_a=factor * record.current_a) for record in records]


def add_context(rows: list[dict[str, Any]], convention: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = {"current_convention": convention, **row}
        if "current_a_discharge_positive" in item:
            item["current_a_model"] = item.pop("current_a_discharge_positive")
        out.append(item)
    return out


def run_one_convention(
    records: list[batt3.DynRecord],
    ocv_model: batt3.OcvModel,
    convention: str,
) -> dict[str, Any]:
    signed_records = records_for_current_convention(records, convention)
    ocv_states = batt3.fit_ocv_train_states(signed_records, ocv_model)
    ohmic_params, ohmic_states, ohmic_result = batt3.fit_ohmic(signed_records, ocv_model)
    ecm_params, ecm_states, ecm_result = batt3.fit_constant_ecm(signed_records, ocv_model)
    model_params = {
        "ocv_only": {},
        "ohmic_only": ohmic_params,
        "constant_ecm": ecm_params,
    }
    train_states = {
        "ocv_only": ocv_states,
        "ohmic_only": ohmic_states,
        "constant_ecm": ecm_states,
    }
    trajectory_rows, split_rows, prediction_rows, split_summary = batt3.evaluate_models(
        signed_records,
        ocv_model,
        model_params,
        train_states,
    )
    return {
        "current_convention": convention,
        "trajectory_rows": add_context(trajectory_rows, convention),
        "split_rows": add_context(split_rows, convention),
        "prediction_rows": add_context(prediction_rows, convention),
        "split_summary": {f"{convention}:{key}": value for key, value in split_summary.items()},
        "fit_summary": {
            "current_convention": convention,
            "ohmic_only": {
                "params": ohmic_params,
                "cost": float(ohmic_result.cost),
                "optimality": float(ohmic_result.optimality),
                "nfev": int(ohmic_result.nfev),
                "success": bool(ohmic_result.success),
                "message": str(ohmic_result.message),
            },
            "constant_ecm": {
                "params": ecm_params,
                "cost": float(ecm_result.cost),
                "optimality": float(ecm_result.optimality),
                "nfev": int(ecm_result.nfev),
                "success": bool(ecm_result.success),
                "message": str(ecm_result.message),
            },
        },
        "train_initial_states": train_states,
    }


def fit_summary_rows(fit_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in fit_summaries:
        convention = summary["current_convention"]
        for model in ["ohmic_only", "constant_ecm"]:
            item = summary[model]
            row = {
                "current_convention": convention,
                "model": model,
                "cost": item["cost"],
                "optimality": item["optimality"],
                "nfev": item["nfev"],
                "success": item["success"],
                "message": item["message"],
            }
            for key, value in item["params"].items():
                row[key] = value
            rows.append(row)
    return rows


def validation_rank(split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [row for row in split_rows if row["split"] == "validation_temperature"],
        key=lambda row: float(row["rmse_mv"]),
    )


def improvement_rows(split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for row in split_rows:
        grouped.setdefault((str(row["current_convention"]), str(row["split"])), {})[str(row["model"])] = float(row["rmse_mv"])
    rows: list[dict[str, Any]] = []
    for (convention, split), values in sorted(grouped.items()):
        base = values.get("ocv_only")
        if base is None:
            continue
        for model in ["ohmic_only", "constant_ecm"]:
            if model not in values:
                continue
            rows.append(
                {
                    "current_convention": convention,
                    "split": split,
                    "model": model,
                    "baseline_model": "ocv_only",
                    "baseline_rmse_mv": base,
                    "model_rmse_mv": values[model],
                    "improvement_percent": 100.0 * (base - values[model]) / base if base else float("nan"),
                }
            )
    return rows


def write_rmse_svg(path: Path, split_rows: list[dict[str, Any]], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    splits = ["train", "validation_temperature", "test_high_amplitude", "test_cell_transfer"]
    bar_specs = [
        ("I_model=-I_raw", "ocv_only", "#6b7280"),
        ("I_model=-I_raw", "ohmic_only", "#2563eb"),
        ("I_model=-I_raw", "constant_ecm", "#1d4ed8"),
        ("I_model=I_raw", "ohmic_only", "#dc2626"),
        ("I_model=I_raw", "constant_ecm", "#7c3aed"),
    ]
    labels_en = {
        "train": "Train",
        "validation_temperature": "Held-out T",
        "test_high_amplitude": "High amp.",
        "test_cell_transfer": "Cell transfer",
        ("I_model=-I_raw", "ocv_only"): "discharge OCV",
        ("I_model=-I_raw", "ohmic_only"): "ohmic, -raw",
        ("I_model=-I_raw", "constant_ecm"): "ECM, -raw",
        ("I_model=I_raw", "ohmic_only"): "ohmic, raw",
        ("I_model=I_raw", "constant_ecm"): "ECM, raw",
    }
    labels_zh = {
        "train": "训练",
        "validation_temperature": "留出温度",
        "test_high_amplitude": "高倍率",
        "test_cell_transfer": "跨电芯",
        ("I_model=-I_raw", "ocv_only"): "放电 OCV",
        ("I_model=-I_raw", "ohmic_only"): "欧姆，-原始电流",
        ("I_model=-I_raw", "constant_ecm"): "ECM，-原始电流",
        ("I_model=I_raw", "ohmic_only"): "欧姆，原始电流",
        ("I_model=I_raw", "constant_ecm"): "ECM，原始电流",
    }
    labels = labels_zh if zh else labels_en
    lookup = {
        (row["current_convention"], row["model"], row["split"]): float(row["rmse_mv"])
        for row in split_rows
    }
    y_max = max(lookup.values()) * 1.12
    left, top, width, height = 86.0, 70.0, 760.0, 320.0
    group_w = width / len(splits)
    bar_w = 19.0
    gap = 4.0
    bars: list[str] = []
    for i, split in enumerate(splits):
        base_x = left + i * group_w + group_w / 2 - (len(bar_specs) * bar_w + (len(bar_specs) - 1) * gap) / 2
        for j, (convention, model, color) in enumerate(bar_specs):
            value = lookup.get((convention, model, split), 0.0)
            bar_h = value / y_max * height
            x = base_x + j * (bar_w + gap)
            y = top + height - bar_h
            bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{color}"/>')
        bars.append(f'<text x="{left + i * group_w + group_w / 2:.2f}" y="420" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#374151">{labels[split]}</text>')
    legend: list[str] = []
    for i, (convention, model, color) in enumerate(bar_specs):
        y = 92 + i * 24
        legend.append(f'<rect x="858" y="{y - 12}" width="17" height="11" fill="{color}"/>')
        legend.append(f'<text x="883" y="{y - 2}" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="11" fill="#111827">{labels[(convention, model)]}</text>')
    title = "BATT-3c current-sign RMSE comparison" if not zh else "BATT-3c 电流符号 RMSE 对比"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="460" viewBox="0 0 1060 460">
  <rect width="1060" height="460" fill="#ffffff"/>
  <text x="86" y="36" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  <rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#6b7280"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#6b7280"/>
  <text x="25" y="{top + height / 2}" transform="rotate(-90 25 {top + height / 2})" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">RMSE (mV)</text>
  {''.join(bars)}
  {''.join(legend)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_parameter_svg(path: Path, fit_rows: list[dict[str, Any]], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in fit_rows if row["model"] == "ohmic_only"]
    labels = [row["current_convention"] for row in rows]
    values = [float(row.get("R0_ohm", 0.0)) for row in rows]
    y_max = max(values + [1e-9]) * 1.25
    left, top, width, height = 92.0, 68.0, 560.0, 300.0
    colors = ["#2563eb", "#dc2626"]
    bars: list[str] = []
    for i, value in enumerate(values):
        bar_w = 110.0
        x = left + 120.0 + i * 220.0
        bar_h = value / y_max * height if y_max > 0 else 0.0
        y = top + height - bar_h
        bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{colors[i % len(colors)]}"/>')
        bars.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">{value:.4g}</text>')
        bars.append(f'<text x="{x + bar_w / 2:.2f}" y="404" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#374151">{labels[i]}</text>')
    title = "Fitted ohmic resistance by current convention" if not zh else "不同电流约定下的欧姆电阻"
    y_label = "R0 (ohm)" if not zh else "R0 (欧姆)"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="440" viewBox="0 0 760 440">
  <rect width="760" height="440" fill="#ffffff"/>
  <text x="92" y="34" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  <rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#6b7280"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#6b7280"/>
  <text x="25" y="{top + height / 2}" transform="rotate(-90 25 {top + height / 2})" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{y_label}</text>
  {''.join(bars)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    records = batt3.load_dynamic_records()
    ocv_model = load_ocv_model(str(CONFIG["ocv_curve"]))
    convention_results = [
        run_one_convention(records, ocv_model, convention)
        for convention in CONFIG["current_conventions"]
    ]
    trajectory_rows = [row for result in convention_results for row in result["trajectory_rows"]]
    split_rows = [row for result in convention_results for row in result["split_rows"]]
    prediction_rows = [row for result in convention_results for row in result["prediction_rows"]]
    fit_summaries = [result["fit_summary"] for result in convention_results]
    fit_rows = fit_summary_rows(fit_summaries)
    improvements = improvement_rows(split_rows)
    split_summary: dict[str, dict[str, Any]] = {}
    for result in convention_results:
        split_summary.update(result["split_summary"])

    metrics_by_trajectory_path = RESULT_DIR / "metrics_by_trajectory.csv"
    metrics_by_split_path = RESULT_DIR / "metrics_by_split.csv"
    predictions_path = RESULT_DIR / "predictions_sample.csv"
    fit_summary_path = RESULT_DIR / "fit_summary.csv"
    improvement_path = RESULT_DIR / "improvement_vs_ocv.csv"
    metrics_path = RESULT_DIR / "metrics.json"
    provenance_path = PROVENANCE_DIR / "batt3c_discharge_ocv_ecm_provenance.json"

    write_csv(metrics_by_trajectory_path, trajectory_rows)
    write_csv(metrics_by_split_path, split_rows)
    write_csv(predictions_path, prediction_rows)
    write_csv(fit_summary_path, fit_rows)
    write_csv(improvement_path, improvements)
    write_rmse_svg(FIGURE_DIR / "current_sign_rmse_by_split.svg", split_rows, zh=False)
    write_rmse_svg(FIGURE_DIR / "current_sign_rmse_by_split_zh.svg", split_rows, zh=True)
    write_parameter_svg(FIGURE_DIR / "ohmic_resistance_by_sign.svg", fit_rows, zh=False)
    write_parameter_svg(FIGURE_DIR / "ohmic_resistance_by_sign_zh.svg", fit_rows, zh=True)

    rank = validation_rank(split_rows)
    best = rank[0] if rank else {}
    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "complete",
        "config": CONFIG,
        "raw_zip_sha256": batt3.sha256_file(batt3.RAW_ZIP),
        "record_count": len(records),
        "fit_summary": fit_summaries,
        "validation_rank_by_rmse": rank,
        "best_validation_model": best,
        "split_summary": split_summary,
        "outputs": {
            "metrics_by_trajectory_csv": str(metrics_by_trajectory_path.relative_to(ROOT)),
            "metrics_by_split_csv": str(metrics_by_split_path.relative_to(ROOT)),
            "predictions_sample_csv": str(predictions_path.relative_to(ROOT)),
            "fit_summary_csv": str(fit_summary_path.relative_to(ROOT)),
            "improvement_vs_ocv_csv": str(improvement_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
            "figures": [
                str((FIGURE_DIR / "current_sign_rmse_by_split.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "current_sign_rmse_by_split_zh.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "ohmic_resistance_by_sign.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "ohmic_resistance_by_sign_zh.svg").relative_to(ROOT)),
            ],
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    provenance = {
        **CONFIG,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "script": "scripts/run_batt3c_discharge_ocv_ecm.py",
        "outputs": metrics["outputs"] | {
            "provenance_json": str(provenance_path.relative_to(ROOT)),
        },
        "test_usage_rule": "Global ECM parameters are fitted only on split=train for each current convention. Non-train records use only the initialization prefix for initial-state estimation.",
        "next_action": "Use the better current convention as the BATT-4 baseline if the corrected ECM produces physically plausible parameters and improves validation RMSE.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
