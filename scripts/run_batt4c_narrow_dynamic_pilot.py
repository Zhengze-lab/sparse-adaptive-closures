#!/usr/bin/env python3
"""Run BATT-4c: narrow no-intercept dynamic residual pilot after R0(T)."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

import run_batt3_constant_ecm as batt3
import run_batt3c_discharge_ocv_ecm as batt3c
import run_batt4a_r0_slot_ecm as b4
import run_batt4a_filtered_r0_slot_ecm as b4f
import run_batt4b_r0_residual_diagnostics as b4b


ROOT = batt3.ROOT
RESULT_DIR = ROOT / "results" / "battery_lfp_narrow_dynamic_pilot"
FIGURE_DIR = RESULT_DIR / "figures"
PROVENANCE_DIR = ROOT / "data" / "provenance"


CONFIG: dict[str, Any] = {
    "experiment_id": "BATT4c_lfp_narrow_dynamic_pilot",
    "description": "Narrow no-intercept q_tau dynamic residual pilot on top of the physics-filtered BATT-4a R0(T) slot.",
    "source_dataset": "battery_lfp_ocv_dyn",
    "source_url": "https://data.mendeley.com/datasets/p8kf893yv3/1",
    "source_doi": "10.17632/p8kf893yv3.1",
    "license": "CC BY 4.0",
    "baseline_experiment": "BATT4a2_lfp_filtered_r0_slot_ecm",
    "diagnostic_experiment": "BATT4b_lfp_r0_residual_diagnostics",
    "base_model": "BATT4a2_R0T_physical",
    "selected_r0_model": b4b.CONFIG["selected_r0_model"],
    "selected_feature_set": b4b.CONFIG["selected_feature_set"],
    "ocv_curve": "discharge",
    "current_convention": "I_model=I_raw",
    "dynamic_model_form": "Vhat = OCV_discharge(s,T) - R0(T) I - k q_tau; q_tau[k+1]=exp(-dt/tau) q_tau[k] + (1-exp(-dt/tau)) I[k]",
    "dynamic_tau_grid_s": b4b.CONFIG["dynamic_tau_grid_s"],
    "gain_fit": "no_intercept_train_residual_lstsq",
    "selection_rule": "Select the lowest validation-temperature RMSE among q_tau candidates that do not worsen validation RMSE relative to R0(T).",
    "fit_stride": batt3.CONFIG["fit_stride"],
    "prediction_sample_stride": batt3.CONFIG["prediction_sample_stride"],
    "score_start_s": batt3.CONFIG["score_start_s"],
    "initialization_window_s": batt3.CONFIG["initialization_window_s"],
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    batt3.write_csv(path, rows)


def model_name(tau_s: float) -> str:
    return f"BATT4c_R0T_qtau{int(tau_s)}s_no_intercept"


def label_for_model(model: str, zh: bool = False) -> str:
    if model == str(CONFIG["base_model"]):
        return "BATT-4a2 R0(T)" if not zh else "BATT-4a2 R0(T)"
    if model.startswith("BATT4c_R0T_qtau"):
        tau = model.replace("BATT4c_R0T_qtau", "").replace("s_no_intercept", "")
        return f"BATT-4c q_tau={tau}s" if not zh else f"BATT-4c q_tau={tau}秒"
    return model


def prepare_base_cache(
    records: list[batt3.DynRecord],
    ocv_model: batt3.OcvModel,
    coeffs: np.ndarray,
    states: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    cache: list[dict[str, Any]] = []
    for record in records:
        state = states[record.label]
        pred, soc, r0 = b4.predict_slot_voltage(record, ocv_model, coeffs, float(state["s0"]))
        idx = batt3.score_indices(record)
        valid = (
            np.isfinite(pred[idx])
            & np.isfinite(record.voltage_v[idx])
            & np.isfinite(soc[idx])
            & np.isfinite(r0[idx])
        )
        score_idx = idx[valid]
        q_by_tau = {
            float(tau): b4b.filtered_current(record.time_s, record.current_a, float(tau))
            for tau in CONFIG["dynamic_tau_grid_s"]
        }
        cache.append(
            {
                "record": record,
                "state": state,
                "base_prediction_v": pred,
                "base_residual_v": pred - record.voltage_v,
                "soc": soc,
                "r0": r0,
                "score_idx": score_idx,
                "q_by_tau": q_by_tau,
            }
        )
    return cache


def fit_gain(cache: list[dict[str, Any]], tau_s: float) -> float:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for item in cache:
        record: batt3.DynRecord = item["record"]
        if record.split != "train":
            continue
        idx = item["score_idx"]
        x_parts.append(item["q_by_tau"][float(tau_s)][idx])
        y_parts.append(item["base_residual_v"][idx])
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def prediction_for_item(item: dict[str, Any], tau_s: float | None = None, gain: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    base = item["base_prediction_v"]
    if tau_s is None:
        return base, np.zeros_like(base)
    correction = gain * item["q_by_tau"][float(tau_s)]
    return base - correction, correction


def evaluate_model(
    cache: list[dict[str, Any]],
    model: str,
    tau_s: float | None = None,
    gain: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    trajectory_rows: list[dict[str, Any]] = []
    split_acc: dict[str, dict[str, list[float]]] = {}
    for item in cache:
        record: batt3.DynRecord = item["record"]
        idx = item["score_idx"]
        pred, correction = prediction_for_item(item, tau_s, gain)
        y = record.voltage_v[idx]
        score_pred = pred[idx]
        score_soc = item["soc"][idx]
        score_r0 = item["r0"][idx]
        score_correction = correction[idx]
        met = batt3.error_metrics(y, score_pred)
        trajectory_rows.append(
            {
                "model": model,
                "split": record.split,
                "record": record.label,
                "filename": record.filename,
                "temperature_c": int(record.temperature_c),
                "amplitude_token": int(record.amplitude_token),
                "tau_s": "" if tau_s is None else float(tau_s),
                "dynamic_gain_ohm": float(gain),
                "n_score": int(len(idx)),
                "s0": float(item["state"]["s0"]),
                "soc_min": float(np.min(score_soc)),
                "soc_max": float(np.max(score_soc)),
                "r0_min_ohm": float(np.min(score_r0)),
                "r0_max_ohm": float(np.max(score_r0)),
                "dynamic_correction_min_v": float(np.min(score_correction)),
                "dynamic_correction_max_v": float(np.max(score_correction)),
                "dynamic_correction_rms_mv": 1000.0 * float(np.sqrt(np.mean(score_correction * score_correction))),
                **met,
            }
        )
        split_acc.setdefault(record.split, {"y": [], "pred": [], "correction": [], "r0": []})
        split_acc[record.split]["y"].extend(float(v) for v in y)
        split_acc[record.split]["pred"].extend(float(v) for v in score_pred)
        split_acc[record.split]["correction"].extend(float(v) for v in score_correction)
        split_acc[record.split]["r0"].extend(float(v) for v in score_r0)

    split_rows: list[dict[str, Any]] = []
    split_summary: dict[str, dict[str, Any]] = {}
    for split, values in sorted(split_acc.items()):
        y = np.array(values["y"], dtype=float)
        pred = np.array(values["pred"], dtype=float)
        corr = np.array(values["correction"], dtype=float)
        r0 = np.array(values["r0"], dtype=float)
        row = {
            "model": model,
            "split": split,
            "tau_s": "" if tau_s is None else float(tau_s),
            "dynamic_gain_ohm": float(gain),
            "n_score": int(len(y)),
            "dynamic_correction_rms_mv": 1000.0 * float(np.sqrt(np.mean(corr * corr))),
            "r0_min_ohm": float(np.min(r0)),
            "r0_max_ohm": float(np.max(r0)),
            "r0_negative_fraction": float(np.mean(r0 < 0.0)),
            **batt3.error_metrics(y, pred),
        }
        split_rows.append(row)
        split_summary[f"{model}:{split}"] = row
    return trajectory_rows, split_rows, split_summary


def candidate_scan_rows(
    candidate_split_rows: list[dict[str, Any]],
    base_split_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base = {row["split"]: float(row["rmse_mv"]) for row in base_split_rows}
    rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    for row in candidate_split_rows:
        grouped.setdefault(row["model"], {"model": row["model"], "tau_s": row["tau_s"], "dynamic_gain_ohm": row["dynamic_gain_ohm"]})
        split = str(row["split"])
        rmse = float(row["rmse_mv"])
        grouped[row["model"]][f"{split}_rmse_mv"] = rmse
        grouped[row["model"]][f"{split}_improvement_percent"] = 100.0 * (base[split] - rmse) / base[split] if base.get(split) else float("nan")
    for model in sorted(grouped, key=lambda name: float(grouped[name]["tau_s"])):
        rows.append(grouped[model])
    return rows


def select_candidate(scan_rows: list[dict[str, Any]], base_validation_rmse: float) -> dict[str, Any]:
    feasible = [
        row for row in scan_rows
        if float(row["validation_temperature_rmse_mv"]) <= base_validation_rmse
    ]
    if not feasible:
        feasible = scan_rows
    return min(feasible, key=lambda row: float(row["validation_temperature_rmse_mv"]))


def prediction_sample_rows(
    cache: list[dict[str, Any]],
    selected: list[tuple[str, float | None, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stride = int(CONFIG["prediction_sample_stride"])
    for model, tau_s, gain in selected:
        for item in cache:
            record: batt3.DynRecord = item["record"]
            pred, correction = prediction_for_item(item, tau_s, gain)
            idx = np.arange(0, len(record.time_s), stride, dtype=int)
            for j in idx:
                rows.append(
                    {
                        "model": model,
                        "split": record.split,
                        "record": record.label,
                        "time_s": float(record.time_s[j]),
                        "temperature_c": int(record.temperature_c),
                        "current_a_model": float(record.current_a[j]),
                        "voltage_v": float(record.voltage_v[j]),
                        "prediction_v": float(pred[j]),
                        "error_mv": 1000.0 * float(pred[j] - record.voltage_v[j]),
                        "soc": float(item["soc"][j]),
                        "r0_ohm": float(item["r0"][j]),
                        "dynamic_correction_v": float(correction[j]),
                    }
                )
    return rows


def write_candidate_scan_svg(path: Path, scan_rows: list[dict[str, Any]], base_rows: list[dict[str, Any]], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    taus = np.array([float(row["tau_s"]) for row in scan_rows], dtype=float)
    val = np.array([float(row["validation_temperature_rmse_mv"]) for row in scan_rows], dtype=float)
    high = np.array([float(row["test_high_amplitude_rmse_mv"]) for row in scan_rows], dtype=float)
    base_lookup = {row["split"]: float(row["rmse_mv"]) for row in base_rows}
    base_val = base_lookup["validation_temperature"]
    base_high = base_lookup["test_high_amplitude"]
    left, top, panel_w, panel_h = 82.0, 64.0, 720.0, 300.0
    y_max = max(float(np.max(val)), float(np.max(high)), base_val, base_high) * 1.1
    y_min = 0.0
    x_min = float(np.min(taus))
    x_max = float(np.max(taus))

    def points(values: np.ndarray) -> str:
        xp = left + (taus - x_min) / (x_max - x_min) * panel_w
        yp = top + panel_h - (values - y_min) / (y_max - y_min) * panel_h
        return " ".join(f"{float(xx):.2f},{float(yy):.2f}" for xx, yy in zip(xp, yp))

    title = "BATT-4c narrow dynamic pilot scan" if not zh else "BATT-4c 窄动态 pilot 扫描"
    xlab = "tau (s)" if not zh else "tau (秒)"
    ylab = "RMSE (mV)" if not zh else "RMSE (mV)"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="990" height="430" viewBox="0 0 990 430">
  <rect width="990" height="430" fill="#ffffff"/>
  <text x="82" y="36" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="20" fill="#111827">{title}</text>
  <rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#ffffff" stroke="#d1d5db"/>
  <line x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" y2="{top + panel_h}" stroke="#6b7280"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_h}" stroke="#6b7280"/>
  <polyline points="{points(val)}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
  <polyline points="{points(high)}" fill="none" stroke="#dc2626" stroke-width="2.5"/>
  <line x1="{left}" y1="{top + panel_h - base_val / y_max * panel_h:.2f}" x2="{left + panel_w}" y2="{top + panel_h - base_val / y_max * panel_h:.2f}" stroke="#2563eb" stroke-width="1.3" stroke-dasharray="5 5"/>
  <line x1="{left}" y1="{top + panel_h - base_high / y_max * panel_h:.2f}" x2="{left + panel_w}" y2="{top + panel_h - base_high / y_max * panel_h:.2f}" stroke="#dc2626" stroke-width="1.3" stroke-dasharray="5 5"/>
  <text x="{left + panel_w / 2}" y="404" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{xlab}</text>
  <text x="24" y="{top + panel_h / 2}" transform="rotate(-90 24 {top + panel_h / 2})" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="13" fill="#374151">{ylab}</text>
  <rect x="830" y="96" width="18" height="11" fill="#2563eb"/>
  <text x="856" y="106" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">validation</text>
  <rect x="830" y="124" width="18" height="11" fill="#dc2626"/>
  <text x="856" y="134" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#111827">high amplitude</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_rmse_svg(path: Path, split_rows: list[dict[str, Any]], models: list[str], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    splits = ["train", "validation_temperature", "test_high_amplitude", "test_cell_transfer"]
    split_labels = {
        "train": "训练" if zh else "Train",
        "validation_temperature": "留出温度" if zh else "Held-out T",
        "test_high_amplitude": "高倍率" if zh else "High amp.",
        "test_cell_transfer": "跨电芯" if zh else "Cell transfer",
    }
    colors = ["#6b7280", "#2563eb", "#dc2626"]
    lookup = {(row["model"], row["split"]): float(row["rmse_mv"]) for row in split_rows}
    y_max = max(lookup.get((model, split), 0.0) for model in models for split in splits) * 1.12
    left, top, width, height = 86.0, 70.0, 820.0, 320.0
    group_w = width / len(splits)
    bar_w = 28.0
    gap = 6.0
    bars: list[str] = []
    for i, split in enumerate(splits):
        base_x = left + i * group_w + group_w / 2 - (len(models) * bar_w + (len(models) - 1) * gap) / 2
        for j, model in enumerate(models):
            value = lookup.get((model, split), 0.0)
            bar_h = value / y_max * height if y_max > 0 else 0.0
            x = base_x + j * (bar_w + gap)
            y = top + height - bar_h
            bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{colors[j % len(colors)]}"/>')
        bars.append(f'<text x="{left + i * group_w + group_w / 2:.2f}" y="420" text-anchor="middle" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="12" fill="#374151">{split_labels[split]}</text>')
    legend: list[str] = []
    for i, model in enumerate(models):
        y = 90 + i * 26
        legend.append(f'<rect x="930" y="{y - 12}" width="17" height="11" fill="{colors[i % len(colors)]}"/>')
        legend.append(f'<text x="956" y="{y - 2}" font-family="Noto Sans CJK SC, DejaVu Sans, Arial" font-size="11" fill="#111827">{label_for_model(model, zh=zh)}</text>')
    title = "BATT-4c selected dynamic pilot RMSE" if not zh else "BATT-4c 选中动态 pilot RMSE"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="460" viewBox="0 0 1260 460">
  <rect width="1260" height="460" fill="#ffffff"/>
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


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    b4f.configure_feature_set(str(CONFIG["selected_feature_set"]))
    records = batt3c.records_for_current_convention(batt3.load_dynamic_records(), str(CONFIG["current_convention"]))
    ocv_model = batt3c.load_ocv_model(str(CONFIG["ocv_curve"]))
    coeffs = b4b.read_selected_r0_coeffs()
    states = b4b.fit_r0t_states(records, ocv_model, coeffs)
    cache = prepare_base_cache(records, ocv_model, coeffs, states)

    all_trajectory_rows: list[dict[str, Any]] = []
    all_split_rows: list[dict[str, Any]] = []
    split_summary: dict[str, dict[str, Any]] = {}

    base_trajectory, base_split, base_summary = evaluate_model(cache, str(CONFIG["base_model"]))
    all_trajectory_rows.extend(base_trajectory)
    all_split_rows.extend(base_split)
    split_summary.update(base_summary)

    fit_rows: list[dict[str, Any]] = []
    candidate_split_rows: list[dict[str, Any]] = []
    candidate_trajectory_rows: list[dict[str, Any]] = []
    candidate_specs: dict[str, tuple[float, float]] = {}
    for tau in CONFIG["dynamic_tau_grid_s"]:
        tau_f = float(tau)
        gain = fit_gain(cache, tau_f)
        model = model_name(tau_f)
        traj_rows, split_rows, summary = evaluate_model(cache, model, tau_f, gain)
        candidate_trajectory_rows.extend(traj_rows)
        candidate_split_rows.extend(split_rows)
        all_trajectory_rows.extend(traj_rows)
        all_split_rows.extend(split_rows)
        split_summary.update(summary)
        candidate_specs[model] = (tau_f, gain)
        fit_rows.append(
            {
                "model": model,
                "tau_s": tau_f,
                "dynamic_gain_ohm": gain,
                "gain_fit": CONFIG["gain_fit"],
            }
        )

    scan_rows = candidate_scan_rows(candidate_split_rows, base_split)
    base_validation_rmse = next(float(row["rmse_mv"]) for row in base_split if row["split"] == "validation_temperature")
    selected = select_candidate(scan_rows, base_validation_rmse)
    selected_model = str(selected["model"])
    selected_tau, selected_gain = candidate_specs[selected_model]
    best_high = min(scan_rows, key=lambda row: float(row["test_high_amplitude_rmse_mv"]))
    best_high_model = str(best_high["model"])
    best_high_tau, best_high_gain = candidate_specs[best_high_model]

    selected_models = [(str(CONFIG["base_model"]), None, 0.0), (selected_model, selected_tau, selected_gain)]
    if best_high_model != selected_model:
        selected_models.append((best_high_model, best_high_tau, best_high_gain))

    prediction_rows = prediction_sample_rows(cache, selected_models)

    metrics_by_trajectory_path = RESULT_DIR / "metrics_by_trajectory.csv"
    metrics_by_split_path = RESULT_DIR / "metrics_by_split.csv"
    candidate_scan_path = RESULT_DIR / "candidate_scan.csv"
    predictions_path = RESULT_DIR / "predictions_sample.csv"
    fit_summary_path = RESULT_DIR / "fit_summary.csv"
    metrics_path = RESULT_DIR / "metrics.json"
    provenance_path = PROVENANCE_DIR / "batt4c_narrow_dynamic_pilot_provenance.json"

    write_csv(metrics_by_trajectory_path, all_trajectory_rows)
    write_csv(metrics_by_split_path, all_split_rows)
    write_csv(candidate_scan_path, scan_rows)
    write_csv(predictions_path, prediction_rows)
    write_csv(fit_summary_path, fit_rows)
    write_candidate_scan_svg(FIGURE_DIR / "candidate_scan.svg", scan_rows, base_split, zh=False)
    write_candidate_scan_svg(FIGURE_DIR / "candidate_scan_zh.svg", scan_rows, base_split, zh=True)
    write_rmse_svg(FIGURE_DIR / "rmse_by_split.svg", all_split_rows, [item[0] for item in selected_models], zh=False)
    write_rmse_svg(FIGURE_DIR / "rmse_by_split_zh.svg", all_split_rows, [item[0] for item in selected_models], zh=True)

    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "status": "complete",
        "config": CONFIG,
        "r0_coefficients_ohm": coeffs.tolist(),
        "base_model_split_rows": base_split,
        "candidate_scan": scan_rows,
        "selected_by_validation": selected,
        "selected_by_high_amplitude": best_high,
        "split_summary": split_summary,
        "outputs": {
            "metrics_by_trajectory_csv": str(metrics_by_trajectory_path.relative_to(ROOT)),
            "metrics_by_split_csv": str(metrics_by_split_path.relative_to(ROOT)),
            "candidate_scan_csv": str(candidate_scan_path.relative_to(ROOT)),
            "predictions_sample_csv": str(predictions_path.relative_to(ROOT)),
            "fit_summary_csv": str(fit_summary_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
            "figures": [
                str((FIGURE_DIR / "candidate_scan.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "candidate_scan_zh.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "rmse_by_split.svg").relative_to(ROOT)),
                str((FIGURE_DIR / "rmse_by_split_zh.svg").relative_to(ROOT)),
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
        "script": "scripts/run_batt4c_narrow_dynamic_pilot.py",
        "inputs": {
            "batt4a2_coefficients_csv": str((b4b.BATT4A2_RESULT_DIR / "coefficients.csv").relative_to(ROOT)),
            "raw_zip": str(batt3.RAW_ZIP.relative_to(ROOT)),
            "ocv_grid_csv": str(batt3.OCV_GRID_CSV.relative_to(ROOT)),
        },
        "outputs": metrics["outputs"] | {
            "provenance_json": str(provenance_path.relative_to(ROOT)),
        },
        "test_usage_rule": "R0(T) coefficients are fixed from BATT-4a2. The q_tau gain is fitted only on train residuals without an intercept; non-train records use only prefix SOC initialization.",
        "next_action": "Compare the selected narrow dynamic pilot against a cell-transfer/SOC/OCV correction route before claiming broad battery ECM transfer.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
