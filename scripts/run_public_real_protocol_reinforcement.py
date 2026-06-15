#!/usr/bin/env python3
"""Reinforce public real-system evidence with validation screening and segment tests.

This script does not replace the main E6/E7 experiments. It adds a protocol
check that mirrors the synthetic slot-screening logic on public real systems:
enumerate candidates before the final test, select among physically admissible
slot candidates on a validation split, and report official-test performance and
segment-level robustness after the decision is frozen.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "public_real_protocol_reinforcement"
FIGURE_DIR = RESULT_DIR / "figures"
PROVENANCE_DIR = ROOT / "data" / "provenance"


CONFIG: dict[str, Any] = {
    "experiment_id": "public_real_protocol_reinforcement",
    "description": "Validation-screening and segment-robustness checks for public real-system E6/E7 cases.",
    "created_for": "public real-system slot-selection protocol reinforcement.",
    "e6": {
        "train_fraction": 0.70,
        "segment_length": 128,
        "physical_slot_candidates": ["B0_linear_outflow", "B3_sqrt_outflow_slot"],
        "free_predictive_baselines": [],
        "selection_rule": "choose the physically admissible candidate with the lowest validation output-error NRMSE; official test is used only after selection",
    },
    "e7": {
        "train_fraction": 0.70,
        "segment_length": 248,
        "physical_slot_candidates": ["B0_viscous", "B0_symmetric_coulomb", "B3_asymmetric_friction_slot"],
        "free_predictive_baselines": ["B1_full_polynomial_inverse"],
        "selection_rule": "choose among physical friction candidates by validation inverse-dynamics NRMSE; B1 is reported as a freer predictive baseline, not as a coefficient-slot candidate",
    },
}


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E6 = load_script("run_e6_cascaded_tanks_slot_oe", ROOT / "scripts" / "run_e6_cascaded_tanks_slot_oe.py")
E7 = load_script("run_e7_emps_friction_slot", ROOT / "scripts" / "run_e7_emps_friction_slot.py")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_e6_record(record: dict[str, Any], train_fraction: float) -> tuple[dict[str, Any], dict[str, Any]]:
    n = len(record["y"])
    cut = int(round(n * train_fraction))
    cut = min(max(cut, 128), n - 128)

    def make(start: int, stop: int, split: str, init_window: int) -> dict[str, Any]:
        return {
            "split": split,
            "u": np.asarray(record["u"][start:stop], dtype=float),
            "y": np.asarray(record["y"][start:stop], dtype=float),
            "sampling_time": float(record["sampling_time"]),
            "state_initialization_window": init_window,
        }

    return make(0, cut, "protocol_train", 0), make(cut, n, "protocol_validation", int(E6.CONFIG["state_initialization_window"]))


def e6_predict_on_record(model: str, fit: dict[str, Any], record: dict[str, Any]) -> tuple[np.ndarray, int]:
    score_init = int(record.get("state_initialization_window", E6.CONFIG["state_initialization_window"]))
    if score_init > 0:
        x10 = E6.initialize_x10(
            model,
            fit["params"],
            record["u"],
            record["y"],
            float(record["sampling_time"]),
            float(fit["x10_estimation"]),
            score_init,
        )
    else:
        score_init = int(E6.CONFIG["state_initialization_window"])
        x10 = float(fit["x10_estimation"])
    pred = E6.simulate(record["u"], float(record["y"][0]), x10, float(record["sampling_time"]), model, fit["params"])
    return pred, score_init


def segment_nrmse(y: np.ndarray, pred: np.ndarray, start: int, stop: int) -> float:
    truth = y[start:stop]
    estimate = pred[start:stop]
    denom = float(np.std(truth)) or 1.0
    return float(np.sqrt(np.mean((estimate - truth) ** 2)) / denom)


def run_e6_protocol() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records = E6.load_records()
    train_record, validation_record = split_e6_record(records["estimation"], float(CONFIG["e6"]["train_fraction"]))
    official_test = records["test"]
    models = ["B0_linear_outflow", "B3_sqrt_outflow_slot"]

    rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, np.ndarray]] = {model: {} for model in models}
    fits = {}

    for model in models:
        fit = E6.fit_model(
            model,
            train_record["u"],
            train_record["y"],
            float(train_record["sampling_time"]),
            E6.make_starts(model, train_record["y"], train_record["u"]),
        )
        fits[model] = fit
        for split_name, record in [
            ("protocol_train", train_record),
            ("protocol_validation", validation_record),
            ("official_test", official_test),
        ]:
            pred, score_init = e6_predict_on_record(model, fit, record)
            predictions[model][split_name] = pred
            metrics = E6.error_metrics(record["y"], pred, score_init)
            rows.append(
                {
                    "system": "E6_Cascaded_Tanks",
                    "model": model,
                    "model_role": "physical_slot_candidate",
                    "split": split_name,
                    "n_samples": int(len(record["y"])),
                    "score_start_index": int(score_init),
                    "nrmse": metrics["nrmse"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "fit_percent": metrics["fit_percent"],
                }
            )

    validation_by_model = {row["model"]: float(row["nrmse"]) for row in rows if row["split"] == "protocol_validation"}
    selected = min(CONFIG["e6"]["physical_slot_candidates"], key=lambda name: validation_by_model[name])

    test_y = official_test["y"]
    start = int(official_test.get("state_initialization_window", E6.CONFIG["state_initialization_window"]))
    seg_len = int(CONFIG["e6"]["segment_length"])
    for segment_id, seg_start in enumerate(range(start, len(test_y) - seg_len + 1, seg_len)):
        seg_stop = seg_start + seg_len
        b0 = segment_nrmse(test_y, predictions["B0_linear_outflow"]["official_test"], seg_start, seg_stop)
        b3 = segment_nrmse(test_y, predictions["B3_sqrt_outflow_slot"]["official_test"], seg_start, seg_stop)
        segment_rows.append(
            {
                "system": "E6_Cascaded_Tanks",
                "segment_id": segment_id,
                "start_index": seg_start,
                "stop_index": seg_stop,
                "B0_nrmse": b0,
                "B3_nrmse": b3,
                "improvement_percent": 100.0 * (b0 - b3) / b0 if b0 else float("nan"),
                "selected_model_better": bool(b3 < b0),
            }
        )

    test_by_model = {row["model"]: float(row["nrmse"]) for row in rows if row["split"] == "official_test"}
    summary = {
        "selected_physical_slot_model": selected,
        "validation_b0_nrmse": validation_by_model["B0_linear_outflow"],
        "validation_b3_nrmse": validation_by_model["B3_sqrt_outflow_slot"],
        "official_test_b0_nrmse": test_by_model["B0_linear_outflow"],
        "official_test_b3_nrmse": test_by_model["B3_sqrt_outflow_slot"],
        "official_test_improvement_percent": 100.0
        * (test_by_model["B0_linear_outflow"] - test_by_model["B3_sqrt_outflow_slot"])
        / test_by_model["B0_linear_outflow"],
        "test_segments": len(segment_rows),
        "test_segments_improved": int(sum(row["selected_model_better"] for row in segment_rows)),
    }
    return rows, segment_rows, summary


def slice_e7_record(record: dict[str, Any], start: int, stop: int, split: str) -> dict[str, Any]:
    return {
        "split": split,
        "path": record["path"],
        "dt": float(record["dt"]),
        "t": np.asarray(record["t"][start:stop], dtype=float),
        "q": np.asarray(record["q"][start:stop], dtype=float),
        "v": np.asarray(record["v"][start:stop], dtype=float),
        "acc": np.asarray(record["acc"][start:stop], dtype=float),
        "force": np.asarray(record["force"][start:stop], dtype=float),
        "gtau": float(record["gtau"]),
        "n": int(stop - start),
    }


def split_e7_record(record: dict[str, Any], train_fraction: float) -> tuple[dict[str, Any], dict[str, Any]]:
    n = int(record["n"])
    cut = int(round(n * train_fraction))
    cut = min(max(cut, 256), n - 256)
    return slice_e7_record(record, 0, cut, "protocol_train"), slice_e7_record(record, cut, n, "protocol_validation")


def run_e7_protocol() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    E7.ensure_data()
    full_train = E7.load_record(E7.RAW_ROOT / "EMPS" / "DATA_EMPS.mat", "train")
    official_test = E7.load_record(E7.RAW_ROOT / "EMPS" / "DATA_EMPS_PULSES.mat", "test")
    train_record, validation_record = split_e7_record(full_train, float(CONFIG["e7"]["train_fraction"]))
    models = [
        "B0_viscous",
        "B0_symmetric_coulomb",
        "B3_asymmetric_friction_slot",
        "B1_full_polynomial_inverse",
    ]

    rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    fits = {model: E7.fit_ls(model, train_record) for model in models}
    predictions: dict[str, dict[str, np.ndarray]] = {model: {} for model in models}

    for model, fit in fits.items():
        role = "free_predictive_baseline" if model in CONFIG["e7"]["free_predictive_baselines"] else "physical_slot_candidate"
        for split_name, record in [
            ("protocol_train", train_record),
            ("protocol_validation", validation_record),
            ("official_test", official_test),
        ]:
            pred = E7.predict(fit, record)
            predictions[model][split_name] = pred
            metric = E7.metrics(record["force"], pred)
            rows.append(
                {
                    "system": "E7_EMPS",
                    "model": model,
                    "model_role": role,
                    "split": split_name,
                    "n_samples": int(record["n"]),
                    "score_start_index": 0,
                    "nrmse": metric["nrmse"],
                    "rmse": metric["rmse_n"],
                    "mae": metric["mae_n"],
                    "fit_percent": metric["fit_percent"],
                }
            )

    validation_by_model = {row["model"]: float(row["nrmse"]) for row in rows if row["split"] == "protocol_validation"}
    selected = min(CONFIG["e7"]["physical_slot_candidates"], key=lambda name: validation_by_model[name])

    test_y = official_test["force"]
    seg_len = int(CONFIG["e7"]["segment_length"])
    for segment_id, seg_start in enumerate(range(0, len(test_y) - seg_len + 1, seg_len)):
        seg_stop = seg_start + seg_len
        b0 = segment_nrmse(test_y, predictions["B0_symmetric_coulomb"]["official_test"], seg_start, seg_stop)
        b3 = segment_nrmse(test_y, predictions["B3_asymmetric_friction_slot"]["official_test"], seg_start, seg_stop)
        b1 = segment_nrmse(test_y, predictions["B1_full_polynomial_inverse"]["official_test"], seg_start, seg_stop)
        segment_rows.append(
            {
                "system": "E7_EMPS",
                "segment_id": segment_id,
                "start_index": seg_start,
                "stop_index": seg_stop,
                "B0_nrmse": b0,
                "B3_nrmse": b3,
                "B1_nrmse": b1,
                "improvement_percent": 100.0 * (b0 - b3) / b0 if b0 else float("nan"),
                "selected_model_better": bool(b3 < b0),
                "free_model_best": bool(b1 < b3),
            }
        )

    test_by_model = {row["model"]: float(row["nrmse"]) for row in rows if row["split"] == "official_test"}
    summary = {
        "selected_physical_slot_model": selected,
        "validation_b0_symmetric_nrmse": validation_by_model["B0_symmetric_coulomb"],
        "validation_b3_nrmse": validation_by_model["B3_asymmetric_friction_slot"],
        "validation_b1_full_nrmse": validation_by_model["B1_full_polynomial_inverse"],
        "official_test_b0_symmetric_nrmse": test_by_model["B0_symmetric_coulomb"],
        "official_test_b3_nrmse": test_by_model["B3_asymmetric_friction_slot"],
        "official_test_b1_full_nrmse": test_by_model["B1_full_polynomial_inverse"],
        "official_test_improvement_percent": 100.0
        * (test_by_model["B0_symmetric_coulomb"] - test_by_model["B3_asymmetric_friction_slot"])
        / test_by_model["B0_symmetric_coulomb"],
        "test_segments": len(segment_rows),
        "test_segments_improved": int(sum(row["selected_model_better"] for row in segment_rows)),
        "test_segments_free_model_best": int(sum(row["free_model_best"] for row in segment_rows)),
    }
    return rows, segment_rows, summary


def write_summary_svg(path: Path, candidate_rows: list[dict[str, Any]], segment_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    systems = ["E6_Cascaded_Tanks", "E7_EMPS"]
    labels = {"E6_Cascaded_Tanks": "Cascaded Tanks", "E7_EMPS": "EMPS"}
    selected = {"E6_Cascaded_Tanks": "B3_sqrt_outflow_slot", "E7_EMPS": "B3_asymmetric_friction_slot"}
    baselines = {"E6_Cascaded_Tanks": "B0_linear_outflow", "E7_EMPS": "B0_symmetric_coulomb"}

    width, height = 980, 430
    left, top = 82, 58
    panel_w, panel_h = 360, 260
    gap = 78
    colors = {"B0": "#9E7B55", "B3": "#2E8B6F", "B1": "#4E79A7"}

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">Public real-system validation screening and segment robustness</text>',
    ]

    for panel, system in enumerate(systems):
        x0 = left + panel * (panel_w + gap)
        rows = [row for row in candidate_rows if row["system"] == system and row["split"] in {"protocol_validation", "official_test"}]
        base = baselines[system]
        slot = selected[system]
        vals = {
            ("validation", "B0"): next(float(row["nrmse"]) for row in rows if row["split"] == "protocol_validation" and row["model"] == base),
            ("validation", "B3"): next(float(row["nrmse"]) for row in rows if row["split"] == "protocol_validation" and row["model"] == slot),
            ("test", "B0"): next(float(row["nrmse"]) for row in rows if row["split"] == "official_test" and row["model"] == base),
            ("test", "B3"): next(float(row["nrmse"]) for row in rows if row["split"] == "official_test" and row["model"] == slot),
        }
        e7_b1 = None
        if system == "E7_EMPS":
            e7_b1 = {
                "validation": next(float(row["nrmse"]) for row in rows if row["split"] == "protocol_validation" and row["model"] == "B1_full_polynomial_inverse"),
                "test": next(float(row["nrmse"]) for row in rows if row["split"] == "official_test" and row["model"] == "B1_full_polynomial_inverse"),
            }
        all_values = list(vals.values())
        if e7_b1:
            all_values.extend(e7_b1.values())
        ymax = max(all_values) * 1.28

        def sx(group: int, offset: float) -> float:
            return x0 + 80 + group * 150 + offset

        def sy(value: float) -> float:
            return top + panel_h * (1.0 - value / ymax)

        lines.append(f'<text x="{x0 + panel_w/2}" y="{top - 18}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold">{labels[system]}</text>')
        lines.append(f'<line x1="{x0}" y1="{top + panel_h}" x2="{x0 + panel_w}" y2="{top + panel_h}" stroke="#333"/>')
        lines.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{top + panel_h}" stroke="#333"/>')
        for group, split in enumerate(["validation", "test"]):
            for label, dx, color_key in [("B0", -22, "B0"), ("B3", 22, "B3")]:
                value = vals[(split, label)]
                y = sy(value)
                x = sx(group, dx)
                lines.append(f'<rect x="{x-18}" y="{y}" width="36" height="{top + panel_h - y}" fill="{colors[color_key]}"/>')
                lines.append(f'<text x="{x}" y="{y-5}" text-anchor="middle" font-family="Arial" font-size="10">{value:.3f}</text>')
            if e7_b1:
                value = e7_b1[split]
                y = sy(value)
                x = sx(group, 61)
                lines.append(f'<rect x="{x-15}" y="{y}" width="30" height="{top + panel_h - y}" fill="{colors["B1"]}"/>')
                lines.append(f'<text x="{x}" y="{y-5}" text-anchor="middle" font-family="Arial" font-size="10">{value:.3f}</text>')
            lines.append(f'<text x="{sx(group, 0)}" y="{top + panel_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{split}</text>')

        segs = [row for row in segment_rows if row["system"] == system]
        improved = sum(bool(row["selected_model_better"]) for row in segs)
        pct = 100.0 * improved / len(segs) if segs else 0.0
        lines.append(f'<text x="{x0 + panel_w/2}" y="{top + panel_h + 56}" text-anchor="middle" font-family="Arial" font-size="12">official-test segments improved: {improved}/{len(segs)} ({pct:.0f}%)</text>')

    legend_y = height - 24
    for i, (label, color_key) in enumerate([("B0 physical baseline", "B0"), ("selected slot", "B3"), ("free B1 baseline", "B1")]):
        x = left + i * 260
        lines.append(f'<rect x="{x}" y="{legend_y - 12}" width="18" height="12" fill="{colors[color_key]}"/>')
        lines.append(f'<text x="{x + 26}" y="{legend_y - 2}" font-family="Arial" font-size="11">{label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    e6_rows, e6_segments, e6_summary = run_e6_protocol()
    e7_rows, e7_segments, e7_summary = run_e7_protocol()
    candidate_rows = e6_rows + e7_rows
    segment_rows = e6_segments + e7_segments

    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": CONFIG,
        "summary": {
            "E6_Cascaded_Tanks": e6_summary,
            "E7_EMPS": e7_summary,
            "n_positive_public_systems_with_validation_selected_slots": int(
                e6_summary["selected_physical_slot_model"] == "B3_sqrt_outflow_slot"
            )
            + int(e7_summary["selected_physical_slot_model"] == "B3_asymmetric_friction_slot"),
        },
        "candidate_screening_rows": candidate_rows,
        "segment_rows": segment_rows,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }

    write_csv(RESULT_DIR / "candidate_screening.csv", candidate_rows)
    write_csv(RESULT_DIR / "segment_metrics.csv", segment_rows)
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary_svg(FIGURE_DIR / "public_real_protocol_reinforcement.svg", candidate_rows, segment_rows)

    raw_files = []
    for root in [E6.RAW_ROOT, E7.RAW_ROOT]:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                raw_files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    provenance = {
        **CONFIG,
        "created_at_utc": metrics["created_at_utc"],
        "outputs": {
            "metrics_json": str((RESULT_DIR / "metrics.json").relative_to(ROOT)),
            "candidate_screening_csv": str((RESULT_DIR / "candidate_screening.csv").relative_to(ROOT)),
            "segment_metrics_csv": str((RESULT_DIR / "segment_metrics.csv").relative_to(ROOT)),
            "figure_svg": str((FIGURE_DIR / "public_real_protocol_reinforcement.svg").relative_to(ROOT)),
        },
        "raw_files": raw_files,
    }
    (PROVENANCE_DIR / "public_real_protocol_reinforcement_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
