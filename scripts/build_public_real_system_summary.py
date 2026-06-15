#!/usr/bin/env python3
"""Build a compact E6-E9 public real-system experiment summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "public_real_system_summary"
FIGURE_DIR = RESULT_DIR / "figures"


CASES = [
    {
        "case_id": "E6",
        "system": "Cascaded Tanks",
        "role": "real positive",
        "metric": "rollout NRMSE",
        "source": "Nonlinear Benchmark / 4TU ResearchData",
        "metrics_path": ROOT / "results" / "e6_cascaded_tanks_slot_oe" / "metrics.json",
        "baseline_key": "test_b0_nrmse",
        "slot_key": "test_b3_nrmse",
        "improvement_key": "test_improvement_percent",
        "baseline": "B0 linear outflow",
        "slot": "B3 level-adaptive sqrt outflow slot",
    },
    {
        "case_id": "E7",
        "system": "EMPS",
        "role": "real positive",
        "metric": "inverse-dynamics force NRMSE",
        "source": "Nonlinear Benchmark EMPS",
        "metrics_path": ROOT / "results" / "e7_emps_friction_slot" / "metrics.json",
        "baseline_key": "test_b0_symmetric_nrmse",
        "slot_key": "test_b3_asymmetric_slot_nrmse",
        "improvement_key": "test_b3_vs_b0_improvement_percent",
        "baseline": "B0 symmetric Coulomb friction",
        "slot": "B3 asymmetric friction slot",
    },
    {
        "case_id": "E8",
        "system": "PMSM motor temperature",
        "role": "engineering boundary",
        "metric": "PM-temperature rollout NRMSE",
        "source": "Kaggle Electric Motor Temperature",
        "metrics_path": ROOT / "results" / "e8_pmsm_temperature_slot" / "metrics.json",
        "baseline_key": "test_b0_rollout_nrmse",
        "slot_key": "test_b3_operating_rollout_nrmse",
        "improvement_key": "test_b3_operating_vs_b0_improvement_percent",
        "baseline": "B0 fixed cooling coefficient",
        "slot": "B3 operating-adaptive cooling slot",
    },
    {
        "case_id": "E9",
        "system": "ORNL building thermal",
        "role": "application boundary",
        "metric": "mean-zone-temperature rollout NRMSE",
        "source": "figshare / Scientific Data",
        "metrics_path": ROOT / "results" / "e9_ornl_building_thermal_slot" / "metrics.json",
        "baseline_key": "test_b0_rollout_nrmse",
        "slot_key": "test_b3_rollout_nrmse",
        "improvement_key": "test_b3_vs_b0_improvement_percent",
        "baseline": "B0 fixed envelope coefficient",
        "slot": "B3 weather-adaptive envelope slot",
    },
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rows() -> list[dict[str, Any]]:
    rows = []
    for case in CASES:
        with case["metrics_path"].open(encoding="utf-8") as handle:
            summary = json.load(handle)["summary"]
        baseline = float(summary[case["baseline_key"]])
        slot = float(summary[case["slot_key"]])
        improvement = float(summary[case["improvement_key"]])
        rows.append(
            {
                "case_id": case["case_id"],
                "system": case["system"],
                "role": case["role"],
                "source": case["source"],
                "metric": case["metric"],
                "baseline": case["baseline"],
                "slot_model": case["slot"],
                "baseline_error": baseline,
                "slot_error": slot,
                "improvement_percent": improvement,
            }
        )
    return rows


def write_svg(path: Path, rows: list[dict[str, Any]], zh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 880, 430
    left, right, top, bottom = 180, 50, 50, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [float(row["improvement_percent"]) for row in rows]
    v_min = min(min(values), -15.0)
    v_max = max(max(values), 45.0)
    title = "公开真实系统：系数槽相对固定系数改善率" if zh else "Public real systems: coefficient-slot improvement over fixed coefficients"
    x_zero = left + plot_w * (0.0 - v_min) / (v_max - v_min)

    def sx(value: float) -> float:
        return left + plot_w * (value - v_min) / (v_max - v_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">{title}</text>',
        f'<line x1="{x_zero:.2f}" y1="{top}" x2="{x_zero:.2f}" y2="{height-bottom}" stroke="#777" stroke-dasharray="4 3"/>',
    ]
    bar_h = 44
    gap = 28
    labels_zh = {
        "E6": "E6 串联水箱",
        "E7": "E7 EMPS",
        "E8": "E8 PMSM 温度",
        "E9": "E9 建筑热动态",
    }
    for idx, row in enumerate(rows):
        y = top + idx * (bar_h + gap)
        value = float(row["improvement_percent"])
        x_val = sx(value)
        x = min(x_zero, x_val)
        w = abs(x_val - x_zero)
        color = "#0072B2" if value >= 0 else "#D55E00"
        label = labels_zh[row["case_id"]] if zh else f'{row["case_id"]} {row["system"]}'
        role = "正例" if (zh and "positive" in row["role"]) else "边界" if zh else row["role"]
        parts.append(f'<text x="{left-12}" y="{y+bar_h/2+5}" text-anchor="end" font-family="Arial" font-size="13">{label}</text>')
        parts.append(f'<rect x="{x:.2f}" y="{y}" width="{max(w, 1):.2f}" height="{bar_h}" fill="{color}" opacity="0.82"/>')
        parts.append(f'<text x="{x_val + (6 if value >= 0 else -6):.2f}" y="{y+bar_h/2+5}" text-anchor="{"start" if value >= 0 else "end"}" font-family="Arial" font-size="12">{value:.2f}% ({role})</text>')
    parts.append(f'<text x="{left}" y="{height-18}" font-family="Arial" font-size="12">{"改善率为正表示系数槽优于固定系数基线" if zh else "Positive values indicate improvement over the fixed-coefficient baseline."}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    write_csv(RESULT_DIR / "public_real_system_summary.csv", rows)
    payload = {"rows": rows}
    (RESULT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_svg(FIGURE_DIR / "public_real_system_improvement.svg", rows, zh=False)
    write_svg(FIGURE_DIR / "public_real_system_improvement_zh.svg", rows, zh=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
