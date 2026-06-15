#!/usr/bin/env python3
"""Run a lightweight reproducibility smoke test.

The smoke test intentionally avoids third-party raw datasets. It runs the
local-linear pendulum experiment and the coefficient-slot screening protocol,
then checks a few machine-readable outputs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(relative_path: str) -> None:
    print(f"[smoke] running {relative_path}", flush=True)
    subprocess.run([sys.executable, relative_path], cwd=ROOT, check=True)


def load_json(relative_path: str) -> dict:
    path = ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Expected output not found: {relative_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    run_script("scripts/run_e3_local_linearization_pendulum.py")
    run_script("scripts/run_slot_screening.py")

    e3_metrics = load_json("results/e3_local_linearization_pendulum/metrics.json")
    slot_metrics = load_json("results/slot_screening/metrics.json")

    e3_error = float(
        e3_metrics["rollout_nrmse_by_split"]["test_extrap"]["B3_slot_constrained"]["mean_nrmse_all"]
    )
    if e3_error >= 0.01:
        raise RuntimeError(f"E3 B3 extrapolation error is unexpectedly high: {e3_error}")

    if not bool(slot_metrics["all_selected_are_expected_correct"]):
        raise RuntimeError("Slot-screening protocol did not recover the expected candidate set.")

    expected = {
        "E3_pendulum": "stiffness_theta_even",
        "E4_drag": "drag_v_abs",
        "E5_monod": "growth_rate_rational_S",
    }
    if slot_metrics["selected_candidates"] != expected:
        raise RuntimeError(f"Unexpected slot-screening decisions: {slot_metrics['selected_candidates']}")

    print(
        json.dumps(
            {
                "status": "ok",
                "e3_test_extrap_B3_mean_nrmse": e3_error,
                "selected_candidates": slot_metrics["selected_candidates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
