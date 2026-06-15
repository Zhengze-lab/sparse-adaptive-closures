#!/usr/bin/env python3
"""Run coefficient-slot screening experiments.

This script turns the previously manual slot choices into a reproducible
train/validation/test screening protocol. It does not use the final
extrapolation test split for slot selection.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "slot_screening"
FIGURE_DIR = RESULT_DIR / "figures"
PROVENANCE_DIR = ROOT / "data" / "provenance"


CONFIG: dict[str, Any] = {
    "experiment_id": "slot_screening",
    "random_seed": 20260615,
    "stlsq_threshold": 1e-8,
    "stlsq_max_iter": 30,
    "condition_limit_for_selection": 1e10,
    "min_train_residual_reduction": 1e-3,
    "selection_rule": "choose the physically admissible candidate with finite identifiable design and lowest validation rollout NRMSE",
    "test_usage_rule": "test_extrap splits are used only after the slot/library decision is frozen",
}


COLORS = {
    "selected": "#2E8B6F",
    "eligible": "#8FB9A8",
    "ineligible": "#B8BDC5",
    "bad": "#C8584A",
    "line": "#8F98A3",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": 160,
        "savefig.dpi": 600,
    }
)


def load_script_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e3 = load_script_module("e3_main", "scripts/run_e3_local_linearization_pendulum.py")
e4 = load_script_module("e4_main", "scripts/run_e4_velocity_drag.py")
e5 = load_script_module("e5_main", "scripts/run_e5_monod_rate_slot.py")


@dataclass
class CandidateResult:
    case_id: str
    candidate_id: str
    candidate_label: str
    coefficient_slot: str
    slot_input_z: str
    candidate_library: str
    physical_admissible: bool
    expected_correct: bool
    train_residual_reduction: float
    train_residual_nrmse: float
    validation_rollout_nrmse: float
    test_rollout_nrmse: float
    condition_number: float
    sensitivity_energy: float
    active_terms: int
    selected_by_protocol: bool = False
    validation_rank: int = 0

    def as_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "candidate_id": self.candidate_id,
            "candidate_label": self.candidate_label,
            "coefficient_slot": self.coefficient_slot,
            "slot_input_z": self.slot_input_z,
            "candidate_library": self.candidate_library,
            "physical_admissible": str(self.physical_admissible),
            "expected_correct": str(self.expected_correct),
            "train_residual_reduction": f"{self.train_residual_reduction:.10g}",
            "train_residual_nrmse": f"{self.train_residual_nrmse:.10g}",
            "validation_rollout_nrmse": f"{self.validation_rollout_nrmse:.10g}",
            "test_rollout_nrmse": f"{self.test_rollout_nrmse:.10g}",
            "condition_number": f"{self.condition_number:.10g}",
            "sensitivity_energy": f"{self.sensitivity_energy:.10g}",
            "active_terms": self.active_terms,
            "validation_rank": self.validation_rank,
            "selected_by_protocol": str(self.selected_by_protocol),
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stlsq(a: np.ndarray, y: np.ndarray, threshold: float | None = None, max_iter: int | None = None) -> np.ndarray:
    threshold = CONFIG["stlsq_threshold"] if threshold is None else threshold
    max_iter = CONFIG["stlsq_max_iter"] if max_iter is None else max_iter
    active = np.ones(a.shape[1], dtype=bool)
    coeffs = np.zeros(a.shape[1], dtype=float)
    for _ in range(max_iter):
        if not np.any(active):
            break
        local, *_ = np.linalg.lstsq(a[:, active], y, rcond=None)
        next_coeffs = np.zeros(a.shape[1], dtype=float)
        next_coeffs[active] = local
        next_active = np.abs(next_coeffs) >= threshold
        coeffs = next_coeffs
        if np.array_equal(active, next_active):
            break
        active = next_active
    if np.any(active):
        local, *_ = np.linalg.lstsq(a[:, active], y, rcond=None)
        coeffs = np.zeros(a.shape[1], dtype=float)
        coeffs[active] = local
    return coeffs


def nrmse_nd(true: np.ndarray, pred: np.ndarray) -> float:
    err = pred - true
    return float(np.sqrt(np.mean(err * err)) / (np.std(true) or 1.0))


def nrmse_1d(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred.reshape(-1) - true.reshape(-1)) ** 2)) / (np.std(true) or 1.0))


def train_score(target: np.ndarray, pred: np.ndarray, baseline_residual: np.ndarray) -> tuple[float, float]:
    residual = target - pred
    sse0 = float(np.dot(baseline_residual.reshape(-1), baseline_residual.reshape(-1)))
    sse = float(np.dot(residual.reshape(-1), residual.reshape(-1)))
    reduction = 1.0 - sse / sse0 if sse0 > 0 else 0.0
    denom = float(np.std(target)) or 1.0
    return reduction, float(np.sqrt(np.mean(residual.reshape(-1) ** 2)) / denom)


def condition_number(a: np.ndarray) -> float:
    if a.size == 0:
        return math.inf
    centered = a - np.mean(a, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, keepdims=True)
    keep = scale.reshape(-1) > 1e-12
    if np.sum(keep) == 0:
        return math.inf
    normalized = centered[:, keep] / scale[:, keep]
    try:
        return float(np.linalg.cond(normalized))
    except np.linalg.LinAlgError:
        return math.inf


def sensitivity_energy(a: np.ndarray) -> float:
    return float(np.linalg.norm(a) / math.sqrt(max(a.shape[0], 1)))


def safe_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else 99.0


def rank_and_select(rows: list[CandidateResult]) -> list[CandidateResult]:
    ranked = sorted(rows, key=lambda row: row.validation_rollout_nrmse)
    for rank, row in enumerate(ranked, start=1):
        row.validation_rank = rank
    eligible = [
        row
        for row in ranked
        if row.physical_admissible
        and math.isfinite(row.condition_number)
        and row.condition_number <= CONFIG["condition_limit_for_selection"]
        and row.train_residual_reduction >= CONFIG["min_train_residual_reduction"]
        and row.active_terms > 0
    ]
    if eligible:
        eligible[0].selected_by_protocol = True
    return rows


def rollout_e3(rhs: Callable[[float, np.ndarray], np.ndarray], trajectories: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for traj in trajectories:
        try:
            _, pred = e3.simulate(rhs, np.array(traj["x0"], dtype=float), e3.CONFIG["t_end"], e3.CONFIG["dt"])
            values.append(nrmse_nd(traj["x"], pred))
        except Exception:
            values.append(99.0)
    return safe_mean(values)


def phi_e3(x: np.ndarray, mode: str) -> tuple[np.ndarray, list[str]]:
    theta = x[:, 0]
    omega = x[:, 1]
    if mode == "theta_even":
        return np.column_stack([np.ones_like(theta), theta**2, theta**4, theta**6]), ["1", "theta^2", "theta^4", "theta^6"]
    if mode == "omega_even":
        return np.column_stack([np.ones_like(omega), omega**2, omega**4, omega**6]), ["1", "omega^2", "omega^4", "omega^6"]
    if mode == "theta_odd":
        return np.column_stack([theta, theta**3, theta**5]), ["theta", "theta^3", "theta^5"]
    raise ValueError(mode)


def run_e3_screening() -> list[CandidateResult]:
    trajectories = e3.build_trajectories()
    train_local = [traj for traj in trajectories if traj["split"] == "train_small"]
    train_screen = [traj for traj in trajectories if traj["split"] == "train_medium"]
    validation = [traj for traj in trajectories if traj["split"] == "test_interp"]
    test = [traj for traj in trajectories if traj["split"] == "test_extrap"]
    x_local, dx_local = e3.stack_data(train_local)
    x_train, dx_train = e3.stack_data(train_screen)
    c = float(e3.CONFIG["damping_c"])
    k0 = e3.fit_local_k0(x_local, dx_local, c)
    baseline_train = -c * x_train[:, 1] - k0 * x_train[:, 0]
    baseline_residual = dx_train[:, 1] - baseline_train

    specs = [
        ("stiffness_theta_even", "stiffness coefficient", "theta", "theta-even polynomial", True, True, "-theta", "theta_even"),
        ("damping_theta_even", "damping coefficient", "theta", "theta-even polynomial", True, False, "-omega", "theta_even"),
        ("stiffness_omega_even", "stiffness coefficient", "omega", "omega-even polynomial", True, False, "-theta", "omega_even"),
        ("damping_omega_even", "damping coefficient", "omega", "omega-even polynomial", True, False, "-omega", "omega_even"),
        ("free_additive_theta_odd", "additive residual", "theta", "theta-odd polynomial", False, False, "one", "theta_odd"),
    ]
    results: list[CandidateResult] = []
    for candidate_id, slot, z_name, library_name, admissible, expected, g_name, phi_name in specs:
        phi, terms = phi_e3(x_train, phi_name)
        if g_name == "-theta":
            g = -x_train[:, 0]
        elif g_name == "-omega":
            g = -x_train[:, 1]
        elif g_name == "one":
            g = np.ones(x_train.shape[0])
        else:
            raise ValueError(g_name)
        design = g[:, None] * phi
        coeffs = stlsq(design, baseline_residual)
        pred_delta = design @ coeffs
        reduction, residual_nrmse = train_score(dx_train[:, 1], baseline_train + pred_delta, baseline_residual)

        def rhs(_: float, xx: np.ndarray, *, coeffs: np.ndarray = coeffs, g_name: str = g_name, phi_name: str = phi_name) -> np.ndarray:
            phi_x, _ = phi_e3(xx.reshape(1, 2), phi_name)
            if g_name == "-theta":
                gx = -float(xx[0])
            elif g_name == "-omega":
                gx = -float(xx[1])
            else:
                gx = 1.0
            delta = gx * float((phi_x @ coeffs).reshape(-1)[0])
            return np.array([xx[1], -c * xx[1] - k0 * xx[0] + delta], dtype=float)

        results.append(
            CandidateResult(
                case_id="E3_pendulum",
                candidate_id=candidate_id,
                candidate_label=candidate_id.replace("_", " "),
                coefficient_slot=slot,
                slot_input_z=z_name,
                candidate_library=", ".join(terms),
                physical_admissible=admissible,
                expected_correct=expected,
                train_residual_reduction=reduction,
                train_residual_nrmse=residual_nrmse,
                validation_rollout_nrmse=rollout_e3(rhs, validation),
                test_rollout_nrmse=rollout_e3(rhs, test),
                condition_number=condition_number(design),
                sensitivity_energy=sensitivity_energy(design),
                active_terms=int(np.sum(np.abs(coeffs) >= 1e-6)),
            )
        )
    return rank_and_select(results)


def rollout_e4(rhs: Callable[[float, np.ndarray, dict[str, Any]], np.ndarray], trajectories: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for traj in trajectories:
        amp = float(traj["amplitude"])
        phase = float(traj["phase"])
        x0 = np.array([float(traj["v0"])], dtype=float)

        def local_rhs(tt: float, xx: np.ndarray) -> np.ndarray:
            return rhs(tt, xx, {"amplitude": amp, "phase": phase})

        try:
            _, pred = e4.simulate(local_rhs, x0, e4.CONFIG["t_end"], e4.CONFIG["dt"])
            values.append(nrmse_1d(traj["x"], pred))
        except Exception:
            values.append(99.0)
    return safe_mean(values)


def phi_e4(v: np.ndarray, u: np.ndarray, mode: str) -> tuple[np.ndarray, list[str]]:
    if mode == "v_abs":
        return np.column_stack([np.ones_like(v), np.abs(v)]), ["1", "abs(v)"]
    if mode == "u_abs":
        return np.column_stack([np.ones_like(v), np.abs(u)]), ["1", "abs(u)"]
    if mode == "v_poly":
        return np.column_stack([v, v * np.abs(v)]), ["v", "v*abs(v)"]
    if mode == "u_gain":
        return np.column_stack([np.ones_like(v), np.abs(u), u]), ["1", "abs(u)", "u"]
    raise ValueError(mode)


def run_e4_screening() -> list[CandidateResult]:
    trajectories = e4.build_trajectories()
    train_local = [traj for traj in trajectories if traj["split"] == "train_low"]
    train_screen = [traj for traj in trajectories if traj["split"] in {"train_low", "train_medium"}]
    validation = [traj for traj in trajectories if traj["split"] == "test_interp"]
    test = [traj for traj in trajectories if traj["split"] == "test_extrap"]
    v_local, dx_local, u_local = e4.stack_data(train_local)
    v_train, dx_train, u_train = e4.stack_data(train_screen)
    c0 = e4.fit_local_c(v_local, dx_local, u_local)
    baseline_train = u_train - c0 * v_train
    baseline_residual = dx_train - baseline_train

    specs = [
        ("drag_v_abs", "drag coefficient", "v", "1, abs(v)", True, True, "-v", "v_abs"),
        ("drag_u_abs", "drag coefficient", "u", "1, abs(u)", True, False, "-v", "u_abs"),
        ("input_gain_u", "input gain", "u", "1, abs(u), u", True, False, "u", "u_gain"),
        ("free_additive_v", "additive residual", "v", "v, v*abs(v)", False, False, "one", "v_poly"),
    ]
    results: list[CandidateResult] = []
    for candidate_id, slot, z_name, library_name, admissible, expected, g_name, phi_name in specs:
        phi, terms = phi_e4(v_train, u_train, phi_name)
        if g_name == "-v":
            g = -v_train
        elif g_name == "u":
            g = u_train
        elif g_name == "one":
            g = np.ones_like(v_train)
        else:
            raise ValueError(g_name)
        design = g[:, None] * phi
        coeffs = stlsq(design, baseline_residual)
        pred_delta = design @ coeffs
        reduction, residual_nrmse = train_score(dx_train, baseline_train + pred_delta, baseline_residual)

        def rhs(tt: float, xx: np.ndarray, spec: dict[str, Any], *, coeffs: np.ndarray = coeffs, g_name: str = g_name, phi_name: str = phi_name) -> np.ndarray:
            amp = float(spec["amplitude"])
            phase = float(spec["phase"])
            v = float(xx[0])
            u = e4.input_signal(tt, amp, phase)
            phi_x, _ = phi_e4(np.array([v]), np.array([u]), phi_name)
            if g_name == "-v":
                gx = -v
            elif g_name == "u":
                gx = u
            else:
                gx = 1.0
            delta = gx * float((phi_x @ coeffs).reshape(-1)[0])
            return np.array([u - c0 * v + delta], dtype=float)

        results.append(
            CandidateResult(
                case_id="E4_drag",
                candidate_id=candidate_id,
                candidate_label=candidate_id.replace("_", " "),
                coefficient_slot=slot,
                slot_input_z=z_name,
                candidate_library=", ".join(terms),
                physical_admissible=admissible,
                expected_correct=expected,
                train_residual_reduction=reduction,
                train_residual_nrmse=residual_nrmse,
                validation_rollout_nrmse=rollout_e4(rhs, validation),
                test_rollout_nrmse=rollout_e4(rhs, test),
                condition_number=condition_number(design),
                sensitivity_energy=sensitivity_energy(design),
                active_terms=int(np.sum(np.abs(coeffs) >= 1e-6)),
            )
        )
    return rank_and_select(results)


def rollout_e5(rhs: Callable[[float, np.ndarray], np.ndarray], trajectories: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for traj in trajectories:
        try:
            _, pred = e5.simulate(rhs, np.array(traj["x0"], dtype=float), e5.CONFIG["t_end"], e5.CONFIG["dt"])
            values.append(nrmse_nd(traj["x"], pred))
        except Exception:
            values.append(99.0)
    return safe_mean(values)


def fit_e5_poly_growth(x: np.ndarray, dx: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    biomass = x[:, 0]
    substrate = x[:, 1]
    if mode == "S_poly":
        phi = e5.poly_slot_library(substrate)
    elif mode == "X_poly":
        phi = np.column_stack([np.ones_like(biomass), biomass, biomass**2, biomass**3])
    else:
        raise ValueError(mode)
    design = np.vstack([biomass[:, None] * phi, -(1.0 / e5.CONFIG["yield_y"]) * biomass[:, None] * phi])
    target = np.concatenate([dx[:, 0], dx[:, 1]])
    coeffs = stlsq(design, target)
    return coeffs, design, target


def fit_e5_substrate_only(x: np.ndarray, dx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    biomass = x[:, 0]
    substrate = x[:, 1]
    phi = e5.poly_slot_library(substrate)
    zeros = np.zeros_like(phi)
    design = np.vstack([zeros, -(1.0 / e5.CONFIG["yield_y"]) * biomass[:, None] * phi])
    target = np.concatenate([dx[:, 0], dx[:, 1]])
    coeffs = stlsq(design, target)
    return coeffs, design, target


def fit_e5_additive(x: np.ndarray, dx: np.ndarray, mu0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    substrate = x[:, 1]
    phi = e5.poly_slot_library(substrate)
    design = np.vstack([phi, phi])
    baseline = np.array([e5.b0_rhs(0.0, xx, mu0, e5.CONFIG["yield_y"]) for xx in x])
    target = np.concatenate([dx[:, 0] - baseline[:, 0], dx[:, 1] - baseline[:, 1]])
    coeffs = stlsq(design, target)
    return coeffs, design, target, baseline


def e5_rational_sensitivity(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    biomass = x[:, 0]
    substrate = np.maximum(x[:, 1], 0.0)
    a, b = float(params[0]), float(params[1])
    dmu_da = substrate / (1.0 + b * substrate + 1e-15)
    dmu_db = -a * substrate**2 / (1.0 + b * substrate + 1e-15) ** 2
    phi = np.column_stack([dmu_da, dmu_db])
    return np.vstack([biomass[:, None] * phi, -(1.0 / e5.CONFIG["yield_y"]) * biomass[:, None] * phi])


def run_e5_screening() -> list[CandidateResult]:
    trajectories = e5.build_trajectories()
    train_local = [traj for traj in trajectories if traj["split"] == "train_local"]
    train_screen = [traj for traj in trajectories if traj["split"] in {"train_local", "train_wide"}]
    validation = [traj for traj in trajectories if traj["split"] == "test_interp"]
    test = [traj for traj in trajectories if traj["split"] == "test_extrap"]
    x_local, dx_local = e5.stack_data(train_local)
    x_train, dx_train = e5.stack_data(train_screen)
    mu0 = e5.fit_constant_mu(x_local, dx_local)
    baseline = np.array([e5.b0_rhs(0.0, xx, mu0, e5.CONFIG["yield_y"]) for xx in x_train])
    baseline_target = dx_train
    baseline_residual = baseline_target - baseline
    flat_baseline_residual = baseline_residual.reshape(-1)
    results: list[CandidateResult] = []

    poly_coeffs, poly_design, poly_target = fit_e5_poly_growth(x_train, dx_train, "S_poly")
    poly_pred_flat = poly_design @ poly_coeffs
    poly_pred = np.column_stack([poly_pred_flat[: len(x_train)], poly_pred_flat[len(x_train) :]])
    reduction, residual_nrmse = train_score(baseline_target, poly_pred, baseline_residual)

    def poly_rhs(_: float, xx: np.ndarray, *, coeffs: np.ndarray = poly_coeffs) -> np.ndarray:
        growth = float(e5.poly_mu(float(xx[1]), coeffs)) * float(xx[0])
        return np.array([growth, -(1.0 / e5.CONFIG["yield_y"]) * growth], dtype=float)

    results.append(
        CandidateResult(
            case_id="E5_monod",
            candidate_id="growth_rate_poly_S",
            candidate_label="growth rate poly S",
            coefficient_slot="growth rate",
            slot_input_z="S",
            candidate_library=", ".join(e5.CONFIG["slot_poly_terms"]),
            physical_admissible=True,
            expected_correct=False,
            train_residual_reduction=reduction,
            train_residual_nrmse=residual_nrmse,
            validation_rollout_nrmse=rollout_e5(poly_rhs, validation),
            test_rollout_nrmse=rollout_e5(poly_rhs, test),
            condition_number=condition_number(poly_design),
            sensitivity_energy=sensitivity_energy(poly_design),
            active_terms=int(np.sum(np.abs(poly_coeffs) >= 1e-6)),
        )
    )

    rat_params = e5.fit_rational_slot(x_train, dx_train)
    rat_pred = np.array([e5.b5_rational_rhs(0.0, xx, rat_params, e5.CONFIG["yield_y"]) for xx in x_train])
    rat_design = e5_rational_sensitivity(x_train, rat_params)
    reduction, residual_nrmse = train_score(baseline_target, rat_pred, baseline_residual)

    def rat_rhs(_: float, xx: np.ndarray, *, params: np.ndarray = rat_params) -> np.ndarray:
        return e5.b5_rational_rhs(0.0, xx, params, e5.CONFIG["yield_y"])

    results.append(
        CandidateResult(
            case_id="E5_monod",
            candidate_id="growth_rate_rational_S",
            candidate_label="growth rate rational S",
            coefficient_slot="growth rate",
            slot_input_z="S",
            candidate_library="a*S/(1+b*S)",
            physical_admissible=True,
            expected_correct=True,
            train_residual_reduction=reduction,
            train_residual_nrmse=residual_nrmse,
            validation_rollout_nrmse=rollout_e5(rat_rhs, validation),
            test_rollout_nrmse=rollout_e5(rat_rhs, test),
            condition_number=condition_number(rat_design),
            sensitivity_energy=sensitivity_energy(rat_design),
            active_terms=int(np.sum(np.abs(rat_params) >= 1e-6)),
        )
    )

    x_coeffs, x_design, _ = fit_e5_poly_growth(x_train, dx_train, "X_poly")
    x_pred_flat = x_design @ x_coeffs
    x_pred = np.column_stack([x_pred_flat[: len(x_train)], x_pred_flat[len(x_train) :]])
    reduction, residual_nrmse = train_score(baseline_target, x_pred, baseline_residual)

    def x_rhs(_: float, xx: np.ndarray, *, coeffs: np.ndarray = x_coeffs) -> np.ndarray:
        phi = np.array([1.0, xx[0], xx[0] ** 2, xx[0] ** 3])
        mu = float(phi @ coeffs)
        growth = mu * float(xx[0])
        return np.array([growth, -(1.0 / e5.CONFIG["yield_y"]) * growth], dtype=float)

    results.append(
        CandidateResult(
            case_id="E5_monod",
            candidate_id="growth_rate_poly_X",
            candidate_label="growth rate poly X",
            coefficient_slot="growth rate",
            slot_input_z="X",
            candidate_library="1, X, X^2, X^3",
            physical_admissible=True,
            expected_correct=False,
            train_residual_reduction=reduction,
            train_residual_nrmse=residual_nrmse,
            validation_rollout_nrmse=rollout_e5(x_rhs, validation),
            test_rollout_nrmse=rollout_e5(x_rhs, test),
            condition_number=condition_number(x_design),
            sensitivity_energy=sensitivity_energy(x_design),
            active_terms=int(np.sum(np.abs(x_coeffs) >= 1e-6)),
        )
    )

    s_coeffs, s_design, _ = fit_e5_substrate_only(x_train, dx_train)
    s_pred_flat = s_design @ s_coeffs
    s_pred = np.column_stack([s_pred_flat[: len(x_train)], s_pred_flat[len(x_train) :]])
    reduction, residual_nrmse = train_score(baseline_target, s_pred, baseline_residual)

    def s_rhs(_: float, xx: np.ndarray, *, coeffs: np.ndarray = s_coeffs) -> np.ndarray:
        mu = float(e5.poly_mu(float(xx[1]), coeffs))
        return np.array([mu0 * float(xx[0]), -(1.0 / e5.CONFIG["yield_y"]) * mu * float(xx[0])], dtype=float)

    results.append(
        CandidateResult(
            case_id="E5_monod",
            candidate_id="substrate_consumption_poly_S",
            candidate_label="substrate consumption poly S",
            coefficient_slot="substrate consumption",
            slot_input_z="S",
            candidate_library=", ".join(e5.CONFIG["slot_poly_terms"]),
            physical_admissible=True,
            expected_correct=False,
            train_residual_reduction=reduction,
            train_residual_nrmse=residual_nrmse,
            validation_rollout_nrmse=rollout_e5(s_rhs, validation),
            test_rollout_nrmse=rollout_e5(s_rhs, test),
            condition_number=condition_number(s_design),
            sensitivity_energy=sensitivity_energy(s_design),
            active_terms=int(np.sum(np.abs(s_coeffs) >= 1e-6)),
        )
    )

    add_coeffs, add_design, add_target, _ = fit_e5_additive(x_train, dx_train, mu0)
    add_pred_flat = add_design @ add_coeffs
    add_pred = baseline + np.column_stack([add_pred_flat[: len(x_train)], add_pred_flat[len(x_train) :]])
    reduction, residual_nrmse = train_score(baseline_target, add_pred, baseline_residual)

    def add_rhs(_: float, xx: np.ndarray, *, coeffs: np.ndarray = add_coeffs) -> np.ndarray:
        phi = e5.poly_slot_library(np.array([float(xx[1])]))
        residual = float((phi @ coeffs[: phi.shape[1]]).reshape(-1)[0])
        residual_s = float((phi @ coeffs[phi.shape[1] :]).reshape(-1)[0]) if len(coeffs) > phi.shape[1] else residual
        return e5.b0_rhs(0.0, xx, mu0, e5.CONFIG["yield_y"]) + np.array([residual, residual_s])

    results.append(
        CandidateResult(
            case_id="E5_monod",
            candidate_id="free_additive_poly_S",
            candidate_label="free additive poly S",
            coefficient_slot="additive residual",
            slot_input_z="S",
            candidate_library=", ".join(e5.CONFIG["slot_poly_terms"]),
            physical_admissible=False,
            expected_correct=False,
            train_residual_reduction=reduction,
            train_residual_nrmse=residual_nrmse,
            validation_rollout_nrmse=rollout_e5(add_rhs, validation),
            test_rollout_nrmse=rollout_e5(add_rhs, test),
            condition_number=condition_number(add_design),
            sensitivity_energy=sensitivity_energy(add_design),
            active_terms=int(np.sum(np.abs(add_coeffs) >= 1e-6)),
        )
    )

    # Keep this variable referenced for static checkers and provenance clarity.
    _ = flat_baseline_residual
    return rank_and_select(results)


def panel_label(ax: plt.Axes, label: str, x: float = -0.11, y: float = 1.12) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.9},
    )


def build_screening_figure(rows: list[CandidateResult]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    case_order = ["E3_pendulum", "E4_drag", "E5_monod"]
    titles = {
        "E3_pendulum": "E3 pendulum slot screening",
        "E4_drag": "E4 drag slot screening",
        "E5_monod": "E5 Monod slot/library screening",
    }
    display_labels = {
        "stiffness_theta_even": "stiffness\nz=theta\neven",
        "free_additive_theta_odd": "free\nresidual\nz=theta",
        "stiffness_omega_even": "stiffness\nz=omega\neven",
        "damping_theta_even": "damping\nz=theta\neven",
        "damping_omega_even": "damping\nz=omega\neven",
        "free_additive_v": "free\nresidual\nz=v",
        "drag_v_abs": "drag\nz=|v|",
        "drag_u_abs": "drag\nz=|u|",
        "input_gain_u": "input gain\nz=u",
        "growth_rate_rational_S": "rate\nrational\nS",
        "growth_rate_poly_S": "rate\npoly\nS",
        "growth_rate_poly_X": "rate\npoly\nX",
        "substrate_consumption_poly_S": "substrate\npoly\nS",
        "free_additive_poly_S": "free\nresid.\npoly S",
    }
    fig = plt.figure(figsize=(7.2, 5.05))
    gs = fig.add_gridspec(2, 2, hspace=0.74, wspace=0.32)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
    ax_summary = fig.add_subplot(gs[1, 1])

    selected_rows: list[CandidateResult] = []
    for ax, case_id, label in zip(axes, case_order, ["a", "b", "c"]):
        case_rows = sorted([row for row in rows if row.case_id == case_id], key=lambda row: row.validation_rollout_nrmse)
        labels = [display_labels.get(row.candidate_id, row.candidate_id.replace("_", "\n")) for row in case_rows]
        values = [max(row.validation_rollout_nrmse, 1e-8) for row in case_rows]
        colors = [
            COLORS["selected"] if row.selected_by_protocol else COLORS["eligible"] if row.physical_admissible else COLORS["ineligible"]
            for row in case_rows
        ]
        ax.bar(np.arange(len(case_rows)), values, color=colors)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(case_rows)), labels, rotation=0)
        ax.set_ylabel("validation rollout NRMSE")
        ax.set_title(titles[case_id])
        panel_label(ax, label)
        for idx, row in enumerate(case_rows):
            if row.selected_by_protocol:
                selected_rows.append(row)
                ax.text(idx, values[idx] * 1.5, "selected", ha="center", va="bottom", fontsize=6.4, color=COLORS["selected"])

    ax_summary.axhline(1.0, color=COLORS["line"], lw=0.8, ls="--")
    x = np.arange(len(selected_rows))
    ax_summary.bar(x - 0.17, [max(row.train_residual_reduction, 1e-8) for row in selected_rows], width=0.34, color="#B9D9CD", label="train residual reduction")
    ax_summary.bar(x + 0.17, [max(row.test_rollout_nrmse, 1e-8) for row in selected_rows], width=0.34, color=COLORS["selected"], label="test NRMSE")
    ax_summary.set_yscale("log")
    ax_summary.set_xticks(x, [row.case_id.replace("_", "\n") for row in selected_rows])
    ax_summary.set_title("Frozen choices: train reduction and final test error", pad=4)
    ax_summary.set_ylabel("score (log scale)")
    ax_summary.legend(
        loc="lower center",
        bbox_to_anchor=(0.58, 1.17),
        ncol=2,
        borderaxespad=0.0,
        frameon=False,
        handlelength=1.8,
    )
    panel_label(ax_summary, "d", y=1.30)

    for suffix in [".pdf", ".svg"]:
        path = FIGURE_DIR / f"slot_screening{suffix}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_provenance(paths: list[Path]) -> None:
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    provenance = {
        "experiment_id": CONFIG["experiment_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "config": CONFIG,
        "inputs": [
            "scripts/run_e3_local_linearization_pendulum.py",
            "scripts/run_e4_velocity_drag.py",
            "scripts/run_e5_monod_rate_slot.py",
        ],
        "outputs": [str(path.relative_to(ROOT)) for path in paths],
    }
    (PROVENANCE_DIR / "slot_screening_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rows = run_e3_screening() + run_e4_screening() + run_e5_screening()
    rows = sorted(rows, key=lambda row: (row.case_id, row.validation_rank))
    candidate_path = RESULT_DIR / "slot_screening_candidates.csv"
    write_csv(candidate_path, [row.as_row() for row in rows])

    selected_rows = [row for row in rows if row.selected_by_protocol]
    decision_rows = []
    for row in selected_rows:
        decision_rows.append(
            {
                "case_id": row.case_id,
                "selected_candidate": row.candidate_id,
                "coefficient_slot": row.coefficient_slot,
                "slot_input_z": row.slot_input_z,
                "candidate_library": row.candidate_library,
                "validation_rollout_nrmse": f"{row.validation_rollout_nrmse:.10g}",
                "test_rollout_nrmse": f"{row.test_rollout_nrmse:.10g}",
                "train_residual_reduction": f"{row.train_residual_reduction:.10g}",
                "condition_number": f"{row.condition_number:.10g}",
            }
        )
    decision_path = RESULT_DIR / "slot_screening_decisions.csv"
    write_csv(decision_path, decision_rows)

    metrics = {
        "experiment_id": CONFIG["experiment_id"],
        "selection_rule": CONFIG["selection_rule"],
        "test_usage_rule": CONFIG["test_usage_rule"],
        "n_candidates": len(rows),
        "n_selected": len(selected_rows),
        "selected_candidates": {row.case_id: row.candidate_id for row in selected_rows},
        "all_selected_are_expected_correct": all(row.expected_correct for row in selected_rows),
        "case_metrics": {
            row.case_id: {
                "selected_candidate": row.candidate_id,
                "validation_rollout_nrmse": row.validation_rollout_nrmse,
                "test_rollout_nrmse": row.test_rollout_nrmse,
                "train_residual_reduction": row.train_residual_reduction,
                "condition_number": row.condition_number,
            }
            for row in selected_rows
        },
    }
    metrics_path = RESULT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    build_screening_figure(rows)
    figure_path = FIGURE_DIR / "slot_screening.pdf"
    write_provenance([candidate_path, decision_path, metrics_path, figure_path])

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
