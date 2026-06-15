#!/usr/bin/env python3
"""Audit the official Silverbox input-output dataset without fitting a model."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata as metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nonlinear_benchmarks as nb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "external" / "silverbox"
PROVENANCE_DIR = ROOT / "data" / "provenance"
RESULT_ROOT = ROOT / "results" / "silverbox_data_audit"


CONFIG: dict[str, Any] = {
    "dataset_id": "silverbox_data_audit",
    "experiment_id": "SB-0",
    "description": "Official Silverbox data loading and split audit; no model is trained.",
    "source_url": "https://www.nonlinearbenchmark.org/benchmarks/silverbox",
    "dataloader_url": "https://github.com/MaartenSchoukens/nonlinear_benchmarks",
    "citation": (
        "T. Wigren and J. Schoukens. Three free data sets for development and "
        "benchmarking in nonlinear system identification. 2013 European Control "
        "Conference (ECC), pp. 2933-2938. doi:10.23919/ECC.2013.6669201."
    ),
    "raw_dir": "data/raw/external/silverbox",
    "official_test_initialization_window": 50,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def summarize_record(name: str, record: Any, split: str) -> dict[str, Any]:
    u = np.asarray(record.u, dtype=float)
    y = np.asarray(record.y, dtype=float)
    return {
        "record": name,
        "split": split,
        "sample_count": int(len(y)),
        "u_shape": str(tuple(u.shape)),
        "y_shape": str(tuple(y.shape)),
        "sampling_time_s": float(record.sampling_time),
        "state_initialization_window_length": getattr(record, "state_initialization_window_length", None),
        "u_mean": float(np.mean(u)),
        "u_std": float(np.std(u)),
        "u_min": float(np.min(u)),
        "u_max": float(np.max(u)),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
    }


def collect_raw_files(raw_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(raw_root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def main() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    data_file_locations = nb.Silverbox(
        data_file_locations=True,
        dir_placement=str(RAW_ROOT),
        force_download=False,
    )
    train_val, test = nb.Silverbox(
        dir_placement=str(RAW_ROOT),
        force_download=False,
    )
    test_multisine, test_arrow_full, test_arrow_no_extrapolation = test

    summary_rows = [
        summarize_record("train_val", train_val, "train_val"),
        summarize_record("test_multisine", test_multisine, "test"),
        summarize_record("test_arrow_full", test_arrow_full, "test"),
        summarize_record("test_arrow_no_extrapolation", test_arrow_no_extrapolation, "test"),
    ]
    raw_file_rows = collect_raw_files(RAW_ROOT)

    summary_path = RESULT_ROOT / "dataset_summary.csv"
    raw_files_path = RESULT_ROOT / "raw_files.csv"
    metrics_path = RESULT_ROOT / "metrics.json"
    provenance_path = PROVENANCE_DIR / "silverbox_data_audit_provenance.json"
    hashes_path = RESULT_ROOT / "hashes.json"

    write_csv(summary_path, summary_rows)
    write_csv(raw_files_path, raw_file_rows)

    metrics = {
        "dataset_id": CONFIG["dataset_id"],
        "records": summary_rows,
        "raw_files": raw_file_rows,
        "data_file_locations": [str(path) for path in data_file_locations],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    provenance = {
        **CONFIG,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "nonlinear-benchmarks": package_version("nonlinear-benchmarks"),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
        },
        "data_file_locations": [str(path) for path in data_file_locations],
        "outputs": {
            "dataset_summary_csv": str(summary_path.relative_to(ROOT)),
            "raw_files_csv": str(raw_files_path.relative_to(ROOT)),
            "metrics_json": str(metrics_path.relative_to(ROOT)),
            "hashes_json": str(hashes_path.relative_to(ROOT)),
        },
        "model_training": "none",
        "test_usage_rule": "Test records are audited only; they must not be used for model selection.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    hashes = {
        "dataset_summary_csv_sha256": sha256_file(summary_path),
        "raw_files_csv_sha256": sha256_file(raw_files_path),
        "metrics_json_sha256": sha256_file(metrics_path),
        "provenance_json_sha256": sha256_file(provenance_path),
    }
    hashes_path.write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
