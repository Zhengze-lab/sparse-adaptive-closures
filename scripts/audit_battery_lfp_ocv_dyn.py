#!/usr/bin/env python3
"""Audit the LiFePO4 OCV/DYN Mendeley dataset zip without full extraction."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, whosmat


DATASET_ID = "battery_lfp_ocv_dyn"
SOURCE_URL = "https://data.mendeley.com/datasets/p8kf893yv3/1"
DOWNLOAD_URL = "https://data.mendeley.com/public-api/zip/p8kf893yv3/download/1"
DOI = "10.17632/p8kf893yv3.1"
RAW_ZIP = Path("data/raw/external/battery_lfp_ocv_dyn/p8kf893yv3_v1.zip")
RESULT_DIR = Path("results/battery_lfp_ocv_dyn_audit")
PROVENANCE_PATH = Path("data/provenance/battery_lfp_ocv_dyn_audit_provenance.json")


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def norm_folder(name: str) -> str:
    normalized = name.strip("/")
    if "/" not in normalized:
        return ""
    return normalized.rsplit("/", 1)[0].strip()


def ext(name: str) -> str:
    suffix = Path(name.strip()).suffix.lower()
    return suffix if suffix else "<none>"


def value_fields(value: Any) -> list[str]:
    if hasattr(value, "_fieldnames") and value._fieldnames:
        return list(value._fieldnames)
    if isinstance(value, np.ndarray) and value.dtype.names:
        return list(value.dtype.names)
    return []


def array_stats(value: Any) -> dict[str, Any]:
    arr = np.asarray(value).astype(float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": "", "max": "", "median": ""}
    return {"min": float(np.min(arr)), "max": float(np.max(arr)), "median": float(np.median(arr))}


def infer_temperature_c(filename: str) -> int | None:
    match = re.search(r"_([NP])(\d{2})(?:_|\.)", filename)
    if not match:
        return None
    sign = -1 if match.group(1) == "N" else 1
    return sign * int(match.group(2))


def summarize_series(filename: str, root_variable: str, series_name: str, value: Any) -> dict[str, Any] | None:
    fields = value_fields(value)
    if "time" not in fields:
        return None

    time = np.asarray(getattr(value, "time")).astype(float).ravel()
    finite_time = time[np.isfinite(time)]
    if finite_time.size >= 2:
        dt = np.diff(finite_time)
        dt = dt[np.isfinite(dt) & (dt > 0)]
    else:
        dt = np.array([])

    row: dict[str, Any] = {
        "filename": filename,
        "temperature_c_from_name": infer_temperature_c(filename),
        "root_variable": root_variable,
        "series": series_name,
        "fields": "|".join(fields),
        "n_samples": int(finite_time.size),
        "time_start": float(finite_time[0]) if finite_time.size else "",
        "time_end": float(finite_time[-1]) if finite_time.size else "",
        "median_dt": float(np.median(dt)) if dt.size else "",
    }
    for field in ["current", "voltage", "chgAh", "disAh", "Ts", "Tf", "Ts1", "SurfaceTemperature", "AirTemperature"]:
        if field in fields:
            stats = array_stats(getattr(value, field))
            row[f"{field}_min"] = stats["min"]
            row[f"{field}_max"] = stats["max"]
            row[f"{field}_median"] = stats["median"]
        else:
            row[f"{field}_min"] = ""
            row[f"{field}_max"] = ""
            row[f"{field}_median"] = ""
    return row


def audit_mat_series(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = zf.read(info)
    try:
        loaded = loadmat(io.BytesIO(data), squeeze_me=True, struct_as_record=False)
    except Exception:
        return rows

    for root_variable, root_value in loaded.items():
        if root_variable.startswith("__"):
            continue
        root_fields = value_fields(root_value)
        root_row = summarize_series(info.filename, root_variable, "<root>", root_value)
        if root_row:
            rows.append(root_row)
        for field in root_fields:
            child = getattr(root_value, field)
            child_row = summarize_series(info.filename, root_variable, field, child)
            if child_row:
                rows.append(child_row)
    return rows


def audit_mat_file(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = zf.read(info)
    try:
        variables = whosmat(io.BytesIO(data))
    except Exception as exc:  # pragma: no cover - diagnostic path
        return [
            {
                "filename": info.filename,
                "variable": "<whosmat_error>",
                "shape": "",
                "class": "",
                "fields": "",
                "error": repr(exc),
            }
        ]

    loaded: dict[str, Any] = {}
    try:
        loaded = loadmat(io.BytesIO(data), squeeze_me=True, struct_as_record=False)
    except Exception:
        loaded = {}

    for name, shape, class_name in variables:
        fields = value_fields(loaded.get(name)) if loaded else []
        rows.append(
            {
                "filename": info.filename,
                "variable": name,
                "shape": "x".join(str(v) for v in shape),
                "class": class_name,
                "fields": "|".join(fields),
                "error": "",
            }
        )
    return rows


def main() -> None:
    if not RAW_ZIP.exists():
        raise FileNotFoundError(f"Missing raw zip: {RAW_ZIP}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)

    zip_hash = sha256_path(RAW_ZIP)
    file_rows: list[dict[str, Any]] = []
    mat_rows: list[dict[str, Any]] = []
    series_rows: list[dict[str, Any]] = []
    folder_summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"file_count": 0, "total_file_size": 0, "total_compress_size": 0, "extensions": set()}
    )

    with zipfile.ZipFile(RAW_ZIP) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        for info in infos:
            data = zf.read(info)
            file_hash = sha256_bytes(data)
            folder = norm_folder(info.filename)
            extension = ext(info.filename)
            file_rows.append(
                {
                    "filename": info.filename,
                    "folder": folder,
                    "extension": extension,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "sha256": file_hash,
                }
            )
            summary = folder_summary[folder]
            summary["file_count"] += 1
            summary["total_file_size"] += info.file_size
            summary["total_compress_size"] += info.compress_size
            summary["extensions"].add(extension)
            if extension == ".mat":
                mat_rows.extend(audit_mat_file(zf, info))
                series_rows.extend(audit_mat_series(zf, info))

    with (RESULT_DIR / "zip_files.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "folder", "extension", "file_size", "compress_size", "sha256"],
        )
        writer.writeheader()
        writer.writerows(file_rows)

    folder_rows = []
    for folder, summary in sorted(folder_summary.items()):
        folder_rows.append(
            {
                "folder": folder,
                "file_count": summary["file_count"],
                "total_file_size": summary["total_file_size"],
                "total_compress_size": summary["total_compress_size"],
                "extensions": "|".join(sorted(summary["extensions"])),
            }
        )
    with (RESULT_DIR / "folder_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["folder", "file_count", "total_file_size", "total_compress_size", "extensions"],
        )
        writer.writeheader()
        writer.writerows(folder_rows)

    with (RESULT_DIR / "mat_variables.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "variable", "shape", "class", "fields", "error"],
        )
        writer.writeheader()
        writer.writerows(mat_rows)

    series_fieldnames = [
        "filename",
        "temperature_c_from_name",
        "root_variable",
        "series",
        "fields",
        "n_samples",
        "time_start",
        "time_end",
        "median_dt",
    ]
    for field in ["current", "voltage", "chgAh", "disAh", "Ts", "Tf", "Ts1", "SurfaceTemperature", "AirTemperature"]:
        series_fieldnames.extend([f"{field}_min", f"{field}_max", f"{field}_median"])
    with (RESULT_DIR / "mat_series_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=series_fieldnames)
        writer.writeheader()
        writer.writerows(series_rows)

    metadata = {
        "dataset_id": DATASET_ID,
        "source_url": SOURCE_URL,
        "download_url": DOWNLOAD_URL,
        "doi": DOI,
        "license": "CC BY 4.0",
        "audit_date": date.today().isoformat(),
        "raw_zip": str(RAW_ZIP),
        "zip_size_bytes": RAW_ZIP.stat().st_size,
        "zip_sha256": zip_hash,
        "file_count": len(file_rows),
        "mat_file_count": sum(1 for row in file_rows if row["extension"] == ".mat"),
        "xlsx_file_count": sum(1 for row in file_rows if row["extension"] == ".xlsx"),
        "mat_series_count": len(series_rows),
        "top_level_folders": sorted({row["folder"].split("/", 1)[0] for row in file_rows if row["folder"]}),
        "result_files": {
            "zip_files": str(RESULT_DIR / "zip_files.csv"),
            "folder_summary": str(RESULT_DIR / "folder_summary.csv"),
            "mat_variables": str(RESULT_DIR / "mat_variables.csv"),
            "mat_series_summary": str(RESULT_DIR / "mat_series_summary.csv"),
        },
        "next_action": "Inspect OCVdata and DYNdata variables to implement BATT-2 OCV calibration and data-loading adapters.",
    }

    with (RESULT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    provenance = {
        **metadata,
        "script": "scripts/audit_battery_lfp_ocv_dyn.py",
        "notes": [
            "The audit reads files from the zip archive without extracting the full dataset.",
            "Per-file SHA256 values are computed from uncompressed file bytes inside the zip.",
        ],
    }
    with PROVENANCE_PATH.open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
