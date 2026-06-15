#!/usr/bin/env python3
"""Check the experiment-code release package.

The checks are intentionally conservative: they verify that required release
files exist, raw third-party archives are absent, local caches are absent, and
article-writing/publishing-administration terms have not leaked into the
experiment-code package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "MANIFEST.sha256"

REQUIRED_FILES = [
    ".gitignore",
    "CITATION.cff",
    "DATA_SOURCES.md",
    "KNOWN_LIMITATIONS.md",
    "LICENSE",
    "OPEN_SOURCE_CHECKLIST.md",
    "README.md",
    "RELEASE_NOTES.md",
    "REPRODUCE.md",
    "environment.yml",
    "requirements.txt",
    "scripts/smoke_test.py",
    "scripts/run_slot_screening.py",
    "scripts/run_ablation_suite.py",
    "data/raw/README.md",
]

REQUIRED_DIRS = [
    "scripts",
    "data/generated",
    "data/processed",
    "data/provenance",
    "data/raw",
    "results",
]

FORBIDDEN_DIR_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "paper",
    "papers",
    "templates",
    "references",
    "reports",
    "docs",
}

FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".zip",
    ".tif",
    ".tiff",
    ".png",
    ".mat",
    ".h5",
    ".hdf5",
}

FORBIDDEN_TERMS = [
    "paper1",
    "Paper 1",
    "manuscript",
    "submission",
    "journal",
    "cover letter",
    "highlights",
    "reference-management",
    "reviewer",
    "caption table",
    "LaTeX",
    "BibTeX",
]

TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yml",
    ".yaml",
}


def iter_files() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*") if path.is_file())


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_required() -> list[str]:
    errors: list[str] = []
    for item in REQUIRED_FILES:
        if not (ROOT / item).is_file():
            errors.append(f"Missing required file: {item}")
    for item in REQUIRED_DIRS:
        if not (ROOT / item).is_dir():
            errors.append(f"Missing required directory: {item}")
    return errors


def check_forbidden_paths(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        rel = relative(path)
        parts = set(path.relative_to(ROOT).parts[:-1])
        bad_parts = sorted(parts & FORBIDDEN_DIR_PARTS)
        if bad_parts:
            errors.append(f"Forbidden directory component {bad_parts} in {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"Forbidden file suffix {path.suffix} in {rel}")
    raw_files = sorted(path for path in (ROOT / "data" / "raw").rglob("*") if path.is_file())
    allowed_raw = {ROOT / "data" / "raw" / "README.md"}
    for path in raw_files:
        if path not in allowed_raw:
            errors.append(f"Raw third-party data should not be bundled: {relative(path)}")
    return errors


def check_forbidden_terms(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        if path.name in {MANIFEST_NAME, "check_release_package.py"} or path.suffix not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for term in FORBIDDEN_TERMS:
            if term in text:
                errors.append(f"Forbidden term {term!r} in {relative(path)}")
    return errors


def build_manifest(files: list[Path]) -> str:
    lines: list[str] = []
    for path in files:
        rel = relative(path)
        if rel == MANIFEST_NAME:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true", help="write MANIFEST.sha256 after checks pass")
    args = parser.parse_args()

    files = iter_files()
    errors = []
    errors.extend(check_required())
    errors.extend(check_forbidden_paths(files))
    errors.extend(check_forbidden_terms(files))

    summary = {
        "root": str(ROOT),
        "file_count": len(files),
        "errors": errors,
    }
    if errors:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        raise SystemExit(1)

    if args.write_manifest:
        (ROOT / MANIFEST_NAME).write_text(build_manifest(files), encoding="utf-8")
        summary["manifest"] = MANIFEST_NAME

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
