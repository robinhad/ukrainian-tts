#!/usr/bin/env python3
"""Validate the expanded-v3 source policy and its records."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TOKEN_FILE = Path("/home/ballvan/Projects/hf_token.txt")
PER_FILE_LICENSES = {
    "CC0-1.0",
    "CC-BY-2.0-FR",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
    "Public-Domain",
}


def load_registry(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("The source registry has an unsupported schema.")
    return data


def read_hf_token(path: Path = DEFAULT_TOKEN_FILE) -> str:
    """Read a token without putting it in an environment variable."""
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"The Hugging Face token file is empty: {path}")
    return token


def token_file_mode(path: Path = DEFAULT_TOKEN_FILE) -> int:
    return path.stat().st_mode & 0o777


def free_disk_gib(path: Path) -> float:
    probe = path.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024**3


def check_disk(path: Path, registry: dict[str, Any]) -> dict[str, Any]:
    policy = registry["policy"]
    free = free_disk_gib(path)
    stop = float(policy["free_disk_stop_gib"])
    warn = float(policy["free_disk_warn_gib"])
    status = "PASS"
    if free < stop:
        status = "STOP"
    elif free < warn:
        status = "WARN"
    return {
        "status": status,
        "free_gib": round(free, 2),
        "warn_gib": warn,
        "stop_gib": stop,
    }


def validate_registry(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    policy = registry.get("policy", {})
    allowed = set(policy.get("allow_licenses", []))
    if float(policy.get("free_disk_stop_gib", 0)) != 60:
        errors.append("The free disk stop threshold must be 60 GiB.")
    if float(policy.get("free_disk_warn_gib", 0)) <= 60:
        errors.append("The free disk warning threshold must exceed 60 GiB.")
    for source_id, source in registry.get("sources", {}).items():
        decision = source.get("decision")
        if source.get("enabled") and decision not in {"allow", "allow_per_file"}:
            errors.append(f"{source_id}: an enabled source is not allowed.")
        if decision == "allow" and source.get("license") not in allowed:
            errors.append(f"{source_id}: the source license is not in the allow list.")
        if decision == "allow_per_file" and source.get("license") != "per-file":
            errors.append(f"{source_id}: per-file review is not explicit.")
        if source.get("raw_redistribution") is not False:
            errors.append(f"{source_id}: raw redistribution must be false.")
        if source.get("commercial_use") is False:
            errors.append(f"{source_id}: commercial use is not allowed.")
    return errors


def validate_record(record: dict[str, Any], registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source_id = str(record.get("source_id") or "")
    source = registry.get("sources", {}).get(source_id)
    if source is None:
        return [f"{source_id or '<empty>'}: the source is not in the registry."]
    if not source.get("enabled") or source.get("decision") == "deny":
        errors.append(f"{source_id}: the source is denied.")
    record_license = str(record.get("source_license") or "")
    if source.get("decision") == "allow":
        if record_license != source.get("license"):
            errors.append(f"{source_id}: the record license does not match the registry.")
    elif source.get("decision") == "allow_per_file":
        if record_license not in PER_FILE_LICENSES:
            errors.append(f"{source_id}: the per-file license is not allowed.")
        if not record.get("license_evidence"):
            errors.append(f"{source_id}: per-file license evidence is missing.")
    if not record.get("audio_sha256_source"):
        errors.append(f"{source_id}: the source audio hash is missing.")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    errors = validate_registry(registry)
    record_count = 0
    if args.records:
        for line_number, line in enumerate(
            args.records.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            record_count += 1
            record_errors = validate_record(json.loads(line), registry)
            errors.extend(f"line {line_number}: {item}" for item in record_errors)
    disk = check_disk(args.workspace, registry)
    if disk["status"] == "STOP":
        errors.append(
            f"Free disk space is {disk['free_gib']} GiB. The stop threshold is 60 GiB."
        )
    token = {
        "path": str(DEFAULT_TOKEN_FILE),
        "exists": DEFAULT_TOKEN_FILE.is_file(),
        "mode": (
            f"{token_file_mode(DEFAULT_TOKEN_FILE):03o}"
            if DEFAULT_TOKEN_FILE.is_file()
            else None
        ),
        "mode_warning": (
            DEFAULT_TOKEN_FILE.is_file()
            and bool(token_file_mode(DEFAULT_TOKEN_FILE) & 0o077)
        ),
        "exported": "HF_TOKEN" in os.environ or "HUGGING_FACE_HUB_TOKEN" in os.environ,
    }
    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "record_count": record_count,
        "disk": disk,
        "token_file": token,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
