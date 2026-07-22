#!/usr/bin/env python3
"""Record or verify the pinned eSpeak-ng runtime and language data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from training.frontend.phonemize import detect_espeak_version, hash_tree


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-hash", action="store_true")
    args = parser.parse_args()
    expected_version = (root / "vendor/ESPEAK_NG_VERSION").read_text().strip()
    hash_path = root / "vendor/ESPEAK_NG_DATA_HASH"
    data_path = Path(os.environ["ESPEAK_DATA_PATH"])
    actual_version = detect_espeak_version()
    actual_hash = hash_tree(data_path)
    if args.write_hash:
        hash_path.write_text(actual_hash + "\n", encoding="utf-8")
    expected_hash = hash_path.read_text().strip() if hash_path.exists() else ""
    errors = []
    if actual_version != expected_version:
        errors.append(f"version {actual_version} != {expected_version}")
    if not expected_hash:
        errors.append("ESPEAK_NG_DATA_HASH is not recorded")
    elif actual_hash != expected_hash:
        errors.append("eSpeak language data hash mismatch")
    result = {
        "status": "PASS" if not errors else "FAIL", "version": actual_version,
        "data_path": str(data_path), "data_hash": actual_hash, "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
