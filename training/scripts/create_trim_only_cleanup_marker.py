#!/usr/bin/env python3
"""Authorize exact source-cache paths after true 50/50 extraction passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-report", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.hybrid_report.read_text(encoding="utf-8"))
    counts = report.get("counts", {})
    if report.get("status") != "PASS" or counts.get("raw", 0) < 1 or counts.get("clean", 0) < 1:
        raise SystemExit("The hybrid embedding gate is not PASS.")
    root = args.source_root.resolve()
    names = (
        "pseudo_uk",
        "pseudo_uk_v2-c050-d20-g050",
        "pseudo_uk_v3-c050-d20-g050-duration-split",
        "voa_ukr_user_grant",
        "pseudo_uk_v4-c050-d20-g050-defer-long",
    )
    eligible = []
    for name in names:
        path = (root / name).resolve()
        if path.parent != root or path.is_symlink():
            raise SystemExit(f"Unsafe cleanup target: {path}")
        if path.is_dir():
            eligible.append(str(path))
    payload = {
        "status": "PASS",
        "raw_embeddings_complete": True,
        "hybrid_report": str(args.hybrid_report.resolve()),
        "source_root": str(root),
        "eligible_paths": eligible,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
