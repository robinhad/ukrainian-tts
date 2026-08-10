#!/usr/bin/env python3
"""Write a resource and phase status for the full v6 pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def is_active(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def commands() -> str:
    return subprocess.run(
        ["ps", "-eo", "cmd"], check=True, text=True, capture_output=True
    ).stdout


def phase(processes: str, root: Path) -> str:
    if "tts_train" in processes or "torchrun" in processes:
        return "training"
    if "run_expanded_v6_sidon_deess_novoa_smoke" in processes:
        return "smoke_training_or_inference"
    if "extract_spk_embed" in processes:
        return "clean_speaker_embeddings"
    if "prepare_expanded_v6_sidon_deess_novoa" in processes:
        return "manifests_tokens_or_statistics"
    if (root / "reports/expanded_v6_sidon_deess_novoa_smoke_inference.json").is_file():
        return "post_training_evaluation"
    return "waiting_or_finalizing"


def gpu() -> list[dict]:
    query = "index,utilization.gpu,memory.used,power.draw,power.limit,temperature.gpu"
    output = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    ).stdout
    result = []
    for line in output.splitlines():
        index, utilization, memory, power, limit, temperature = [
            value.strip() for value in line.split(",")
        ]
        result.append({
            "index": int(index), "utilization_percent": float(utilization),
            "memory_used_mib": float(memory), "power_w": float(power),
            "power_limit_w": float(limit), "temperature_c": float(temperature),
        })
    return result


def latest_training_status(root: Path) -> dict | None:
    path = root / "reports/training_status_expanded_v6_sidon_deess_novoa_100k.jsonl"
    if not path.is_file():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.interval_seconds <= 1800:
        raise SystemExit("The monitor interval must be from 1 to 1,800 seconds.")
    args.status.parent.mkdir(parents=True, exist_ok=True)
    while True:
        active = is_active(args.watch_pid)
        current_phase = phase(commands(), args.root)
        training = latest_training_status(args.root)
        now = datetime.now(ZoneInfo("Europe/Kyiv"))
        record = {
            "timestamp_kyiv": now.isoformat(), "watch_pid": args.watch_pid,
            "pipeline_active": active, "phase": current_phase,
            "free_disk_gib": round(shutil.disk_usage(args.root).free / 1024**3, 2),
            "gpu": gpu(),
            "eta_kyiv": training.get("eta_kyiv") if training else None,
            "training": training,
        }
        with args.status.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        if not active:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
