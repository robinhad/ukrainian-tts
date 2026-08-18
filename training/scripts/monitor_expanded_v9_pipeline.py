#!/usr/bin/env python3
"""Write v9 phase, resource, progress, and Kyiv ETA records."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

NAME = "expanded_v9_cascade_with_voa"
TOTAL = 208_867


def active(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def commands() -> str:
    return subprocess.run(
        ["ps", "-eo", "cmd"], check=True, text=True, capture_output=True
    ).stdout


def phase(processes: str, running: bool) -> str:
    if not running:
        return "complete_or_failed"
    if "evaluate_expanded_v9" in processes:
        return "milestone_evaluation"
    if "espnet2.bin.gan_tts_train --collect_stats true" in processes:
        return "statistics"
    if "run_expanded_v9_with_voa_smoke" in processes:
        return "smoke_training_or_inference"
    if "espnet2.bin.gan_tts_train" in processes or "torchrun" in processes:
        return "training"
    if "extract_spk_embed" in processes:
        return "speaker_embeddings"
    if "prepare_expanded_v9_with_voa" in processes:
        return "manifests_embeddings_tokens_or_statistics"
    if "preprocess_training_cascade_audio" in processes:
        return "audio_preprocessing"
    return "initializing_or_transition"


def latest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return json.loads(lines[-1]) if lines else None


def result_count(root: Path) -> int:
    directory = root / f"reports/{NAME}_processing"
    count = 0
    for path in directory.glob("shard-*.jsonl"):
        with path.open("rb") as stream:
            count += sum(1 for _ in stream)
    return count


def gpu() -> list[dict]:
    query = "index,utilization.gpu,memory.used,power.draw,power.limit,temperature.gpu"
    output = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        index, utilization, memory, power, limit, temperature = [
            value.strip() for value in line.split(",")
        ]
        rows.append(
            {
                "index": int(index),
                "utilization_percent": float(utilization),
                "memory_used_mib": float(memory),
                "power_w": float(power),
                "power_limit_w": float(limit),
                "power_utilization_percent": round(100 * float(power) / float(limit), 1),
                "temperature_c": float(temperature),
            }
        )
    return rows


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
        running = active(args.watch_pid)
        now = datetime.now(ZoneInfo("Europe/Kyiv"))
        current_phase = phase(commands(), running)
        processed = result_count(args.root)
        prep = latest(args.root / f"reports/{NAME}_preparation_status.jsonl")
        duration_progress = latest(
            args.root / f"reports/{NAME}_duration_status.jsonl"
        )
        training = latest(args.root / f"reports/training_status_{NAME}_500k.jsonl")
        phase_eta = None
        total_eta = None
        eta_basis = "not_available"
        if current_phase == "audio_preprocessing":
            phase_eta = (
                duration_progress.get("eta_kyiv")
                if duration_progress
                else prep.get("eta_kyiv") if prep else None
            )
            if phase_eta:
                total_eta = (
                    datetime.fromisoformat(phase_eta) + timedelta(hours=114)
                ).isoformat()
            eta_basis = "observed_audio_duration_rate_plus_114h_downstream_estimate"
        elif current_phase in {
            "manifests_embeddings_tokens_or_statistics",
            "statistics",
            "speaker_embeddings",
        }:
            phase_eta = (now + timedelta(hours=8)).isoformat()
            total_eta = (now + timedelta(hours=114)).isoformat()
            eta_basis = "prior_pipeline_runtime_estimate"
        elif current_phase == "smoke_training_or_inference":
            phase_eta = (now + timedelta(hours=1)).isoformat()
            total_eta = (now + timedelta(hours=111)).isoformat()
            eta_basis = "prior_pipeline_runtime_estimate"
        elif current_phase == "training" and training:
            phase_eta = training.get("eta_kyiv")
            total_eta = phase_eta
            eta_basis = "observed_training_rate"
        elif current_phase == "milestone_evaluation":
            phase_eta = (now + timedelta(hours=2)).isoformat()
            total_eta = phase_eta
            eta_basis = "evaluation_runtime_estimate"
        elif current_phase == "initializing_or_transition":
            phase_eta = (now + timedelta(minutes=15)).isoformat()
            total_eta = (now + timedelta(hours=140)).isoformat()
            eta_basis = "initial_runtime_estimate"
        record = {
            "timestamp_kyiv": now.isoformat(),
            "watch_pid": args.watch_pid,
            "pipeline_active": running,
            "phase": current_phase,
            "processed": processed,
            "total": TOTAL,
            "progress_percent": round(100 * processed / TOTAL, 4),
            "free_disk_gib": round(shutil.disk_usage(args.root).free / 1024**3, 2),
            "gpu": gpu(),
            "phase_eta_kyiv": phase_eta,
            "total_eta_kyiv": total_eta,
            "eta_basis": eta_basis,
            "preprocessing": prep,
            "duration_progress": duration_progress,
            "training": training,
        }
        with args.status.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        if not running:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
