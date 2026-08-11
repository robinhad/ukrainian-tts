#!/usr/bin/env python3
"""Write v7 phase, resource, power, progress, and Kyiv ETA records."""

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

NAME = "expanded_v7_sidon_deess_declick_limit_novoa"
TOTAL_AUDIO = 74_156


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


def phase(processes: str, active: bool) -> str:
    if not active:
        return "complete_or_failed"
    if "evaluate_expanded_v7_declick_limited" in processes:
        return "milestone_evaluation"
    if "espnet2.bin.gan_tts_train" in processes or "torchrun" in processes:
        return "training"
    if "run_expanded_v7_declick_limited_smoke" in processes:
        return "smoke_training_or_inference"
    if "extract_spk_embed" in processes:
        return "clean_speaker_embeddings"
    if "prepare_expanded_v7_declick_limited" in processes:
        return "manifests_tokens_or_statistics"
    if "declick_and_limit_audio.py" in processes:
        return "declick_and_peak_limit"
    return "initializing_or_transition"


def gpu() -> list[dict]:
    query = "index,utilization.gpu,memory.used,power.draw,power.limit,temperature.gpu"
    output = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    result = []
    for line in output.splitlines():
        index, utilization, memory, power, limit, temperature = [
            value.strip() for value in line.split(",")
        ]
        result.append(
            {
                "index": int(index),
                "utilization_percent": float(utilization),
                "memory_used_mib": float(memory),
                "power_w": float(power),
                "power_limit_w": float(limit),
                "power_utilization_percent": round(
                    100 * float(power) / float(limit), 1
                ),
                "temperature_c": float(temperature),
            }
        )
    return result


def latest_jsonl(path: Path) -> dict | None:
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return json.loads(lines[-1]) if lines else None


def audio_count(root: Path) -> int:
    directory = root / f"data/{NAME}/audio_24k"
    if not directory.is_dir():
        return 0
    return sum(
        entry.is_file() and entry.name.lower().endswith(".wav")
        for entry in os.scandir(directory)
    )


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
    started = time.monotonic()
    initial_audio = audio_count(args.root)
    while True:
        active = is_active(args.watch_pid)
        current_phase = phase(commands(), active)
        now = datetime.now(ZoneInfo("Europe/Kyiv"))
        count = audio_count(args.root)
        elapsed = max(time.monotonic() - started, 1.0)
        new_audio = max(count - initial_audio, 0)
        rate = new_audio / elapsed
        phase_eta = None
        total_eta = None
        eta_basis = "not_available"
        training = latest_jsonl(
            args.root / f"reports/training_status_{NAME}_100k.jsonl"
        )
        if current_phase == "declick_and_peak_limit":
            remaining = (TOTAL_AUDIO - count) / rate if rate > 0 else 3600
            phase_eta = now + timedelta(seconds=max(remaining, 0))
            total_eta = phase_eta + timedelta(hours=26)
            eta_basis = "observed_audio_rate_plus_26h_downstream_estimate"
        elif current_phase in {
            "manifests_tokens_or_statistics",
            "clean_speaker_embeddings",
        }:
            phase_eta = now + timedelta(hours=2)
            total_eta = now + timedelta(hours=26)
            eta_basis = "v6_runtime_estimate"
        elif current_phase == "smoke_training_or_inference":
            phase_eta = now + timedelta(hours=1)
            total_eta = now + timedelta(hours=24)
            eta_basis = "v6_runtime_estimate"
        elif current_phase == "training" and training:
            phase_eta = training.get("eta_kyiv")
            if phase_eta:
                total_eta = datetime.fromisoformat(phase_eta) + timedelta(hours=2)
            eta_basis = "observed_training_rate_plus_2h_evaluation_estimate"
        elif current_phase == "milestone_evaluation":
            phase_eta = now + timedelta(hours=2)
            total_eta = phase_eta
            eta_basis = "v6_runtime_estimate"
        elif current_phase == "initializing_or_transition":
            phase_eta = now + timedelta(minutes=10)
            total_eta = now + timedelta(hours=27)
            eta_basis = "initial_v6_runtime_estimate"
        record = {
            "timestamp_kyiv": now.isoformat(),
            "watch_pid": args.watch_pid,
            "pipeline_active": active,
            "phase": current_phase,
            "audio_processed": count,
            "audio_total": TOTAL_AUDIO,
            "audio_progress_percent": round(100 * count / TOTAL_AUDIO, 4),
            "audio_rate_files_per_second": round(rate, 4),
            "free_disk_gib": round(shutil.disk_usage(args.root).free / 1024**3, 2),
            "gpu": gpu(),
            "phase_eta_kyiv": (
                phase_eta.isoformat() if isinstance(phase_eta, datetime) else phase_eta
            ),
            "total_eta_kyiv": (
                total_eta.isoformat() if isinstance(total_eta, datetime) else None
            ),
            "eta_basis": eta_basis,
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
