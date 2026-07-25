#!/usr/bin/env python3
"""Fail-fast host resource checks for smoke or full JETS training."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_GPU_UUID = "GPU-be591530-39fd-0b1c-50af-8c75548cb6b8"


def run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def available_ram_gib() -> float:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0])
    return values["MemAvailable"] / 1024**2


def gpu_rows() -> list[dict[str, str]]:
    fields = [
        "index", "name", "uuid", "driver_version", "pstate", "memory.total",
        "memory.used", "memory.free", "utilization.gpu", "temperature.gpu",
        "power.draw", "power.limit", "compute_mode",
    ]
    output = run([
        "nvidia-smi", f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ])
    return [dict(zip(fields, row)) for row in csv.reader(io.StringIO(output), skipinitialspace=True)]


def compute_processes() -> list[dict[str, str]]:
    fields = ["gpu_uuid", "pid", "process_name", "used_memory"]
    output = run([
        "nvidia-smi", f"--query-compute-apps={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return []
    return [dict(zip(fields, row)) for row in csv.reader(io.StringIO(output), skipinitialspace=True)]


def torch_probe(expected_uuids: list[str]) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    if torch.cuda.device_count() < len(expected_uuids):
        raise RuntimeError(
            f"PyTorch sees {torch.cuda.device_count()} CUDA devices, expected {len(expected_uuids)}"
        )
    devices = []
    for index, expected_uuid in enumerate(expected_uuids):
        properties = torch.cuda.get_device_properties(index)
        tensor = torch.randn((2048, 2048), device=f"cuda:{index}")
        result = tensor @ tensor.T
        torch.cuda.synchronize(index)
        if not torch.isfinite(result).all().item():
            raise RuntimeError(f"CUDA matmul returned non-finite values on device {index}")
        devices.append({
            "index": index,
            "device_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_mib": round(properties.total_memory / 1024**2),
            "expected_uuid": expected_uuid,
        })
        del tensor, result
        torch.cuda.empty_cache()
    return {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": devices,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--gpu-uuid", default=DEFAULT_GPU_UUID)
    parser.add_argument(
        "--gpu-uuids",
        help="comma-separated GPU UUIDs; overrides --gpu-uuid",
    )
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-torch", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    minimum_ram = 32 if args.mode == "smoke" else 64
    minimum_disk = 60
    expected_uuids = (
        [value.strip() for value in args.gpu_uuids.split(",") if value.strip()]
        if args.gpu_uuids else [args.gpu_uuid]
    )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "gpu_uuids": expected_uuids,
        "checks": {},
        "errors": [],
    }
    try:
        rows = gpu_rows()
        targets = [next(row for row in rows if row["uuid"] == uuid) for uuid in expected_uuids]
        processes = [row for row in compute_processes() if row["gpu_uuid"] in expected_uuids]
        report["checks"]["gpus"] = rows
        report["checks"]["target_processes"] = processes
        for target in targets:
            free_mib = float(target["memory.free"])
            temperature = float(target["temperature.gpu"])
            if free_mib < 22 * 1024:
                report["errors"].append(
                    f"GPU {target['uuid']} has only {free_mib:.0f} MiB free"
                )
            if temperature >= 75:
                report["errors"].append(
                    f"GPU {target['uuid']} temperature is {temperature:.0f} C"
                )
        if processes:
            report["errors"].append("target GPU has active compute processes")
    except Exception as error:
        report["errors"].append(f"GPU probe failed: {error}")

    ram_gib = available_ram_gib()
    disk_gib = shutil.disk_usage(args.workspace).free / 1024**3
    report["checks"]["available_ram_gib"] = round(ram_gib, 2)
    report["checks"]["free_disk_gib"] = round(disk_gib, 2)
    report["checks"]["free_disk_warning_gib"] = 80
    report["checks"]["free_disk_stop_gib"] = minimum_disk
    if ram_gib < minimum_ram:
        report["errors"].append(f"available RAM {ram_gib:.1f} GiB < {minimum_ram} GiB")
    if disk_gib < minimum_disk:
        report["errors"].append(f"free disk {disk_gib:.1f} GiB < {minimum_disk} GiB")
    elif args.mode == "full" and disk_gib < 80:
        report["checks"]["disk_warning"] = (
            f"Free disk space is below the 80 GiB cleanup warning threshold."
        )

    if args.require_torch and not report["errors"]:
        try:
            report["checks"]["torch"] = torch_probe(expected_uuids)
        except Exception as error:
            report["errors"].append(f"PyTorch probe failed: {error}")

    report["status"] = "PASS" if not report["errors"] else "FAIL"
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
