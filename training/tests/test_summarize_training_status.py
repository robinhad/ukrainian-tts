import json
import subprocess
import sys
from pathlib import Path

from training.scripts.summarize_training_status import summarize


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_training_status.py"


def test_summarize_keeps_the_original_report_fields() -> None:
    rows = [
        {
            "timestamp_kyiv": "2026-07-23T11:00:00+03:00",
            "gpus": [
                {
                    "index": 0,
                    "power_draw_w": 200.0,
                    "power_limit_w": 350.0,
                    "utilization_percent": 90,
                    "temperature_c": 80,
                    "memory_used_mib": 20000,
                }
            ],
        },
        {
            "timestamp_kyiv": "2026-07-23T11:05:00+03:00",
            "gpus": [
                {
                    "index": 0,
                    "power_draw_w": 220.0,
                    "power_limit_w": 350.0,
                    "utilization_percent": 100,
                    "temperature_c": 82,
                    "memory_used_mib": 21000,
                }
            ],
        },
    ]
    report = summarize(rows)
    gpu = report["gpus"][0]
    assert report["status"] == "PASS"
    assert report["first_sample_kyiv"] == "2026-07-23T11:00:00+03:00"
    assert gpu["power_limit_w"] == 350.0
    assert gpu["mean_sampled_power_w"] == 210.0
    assert gpu["maximum_sampled_utilization_percent"] == 100


def test_empty_status_is_failure() -> None:
    assert summarize([])["status"] == "FAIL"


def test_status_summary_checks_two_gpus_and_monitor_gap(tmp_path: Path) -> None:
    source = tmp_path / "status.jsonl"
    output = tmp_path / "summary.json"
    rows = []
    for minute, steps in ((0, 100), (15, 200), (30, 300)):
        rows.append(
            {
                "timestamp_kyiv": f"2026-08-05T04:{minute:02d}:00+03:00",
                "target_iterations": 100000,
                "total_iterations": steps,
                "eta_kyiv": "2026-08-06T01:00:00+03:00",
                "free_disk_gib": 40.0,
                "critical_conditions": [],
                "error_matches": 0,
                "gpus": [
                    {
                        "index": index,
                        "power_draw_w": 220.0 + index,
                        "power_utilization_percent": 73.0,
                        "temperature_c": 80 - index,
                        "memory_used_mib": 22000,
                        "utilization_percent": 100,
                    }
                    for index in (0, 1)
                ],
            }
        )
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(source),
            "--output",
            str(output),
            "--target-iterations",
            "100000",
        ],
        check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["maximum_observed_gap_minutes"] == 15.0
    assert [gpu["index"] for gpu in report["gpus"]] == [0, 1]
