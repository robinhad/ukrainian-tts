from datetime import datetime
from zoneinfo import ZoneInfo

from training.scripts.monitor_expanded_v5_preparation import (
    completed_iterations,
    first_log_time,
)


def test_completed_iterations_handles_split_counter_reset() -> None:
    text = "\n".join(
        [
            "INFO: Niter: 10",
            "INFO: Niter: 20",
            "INFO: Niter: 30",
            "INFO: Niter: 10",
            "INFO: Niter: 20",
        ]
    )
    assert completed_iterations(text) == 50


def test_completed_iterations_handles_empty_log() -> None:
    assert completed_iterations("") == 0


def test_first_log_time_uses_espnet_timestamp() -> None:
    text = "[host] 2026-08-08 19:12:43,595 (gan_tts:290) INFO: ready"
    expected = datetime(2026, 8, 8, 19, 12, 43, tzinfo=ZoneInfo("Europe/Kyiv"))
    assert first_log_time(text) == expected.timestamp()
