from types import SimpleNamespace

import pytest

from training.scripts.summarize_tensorboard import summarize_events


def event(step: int, value: float, wall_time: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(step=step, value=value, wall_time=wall_time)


def test_summarize_events_keeps_latest_duplicate_step():
    result = summarize_events(
        [
            event(2, 7.0),
            event(1, 10.0),
            event(2, 6.0, wall_time=2.0),
            event(3, 5.0),
        ]
    )

    assert result == {
        "count": 3,
        "first_step": 1,
        "first_value": 10.0,
        "last_step": 3,
        "last_value": 5.0,
        "minimum_value": 5.0,
        "maximum_value": 10.0,
        "absolute_change": -5.0,
    }


def test_summarize_events_rejects_non_finite_value():
    with pytest.raises(ValueError, match="non-finite"):
        summarize_events([event(1, float("nan"))])
