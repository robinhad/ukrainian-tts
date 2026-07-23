from training.scripts.training_status import parse_log


def test_parse_training_status():
    text = """
    3/25epoch started. Estimated time to finish: 6 hours, 31 minutes and 14.06 seconds
    3epoch:train:111-120batch: generator_loss=61.084
    """
    status = parse_log(text, iterations_per_epoch=1000)
    assert status == {
        "epoch": 3,
        "batch": 120,
        "total_iterations": 2120,
        "estimated_seconds_remaining": 23474.06,
        "observed_seconds_per_iteration": None,
        "observed_iterations_per_minute": None,
        "error_matches": 0,
    }


def test_parse_training_error():
    status = parse_log("Traceback: RuntimeError: out of memory", iterations_per_epoch=1000)
    assert status["error_matches"] == 3


def test_parse_observed_training_rate():
    text = """
    [host:0/2] 2026-07-23 11:30:00,000 INFO: 1epoch:train:91-100batch
    [host:0/2] 2026-07-23 11:30:16,000 INFO: 1epoch:train:101-110batch
    [host:0/2] 2026-07-23 11:30:32,000 INFO: 1epoch:train:111-120batch
    """
    status = parse_log(text, iterations_per_epoch=1000)
    assert status["observed_seconds_per_iteration"] == 1.6
    assert status["observed_iterations_per_minute"] == 37.5
