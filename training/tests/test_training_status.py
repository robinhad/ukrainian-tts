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
        "estimate_timestamp": None,
        "observed_seconds_per_iteration": None,
        "observed_iterations_per_minute": None,
        "latest_train_metrics": {"generator_loss": 61.084},
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


def test_parse_estimate_timestamp():
    text = (
        "[host:0/2] 2026-07-23 11:54:00,420 INFO: "
        "2/25epoch started. Estimated time to finish: "
        "11 hours, 23 minutes and 32.43 seconds"
    )
    status = parse_log(text, iterations_per_epoch=1000)
    assert status["estimate_timestamp"] == "2026-07-23 11:54:00,420"


def test_parse_latest_train_metrics():
    text = (
        "1epoch:train:11-20batch: generator_loss=9.125, "
        "generator_align_loss=5.5, generator_g_mel_loss=7.250e+01, "
        "discriminator_loss=1.25, train_time=1.5"
    )

    status = parse_log(text, iterations_per_epoch=1000)

    assert status["latest_train_metrics"] == {
        "generator_loss": 9.125,
        "generator_g_mel_loss": 72.5,
        "generator_align_loss": 5.5,
        "discriminator_loss": 1.25,
        "train_time": 1.5,
    }
