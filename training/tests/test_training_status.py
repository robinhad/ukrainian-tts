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
        "error_matches": 0,
    }


def test_parse_training_error():
    status = parse_log("Traceback: RuntimeError: out of memory", iterations_per_epoch=1000)
    assert status["error_matches"] == 3
