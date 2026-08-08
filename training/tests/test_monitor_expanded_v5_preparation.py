from training.scripts.monitor_expanded_v5_preparation import completed_iterations


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
