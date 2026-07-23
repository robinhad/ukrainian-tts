from training.scripts.summarize_training_status import summarize


def test_summarize_gpu_status():
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
    assert report["status"] == "PASS"
    assert report["sample_count"] == 2
    assert report["gpus"] == [
        {
            "index": 0,
            "sample_count": 2,
            "power_limit_w": 350.0,
            "mean_sampled_power_w": 210.0,
            "maximum_sampled_power_w": 220.0,
            "mean_sampled_utilization_percent": 95.0,
            "maximum_sampled_utilization_percent": 100,
            "maximum_sampled_temperature_c": 82,
            "maximum_sampled_memory_mib": 21000,
        }
    ]


def test_empty_status_is_failure():
    report = summarize([])
    assert report["status"] == "FAIL"
    assert report["gpus"] == []
