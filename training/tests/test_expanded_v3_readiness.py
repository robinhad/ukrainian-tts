from pathlib import Path

import pandas as pd

from training.scripts.audit_expanded_v3_readiness import (
    gate,
    hybrid_embeddings_are_balanced,
    minimum_source_duration_gates,
)


def test_readiness_gate_has_required_fields():
    item = gate("Disk reserve", "PASS", "100 GiB is free.", Path("/tmp"), "Recheck.")
    assert item == {
        "gate": "Disk reserve",
        "status": "PASS",
        "evidence": "100 GiB is free.",
        "artifact": "/tmp",
        "next_action": "Recheck.",
    }


def test_minimum_source_duration_gate_fails_below_required_hours():
    frame = pd.DataFrame(
        {
            "source_id": ["voa", "voa", "other"],
            "duration": [1800.0, 1800.0, 3600.0],
        }
    )
    registry = {
        "sources": {
            "voa": {"minimum_retained_hours": 2.0},
            "other": {},
        }
    }

    result = minimum_source_duration_gates(frame, registry)

    assert len(result) == 1
    assert result[0]["status"] == "FAIL"
    assert "1.000 hours" in result[0]["evidence"]


def test_hybrid_embedding_gate_accepts_nearest_split_for_odd_count():
    report = {
        "counts": {"clean": 51, "raw": 50},
        "maximum_speaker_delta": 1,
        "records": 101,
        "status": "PASS",
        "total_delta": 1,
    }

    assert hybrid_embeddings_are_balanced(report, 101)


def test_hybrid_embedding_gate_rejects_unbalanced_split():
    report = {
        "counts": {"clean": 52, "raw": 49},
        "maximum_speaker_delta": 3,
        "records": 101,
        "status": "PASS",
        "total_delta": 3,
    }

    assert not hybrid_embeddings_are_balanced(report, 101)
