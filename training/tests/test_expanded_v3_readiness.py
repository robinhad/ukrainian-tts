from pathlib import Path

from training.scripts.audit_expanded_v3_readiness import gate


def test_readiness_gate_has_required_fields():
    item = gate("Disk reserve", "PASS", "100 GiB is free.", Path("/tmp"), "Recheck.")
    assert item == {
        "gate": "Disk reserve",
        "status": "PASS",
        "evidence": "100 GiB is free.",
        "artifact": "/tmp",
        "next_action": "Recheck.",
    }
