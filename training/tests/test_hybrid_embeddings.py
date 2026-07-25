from pathlib import Path

import pandas as pd

from training.scripts.prepare_hybrid_embeddings import assign_variants


def test_hybrid_assignment_is_exact_and_speaker_stratified():
    frame = pd.DataFrame(
        [
            {"utterance_id": f"a-{index}", "speaker_id": "a"}
            for index in range(5)
        ]
        + [
            {"utterance_id": f"b-{index}", "speaker_id": "b"}
            for index in range(4)
        ]
        + [{"utterance_id": "c-0", "speaker_id": "c"}]
    )
    assignment = assign_variants(frame)
    assert list(assignment.values()).count("raw") == 5
    assert list(assignment.values()).count("clean") == 5
    for speaker, speaker_frame in frame.groupby("speaker_id"):
        values = [assignment[value] for value in speaker_frame["utterance_id"]]
        assert abs(values.count("raw") - values.count("clean")) <= 1
    assert assignment == assign_variants(frame)
