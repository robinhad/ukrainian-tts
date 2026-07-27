from pathlib import Path

import pandas as pd

from training.scripts.prepare_hybrid_embeddings import (
    assign_variants,
    kaldi_speaker_id,
)


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


def test_kaldi_speaker_id_keeps_utterance_sort_order():
    utterances = ["source_0a", "source_0b", "source_ff"]
    lines = [
        f"{utterance} {kaldi_speaker_id(utterance, 'shared-speaker')}"
        for utterance in utterances
    ]
    assert lines == sorted(lines)
    assert lines == sorted(lines, key=lambda line: line.split(maxsplit=1)[1])
