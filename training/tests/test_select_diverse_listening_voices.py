import numpy as np
from kaldiio import save_ark

from training.scripts.select_diverse_listening_voices import (
    load_vectors,
    representative_cluster_medoids,
)


def test_representative_cluster_medoids_select_cluster_centers():
    candidates = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [-1.0, 0.0],
            [-0.99, -0.01],
        ],
        dtype=np.float32,
    )

    selected = representative_cluster_medoids(candidates, 2)

    assert len(selected) == 2
    assert {0 if candidates[index, 0] > 0 else 1 for index in selected} == {0, 1}


def test_load_vectors_reads_sharded_scp(tmp_path):
    first_ark = tmp_path / "part-0.ark"
    first_scp = tmp_path / "part-0.scp"
    second_ark = tmp_path / "part-1.ark"
    second_scp = tmp_path / "part-1.scp"
    save_ark(
        str(first_ark),
        {"utterance-1": np.array([1.0, 0.0], dtype=np.float32)},
        scp=str(first_scp),
    )
    save_ark(
        str(second_ark),
        {"utterance-2": np.array([0.0, 1.0], dtype=np.float32)},
        scp=str(second_scp),
    )
    combined_scp = tmp_path / "xvector.scp"
    combined_scp.write_text(
        first_scp.read_text(encoding="utf-8")
        + second_scp.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    vectors = load_vectors(combined_scp)

    assert sorted(vectors) == ["utterance-1", "utterance-2"]
