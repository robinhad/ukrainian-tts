import numpy as np

from training.scripts.select_diverse_listening_voices import (
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
