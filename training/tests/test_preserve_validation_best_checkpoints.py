import json
from pathlib import Path

from training.scripts.preserve_validation_best_checkpoints import preserve_best


def test_preserve_best_keeps_ranked_available_checkpoints(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    log_path = exp_dir / "train.log"
    rows = [
        (1, 5.0, 10.0, 7.0),
        (2, 6.0, 9.0, 6.0),
        (3, 7.0, 8.0, 5.0),
        (4, 8.0, 7.0, 4.0),
        (5, 9.0, 6.0, 3.0),
    ]
    log_path.write_text(
        "\n".join(
            f"INFO: {epoch}epoch results: [valid] "
            f"generator_loss={generator}, "
            f"generator_g_mel_loss={mel}, "
            f"generator_align_loss={alignment}"
            for epoch, generator, mel, alignment in rows
        ),
        encoding="utf-8",
    )
    for epoch, *_ in rows:
        (exp_dir / f"{epoch}epoch.pth").write_bytes(f"checkpoint-{epoch}".encode())

    payload = preserve_best(log_path, exp_dir, keep=2)

    assert [item["epoch"] for item in payload["metrics"]["generator_loss"]] == [1, 2]
    assert [item["epoch"] for item in payload["metrics"]["generator_g_mel_loss"]] == [5, 4]
    assert [item["epoch"] for item in payload["metrics"]["generator_align_loss"]] == [5, 4]
    assert payload["preserved_epochs"] == [1, 2, 4, 5]
    for epoch in payload["preserved_epochs"]:
        source = exp_dir / f"{epoch}epoch.pth"
        target = exp_dir / "best_checkpoints" / f"{epoch}epoch.pth"
        assert source.stat().st_ino == target.stat().st_ino

    state = json.loads(
        (exp_dir / "best_checkpoints" / "validation_best.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["keep_per_metric"] == 2


def test_preserve_best_ignores_missing_checkpoint(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    log_path = exp_dir / "train.log"
    log_path.write_text(
        "INFO: 1epoch results: [valid] generator_loss=5, "
        "generator_g_mel_loss=4, generator_align_loss=3\n",
        encoding="utf-8",
    )

    payload = preserve_best(log_path, exp_dir, keep=3)

    assert payload["preserved_epochs"] == []
    assert all(not values for values in payload["metrics"].values())
