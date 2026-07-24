import hashlib
from pathlib import Path

import pytest

from training.scripts.package_multispeaker_release import (
    load_pass_report,
    write_checksums,
)


def test_write_checksums_covers_each_file(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/b.txt").write_text("b", encoding="utf-8")

    write_checksums(tmp_path)

    expected = [
        f"{hashlib.sha256(b'a').hexdigest()}  a.txt",
        f"{hashlib.sha256(b'b').hexdigest()}  nested/b.txt",
    ]
    assert (tmp_path / "checksums.txt").read_text().splitlines() == expected


def test_load_pass_report_rejects_fail(tmp_path: Path):
    report = tmp_path / "report.json"
    report.write_text('{"status": "FAIL"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not have PASS"):
        load_pass_report(report, "test report")
