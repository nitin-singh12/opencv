"""The distribution step must retain only the validated wheel."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("retain_test_wheel", Path(__file__).resolve().parents[1] / "dev/retain_test_wheel.py")
retain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retain)


def test_only_successful_matching_wheel_is_retained(tmp_path, monkeypatch):
    work = tmp_path / "work"
    (work / "dist").mkdir(parents=True)
    wheel = work / "dist/opencv_bv_tools-0.10.0.dev13-test.whl"
    wheel.write_bytes(b"test wheel")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    report = {"status": "passed", "commit": "commit-one", "wheel_sha256": digest}
    (work / "package-report.json").write_text(json.dumps(report))
    monkeypatch.setenv("GITHUB_SHA", "commit-one")
    output = tmp_path / "retained"
    retain.main(work, output)
    assert (output / wheel.name).read_bytes() == wheel.read_bytes()
    assert (output / f"{wheel.name}.sha256").read_bytes() == f"{digest}  {wheel.name}\n".encode()
    wheel.write_bytes(b"different wheel")
    with pytest.raises(ValueError, match="checksum"):
        retain.main(work, tmp_path / "bad")
    wheel.write_bytes(b"test wheel")
    monkeypatch.setenv("GITHUB_SHA", "another-commit")
    with pytest.raises(ValueError, match="commit"):
        retain.main(work, tmp_path / "bad")
    report["status"] = "failed"
    (work / "package-report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="successful"):
        retain.main(work, tmp_path / "bad")
