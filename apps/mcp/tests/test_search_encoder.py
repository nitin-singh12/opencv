"""Verify search model integrity before native model parsing."""
from hashlib import sha256

import pytest

from opencv_bv_tools.runtime import safe_files
from opencv_bv_tools.runtime import search_encoder


def test_model_snapshot_checks_content_and_folder(tmp_path, monkeypatch):
    source = tmp_path / "models"
    source.mkdir()
    (source / "model").mkdir()
    contents = {"vision.onnx": b"vision", "model/tokenizer.json": b"vocabulary"}
    for name, value in contents.items():
        (source / name).write_bytes(value)
    monkeypatch.setattr(search_encoder, "ASSETS", {
        name: (len(value), sha256(value).hexdigest()) for name, value in contents.items()
    })
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = search_encoder.snapshot_assets(source, scratch, [source])
    assert all((result / name).read_bytes() == value for name, value in contents.items())

    (source / "vision.onnx").write_bytes(b"wrong!")
    scratch2 = tmp_path / "bad"
    scratch2.mkdir()
    with pytest.raises(ValueError, match="fgclip2_integrity_error"):
        search_encoder.snapshot_assets(source, scratch2, [source])

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(safe_files.UnsafePath):
        search_encoder.snapshot_assets(source, outside, [outside])


def test_search_configuration_rejects_unsupported_device(tmp_path):
    from opencv_bv_tools.runtime.context import ServerConfig

    with pytest.raises(ValueError, match="CPU only"):
        ServerConfig(allow_dirs=(tmp_path,), search_model_dir=tmp_path,
                     search_index_dir=tmp_path, search_device="cuda").validate()
