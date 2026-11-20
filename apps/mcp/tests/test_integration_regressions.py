"""Regression cases discovered while integrating the stacked feature branches."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from opencv_bv_tools import png_compression as png_cli
from opencv_bv_tools.runtime import worker_process as _reader_process
from opencv_bv_tools.qr_reader import QRLimits
from opencv_bv_tools.runtime import safe_files


@pytest.mark.parametrize("module", ["qr_reader", "barcode_reader"])
def test_reader_cli_rejects_fifo_without_waiting_for_writer(tmp_path, module):
    fifo = tmp_path / "input.png"
    os.mkfifo(fifo)
    result = subprocess.run([sys.executable, "-m", f"opencv_bv_tools.{module}", str(fifo), "--timeout", "0.1"],
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 2
    assert "regular file" in result.stderr


@pytest.mark.parametrize("when", ["worker_finished", "temporary_written"])
def test_png_cancel_does_not_publish(tmp_path, monkeypatch, when):
    source, destination = tmp_path / "source.png", tmp_path / "out.png"
    Image.new("RGB", (8, 8), "white").save(source)
    destination.write_bytes(b"keep existing output")
    cancelled = False
    def worker(*args, **kwargs):
        nonlocal cancelled
        if when == "worker_finished":
            cancelled = True
        return {"status": "unchanged", "data": source.read_bytes()}
    monkeypatch.setattr(png_cli, "compress_png", worker)
    publish = png_cli._publish_bytes
    def publish_late(*args, **kwargs):
        nonlocal cancelled
        cancelled = True
        return publish(*args, **kwargs)
    if when == "temporary_written":
        monkeypatch.setattr(png_cli, "_publish_bytes", publish_late)
    result = png_cli.compress_file(source, destination, overwrite=True, cancelled=lambda: cancelled)
    assert result["stop_reason"] == "cancelled" and result["data"] is None
    assert destination.read_bytes() == b"keep existing output"
    assert not list(tmp_path.glob(".opencv-bv-*.tmp"))


def test_safe_file_cancel_after_temp_write_preserves_destination(tmp_path):
    destination = tmp_path / "out.png"
    destination.write_bytes(b"original")
    def cancelled():
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        safe_files.write_file(b"replacement", destination, safe_files.Folders([tmp_path]),
                              overwrite=True, before_publish=cancelled)
    assert destination.read_bytes() == b"original"
    assert not list(tmp_path.glob(".email-redact-*.tmp"))


def test_worker_retries_transient_nonblocking_write(tmp_path, monkeypatch):
    original = os.write
    injected = False
    def write(fd, data):
        nonlocal injected
        if not injected:
            injected = True
            raise BlockingIOError(11, "Resource temporarily unavailable")
        return original(fd, data)
    monkeypatch.setattr(_reader_process.os, "write", write)
    response = {"search_status": "completed", "stop_reason": "configured_search_completed",
                "symbols": [], "undecoded": [], "image_size": None}
    script = "import sys;sys.stdin.buffer.read();print(" + repr(json.dumps(response)) + ")"
    result = _reader_process._run_worker(b"input", {"limits": asdict(QRLimits())}, timeout=3,
                                         command=[sys.executable, "-c", script])
    assert injected and result["search_status"] == "completed", result
