"""Real subprocess/pipe lifecycle checks with an explicitly synthetic encoder."""
from dataclasses import replace
from pathlib import Path
import sys
import threading
import time

import pytest

from opencv_bv_tools.runtime.context import ServerConfig
from opencv_bv_tools.runtime import search_files, search_process
from opencv_bv_tools.runtime.search_process import SearchError

CODE = '''
import time
from pathlib import Path
from opencv_bv_tools.runtime import search_encoder, search_worker
class Fake:
    info = {"fingerprint": "a" * 64}
    def __init__(self, options):
        (Path(options["snapshot_dir"]) / "synthetic-model").write_bytes(b"synthetic")
    def encode(self, raw, kind, max_pixels):
        if raw == b"slow":
            time.sleep(30)
        if raw == b"bad":
            raise ValueError("invalid_image")
        return {**self.info, "vector": [1., 0.], "truncated": False}
search_encoder.Encoder = Fake
search_worker.main()
'''


@pytest.fixture
def encoder(tmp_path, monkeypatch):
    index = search_files.create_private_directory(tmp_path / "index")
    config = ServerConfig(allow_dirs=(tmp_path,), search_model_dir=tmp_path,
                          search_index_dir=index, search_timeout=3, search_idle_seconds=1)
    original_start = search_process.start_worker
    monkeypatch.setattr(search_process, "start_worker",
                        lambda command, **kwargs: original_start([sys.executable, "-c", CODE], **kwargs))
    encoder = search_process.EncoderProcess(config)
    encoder.scratch_parent = index
    yield encoder
    encoder.close()
    assert not list(index.iterdir())


def test_search_worker_reuse_rejection_and_idle_cleanup(encoder):
    assert encoder.request("info")["fingerprint"] == "a" * 64
    process = encoder.process
    scratch = Path(encoder.scratch.name)
    assert (scratch / "synthetic-model").exists()
    with pytest.raises(SearchError, match="invalid_image"):
        encoder.request("image", b"bad")
    assert encoder.process is process
    assert encoder.request("text", b"query")["vector"] == [1., 0.]
    deadline = time.monotonic() + 5
    while (encoder.process is not None or scratch.exists()) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert encoder.process is None and process.poll() is not None
    assert not scratch.exists()
    assert encoder.request("info")["fingerprint"] == "a" * 64
    assert encoder.process is not process


def test_search_worker_timeout_reaps_and_removes_scratch(encoder):
    encoder.request("info")
    encoder.config = replace(encoder.config, search_timeout=0.2)
    process = encoder.process
    scratch = Path(encoder.scratch.name)
    with pytest.raises(SearchError, match="search_timeout"):
        encoder.request("image", b"slow")
    assert process.poll() is not None and encoder.process is None
    assert not scratch.exists()


def test_search_worker_cancellation_reaps_and_removes_scratch(encoder):
    encoder.request("info")
    process = encoder.process
    scratch = Path(encoder.scratch.name)
    cancelled = threading.Event()
    timer = threading.Timer(0.1, cancelled.set)
    timer.start()
    def check():
        if cancelled.is_set():
            raise RuntimeError("cancelled")
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            encoder.request("image", b"slow", check)
    finally:
        timer.cancel()
    assert process.poll() is not None and encoder.process is None
    assert not scratch.exists()
