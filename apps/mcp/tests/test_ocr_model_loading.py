"""Check verified OCR model loading from paths containing non-ASCII text."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import rapidocr

from opencv_bv_tools.runtime import ocr_opencv as backend


def config(role):
    return SimpleNamespace(task_type=SimpleNamespace(value=role),
                           ocr_version=SimpleNamespace(value="PP-OCRv6"),
                           model_type=SimpleNamespace(value="small"),
                           lang_type=SimpleNamespace(value="ch"))


@pytest.mark.parametrize("role", ["det", "rec"])
def test_verified_model_bytes_from_unicode_path(tmp_path, monkeypatch, role):
    folder = tmp_path / "model café 漢字"
    (folder / "models").mkdir(parents=True)
    data = b"verified test model"
    name, _ = backend.MODELS[role]
    (folder / "models" / name).write_bytes(data)
    monkeypatch.setattr(rapidocr, "__file__", str(folder / "__init__.py"))
    monkeypatch.setitem(backend.MODELS, role, (name, hashlib.sha256(data).hexdigest()))
    monkeypatch.setattr(backend, "_characters", lambda: [])
    seen = []

    class Net:
        def setPreferableBackend(self, value):
            pass

        def setPreferableTarget(self, value):
            pass

    def load(model):
        assert isinstance(model, np.ndarray) and model.dtype == np.uint8
        seen.append(model.tobytes())
        return Net()

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", load)
    backend.OpenCVSession(config(role))
    assert seen == [data]
    (folder / "models" / name).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="missing or has changed"):
        backend.OpenCVSession(config(role))
    assert seen == [data]


@pytest.mark.parametrize("role,shape", [("det", (1, 3, 64, 64)), ("rec", (1, 3, 48, 320))])
def test_buffer_and_path_inference_match(tmp_path, monkeypatch, role, shape):
    name, digest = backend.MODELS[role]
    path = Path(rapidocr.__file__).parent / "models" / name
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest
    # Keep the reference path ASCII on the hosted Windows runner; the actual
    # session must load a model in a non-ASCII directory on every platform.
    reference = tmp_path / "reference.onnx"
    reference.write_bytes(data)
    folder = tmp_path / "model café 漢字"
    (folder / "models").mkdir(parents=True)
    (folder / "models" / name).write_bytes(data)
    monkeypatch.setattr(rapidocr, "__file__", str(folder / "__init__.py"))
    previous = cv2.getNumThreads()
    cv2.setNumThreads(2)
    try:
        by_path = cv2.dnn.readNetFromONNX(str(reference))
        by_buffer = backend.OpenCVSession(config(role))
        tensor = np.zeros(shape, dtype=np.float32)
        by_path.setInput(tensor, "x")
        assert np.array_equal(by_path.forward("fetch_name_0"), by_buffer(tensor))
    finally:
        cv2.setNumThreads(previous)
