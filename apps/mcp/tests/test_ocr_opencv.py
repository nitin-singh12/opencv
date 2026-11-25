"""Backend integrity, lazy isolation and real shape-changing OpenCV inference."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import cv2
import numpy as np
import pytest

from opencv_bv_tools.runtime import ocr_opencv as backend
from opencv_bv_tools.runtime.image_redact import Ocr


def test_lazy_instances_are_independent(monkeypatch):
    from opencv_bv_tools.runtime import ocr_native as native
    created = []
    class FakeDetector:
        def __init__(self, path):
            self.session = object()
            created.append(self)
        def __call__(self, image):
            return None
    monkeypatch.setattr(native, "Detector", FakeDetector)
    first, second = Ocr(), Ocr()
    assert first.providers() == second.providers() == {}
    first.engine(np.zeros((32, 32, 3), np.uint8), use_det=True, use_cls=False, use_rec=False)
    assert first.engine.text_det is not None and second.engine.text_det is None
    second.engine(np.zeros((32, 32, 3), np.uint8), use_det=True, use_cls=False, use_rec=False)
    assert first.engine.text_det.session is not second.engine.text_det.session
    assert first.providers() == {"text_det": ["OpenCV DNN CPU"]}


@pytest.mark.parametrize("det,cls,rec", [(True, True, False), (True, False, True), (False, False, False)])
def test_unsupported_ocr_modes_are_rejected(det, cls, rec):
    ocr = Ocr()
    with pytest.raises(ValueError, match="classification is unavailable"):
        ocr.engine(np.zeros((32, 32, 3), np.uint8), use_det=det, use_cls=cls, use_rec=rec)


@pytest.mark.parametrize("kind", ["missing", "changed", "wrong_count"])
def test_dictionary_integrity_before_model_loading(tmp_path, monkeypatch, kind):
    path = tmp_path / "characters.txt"
    if kind != "missing":
        path.write_bytes(b"wrong\ncharacters")
    monkeypatch.setattr(backend, "CHARACTERS", path)
    if kind == "wrong_count":
        monkeypatch.setattr(backend, "CHARACTERS_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *args: pytest.fail("Unverified dictionary"))
    from opencv_bv_tools.runtime import ocr_native as native
    with pytest.raises(RuntimeError, match="missing or has changed|unexpected size"):
        native.Recognizer(tmp_path / "unused.onnx", path)


@pytest.mark.parametrize("exists", [False, True])
def test_missing_or_changed_weights_never_load(tmp_path, monkeypatch, exists):
    from opencv_bv_tools.runtime import ocr_native as native
    path = tmp_path / backend.MODELS["det"][0]
    if exists:
        path.write_bytes(b"changed model")
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *args: pytest.fail("Unverified model"))
    with pytest.raises(RuntimeError, match="missing or has changed"):
        native.Detector(path)


@pytest.mark.parametrize("options", [{"device": "cuda"}, {"models": "v6-tiny"}])
def test_unsupported_configuration_is_explicit(options):
    with pytest.raises(ValueError, match="v6-small models on CPU"):
        Ocr(**options)


def test_recognizer_batches_widths_and_repeated_net():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(2)
    try:
        ocr = Ocr()
        from opencv_bv_tools.runtime import ocr_native as native
        path = backend.model_path(backend.MODELS["rec"][0])
        recognizer = native.Recognizer(path, backend.CHARACTERS)
        session = recognizer.session
        first = np.random.default_rng(20261112).standard_normal((1, 3, 48, 320)).astype(np.float32)
        expected = session(first)
        for batch, width in ((2, 640), (1, 160), (3, 320), (1, 320)):
            output = session(np.zeros((batch, 3, 48, width), np.float32))
            assert output.shape[0] == batch and output.shape[2] == backend.CHARACTER_COUNT + 2
            assert np.isfinite(output).all()
        assert np.array_equal(session(first), expected)
        assert len(recognizer.characters) == expected.shape[2]
    finally:
        cv2.setNumThreads(previous)


def test_ocr_and_pdf_imports_without_runtime_or_network():
    fixture = Path(__file__).parent / "fixtures/smoke/email.png"
    script = r'''
import importlib.abc,json,socket,sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'rapidocr', 'onnxruntime', 'onnx', 'shapely'}:
            raise AssertionError('Forbidden inference dependency: ' + name)
sys.meta_path.insert(0,Deny())
def offline(*args, **kwargs):
    raise AssertionError('Inference must not download anything')
socket.socket.connect=offline
socket.create_connection=offline
import cv2
from opencv_bv_tools.runtime import image_redact as app, pdf_redact
cv2.setNumThreads(2)
app.check_models()
ocr=app.Ocr(faint_pass=True,gate_second_read=True,size_guard=True)
with open(sys.argv[1],'rb') as stream:
    rgb=app.open_image(stream)
result=app.detect(rgb,ocr)
assert len(result.boxes)>=2
assert not hasattr(ocr.engine, "text_cls")
assert not any(k.split('.')[0] in {'rapidocr','onnxruntime','onnx','shapely'} for k in sys.modules)
print(json.dumps({'boxes':len(result.boxes),'backend':ocr.providers()}))
'''
    process = subprocess.run([sys.executable, "-c", script, str(fixture)], capture_output=True,
                             text=True, timeout=90)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["backend"] == {"text_det": ["OpenCV DNN CPU"], "text_rec": ["OpenCV DNN CPU"]}


_GOLDEN_UNCLIP = json.loads((Path(__file__).parent / 'fixtures/ocr/unclip-golden.json').read_text())['cases']


@pytest.mark.parametrize('distance', [float('nan'), float('inf'), float('-inf'), 1e300])
def test_native_clipper_rejects_unsafe_distances(distance):
    from opencv_bv_tools import _clipper_native
    with pytest.raises(ValueError):
        _clipper_native.offset([[0, 0], [10, 0], [10, 10], [0, 10]], distance)


def test_native_clipper_rejects_unsafe_coordinates():
    from opencv_bv_tools import _clipper_native
    with pytest.raises(ValueError):
        _clipper_native.offset([[0, 0], [2**61, 0], [2**61, 10]], 1)


@pytest.mark.parametrize('case', _GOLDEN_UNCLIP, ids=lambda case: case['id'])
def test_unclip_matches_frozen_pre_removal_geometry(monkeypatch, case):
    from opencv_bv_tools.runtime import ocr_native as detector

    original = detector._clipper_native
    distances = []

    def recording_offset(points, distance):
        distances.append(float(distance).hex())
        return original.offset(points, distance)

    monkeypatch.setattr(detector, '_clipper_native', SimpleNamespace(offset=recording_offset))
    operation = detector.DBPostProcess(unclip_ratio=case['unclip_ratio'])
    points = np.array(case['points'], dtype=case['dtype']).reshape(case['shape'])
    if 'exception' in case:
        with pytest.raises(getattr(builtins, case['exception']['type'])):
            operation.unclip(points)
        assert distances == []
        return

    actual = operation.unclip(points)
    expected = case['expanded']
    assert distances == [case['distance_hex']]
    assert actual.dtype == np.dtype(expected['dtype'])
    np.testing.assert_array_equal(
        actual,
        np.array(expected['values'], dtype=expected['dtype']).reshape(expected['shape']),
    )


def test_ctc_keeps_blank_separated_repeats_and_character_positions():
    from opencv_bv_tools.runtime.ocr_native import Recognizer
    recognizer = Recognizer.__new__(Recognizer)
    recognizer.characters = ["blank", "a", "b", " "]
    indices = [1, 1, 0, 1, 3, 2]
    predictions = np.zeros((1, len(indices), len(recognizer.characters)), dtype=np.float32)
    predictions[0, np.arange(len(indices)), indices] = 1.0
    inputs = []
    def inference(tensor):
        inputs.append(tensor)
        return predictions
    recognizer.session = inference
    output = recognizer(np.zeros((48, 96, 3), dtype=np.uint8))
    assert output.txts == ["aa b"]
    assert output.word_results[0].word_cols == [[0, 3, 5]]
    assert output.word_results[0].line_txt_len == pytest.approx(6 * 2 / (320 / 48))
    assert inputs[0].shape == (1, 3, 48, 320)
    assert np.all(inputs[0][:, :, :, 96:] == 0)
