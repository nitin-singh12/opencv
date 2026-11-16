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


@pytest.mark.parametrize("after", [0, 1, 2])
def test_companion_patch_insertion_keeps_original_line_order(monkeypatch, after):
    path = Path(__file__).resolve().parents[1] / "scripts/build_rapidocr.py"
    spec = importlib.util.spec_from_file_location("build_rapidocr", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    name = "rapidocr/main.py"
    monkeypatch.setattr(builder, "PATCHED", {name})
    files = {name: b"first\nsecond\n"}
    patch = f"--- a/{name}\n+++ b/{name}\n@@ -{after},0 +{after+1},1 @@\n+inserted\n"
    builder.apply_patch(files, patch)
    expected = [b"first\n", b"second\n"]
    expected.insert(after, b"inserted\n")
    assert files[name] == b"".join(expected)


def test_lazy_instances_and_classifier_never_falls_back(monkeypatch):
    import rapidocr.ch_ppocr_cls.main as classifier
    import rapidocr.ch_ppocr_det.main as detector
    import rapidocr.ch_ppocr_rec.main as recognizer

    def forbidden(*args):
        pytest.fail("Injected sessions must not select an upstream inference engine")

    for module in (classifier, detector, recognizer):
        monkeypatch.setattr(module, "get_engine", forbidden)
    first, second = Ocr(), Ocr()
    assert first.providers() == second.providers() == {}
    assert first.engine.text_det is second.engine.text_det is None
    first.engine._load_det_model()
    assert second.engine.text_det is None
    second.engine._load_det_model()
    assert first.engine.text_det.session is not second.engine.text_det.session
    assert first.providers() == {"text_det": ["OpenCV DNN CPU"]}
    with pytest.raises(ValueError, match="bundled v6-small"):
        first.engine._load_cls_model()
    assert first.engine.text_cls is None


def test_companion_without_factory_keeps_upstream_engine_selection(monkeypatch):
    from rapidocr import RapidOCR
    import rapidocr.ch_ppocr_det.main as detector
    import rapidocr.ch_ppocr_rec.main as recognizer

    class Session:
        def have_key(self, key="character"):
            return key == "character"

        def get_character_list(self, key="character"):
            return ["a", "b"]

    selected = []
    def select_engine(kind):
        selected.append(kind)
        return lambda cfg: Session()

    for module in (detector, recognizer):
        monkeypatch.setattr(module, "get_engine", select_engine)
    ocr = RapidOCR(params={"Global.use_cls": False})
    assert isinstance(ocr._load_det_model().session, Session)
    assert isinstance(ocr._load_rec_model().session, Session)
    assert len(selected) == 2


@pytest.mark.parametrize("kind", ["missing", "changed", "wrong_count"])
def test_dictionary_integrity_before_model_loading(tmp_path, monkeypatch, kind):
    path = tmp_path / "characters.txt"
    if kind != "missing":
        path.write_bytes(b"wrong\ncharacters")
    monkeypatch.setattr(backend, "CHARACTERS", path)
    if kind == "wrong_count":
        monkeypatch.setattr(backend, "CHARACTERS_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *args: pytest.fail("Unverified dictionary"))
    with pytest.raises(RuntimeError, match="missing or has changed|unexpected size"):
        Ocr().engine._load_rec_model()


@pytest.mark.parametrize("exists", [False, True])
def test_missing_or_changed_weights_never_load(tmp_path, monkeypatch, exists):
    import rapidocr
    monkeypatch.setattr(rapidocr, "__file__", str(tmp_path / "__init__.py"))
    if exists:
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / backend.MODELS["det"][0]).write_bytes(b"changed model")
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *args: pytest.fail("Unverified model"))
    with pytest.raises(RuntimeError, match="missing or has changed"):
        Ocr().engine._load_det_model()


@pytest.mark.parametrize("options", [{"device": "cuda"}, {"models": "v6-tiny"}])
def test_unsupported_configuration_is_explicit(options):
    with pytest.raises(ValueError, match="v6-small models on CPU"):
        Ocr(**options)


def test_recognizer_batches_widths_and_repeated_net():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(2)
    try:
        ocr = Ocr()
        session = ocr.engine._load_rec_model().session
        first = np.random.default_rng(20261112).standard_normal((1, 3, 48, 320)).astype(np.float32)
        expected = session(first)
        for batch, width in ((2, 640), (1, 160), (3, 320), (1, 320)):
            output = session(np.zeros((batch, 3, 48, width), np.float32))
            assert output.shape[0] == batch and output.shape[2] == backend.CHARACTER_COUNT + 2
            assert np.isfinite(output).all()
        assert np.array_equal(session(first), expected)
        assert len(ocr.engine.text_rec.postprocess_op.character) == expected.shape[2]
    finally:
        cv2.setNumThreads(previous)


def test_ocr_and_pdf_imports_without_runtime_or_network():
    fixture = Path(__file__).parent / "fixtures/smoke/email.png"
    script = r'''
import importlib.abc,json,socket,sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'onnxruntime', 'onnx', 'shapely'}:
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
assert ocr.engine.text_cls is None
assert not any(k.split('.')[0] in {'onnxruntime','onnx','shapely'} for k in sys.modules)
print(json.dumps({'boxes':len(result.boxes),'backend':ocr.providers()}))
'''
    process = subprocess.run([sys.executable, "-c", script, str(fixture)], capture_output=True,
                             text=True, timeout=90)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["backend"] == {"text_det": ["OpenCV DNN CPU"], "text_rec": ["OpenCV DNN CPU"]}


_GOLDEN_UNCLIP = json.loads((Path(__file__).parent / 'fixtures/ocr/unclip-golden.json').read_text())['cases']


@pytest.mark.parametrize('case', _GOLDEN_UNCLIP, ids=lambda case: case['id'])
def test_unclip_matches_frozen_pre_removal_geometry(monkeypatch, case):
    import rapidocr.ch_ppocr_det.utils as detector

    original = detector.pyclipper
    distances = []

    class RecordingOffset:
        def __init__(self):
            self.inner = original.PyclipperOffset()

        def AddPath(self, *args):
            return self.inner.AddPath(*args)

        def Execute(self, distance):
            distances.append(float(distance).hex())
            return self.inner.Execute(distance)

    monkeypatch.setattr(detector, 'pyclipper', SimpleNamespace(
        PyclipperOffset=RecordingOffset,
        JT_ROUND=original.JT_ROUND,
        ET_CLOSEDPOLYGON=original.ET_CLOSEDPOLYGON,
    ))
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
