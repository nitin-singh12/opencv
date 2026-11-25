"""Model download and offline cache behavior."""
import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from opencv_bv_tools import model_assets as assets


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    data = {'one.onnx': b'first test model', 'two.onnx': b'second test model'}
    monkeypatch.setattr(assets, 'MODELS', {
        name: (len(value), hashlib.sha256(value).hexdigest(), 'test-id')
        for name, value in data.items()
    })
    monkeypatch.setenv('OPENCV_BV_MODEL_DIR', str(tmp_path / 'cache café'))
    return data


def test_offline_provision_and_runtime_are_verified(catalog, tmp_path, monkeypatch):
    source = tmp_path / 'offline models'
    source.mkdir()
    for name, data in catalog.items():
        (source / name).write_bytes(data)
    result = assets.install_models(source)
    monkeypatch.setattr(assets, 'build_opener', lambda *a: pytest.fail('Unexpected network access'))
    for name, data in catalog.items():
        assert assets.verified_bytes(name) == data
        assert (result / name).read_bytes() == data
    assert assets.install_models() == result
    (result / 'one.onnx').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='invalid'):
        assets.verified_bytes('one.onnx')


def test_invalid_download_does_not_publish(catalog, tmp_path, monkeypatch):
    class Opener:
        def open(self, url, timeout):
            assert url.startswith('https://') and timeout == 60
            return io.BytesIO(b'<html>not a model</html>')
    monkeypatch.setattr(assets, 'build_opener', lambda *a: Opener())
    with pytest.raises(ValueError, match='unexpected size|SHA-256'):
        assets.install_models()
    assert list(assets.cache_directory().iterdir()) == []


def test_missing_model_explains_setup(catalog):
    with pytest.raises(FileNotFoundError, match='opencv-bv-models'):
        assets.verified_bytes('one.onnx')


def test_ocr_resolves_each_model_independently(tmp_path, monkeypatch):
    from opencv_bv_tools.runtime.ocr_native import NativeOCR

    bundled = tmp_path / 'bundled'
    cached = tmp_path / 'cache'
    bundled.mkdir()
    cached.mkdir()
    det = 'PP-OCRv6_det_small.onnx'
    rec = 'PP-OCRv6_rec_small.onnx'
    (cached / det).touch()
    (bundled / rec).touch()
    monkeypatch.delenv('OPENCV_BV_MODEL_DIR', raising=False)
    monkeypatch.setattr(assets, 'BUNDLED', bundled)
    monkeypatch.setattr(assets, 'cache_directory', lambda: cached)
    ocr = NativeOCR(None, tmp_path / 'characters.txt')
    assert ocr.model_path(det) == cached / det
    assert ocr.model_path(rec) == bundled / rec


def test_parallel_installs_use_verified_complete_files(catalog, tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    for name, data in catalog.items():
        (source / name).write_bytes(data)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda _: assets.install_models(source), range(3)))
    for name, data in catalog.items():
        assert assets.verified_bytes(name) == data
    assert sorted(p.name for p in assets.cache_directory().iterdir()) == sorted(catalog)


def test_insecure_redirect_is_rejected():
    with pytest.raises(ValueError, match='insecure'):
        assets.HTTPSRedirects().redirect_request(None, None, 302, '', {}, 'http://example.com/model')


def test_interrupted_download_leaves_no_partial_model(catalog, monkeypatch):
    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError('connection lost')
            return super().read(2)
    class Opener:
        def open(self, url, timeout):
            return Interrupted(b'partial bytes')
    monkeypatch.setattr(assets, 'build_opener', lambda *a: Opener())
    with pytest.raises(OSError, match='connection lost'):
        assets.install_models()
    assert list(assets.cache_directory().iterdir()) == []
