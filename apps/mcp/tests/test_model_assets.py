"""Model download and offline cache behavior."""
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import threading
import time

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
    assert sorted(p.name for p in assets.cache_directory().iterdir()) == ['.install.lock']


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
    assert sorted(p.name for p in assets.cache_directory().iterdir()) == sorted(['.install.lock', *catalog])


def test_installer_waits_before_verifying_cache(catalog, tmp_path, monkeypatch):
    destination = assets.cache_directory()
    destination.mkdir(parents=True)
    for name, data in catalog.items():
        (destination / name).write_bytes(data)
    entered = threading.Event()
    started = threading.Event()
    matches = assets._matches
    def checked(*args):
        entered.set()
        return matches(*args)
    monkeypatch.setattr(assets, '_matches', checked)
    monkeypatch.setattr(assets, 'build_opener', lambda *a: pytest.fail('Already cached'))
    def install():
        started.set()
        return assets.install_models()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with assets._cache_lock(destination):
            future = pool.submit(install)
            assert started.wait(5)
            assert not entered.wait(0.15)
            assert not future.done()
            # Ordinary runtime reads neither acquire nor modify the lock.
            assert assets.verified_bytes('one.onnx') == catalog['one.onnx']
        assert future.result(timeout=5) == destination
    assert entered.is_set()


def test_cache_lock_timeout_releases_descriptor(catalog):
    destination = assets.cache_directory()
    destination.mkdir(parents=True)
    with assets._cache_lock(destination):
        with pytest.raises(TimeoutError, match='Another model installation'):
            with assets._cache_lock(destination, timeout=0):
                pytest.fail('Conflicting lock was granted')
    with assets._cache_lock(destination, timeout=0):
        pass


def test_cache_lock_open_permission_error_is_not_hidden(catalog, monkeypatch):
    destination = assets.cache_directory()
    destination.mkdir(parents=True)
    def denied(*args, **kwargs):
        raise PermissionError('Access denied')
    monkeypatch.setattr(Path, 'open', denied)
    with pytest.raises(PermissionError, match='Access denied'):
        with assets._cache_lock(destination):
            pytest.fail('Permission error was ignored')


@pytest.mark.parametrize('release', ['normal', 'exception', 'terminate'])
def test_process_install_waits_and_recovers(catalog, tmp_path, release):
    destination = assets.cache_directory()
    destination.mkdir(parents=True)
    source = tmp_path / 'source'
    source.mkdir()
    for name, data in catalog.items():
        (source / name).write_bytes(data)
    marker = tmp_path / 'entered-verification'
    holder_ready = tmp_path / 'holder-ready'
    waiter_ready = tmp_path / 'waiter-ready'
    def wait_ready(path, child):
        deadline = time.monotonic() + 10
        while not path.exists():
            if child.poll() is not None:
                pytest.fail(f'Child exited before readiness: {child.communicate(timeout=1)}')
            if time.monotonic() >= deadline:
                pytest.fail(f'Child did not become ready: {path.name}')
            time.sleep(0.01)
    # Use an independent interpreter: threading locks cannot satisfy this test.
    holder = subprocess.Popen([sys.executable, '-u', '-c', '''
import sys
from pathlib import Path
from opencv_bv_tools import model_assets as a
try:
    with a._cache_lock(Path(sys.argv[1])):
        Path(sys.argv[3]).touch()
        sys.stdin.readline()
        if sys.argv[2] == 'exception':
            raise RuntimeError('interrupted installation')
except RuntimeError:
    pass
''', str(destination), release, str(holder_ready)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    waiter = None
    try:
        wait_ready(holder_ready, holder)
        waiter = subprocess.Popen([sys.executable, '-u', '-c', '''
import json,sys
from pathlib import Path
from opencv_bv_tools import model_assets as a
a.MODELS = json.loads(sys.argv[1])
original = a._matches
def checked(*args):
    Path(sys.argv[3]).touch()
    return original(*args)
a._matches = checked
Path(sys.argv[4]).touch()
a.install_models(Path(sys.argv[2]))
''', json.dumps(assets.MODELS), str(source), str(marker), str(waiter_ready)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wait_ready(waiter_ready, waiter)
        with pytest.raises(subprocess.TimeoutExpired):
            waiter.wait(timeout=0.15)
        assert not marker.exists()
        if release == 'terminate':
            holder.terminate()
        else:
            holder.stdin.write(b'continue\n')
            holder.stdin.flush()
        holder.wait(timeout=5)
        stdout, stderr = waiter.communicate(timeout=10)
        assert waiter.returncode == 0, (stdout, stderr)
        assert marker.exists()
        assert all(assets.verified_bytes(name) == data for name, data in catalog.items())
        assert sorted(p.name for p in destination.iterdir()) == sorted(['.install.lock', *catalog])
    finally:
        for child in (waiter, holder):
            if child is None:
                continue
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)


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
    assert sorted(p.name for p in assets.cache_directory().iterdir()) == ['.install.lock']


def test_search_is_separate_optional_streamed_cache(tmp_path, monkeypatch):
    data = b'search model' * 1000
    monkeypatch.setattr(assets, 'SEARCH_MODELS', {
        'model/tokenizer.json': (len(data), hashlib.sha256(data).hexdigest(), 'search-id')})
    monkeypatch.setenv('OPENCV_BV_SEARCH_MODEL_DIR', str(tmp_path / 'search'))
    monkeypatch.setenv('OPENCV_BV_MODEL_DIR', str(tmp_path / 'core'))
    source = tmp_path / 'source'
    (source / 'model').mkdir(parents=True)
    (source / 'model/tokenizer.json').write_bytes(data)
    result = assets.install_models(source, search=True)
    assert (result / 'model/tokenizer.json').read_bytes() == data
    assert not assets.cache_directory().exists()
    monkeypatch.setattr(assets, 'build_opener', lambda *a: pytest.fail('Already verified'))
    assert assets.install_models(search=True) == result


@pytest.mark.parametrize('payload', [b'', b'short', b'x' * len(b'first test model')])
def test_invalid_bytes_preserve_previous_models_and_allow_retry(catalog, monkeypatch, tmp_path, payload):
    class Opener:
        def open(self, url, timeout):
            return io.BytesIO(payload)
    monkeypatch.setattr(assets, 'build_opener', lambda *a: Opener())
    destination = assets.cache_directory()
    destination.mkdir(parents=True)
    (destination / 'two.onnx').write_bytes(catalog['two.onnx'])
    with pytest.raises(ValueError, match='SHA-256'):
        assets.install_models()
    assert sorted(p.name for p in destination.iterdir()) == ['.install.lock', 'two.onnx']
    assert (destination / 'two.onnx').read_bytes() == catalog['two.onnx']
    source = tmp_path / 'retry'
    source.mkdir()
    for name, data in catalog.items():
        (source / name).write_bytes(data)
    assets.install_models(source)
    assert all(assets.verified_bytes(name) == data for name, data in catalog.items())
