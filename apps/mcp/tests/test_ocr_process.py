"""OCR framing, reuse, idle retirement, cancellation and parent isolation."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from PIL import Image
import psutil
import pytest

from opencv_bv_tools.runtime.ocr_process import OCRProcess
from opencv_bv_tools.runtime.ocr_process import OCRWorkerError
from opencv_bv_tools.runtime.ocr_process import parse_header
from opencv_bv_tools.runtime.ocr_process import validate_response
from opencv_bv_tools.runtime.ocr_process import MAX_BYTES
from opencv_bv_tools.runtime.context import ServerConfig
from opencv_bv_tools.runtime.context import CallSlots


@pytest.fixture
def fake(tmp_path):
    path = tmp_path / "worker.py"
    image = io.BytesIO()
    Image.new("RGBA", (4, 4), (255, 255, 255, 255)).save(image, format="PNG")
    path.write_text("""
import hashlib,json,os,sys,time
png = %r
while True:
    line = sys.stdin.buffer.readline()
    if not line: break
    options = json.loads(line)
    data = sys.stdin.buffer.read(options['input_bytes'])
    if data == b'slow': time.sleep(0.3)
    if data == b'stderr': sys.stderr.write('x' * 131072);sys.stderr.flush()
    if data == b'crash': sys.exit(1)
    output = png if options['redact'] else b''
    result = dict(status='ok', png_bytes=len(output), png_sha256=hashlib.sha256(output).hexdigest() if output else None,
                  boxes=[], addresses=['private@example.com'], image_size=[4,4], seconds=0.01, pid=os.getpid(), request_id=options['request_id'])
    if data == b'oversized': result['png_bytes'] = 33554433
    if data == b'boolean': result['png_bytes'] = False
    if data == b'badbox': result['boxes'] = [[0,0,5,5]]
    if data == b'stale': result['request_id'] -= 1
    encoded = json.dumps(result).encode() + b'\\n'
    if data == b'duplicate': encoded = encoded[:-2] + b',"status":"ok"}\\n'
    if data == b'header-eof': sys.stdout.buffer.write(encoded[:-1]);sys.stdout.buffer.flush();sys.exit(0)
    if data == b'body-eof': output = output[:-1]
    if data == b'trailing': output += b'x'
    sys.stdout.buffer.write(encoded + output);sys.stdout.buffer.flush()
    if data == b'body-eof': sys.exit(0)
""" % image.getvalue())
    config = ServerConfig(allow_dirs=(tmp_path,), ocr_idle_seconds=0.15, ocr_timeout=3.0)
    worker = OCRProcess(config, command=[sys.executable, str(path)])
    try:
        yield worker
    finally:
        worker.close()


def request(worker, data=b"valid", **kwargs):
    try:
        return worker.request(data, mode="accurate", redact=True, check_cancelled=lambda: None, **kwargs)
    except OCRWorkerError as error:
        # Synthetic fixtures only: retain the validator reason hidden from users.
        if error.__context__ is not None:
            print(type(error.__context__).__name__, str(error.__context__), flush=True)
        raise


def wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


def test_reuses_worker_then_reaps_idle_and_restarts(fake):
    first, _ = request(fake)
    second, _ = request(fake)
    assert first["pid"] == second["pid"]
    wait_until(lambda: fake.pid is None)
    assert not psutil.pid_exists(first["pid"])
    third, _ = request(fake)
    assert third["pid"] != first["pid"]


def test_stale_idle_callback_cannot_kill_active_request(fake):
    first, _ = request(fake)
    old_generation = fake._generation
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(request, fake, b"slow")
        wait_until(lambda: fake._idle_deadline is None)
        expiry = pool.submit(fake._expire, old_generation)
        result, _ = future.result(timeout=3)
        expiry.result(timeout=3)
    assert result["pid"] == first["pid"] and fake.pid == first["pid"]


@pytest.mark.parametrize("data", [b"crash", b"oversized", b"boolean", b"badbox", b"stale", b"duplicate",
                                   b"header-eof", b"body-eof", b"trailing"])
def test_invalid_response_discards_worker_and_next_call_recovers(fake, data):
    with pytest.raises(OCRWorkerError):
        request(fake, data)
    assert fake.pid is None
    result, output = request(fake)
    assert result["status"] == "ok" and output.startswith(b"\x89PNG")


def test_stderr_cannot_block_worker_or_protocol(fake):
    assert request(fake, b"stderr")[0]["status"] == "ok"


def test_cancellation_reaps_active_worker_and_releases_lock(fake):
    stopped = threading.Event()
    def check():
        if stopped.is_set(): raise RuntimeError("cancelled")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fake.request, b"slow", mode="accurate", redact=True, check_cancelled=check)
        wait_until(lambda: fake.pid is not None)
        pid = fake.pid
        stopped.set()
        with pytest.raises(RuntimeError, match="cancelled"):
            future.result(timeout=3)
    assert fake.pid is None and not psutil.pid_exists(pid)
    assert request(fake)[0]["status"] == "ok"


def test_shutdown_prevents_waiting_request_from_restarting(fake):
    with ThreadPoolExecutor(max_workers=3) as pool:
        active = pool.submit(request, fake, b"slow")
        wait_until(lambda: fake.pid is not None)
        queued = pool.submit(request, fake)
        closing = pool.submit(fake.close)
        for future in (active, queued):
            with pytest.raises(OCRWorkerError, match="closed"):
                future.result(timeout=3)
        closing.result(timeout=3)
    assert fake.pid is None


def test_zero_idle_reaps_after_each_call(fake):
    fake.config = replace(fake.config, ocr_idle_seconds=0)
    result, _ = request(fake)
    assert fake.pid is None and not psutil.pid_exists(result["pid"])


def test_timeout_discards_worker(fake):
    fake.config = replace(fake.config, ocr_timeout=0.05)
    with pytest.raises(OCRWorkerError, match="timeout"):
        request(fake, b"slow")
    assert fake.pid is None


def test_cancelled_queued_call_never_enters():
    stopped = threading.Event()
    def check():
        if stopped.is_set(): raise RuntimeError("cancelled")
    slots = CallSlots(check, 1)
    def queued():
        with slots:
            pytest.fail("cancelled queued call started processing")
    with slots:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(queued)
            stopped.set()
            with pytest.raises(RuntimeError, match="cancelled"):
                future.result(timeout=3)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"n":NaN}', b'{"n":Infinity}', b'\xff'])
def test_strict_json(raw):
    with pytest.raises((ValueError, UnicodeError)):
        parse_header(raw)
