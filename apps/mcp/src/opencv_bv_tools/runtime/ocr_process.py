# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Serialized OCR subprocess with bounded binary frames and idle retirement.

Each frame is one UTF-8 JSON line followed by exactly the declared number of
binary bytes. There are no paths, pickle, shell commands or network operations
in this protocol. A failed exchange discards the worker, without retrying it.
"""
import hashlib
import json
import math
import os
import selectors

import subprocess
import sys
import tempfile
import threading
import time

from .image_io import header as image_header

from .process_support import PipePoller, start_worker, stop_worker

MAX_BYTES = 32 * 1024 * 1024
MAX_HEADER = 1024 * 1024
MAX_PDF_BYTES = 128 * 1024 * 1024
PDF_ERRORS = {
    "pdf_password": "The PDF is protected by a password; remove the password first.",
    "pdf_pages": "The PDF is empty or exceeds the configured page limit.",
    "pdf_pixels": "A PDF page exceeds the configured pixel limit.",
    "pdf_output": "The redacted PDF exceeds the 128 MiB output limit.",
    "pdf_invalid": "The PDF cannot be read; check that the file is complete.",
    "pdf_memory": "PDF rendering exceeded its memory limit.",
    "pdf_timeout": "PDF rendering exceeded its time limit.",
    "pdf_failed": "PDF processing failed; no output was published.",
    "missing_pdf_dependency": "The selected PDF engine dependencies are not installed.",
    "ocr_initialization_failed": "OCR initialization failed.",
    "model_verification_failed": "OCR model verification failed.",
}


class OCRWorkerError(RuntimeError):
    pass


def validate_pdf_response(result, payload, *, pid, request_id, max_pages):
    if not isinstance(result, dict) or type(result.get("request_id")) is not int or result["request_id"] != request_id:
        raise ValueError("Invalid PDF request identity")
    if type(result.get("pdf_bytes")) is not int or not 0 <= result["pdf_bytes"] <= MAX_PDF_BYTES:
        raise ValueError("Invalid PDF length")
    if result.get("status") == "error":
        if set(result) != {"status", "reason", "pdf_bytes", "request_id"} or result["pdf_bytes"] != 0:
            raise ValueError("Invalid PDF error frame")
        if type(result.get("reason")) is not str or result["reason"] not in PDF_ERRORS:
            raise ValueError("Invalid PDF error code")
        raise OCRWorkerError(PDF_ERRORS[result["reason"]])
    if (result.get("status") != "ok" or set(result) != {
            "status", "pdf_bytes", "pdf_sha256", "pages", "boxes_per_page", "addresses",
            "seconds", "pid", "request_id", "engine", "fallback"}):
        raise ValueError("Invalid PDF response")
    if type(result["pid"]) is not int or result["pid"] != pid:
        raise ValueError("Invalid PDF worker identity")
    if type(result["pages"]) is not int or not 1 <= result["pages"] <= max_pages:
        raise ValueError("Invalid page count")
    counts = result["boxes_per_page"]
    if (not isinstance(counts, list) or len(counts) != result["pages"]
            or any(type(n) is not int or not 0 <= n <= 100_000 for n in counts)):
        raise ValueError("Invalid box counts")
    addresses = result["addresses"]
    if (not isinstance(addresses, list) or len(addresses) > 4096
            or any(type(s) is not str or len(s) > 8192 for s in addresses)):
        raise ValueError("Invalid PDF addresses")
    if (type(result["seconds"]) not in (float, int) or not math.isfinite(result["seconds"])
            or result["seconds"] < 0 or result["engine"] not in {"picture", "mupdf"}
            or type(result["fallback"]) is not bool or (result["fallback"] and result["engine"] != "picture")):
        raise ValueError("Invalid PDF metadata")
    digest, size = hashlib.sha256(), 0
    prefix = payload.read(8)
    payload.seek(0)
    for chunk in iter(lambda: payload.read(65536), b""):
        digest.update(chunk)
        size += len(chunk)
    if not prefix.startswith(b"%PDF-") or size != result["pdf_bytes"] or digest.hexdigest() != result["pdf_sha256"]:
        raise ValueError("Invalid PDF payload")
    payload.seek(max(0, size - 32))
    if not payload.read().rstrip().endswith(b"%%EOF"):
        raise ValueError("Incomplete PDF payload")
    payload.seek(0)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key")
        result[key] = value
    return result


def parse_header(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON")))


def validate_response(result, payload, *, redact, max_pixels, pid, request_id):
    if not isinstance(result, dict) or result.get("status") not in {"ok", "error"}:
        raise ValueError("Invalid OCR status")
    if type(result.get("request_id")) is not int or result["request_id"] != request_id:
        raise ValueError("Stale OCR response")
    if type(result.get("png_bytes")) is not int or not 0 <= result["png_bytes"] <= MAX_BYTES:
        raise ValueError("Invalid PNG size")
    if len(payload) != result["png_bytes"]:
        raise ValueError("Truncated PNG")
    if result["status"] == "error":
        if set(result) != {"status", "reason", "png_bytes", "request_id"} or payload or result["png_bytes"] != 0:
            raise ValueError("Invalid error frame")
        if type(result["reason"]) is not str or result["reason"] not in {
            "invalid_image", "image_too_large", "model_verification_failed", "ocr_initialization_failed",
            "ocr_processing_failed", "output_too_large", "invalid_request", "missing_ocr_dependency",
            "cmyk_dependency_required"}:
            raise ValueError("Invalid error reason")
        if result["reason"] == "cmyk_dependency_required":
            raise OCRWorkerError("CMYK JPEG requires the optional cmyk extra (Pillow). See docs/lightweight-branch.md for installation.")
        raise OCRWorkerError(result["reason"].replace("_", " "))
    if set(result) != {"status", "png_bytes", "png_sha256", "boxes", "addresses", "image_size", "seconds", "pid", "request_id"}:
        raise ValueError("Invalid OCR fields")
    if type(result["pid"]) is not int or result["pid"] != pid:
        raise ValueError("Invalid worker identity")
    size = result["image_size"]
    if (not isinstance(size, list) or len(size) != 2 or any(type(n) is not int or n <= 0 for n in size)
            or size[0] * size[1] > max_pixels):
        raise ValueError("Invalid image dimensions")
    seconds = result["seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid inference timing")
    boxes, addresses = result["boxes"], result["addresses"]
    if not isinstance(boxes, list) or len(boxes) > 4096 or not isinstance(addresses, list) or len(addresses) > 4096:
        raise ValueError("Too many OCR results")
    if any(type(s) is not str or len(s) > 8192 for s in addresses):
        raise ValueError("Invalid address text")
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4 or any(type(n) is not int for n in box)
                or box[0] < 0 or box[1] < 0 or box[2] <= 0 or box[3] <= 0
                or box[0] + box[2] > size[0] or box[1] + box[3] > size[1]):
            raise ValueError("Invalid OCR box")
    if redact:
        if not payload.startswith(b"\x89PNG\r\n\x1a\n") or hashlib.sha256(payload).hexdigest() != result["png_sha256"]:
            raise ValueError("Invalid redacted PNG")
        image = image_header(payload)
        if [image.width, image.height] != size or image.animated:
            raise ValueError("Invalid redacted image dimensions")
    elif payload or result["png_sha256"] is not None:
        raise ValueError("Find-only result contains image data")


class OCRProcess:
    def __init__(self, config, *, command=None):
        self.config = config
        self.command = command or [sys.executable, "-m", "opencv_bv_tools.runtime.ocr_worker"]
        self._lock = threading.Lock()
        self._process = None
        self._timer = None
        self._generation = 0
        self._idle_deadline = None
        self._closed = threading.Event()
        self._request_id = 0

    @property
    def pid(self):
        return self._process.pid if self._process is not None else None

    def _stop_locked(self):
        process = self._process
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._idle_deadline = None
        self._generation += 1
        if process is not None:
            stop_worker(process, group=True, grace=0.5)
            process.stdin.close()
            process.stdout.close()
            self._process = None

    def close(self):
        self._closed.set()
        with self._lock:
            self._stop_locked()

    def _expire(self, generation):
        with self._lock:
            if (generation == self._generation and self._idle_deadline is not None
                    and time.monotonic() >= self._idle_deadline):
                self._stop_locked()

    def _schedule_idle_locked(self):
        self._generation += 1
        if self.config.ocr_idle_seconds == 0:
            self._stop_locked()
            return
        self._idle_deadline = time.monotonic() + self.config.ocr_idle_seconds
        self._timer = threading.Timer(self.config.ocr_idle_seconds, self._expire, (self._generation,))
        self._timer.daemon = True
        self._timer.start()

    def request(self, data, *, mode, redact, check_cancelled, pdf=False, progress=None):
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_BYTES:
            raise ValueError("OCR input must be nonempty bytes, at most 32 MiB.")
        while True:
            check_cancelled()
            if self._lock.acquire(timeout=0.05):
                break
        payload = None
        try:
            check_cancelled()
            if self._closed.is_set():
                raise OCRWorkerError("OCR service is closed")
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._generation += 1
            self._idle_deadline = None
            deadline = time.monotonic() + (self.config.pdf_timeout if pdf else self.config.ocr_timeout)
            if self._process is None or self._process.poll() is not None:
                self._stop_locked()
                env = {**os.environ, **{name: str(self.config.threads) for name in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
                # Discard diagnostics continuously, including while idle. Native
                # log output can never block a pipe or disclose OCR text.
                self._process = start_worker(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                  stderr=subprocess.DEVNULL, bufsize=0, env=env, start_new_session=True)
            self._request_id += 1
            options = dict(input_bytes=len(data), mode=mode, redact=redact, request_id=self._request_id,
                           max_pixels=int(self.config.max_megapixels * 1_000_000), threads=self.config.threads)
            if pdf:
                options.update(pdf=True, max_pages=self.config.max_pages, engine=self.config.pdf_engine)
            result, payload = self._exchange(data, options, deadline, check_cancelled, progress=progress)
            if pdf:
                validate_pdf_response(result, payload, pid=self._process.pid, request_id=self._request_id,
                                      max_pages=self.config.max_pages)
            else:
                validate_response(result, payload, redact=redact, max_pixels=options["max_pixels"],
                                  pid=self._process.pid, request_id=self._request_id)
            check_cancelled()
            if time.monotonic() > deadline:
                raise OCRWorkerError("OCR request exceeded the configured timeout")
            self._schedule_idle_locked()
            return result, payload
        except BaseException as exc:
            if pdf and payload is not None:
                payload.close()
            self._stop_locked()
            if isinstance(exc, (OSError, ValueError, KeyError, TypeError, RecursionError)):
                raise OCRWorkerError("OCR worker returned an invalid or incomplete response") from None
            raise
        finally:
            self._lock.release()

    def _exchange(self, data, options, deadline, check_cancelled, *, progress=None):
        process = self._process
        header = json.dumps(options, separators=(",", ":")).encode() + b"\n"
        chunks = [memoryview(header), memoryview(data)]
        part = offset = received = updates = 0
        pending = bytearray()
        result = expected = None
        pdf = options.get("pdf", False)
        field = "pdf_bytes" if pdf else "png_bytes"
        limit = MAX_PDF_BYTES if pdf else MAX_BYTES
        output = tempfile.TemporaryFile() if pdf else None
        complete = False
        try:
            with PipePoller() as poller:
                for stream, events in ((process.stdin, selectors.EVENT_WRITE), (process.stdout, selectors.EVENT_READ)):
                    poller.register(stream, events)
                while True:
                    check_cancelled()
                    if self._closed.is_set():
                        raise OCRWorkerError("OCR service is closed")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise OCRWorkerError("OCR request exceeded the configured timeout")
                    for key, event in poller.select(min(0.02, remaining)):
                        if event == selectors.EVENT_WRITE:
                            try:
                                sent = poller.write(key.fd, chunks[part][offset:offset + 65536])
                            except BlockingIOError:
                                continue
                            offset += sent
                            if offset == len(chunks[part]):
                                part += 1
                                offset = 0
                            if part == len(chunks):
                                poller.unregister(process.stdin)
                            continue
                        try:
                            block = poller.read(key.fd, 65536)
                        except BlockingIOError:
                            continue
                        if not block:
                            raise OCRWorkerError("OCR worker stopped before completing its response")
                        pending.extend(block)
                        while result is None:
                            newline = pending.find(b"\n")
                            if newline > MAX_HEADER or (newline < 0 and len(pending) > MAX_HEADER):
                                raise ValueError("OCR header too large")
                            if newline < 0:
                                break
                            message = parse_header(bytes(pending[:newline]))
                            del pending[:newline + 1]
                            if not isinstance(message, dict):
                                raise ValueError("Invalid response header")
                            if pdf and message.get("status") == "progress":
                                updates += 1
                                if (set(message) != {"status", "request_id", "index", "pages"}
                                        or type(message["request_id"]) is not int
                                        or message["request_id"] != options["request_id"]
                                        or type(message["pages"]) is not int
                                        or not 1 <= message["pages"] <= options["max_pages"]
                                        or type(message["index"]) is not int
                                        or not 0 <= message["index"] < message["pages"]
                                        or updates > 2 * options["max_pages"] or part != len(chunks)):
                                    raise ValueError("Invalid PDF progress")
                                # MuPDF fallback can restart at page zero. Progress
                                # never changes the original request deadline.
                                if progress is not None:
                                    progress(message["index"], message["pages"])
                                continue
                            result = message
                            expected = result[field]
                            if type(expected) is not int or not 0 <= expected <= limit:
                                raise ValueError("Invalid output length")
                        if expected is None:
                            continue
                        if pdf:
                            received += len(pending)
                            if received > expected:
                                raise ValueError("Unexpected trailing response bytes")
                            output.write(pending)
                            pending.clear()
                        else:
                            received = len(pending)
                        if received >= expected:
                            if received != expected or part != len(chunks):
                                raise ValueError("Unexpected trailing or early response bytes")
                            if pdf:
                                output.seek(0)
                            complete = True
                            return result, output if pdf else bytes(pending)
        finally:
            if output is not None and not complete:
                output.close()
