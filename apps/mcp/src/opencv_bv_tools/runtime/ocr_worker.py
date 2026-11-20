# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Private resident OCR worker; never receives filenames or exposes raw logs."""
import hashlib
import io
import json
import os
import signal
import sys
import time

from opencv_bv_tools.runtime.ocr_process import MAX_BYTES
from opencv_bv_tools.runtime.ocr_process import MAX_HEADER
from opencv_bv_tools.runtime.ocr_process import parse_header
from opencv_bv_tools.runtime.email_modes import MODES


def _error(reason):
    return {"status": "error", "reason": reason, "png_bytes": 0}, b""


def _engine(implementation, options, state):
    if "engine" not in state:
        implementation.check_models()
        implementation.cv2.setNumThreads(options["threads"])
        state["engine"] = implementation.Ocr(device="cpu", threads=options["threads"], size_guard=True)
        state["threads"] = options["threads"]
    if state["threads"] != options["threads"]:
        raise ValueError("Thread configuration changed")
    engine = state["engine"]
    for key, value in MODES[options["mode"]].items():
        setattr(engine, key, value)
    engine.pdf_forms = bool(options.get("pdf"))
    return engine


def process(data, options, state, progress=None):
    if options.get("pdf"):
        from opencv_bv_tools.runtime.pdf_worker import process_pdf
        return process_pdf(data, options, state, _engine, progress)
    try:
        from opencv_bv_tools.runtime import image_redact as implementation
    except ImportError:
        return _error("missing_ocr_dependency")
    try:
        image = implementation.open_image(io.BytesIO(data), options["max_pixels"])
    except (implementation.ImageTooLarge,):
        return _error("image_too_large")
    except implementation.CMYKCompatibilityRequired:
        return _error("cmyk_dependency_required")
    except (OSError, ValueError, SyntaxError):
        return _error("invalid_image")
    try:
        engine = _engine(implementation, options, state)
    except RuntimeError:
        return _error("model_verification_failed")
    except Exception:
        return _error("ocr_initialization_failed")
    try:
        started = time.perf_counter()
        detected = implementation.detect_image(image, engine)
        seconds = time.perf_counter() - started
        png = implementation.png_bytes(implementation.redact(image, detected.boxes)) if options["redact"] else b""
        if len(png) > MAX_BYTES:
            return _error("output_too_large")
        return dict(status="ok", png_bytes=len(png), png_sha256=hashlib.sha256(png).hexdigest() if png else None,
                    boxes=detected.boxes, addresses=sorted(set(detected.addresses)), image_size=[image.shape[1], image.shape[0]],
                    seconds=seconds, pid=os.getpid()), png
    except Exception:
        return _error("ocr_processing_failed")


def main():
    # A native library may write directly to fd 1. Preserve a private protocol
    # handle and redirect both Python and native stdout before heavy imports.
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    def terminate(signum, frame):
        # Let PDF finally blocks close/wait for helpers when Python can handle
        # the signal. The parent kills the whole group if native code is stuck.
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    def send(header, chunks=()):
        encoded = json.dumps(header, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
        if len(encoded) > MAX_HEADER:
            raise ValueError("Response header too large")
        for chunk in (encoded + b"\n", *chunks):
            view = memoryview(chunk)
            while view:
                view = view[wire.write(view):]
    state = {}
    while True:
        header = sys.stdin.buffer.readline(4097)
        if not header:
            return
        try:
            if len(header) > 4096 or not header.endswith(b"\n"):
                return
            options = parse_header(header)
            fields = {"input_bytes", "mode", "redact", "max_pixels", "threads", "request_id"}
            pdf = options.get("pdf", False)
            if pdf is True:
                fields |= {"pdf", "max_pages", "engine"}
                if (type(options.get("max_pages")) is not int or not 1 <= options["max_pages"] <= 100
                        or options.get("engine") not in {"picture", "mupdf"} or options.get("redact") is not True):
                    return
            if set(options) != fields:
                return
            if (type(options["input_bytes"]) is not int or not 0 < options["input_bytes"] <= MAX_BYTES
                    or options["mode"] not in MODES or type(options["redact"]) is not bool
                    or type(options["max_pixels"]) is not int or not 0 < options["max_pixels"] <= 40_000_000
                    or type(options["threads"]) is not int or not 1 <= options["threads"] <= 64
                    or type(options["request_id"]) is not int or options["request_id"] <= 0):
                return
            data = sys.stdin.buffer.read(options["input_bytes"])
            if len(data) != options["input_bytes"]:
                return
            def progress(index, count):
                send(dict(status="progress", request_id=options["request_id"], index=index, pages=count))
            result, payload = process(data, options, state, progress)
            result["request_id"] = options["request_id"]
            encoded = json.dumps(result, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
            if len(encoded) > MAX_HEADER:
                if pdf:
                    result, payload = {"status": "error", "reason": "pdf_output", "pdf_bytes": 0}, []
                else:
                    result, payload = _error("output_too_large")
                result["request_id"] = options["request_id"]
                encoded = json.dumps(result).encode()
            send(result, payload if pdf else (payload,))
            del data, result, payload, encoded
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            return


if __name__ == "__main__":
    main()
