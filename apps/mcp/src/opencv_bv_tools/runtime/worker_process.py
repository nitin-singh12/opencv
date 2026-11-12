# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Shared bounded subprocess transport and limits for local image readers."""

from dataclasses import dataclass
from typing import ClassVar
import json
import math
import os
import selectors
import subprocess
import sys
import time

from .process_support import PipePoller, start_worker, stop_worker


class ReaderOptionsError(ValueError):
    """Invalid caller options; image/decoder failures use the result status."""


@dataclass(frozen=True)
class ReaderLimits:
    _error_type: ClassVar[type[ValueError]] = ReaderOptionsError
    max_input_bytes: int = 32 * 1024 * 1024
    max_pixels: int = 8_000_000
    max_dimension: int = 16_384
    max_symbols: int = 32
    max_payload_bytes: int = 8192
    max_total_payload_bytes: int = 262_144
    max_text_chars: int = 32_768
    timeout_seconds: float = 3.0

    def validate(self):
        ceilings = dict(max_input_bytes=32 * 1024 * 1024, max_pixels=20_000_000,
                        max_dimension=32768, max_symbols=128, max_payload_bytes=16384,
                        max_total_payload_bytes=262144, max_text_chars=32768)
        for key, ceiling in ceilings.items():
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise self._error_type(f"{key} must be an integer from 1 to {ceiling}.")
        value = self.timeout_seconds
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 30:
            raise self._error_type("timeout_seconds must be finite, greater than 0 and at most 30.")


MAX_WIRE_BYTES = 1024 * 1024
MAX_DIAGNOSTIC_BYTES = 4096
POLL_SECONDS = 0.02
TEARDOWN_SECONDS = 0.25  # terminate and kill each receive this allowance


def _failure(reason, elapsed=0.0):
    return dict(search_status="failed", stop_reason=reason, symbols=[], undecoded=[],
                image_size=None, coordinate_space="exif_normalized_image",
                elapsed_seconds=elapsed)


def _cleanup(process):
    """Always reap. A stuck OS process can exceed this scheduling/teardown budget."""
    stop_worker(process, grace=TEARDOWN_SECONDS)


def _run_worker(data, options, *, timeout, cancelled=None, command,
                max_output_bytes=MAX_WIRE_BYTES, response_validator=None,
                failure_factory=_failure, status_field="search_status", extra_env=None):
    """Bound both pipes and input transfer; test seam is private, never user input."""
    start = time.monotonic()
    deadline = start + timeout
    process = None
    output = bytearray()
    diagnostic_count = 0
    response = None
    try:
        if cancelled is not None and cancelled():
            return failure_factory("cancelled")
        if sys.platform not in {"linux", "darwin", "win32"}:
            return failure_factory("unsupported_platform")
        # No shell, pickle, image filename, network operation or payload in argv.
        process = start_worker(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1", **(extra_env or {})},
        )
        header = json.dumps(options, separators=(",", ":")).encode() + b"\n"
        chunks = [memoryview(header), memoryview(data)]
        part, offset = 0, 0
        with PipePoller() as poller:
            for stream, event in [(process.stdin, selectors.EVENT_WRITE),
                                  (process.stdout, selectors.EVENT_READ),
                                  (process.stderr, selectors.EVENT_READ)]:
                poller.register(stream, event)
            while poller.get_map() or process.poll() is None:
                if cancelled is not None and cancelled():
                    response = failure_factory("cancelled")
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    response = failure_factory("deadline_exceeded")
                    response[status_field] = "budget_exhausted"
                    break
                events = poller.select(min(POLL_SECONDS, remaining))
                if not poller.get_map():
                    time.sleep(min(POLL_SECONDS, remaining))
                for key, event in events:
                    stream = key.fileobj
                    if event == selectors.EVENT_WRITE:
                        try:
                            sent = poller.write(stream.fileno(), chunks[part][offset:offset + 65536])
                        except BlockingIOError:
                            # Readiness can change before a nonblocking write.
                            # Retry within the same deadline, preserving offset.
                            continue
                        except BrokenPipeError:
                            sent = 0
                            part = len(chunks)
                        else:
                            offset += sent
                            if offset == len(chunks[part]):
                                part += 1
                                offset = 0
                        if part == len(chunks):
                            poller.unregister(stream)
                            stream.close()
                    else:
                        try:
                            block = poller.read(stream.fileno(), 65536)
                        except BlockingIOError:
                            continue
                        if not block:
                            poller.unregister(stream)
                            stream.close()
                        elif stream is process.stdout:
                            output.extend(block)
                            if len(output) > max_output_bytes:
                                response = failure_factory("worker_output_limit")
                        else:
                            diagnostic_count += len(block)
                            if diagnostic_count > MAX_DIAGNOSTIC_BYTES:
                                response = failure_factory("worker_diagnostic_limit")
                if response is not None:
                    break
        if response is None:
            if process.returncode != 0:
                response = failure_factory("worker_failed")
            else:
                try:
                    response = json.loads(output)
                    (response_validator or _validate_response)(response, options["limits"])
                except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
                    response = failure_factory("invalid_worker_response")
    except OSError:
        response = failure_factory("worker_io_error" if process is not None else "worker_start_failed")
    finally:
        if process is not None:
            try:
                _cleanup(process)
            finally:
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close()
    # Parsing/validation can overrun while not yielding (bounded by wire cap).
    if time.monotonic() > deadline and response.get(status_field) != "budget_exhausted":
        response = failure_factory("deadline_exceeded")
        response[status_field] = "budget_exhausted"
    response["elapsed_seconds"] = time.monotonic() - start
    return response


def _validate_response(result, limits):
    """Reject malformed/oversized protocol results before exposing payload data."""
    import base64
    if not isinstance(result, dict) or result["search_status"] not in {
        "completed", "resource_limited", "failed"
    }:
        raise ValueError("Invalid worker status")
    if type(result["stop_reason"]) is not str or len(result["stop_reason"]) > 80:
        raise ValueError("Invalid reason")
    symbols, undecoded = result["symbols"], result["undecoded"]
    if not isinstance(symbols, list) or not isinstance(undecoded, list):
        raise ValueError("Invalid symbols")
    if max(len(symbols), len(undecoded)) > limits["max_symbols"]:
        raise ValueError("Too many symbols")
    size = result["image_size"]
    if size is not None and (not isinstance(size, list) or len(size) != 2 or
                            any(type(x) is not int or not 0 < x <= limits["max_dimension"] for x in size)
                            or size[0] * size[1] > limits["max_pixels"]):
        raise ValueError("Invalid dimensions")
    total, chars = 0, 0
    for symbol in symbols + undecoded:
        polygon = symbol["polygon"]
        if not isinstance(polygon, list) or len(polygon) != 4 or size is None:
            raise ValueError("Invalid polygon")
        for point in polygon:
            if not isinstance(point, list) or len(point) != 2 or any(type(x) is not int for x in point):
                raise ValueError("Invalid point")
            if not (0 <= point[0] <= size[0] and 0 <= point[1] <= size[1]):
                raise ValueError("Out-of-image point")
    for symbol in symbols:
        payload = base64.b64decode(symbol["payload_base64"], validate=True)
        total += len(payload)
        value = symbol["text"]
        if value is not None and type(value) is not str:
            raise ValueError("Invalid text")
        chars += len(value or "")
        if len(payload) > limits["max_payload_bytes"]:
            raise ValueError("Payload too long")
    if total > limits["max_total_payload_bytes"] or chars > limits["max_text_chars"]:
        raise ValueError("Aggregate payload too long")
