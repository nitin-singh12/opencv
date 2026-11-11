"""Strictly lossless, bounded PNG recompression in a disposable Linux worker."""

import argparse
import base64
import hashlib
import json
import math
import os
import stat
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from opencv_bv_tools.image_conversion import ConversionError
from opencv_bv_tools.runtime.file_output import _check_destination, _publish_bytes
from opencv_bv_tools.runtime.worker_process import _run_worker as _supervise


class PNGCompressionError(ValueError):
    """Invalid caller options; input/worker failures are result statuses."""


@dataclass(frozen=True)
class PNGLimits:
    max_input_bytes: int = 32 * 1024 * 1024
    max_pixels: int = 8_000_000
    max_dimension: int = 16384
    max_chunks: int = 4096
    max_metadata_bytes: int = 1024 * 1024
    timeout_seconds: float = 5.0

    def validate(self):
        for key, ceiling in dict(
            max_input_bytes=32 * 1024 * 1024,
            max_pixels=20_000_000,
            max_dimension=32768,
            max_chunks=4096,
            max_metadata_bytes=1024 * 1024,
        ).items():
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise PNGCompressionError(
                    f"{key} must be an integer from 1 to {ceiling}."
                )
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 30
        ):
            raise PNGCompressionError(
                "timeout_seconds must be finite, greater than 0 and at most 30."
            )


def _failure(reason):
    return dict(
        status="failed",
        stop_reason=reason,
        data=None,
        input_bytes=None,
        output_bytes=None,
        bytes_saved=0,
        elapsed_seconds=0.0,
    )


def _validate_response(result, limits):
    if not isinstance(result, dict) or result["status"] not in {
        "optimized",
        "unchanged",
        "failed",
        "resource_limited",
    }:
        raise ValueError("Invalid status")
    if type(result["stop_reason"]) is not str or len(result["stop_reason"]) > 80:
        raise ValueError("Invalid stop reason")
    if result["status"] not in {"optimized", "unchanged"}:
        if result.get("data") is not None or result.get("data_base64") is not None:
            raise ValueError("Failed result contains output")
        return
    source_size, size = result["input_bytes"], result["output_bytes"]
    if (
        type(source_size) is not int
        or type(size) is not int
        or not 1 <= size <= source_size <= limits["max_input_bytes"]
    ):
        raise ValueError("Invalid output size")
    if type(result["data_base64"]) is not str or len(result["data_base64"]) > 4 * (
        (size + 2) // 3
    ):
        raise ValueError("Invalid output encoding")
    output = base64.b64decode(result.pop("data_base64"), validate=True)
    if len(output) != size or not output.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Invalid PNG payload")
    if (
        result["bytes_saved"] != source_size - size
        or type(result["bytes_saved"]) is not int
    ):
        raise ValueError("Invalid savings")
    if (result["status"] == "optimized") != (size < source_size):
        raise ValueError("Status and size disagree")
    if hashlib.sha256(output).hexdigest() != result["output_sha256"]:
        raise ValueError("Output hash mismatch")
    dimensions = result["image_size"]
    if (
        not isinstance(dimensions, list)
        or len(dimensions) != 2
        or any(
            type(x) is not int or not 1 <= x <= limits["max_dimension"]
            for x in dimensions
        )
        or dimensions[0] * dimensions[1] > limits["max_pixels"]
    ):
        raise ValueError("Invalid dimensions")
    result["data"] = output


def _run_worker(data, options, *, timeout, cancelled=None, command=None):
    def validate(result, limits):
        _validate_response(result, limits)
        if result["status"] in {"optimized", "unchanged"}:
            if result["input_bytes"] != len(data):
                raise ValueError("Input size mismatch")
            if result["status"] == "unchanged" and result["data"] != data:
                raise ValueError("Unchanged result must be the original bytes")

    return _supervise(
        data,
        options,
        timeout=timeout,
        cancelled=cancelled,
        command=command or [sys.executable, "-m", "opencv_bv_tools.runtime.png_worker"],
        max_output_bytes=4 * ((len(data) + 2) // 3) + 65536,
        response_validator=validate,
        failure_factory=_failure,
        status_field="status",
        extra_env={"RAYON_NUM_THREADS": "1"},
    )


def compress_png(
    data: bytes, *, limits: PNGLimits | None = None, cancelled=None
) -> dict:
    """Optimize ≤8-bit still PNG, preserving samples/representation and metadata.

    Successful output is never larger than the input. No resizing, palette
    quantization, hidden-RGB modification, EXIF transpose or metadata removal.
    The supervised deadline discards late/partial output; bounded parsing and
    cleanup overhead are not a hard real-time guarantee. Not a memory sandbox.
    """
    limits = PNGLimits() if limits is None else limits
    if not isinstance(limits, PNGLimits):
        raise PNGCompressionError("limits must be PNGLimits.")
    limits.validate()
    if not isinstance(data, bytes) or not data:
        raise PNGCompressionError("Input must be nonempty bytes.")
    if len(data) > limits.max_input_bytes:
        raise PNGCompressionError(f"Input exceeds {limits.max_input_bytes} bytes.")
    if cancelled is not None and not callable(cancelled):
        raise PNGCompressionError("cancelled must be callable or None.")
    return _run_worker(
        data,
        {"limits": asdict(limits)},
        timeout=limits.timeout_seconds,
        cancelled=cancelled,
    )


# MCP interface: registration imports MCP only when the server starts.
def register_tools(server, context):
    from pydantic import StrictBool, StrictFloat

    from opencv_bv_tools import png_compression as implementation
    from opencv_bv_tools.runtime.context import annotations

    @server.tool(annotations=annotations("Compress PNG losslessly", writes=True))
    def compress_png(
        image_path: str,
        output_path: str,
        overwrite: StrictBool = False,
        timeout_seconds: StrictFloat = 5.0,
    ):
        """Compress a still 8-bit-or-lower PNG to a separate .png in allowed folders.

        Preserves pixels, representation and metadata exactly; output never
        grows. Does not remove private metadata or redact image content.
        Existing output requires overwrite=true; input is never overwritten.
        """
        with context.call():
            context.require("png", "oxipng")
            limits = PNGLimits(
                timeout_seconds=timeout_seconds,
                max_pixels=min(context.max_pixels, 8_000_000),
            )
            limits.validate()
            return context.transform(
                image_path,
                output_path,
                overwrite,
                ".png",
                implementation.compress_png,
                {"optimized", "unchanged"},
                limits=limits,
            )


# Command-line interface: the installed executable calls main().
class _Cancelled(Exception):
    pass


def compress_file(
    input_path, output_path, *, overwrite=False, limits=None, cancelled=None
):
    if type(overwrite) is not bool:
        raise PNGCompressionError("overwrite must be a boolean.")
    limits = PNGLimits() if limits is None else limits
    if not isinstance(limits, PNGLimits):
        raise PNGCompressionError("limits must be PNGLimits.")
    limits.validate()
    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".png":
        raise PNGCompressionError("Output extension must be .png.")
    fd = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    try:
        with os.fdopen(fd, "rb") as handle:
            input_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(input_stat.st_mode):
                raise PNGCompressionError("Input must be a regular file.")
            if input_stat.st_size > limits.max_input_bytes:
                raise PNGCompressionError("Input exceeds byte limit.")
            _check_destination(destination, input_stat, overwrite)
            data = handle.read(limits.max_input_bytes + 1)
        result = compress_png(data, limits=limits, cancelled=cancelled)
        if result["status"] in {"optimized", "unchanged"}:

            def check_cancelled():
                if cancelled is not None and cancelled():
                    raise _Cancelled()

            try:
                check_cancelled()
                _publish_bytes(
                    result["data"],
                    destination,
                    input_stat,
                    overwrite,
                    before_publish=check_cancelled,
                )
            except _Cancelled:
                return _failure("cancelled")
        return result
    except ConversionError as exc:
        raise PNGCompressionError(str(exc)) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compress a still PNG losslessly; preserve pixels and metadata."
    )
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--max-pixels", type=int, default=8_000_000)
    args = parser.parse_args(argv)
    try:
        result = compress_file(
            args.input,
            args.output,
            overwrite=args.overwrite,
            limits=PNGLimits(timeout_seconds=args.timeout, max_pixels=args.max_pixels),
        )
    except (PNGCompressionError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 2
    summary = {k: v for k, v in result.items() if k != "data"}
    if result["status"] in {"optimized", "unchanged"}:
        summary["output"] = str(Path(args.output).expanduser())
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] in {"optimized", "unchanged"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
