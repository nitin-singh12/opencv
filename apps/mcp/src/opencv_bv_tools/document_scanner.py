# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Single-image PDF with explicit clockwise rotation; no automatic scanning."""

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

CAPABILITIES = "Single image, manual rotation, image-only PDF; no automatic crop, deskew, perspective correction or OCR."


class DocumentScanError(ValueError):
    """Invalid caller arguments; processing failures are result statuses."""


@dataclass(frozen=True)
class ScanLimits:
    max_input_bytes: int = 32 * 1024 * 1024
    max_output_bytes: int = 32 * 1024 * 1024
    max_pixels: int = 8_000_000
    max_dimension: int = 16384
    timeout_seconds: float = 5.0

    def validate(self):
        for field, ceiling in dict(
            max_input_bytes=32 * 1024 * 1024,
            max_output_bytes=32 * 1024 * 1024,
            max_pixels=20_000_000,
            max_dimension=32768,
        ).items():
            value = getattr(self, field)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise DocumentScanError(
                    f"{field} must be an integer from 1 to {ceiling}."
                )
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 30
        ):
            raise DocumentScanError(
                "timeout_seconds must be finite, greater than 0 and at most 30."
            )


def validate_rotation(rotation, dpi):
    if (
        type(rotation) not in (int, float)
        or not math.isfinite(rotation)
        or not -180 <= rotation <= 180
    ):
        raise DocumentScanError(
            "rotation must be a finite clockwise angle from -180 to 180 degrees."
        )
    if type(dpi) is not int or not 72 <= dpi <= 600:
        raise DocumentScanError("dpi must be an integer from 72 to 600.")


def _failure(reason):
    return dict(
        status="failed",
        stop_reason=reason,
        data=None,
        image_size=None,
        page_size_points=None,
        elapsed_seconds=0.0,
        capabilities=CAPABILITIES,
    )


def _validate_response(result, limits):
    if not isinstance(result, dict) or result["status"] not in {
        "completed",
        "failed",
        "resource_limited",
    }:
        raise ValueError("Invalid status")
    if type(result["stop_reason"]) is not str or len(result["stop_reason"]) > 80:
        raise ValueError("Invalid reason")
    if result["capabilities"] != CAPABILITIES:
        raise ValueError("Invalid capabilities")
    if result["status"] != "completed":
        if result.get("data") is not None or result.get("data_base64") is not None:
            raise ValueError("Failure with output")
        return
    size = result["image_size"]
    page = result["page_size_points"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(
            type(v) is not int or not 1 <= v <= limits["max_dimension"] for v in size
        )
        or size[0] * size[1] > limits["max_pixels"]
    ):
        raise ValueError("Invalid image size")
    if (
        not isinstance(page, list)
        or len(page) != 2
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 14400
            for v in page
        )
    ):
        raise ValueError("Invalid page size")
    validate_rotation(result["rotation_clockwise"], result["dpi"])
    if result["page_count"] != 1 or type(result["page_count"]) is not int:
        raise ValueError("Invalid page count")
    count = result["output_bytes"]
    encoded = result.pop("data_base64")
    if type(count) is not int or not 1 <= count <= limits["max_output_bytes"]:
        raise ValueError("Invalid size")
    if type(encoded) is not str or len(encoded) != 4 * ((count + 2) // 3):
        raise ValueError("Invalid encoding")
    data = base64.b64decode(encoded, validate=True)
    if (
        len(data) != count
        or not data.startswith(b"%PDF-1.4")
        or not data.rstrip().endswith(b"%%EOF")
        or hashlib.sha256(data).hexdigest() != result["output_sha256"]
    ):
        raise ValueError("Invalid PDF bytes")
    result["data"] = data


def _run_worker(data, options, *, timeout, cancelled=None, command=None):
    return _supervise(
        data,
        options,
        timeout=timeout,
        cancelled=cancelled,
        command=command
        or [sys.executable, "-m", "opencv_bv_tools.runtime.scanner_worker"],
        max_output_bytes=4 * ((options["limits"]["max_output_bytes"] + 2) // 3) + 65536,
        response_validator=_validate_response,
        failure_factory=_failure,
        status_field="status",
    )


def scan_document(
    data: bytes,
    *,
    rotation=0,
    dpi=150,
    limits: ScanLimits | None = None,
    cancelled=None,
) -> dict:
    """Convert one still image to one PDF page, rotated clockwise after EXIF.

    Rotation is explicitly chosen by the caller. All normalized image pixels
    are embedded without resampling. PDF placement expands the page with a tiny
    white rounding guard. This does not detect or crop documents, deskew or OCR.
    Linux worker deadlines discard late results; cleanup/parsing are not hard
    real-time bounds. Output limits are not enforced process-memory limits.
    """
    limits = ScanLimits() if limits is None else limits
    if not isinstance(limits, ScanLimits):
        raise DocumentScanError("limits must be ScanLimits.")
    limits.validate()
    validate_rotation(rotation, dpi)
    if not isinstance(data, bytes) or not data:
        raise DocumentScanError("Input must be nonempty bytes.")
    if len(data) > limits.max_input_bytes:
        raise DocumentScanError("Input exceeds byte limit.")
    if cancelled is not None and not callable(cancelled):
        raise DocumentScanError("cancelled must be callable or None.")
    return _run_worker(
        data,
        dict(limits=asdict(limits), rotation=float(rotation), dpi=dpi),
        timeout=limits.timeout_seconds,
        cancelled=cancelled,
    )


# MCP interface: registration imports MCP only when the server starts.
def register_tools(server, context):
    from pydantic import StrictBool, StrictFloat, StrictInt

    from opencv_bv_tools import document_scanner as implementation
    from opencv_bv_tools.runtime.context import annotations

    @server.tool(annotations=annotations("Create PDF from an image", writes=True))
    def scan_document(
        image_path: str,
        output_path: str,
        rotation: StrictFloat = 0.0,
        dpi: StrictInt = 150,
        overwrite: StrictBool = False,
        timeout_seconds: StrictFloat = 5.0,
    ):
        """Create one image-only PDF page from a local image in allowed folders.

        Rotation is explicit clockwise degrees (-180 to 180). Preserves all
        normalized RGB pixels; expands the page without cropping. Does not
        automatically detect edges, deskew, correct perspective or perform OCR.
        Output must be a separate .pdf; existing output requires overwrite=true.
        """
        with context.call():
            context.require("scanner", "cv2", "numpy")
            validate_rotation(rotation, dpi)
            limits = ScanLimits(
                timeout_seconds=timeout_seconds,
                max_pixels=min(context.max_pixels, 8_000_000),
            )
            limits.validate()
            return context.transform(
                image_path,
                output_path,
                overwrite,
                ".pdf",
                implementation.scan_document,
                {"completed"},
                rotation=rotation,
                dpi=dpi,
                limits=limits,
            )


# Command-line interface: the installed executable calls main().
class _Cancelled(Exception):
    pass


def scan_file(
    input_path,
    output_path,
    *,
    rotation=0,
    dpi=150,
    overwrite=False,
    limits=None,
    cancelled=None,
):
    if type(overwrite) is not bool:
        raise DocumentScanError("overwrite must be a boolean.")
    limits = ScanLimits() if limits is None else limits
    if not isinstance(limits, ScanLimits):
        raise DocumentScanError("limits must be ScanLimits.")
    limits.validate()
    validate_rotation(rotation, dpi)
    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".pdf":
        raise DocumentScanError("Output extension must be .pdf.")
    fd = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    try:
        with os.fdopen(fd, "rb") as handle:
            input_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(input_stat.st_mode):
                raise DocumentScanError("Input must be a regular file.")
            if input_stat.st_size > limits.max_input_bytes:
                raise DocumentScanError("Input exceeds byte limit.")
            _check_destination(destination, input_stat, overwrite)
            data = handle.read(limits.max_input_bytes + 1)
        result = scan_document(
            data, rotation=rotation, dpi=dpi, limits=limits, cancelled=cancelled
        )
        if result["status"] == "completed":

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
        raise DocumentScanError(str(exc)) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create a one-page image PDF with manual clockwise rotation; no automatic deskew or OCR."
    )
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--rotation",
        type=float,
        default=0,
        help="Clockwise degrees after EXIF, -180 to 180",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Page-size DPI, 72 to 600; ignores input DPI metadata",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)
    try:
        result = scan_file(
            args.input,
            args.output,
            rotation=args.rotation,
            dpi=args.dpi,
            overwrite=args.overwrite,
            limits=ScanLimits(timeout_seconds=args.timeout),
        )
    except (DocumentScanError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 2
    summary = {k: v for k, v in result.items() if k != "data"}
    if result["status"] == "completed":
        summary["output"] = str(Path(args.output).expanduser())
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
