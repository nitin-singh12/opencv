"""Bounded local face detection and blur; no identity or anonymity guarantee."""

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

MODEL_NAME = "face_detection_yunet_2026may.onnx"
MODEL_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
QUALIFICATION = "Only detected regions were blurred; other faces may remain. Blur does not guarantee anonymity or prevent recognition or recovery."


class FaceBlurError(ValueError):
    """Invalid caller options; processing failures are result statuses."""


@dataclass(frozen=True)
class FaceLimits:
    max_input_bytes: int = 32 * 1024 * 1024
    max_output_bytes: int = 32 * 1024 * 1024
    max_pixels: int = 8_000_000
    max_dimension: int = 16384
    max_regions: int = 64
    timeout_seconds: float = 5.0

    def validate(self):
        for key, ceiling in dict(
            max_input_bytes=32 * 1024 * 1024,
            max_output_bytes=32 * 1024 * 1024,
            max_pixels=20_000_000,
            max_dimension=32768,
            max_regions=128,
        ).items():
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise FaceBlurError(f"{key} must be an integer from 1 to {ceiling}.")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 30
        ):
            raise FaceBlurError(
                "timeout_seconds must be finite, greater than 0 and at most 30."
            )


def _failure(reason):
    return dict(
        status="failed",
        stop_reason=reason,
        data=None,
        regions=[],
        image_size=None,
        elapsed_seconds=0.0,
        qualification=QUALIFICATION,
    )


def _validate_response(result, limits):
    if not isinstance(result, dict) or result["status"] not in {
        "blurred",
        "no_faces",
        "failed",
        "resource_limited",
    }:
        raise ValueError("Invalid status")
    if type(result["stop_reason"]) is not str or len(result["stop_reason"]) > 80:
        raise ValueError("Invalid reason")
    if result["qualification"] != QUALIFICATION:
        raise ValueError("Missing qualification")
    if result["status"] != "blurred":
        if (
            result.get("data") is not None
            or result.get("data_base64") is not None
            or result["regions"]
        ):
            raise ValueError("Non-success has output")
        return
    size = result["image_size"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(
            type(x) is not int or not 1 <= x <= limits["max_dimension"] for x in size
        )
        or size[0] * size[1] > limits["max_pixels"]
    ):
        raise ValueError("Invalid dimensions")
    regions = result["regions"]
    if not isinstance(regions, list) or not 1 <= len(regions) <= limits["max_regions"]:
        raise ValueError("Invalid region count")
    for box in regions:
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(x) is not int for x in box)
            or not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1])
        ):
            raise ValueError("Invalid region")
    n = result["output_bytes"]
    if type(n) is not int or not 1 <= n <= limits["max_output_bytes"]:
        raise ValueError("Invalid output size")
    encoded = result.pop("data_base64")
    if type(encoded) is not str or len(encoded) != 4 * ((n + 2) // 3):
        raise ValueError("Invalid encoding")
    data = base64.b64decode(encoded, validate=True)
    if (
        len(data) != n
        or not data.startswith(b"\x89PNG\r\n\x1a\n")
        or hashlib.sha256(data).hexdigest() != result["output_sha256"]
    ):
        raise ValueError("Invalid output")
    result["data"] = data


def _run_worker(data, options, *, timeout, cancelled=None, command=None):
    return _supervise(
        data,
        options,
        timeout=timeout,
        cancelled=cancelled,
        command=command
        or [sys.executable, "-m", "opencv_bv_tools.runtime.face_worker"],
        max_output_bytes=4 * ((options["limits"]["max_output_bytes"] + 2) // 3) + 65536,
        response_validator=_validate_response,
        failure_factory=_failure,
        status_field="status",
    )


def blur_faces(
    data: bytes, *, limits: FaceLimits | None = None, cancelled=None
) -> dict:
    """Blur detected regions in a still image and return a metadata-free RGB PNG.

    No faces, incomplete detection, timeout or cancellation returns no output.
    Coordinates are half-open rectangles after EXIF orientation. Linux only;
    deadlines discard late output, but bounded parsing and process cleanup are
    not hard real-time guarantees. There is no enforced process memory cap.
    """
    limits = FaceLimits() if limits is None else limits
    if not isinstance(limits, FaceLimits):
        raise FaceBlurError("limits must be FaceLimits.")
    limits.validate()
    if not isinstance(data, bytes) or not data:
        raise FaceBlurError("Input must be nonempty bytes.")
    if len(data) > limits.max_input_bytes:
        raise FaceBlurError("Input exceeds byte limit.")
    if cancelled is not None and not callable(cancelled):
        raise FaceBlurError("cancelled must be callable or None.")
    return _run_worker(
        data,
        {"limits": asdict(limits)},
        timeout=limits.timeout_seconds,
        cancelled=cancelled,
    )


# MCP interface: registration imports MCP only when the server starts.
def register_tools(server, context):
    from pydantic import StrictBool, StrictFloat

    from opencv_bv_tools import face_blurring as implementation
    from opencv_bv_tools.runtime.context import annotations

    @server.tool(annotations=annotations("Blur detected faces", writes=True))
    def blur_faces(
        image_path: str,
        output_path: str,
        overwrite: StrictBool = False,
        timeout_seconds: StrictFloat = 5.0,
    ):
        """Blur detected faces in a local image and save a separate metadata-free PNG.

        Paths must be in allowed folders. No output is written on no faces,
        incomplete processing or failure. Small or difficult faces can be
        missed; review the output. Blur does not guarantee anonymity.
        """
        with context.call():
            context.require("face", "cv2", "numpy")
            limits = FaceLimits(
                timeout_seconds=timeout_seconds,
                max_pixels=min(context.max_pixels, 8_000_000),
            )
            limits.validate()
            return context.transform(
                image_path,
                output_path,
                overwrite,
                ".png",
                implementation.blur_faces,
                {"blurred"},
                limits=limits,
            )


# Command-line interface: the installed executable calls main().
class _Cancelled(Exception):
    pass


def blur_file(input_path, output_path, *, overwrite=False, limits=None, cancelled=None):
    if type(overwrite) is not bool:
        raise FaceBlurError("overwrite must be a boolean.")
    limits = FaceLimits() if limits is None else limits
    if not isinstance(limits, FaceLimits):
        raise FaceBlurError("limits must be FaceLimits.")
    limits.validate()
    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".png":
        raise FaceBlurError("Output extension must be .png.")
    fd = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    try:
        with os.fdopen(fd, "rb") as handle:
            input_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(input_stat.st_mode):
                raise FaceBlurError("Input must be a regular file.")
            if input_stat.st_size > limits.max_input_bytes:
                raise FaceBlurError("Input exceeds byte limit.")
            _check_destination(destination, input_stat, overwrite)
            data = handle.read(limits.max_input_bytes + 1)
        result = blur_faces(data, limits=limits, cancelled=cancelled)
        if result["status"] == "blurred":

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
        raise FaceBlurError(str(exc)) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Blur detected faces locally; review output for missed faces. No anonymity guarantee."
    )
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--max-pixels", type=int, default=8_000_000)
    args = parser.parse_args(argv)
    try:
        result = blur_file(
            args.input,
            args.output,
            overwrite=args.overwrite,
            limits=FaceLimits(timeout_seconds=args.timeout, max_pixels=args.max_pixels),
        )
    except (FaceBlurError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 2
    summary = {k: v for k, v in result.items() if k != "data"}
    if result["status"] == "blurred":
        summary["output"] = str(Path(args.output).expanduser())
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] == "blurred" else 2


if __name__ == "__main__":
    raise SystemExit(main())
