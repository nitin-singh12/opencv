# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Private image normalization and bounded lossless PDF worker."""

import base64
import hashlib
import json
import math
from .process_support import peak_rss_kib
import sys
import time

import zlib
from .pdf_writer import pdf_chunks, PdfWriteError

from opencv_bv_tools.document_scanner import ScanLimits
from opencv_bv_tools.document_scanner import CAPABILITIES
from opencv_bv_tools.document_scanner import validate_rotation
from opencv_bv_tools.document_scanner import _failure
from opencv_bv_tools.runtime.opaque_image import normalize_opaque_image
from opencv_bv_tools.runtime.opaque_image import ImageInputLimitError
from opencv_bv_tools.runtime.png_structure import PNGResourceError


def page_geometry(size, rotation, dpi):
    """Use serialized numbers; round bounds outward to hundredths of a point."""

    def pdf_number(value):
        return float(f"{value:.6f}")

    angle = rotation % 360
    if angle in (0, 90, 180, 270):
        cosine, sine = {0: (1, 0), 90: (0, -1), 180: (-1, 0), 270: (0, 1)}[angle]
    else:
        radians = math.radians(-rotation)
        cosine, sine = math.cos(radians), math.sin(radians)
    a, b, c, d = map(pdf_number, (cosine, sine, -sine, cosine))
    w, h = map(pdf_number, (size[0] * 72 / dpi, size[1] * 72 / dpi))
    corners = [
        (a * x + c * y, b * x + d * y) for x, y in [(0, 0), (w, 0), (w, h), (0, h)]
    ]
    xmin, ymin = min(x for x, y in corners), min(y for x, y in corners)
    xmax, ymax = max(x for x, y in corners), max(y for x, y in corners)
    # Serialize six decimal places, with an outward-rounded page margin.
    tx = pdf_number(math.ceil((-xmin + 0.05) * 100) / 100)
    ty = pdf_number(math.ceil((-ymin + 0.05) * 100) / 100)
    width = pdf_number(math.ceil((xmax + tx + 0.05) * 100) / 100)
    height = pdf_number(math.ceil((ymax + ty + 0.05) * 100) / 100)
    if width > 14400 or height > 14400:
        raise ImageInputLimitError("pdf_page_size_limit")
    if not all(
        0.039 <= x + tx <= width - 0.039 and 0.039 <= y + ty <= height - 0.039
        for x, y in corners
    ):
        raise ValueError("page_rounding_guard_failed")
    return [width, height], [a, b, c, d, tx, ty], [w, h]


def create_pdf(image, rotation, dpi, output_limit):
    size = (image.shape[1], image.shape[0])
    page, matrix, display_size = page_geometry(size, rotation, dpi)
    data = zlib.compress(image.tobytes(), 6)
    pages = [(data, b"/FlateDecode", b"/DeviceRGB", *size, *page, b"")]
    try:
        output = b"".join(
            pdf_chunks(
                pages, output_limit, placements=[(matrix, display_size)], creator=True
            )
        )
    except PdfWriteError as exc:
        raise ImageInputLimitError("output_byte_limit") from exc
    return output, page, matrix, display_size


def process(data, options):
    start = time.monotonic()
    limits = options["limits"]
    try:
        image, source_format = normalize_opaque_image(data, limits)
    except ImportError:
        return _failure("missing_scanner_dependency")
    except (ImageInputLimitError, PNGResourceError) as exc:
        r = _failure(str(exc))
        r["status"] = "resource_limited"
        return r
    except (ValueError, OSError, SyntaxError, EOFError) as exc:
        return _failure(
            "color_profile_not_supported"
            if str(exc) == "color_profile_not_supported"
            else "invalid_or_unsupported_image"
        )
    try:
        encoded, page, matrix, display_size = create_pdf(
            image, options["rotation"], options["dpi"], limits["max_output_bytes"]
        )
    except ImageInputLimitError as exc:
        r = _failure(str(exc))
        r["status"] = "resource_limited"
        return r
    except Exception:
        return _failure("pdf_creation_error")
    return dict(
        status="completed",
        stop_reason="manual_rotation_completed",
        capabilities=CAPABILITIES,
        data_base64=base64.b64encode(encoded).decode(),
        input_format=source_format,
        output_format="PDF",
        page_count=1,
        image_size=[image.shape[1], image.shape[0]],
        page_size_points=page,
        placement_matrix=matrix,
        image_display_size_points=display_size,
        rotation_clockwise=options["rotation"],
        dpi=options["dpi"],
        output_bytes=len(encoded),
        output_sha256=hashlib.sha256(encoded).hexdigest(),
        normalized_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        backend={
            "name": "opencv-bv PDF writer",
            "version": "1",
            "image_encoding": "Flate lossless normalized RGB",
            "rotation": "PDF placement matrix",
            "rounding_guard_points": 0.05,
        },
        worker_seconds=time.monotonic() - start,
        worker_peak_rss_kib=peak_rss_kib(),
    )


def main():
    try:
        header = sys.stdin.buffer.readline(4097)
        if len(header) > 4096 or not header.endswith(b"\n"):
            raise ValueError("Invalid header")
        options = json.loads(header)
        ScanLimits(**options["limits"]).validate()
        validate_rotation(options["rotation"], options["dpi"])
        data = sys.stdin.buffer.read(options["limits"]["max_input_bytes"] + 1)
        if not data or len(data) > options["limits"]["max_input_bytes"]:
            raise ValueError("Invalid input")
        result = process(data, options)
    except Exception:
        result = _failure("worker_error")
    sys.stdout.buffer.write(
        json.dumps(
            result, ensure_ascii=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
