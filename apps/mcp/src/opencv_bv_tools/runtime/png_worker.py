# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Private PNG worker: decode/optimize/verify; never receives filenames."""

import base64
import hashlib
import json
import struct
import zlib
from .process_support import peak_rss_kib
import sys
import time
from .image_io import decode
from .png_structure import _inflate, scanline_bytes, SIGNATURE
from opencv_bv_tools.png_compression import PNGLimits
from opencv_bv_tools.png_compression import _failure
from opencv_bv_tools.runtime.png_structure import parse_png
from opencv_bv_tools.runtime.png_structure import PNGResourceError


def samples(data):
    image, _, _ = decode(data, apply_orientation=False)
    return image.shape, image.dtype.str, hashlib.sha256(image.tobytes()).digest()


def filtered_stream(structure):
    limit = scanline_bytes(
        structure.width,
        structure.height,
        structure.depth,
        structure.color,
        structure.interlace,
    )
    return _inflate(
        b"".join(value for kind, value, _ in structure.chunks if kind == b"IDAT"), limit
    )


def recompress(raw):
    """Choose a small zlib stream while keeping large images responsive."""
    best = None
    chosen = None
    choices = (("default", 1, zlib.Z_DEFAULT_STRATEGY), ("rle", 9, zlib.Z_RLE)) if len(raw) > 4 * 1024 * 1024 else (
        ("default", 9, zlib.Z_DEFAULT_STRATEGY),
        ("filtered", 9, zlib.Z_FILTERED),
        ("rle", 9, zlib.Z_RLE),
    )
    for name, level, strategy in choices:
        encoder = zlib.compressobj(level, zlib.DEFLATED, 15, 8, strategy)
        compressed = encoder.compress(raw) + encoder.flush()
        if best is None or len(compressed) < len(best):
            best, chosen = compressed, (name, level)
    return best, chosen


def replace_idat(source, compressed):
    """Replace consecutive IDAT chunks with one; preserve all other bytes."""
    chunk = (struct.pack(">I", len(compressed)) + b"IDAT" + compressed
             + struct.pack(">I", zlib.crc32(b"IDAT" + compressed) & 0xffffffff))
    output = [SIGNATURE]
    written = False
    for kind, _, raw in source.chunks:
        if kind != b"IDAT":
            output.append(raw)
        elif not written:
            output.append(chunk)
            written = True
    return b"".join(output)


def optimize(data, options):
    limits = options["limits"]
    started = time.monotonic()
    try:
        source = parse_png(data, limits)
        original_samples = samples(data)
        raw = filtered_stream(source)
    except PNGResourceError as exc:
        result = _failure(str(exc))
        result["status"] = "resource_limited"
        return result
    except (ValueError, OSError, SyntaxError, EOFError):
        return _failure("invalid_or_unsupported_png")
    try:
        compressed, (strategy, level) = recompress(raw)
        candidate = replace_idat(source, compressed)
        if len(candidate) >= len(data):
            output = data
            reason = "no_smaller_encoding"
        else:
            verified = parse_png(candidate, limits)
            if (source.non_idat != verified.non_idat
                    or source.representation != verified.representation
                    or filtered_stream(verified) != raw
                    or samples(candidate) != original_samples):
                return _failure("optimizer_verification_failed")
            output = candidate
            reason = "lossless_compression_completed"
    except Exception:
        return _failure("optimizer_or_verification_failed")
    return dict(
        status="optimized" if len(output) < len(data) else "unchanged",
        stop_reason=reason,
        input_bytes=len(data),
        output_bytes=len(output),
        bytes_saved=len(data) - len(output),
        data_base64=base64.b64encode(output).decode("ascii"),
        output_sha256=hashlib.sha256(output).hexdigest(),
        image_size=[source.width, source.height],
        input_bit_depth=source.depth,
        input_color_type=source.color,
        interlace=source.interlace,
        backend={
            "name": "zlib",
            "version": zlib.ZLIB_RUNTIME_VERSION,
            "level": level,
            "strategy": strategy,
            "threads": 1,
            "metadata_policy": "preserve_non_IDAT_bytes",
        },
        verification="identical filtered image data, representation, non-IDAT chunks and decoded samples",
        worker_seconds=time.monotonic() - started,
        worker_peak_rss_kib=peak_rss_kib(),
    )


def main():
    try:
        header = sys.stdin.buffer.readline(4097)
        if len(header) > 4096 or not header.endswith(b"\n"):
            raise ValueError("invalid header")
        options = json.loads(header)
        PNGLimits(**options["limits"]).validate()
        data = sys.stdin.buffer.read(options["limits"]["max_input_bytes"] + 1)
        if not data or len(data) > options["limits"]["max_input_bytes"]:
            raise ValueError("invalid size")
        result = optimize(data, options)
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
