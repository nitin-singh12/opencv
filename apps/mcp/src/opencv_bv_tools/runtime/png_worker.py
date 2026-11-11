"""Private PNG worker: decode/optimize/verify; never receives filenames."""

import base64
import hashlib
from importlib.metadata import version
import json
from .process_support import peak_rss_kib
import sys
import time
from .image_io import decode
from .png_structure import _inflate, scanline_bytes
from opencv_bv_tools.png_compression import PNGLimits
from opencv_bv_tools.png_compression import _failure
from opencv_bv_tools.runtime.png_structure import parse_png
from opencv_bv_tools.runtime.png_structure import transplant_idat
from opencv_bv_tools.runtime.png_structure import PNGResourceError


def samples(data):
    image, _, _ = decode(data, apply_orientation=False)
    return image.shape, image.dtype.str, hashlib.sha256(image.tobytes()).digest()


def duplicate_palette(structure):
    if structure.color != 3:
        return False
    chunks = dict(
        (kind, value)
        for kind, value, _ in structure.chunks
        if kind in {b"PLTE", b"tRNS"}
    )
    palette = chunks[b"PLTE"]
    alpha = chunks.get(b"tRNS", b"")
    entries = [
        palette[i : i + 3] + bytes([alpha[i // 3] if i // 3 < len(alpha) else 255])
        for i in range(0, len(palette), 3)
    ]
    return len(set(entries)) != len(entries)


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


def optimize(data, options):
    limits = options["limits"]
    started = time.monotonic()
    try:
        source = parse_png(data, limits)
        original_samples = samples(data)
    except PNGResourceError as exc:
        result = _failure(str(exc))
        result["status"] = "resource_limited"
        return result
    except (ValueError, OSError, SyntaxError, EOFError):
        return _failure("invalid_or_unsupported_png")
    try:
        import oxipng
    except ImportError:
        return _failure("missing_png_dependency")
    if version("pyoxipng") != "9.1.1":
        return _failure("unsupported_optimizer_version")
    try:
        candidate = oxipng.optimize_from_memory(
            data,
            level=0,
            fix_errors=False,
            force=False,
            optimize_alpha=False,
            bit_depth_reduction=False,
            color_type_reduction=False,
            palette_reduction=False,
            grayscale_reduction=False,
            interlace=None,
            scale_16=False,
            strip=oxipng.StripChunks.none(),
        )
        if len(candidate) > limits["max_input_bytes"]:
            # A larger result cannot improve the validated original.
            output = data
            reason = "no_smaller_encoding"
        else:
            structure = parse_png(candidate, limits)
            if source.representation != structure.representation:
                # Native 9.1.1 may shorten tRNS even with reductions disabled.
                # Preserve the exact representation by returning the valid source.
                output = data
                reason = "representation_preservation_guard"
            elif duplicate_palette(source) and filtered_stream(
                source
            ) != filtered_stream(structure):
                # Equal visible colors cannot prove equal indices for duplicate palette entries.
                output = data
                reason = "palette_index_preservation_guard"
            else:
                output = transplant_idat(source, structure)
                if len(output) >= len(data):
                    output = data
                    reason = "no_smaller_encoding"
                else:
                    verified = parse_png(output, limits)
                    if (
                        source.non_idat != verified.non_idat
                        or samples(output) != original_samples
                    ):
                        return _failure("optimizer_verification_failed")
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
            "name": "pyoxipng",
            "version": version("pyoxipng"),
            "level": 0,
            "threads": 1,
            "representation_reductions": False,
            "optimize_alpha": False,
            "metadata_policy": "preserve_non_IDAT_bytes",
        },
        verification="same decoded shape/dtype/sample hash; identical representation and non-IDAT chunks; duplicate palette filtered-stream guard",
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
