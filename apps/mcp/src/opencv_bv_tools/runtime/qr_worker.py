"""Private JSON/stdin protocol. Image parsing and native decoding stay here."""

import base64
from importlib.metadata import version
import json
from .process_support import peak_rss_kib
import sys
import time


from opencv_bv_tools.qr_reader import MAX_WIRE_BYTES
from opencv_bv_tools.qr_reader import QRLimits
from opencv_bv_tools.qr_reader import _failure


from opencv_bv_tools.runtime.reader_image import ImageLimitError
from opencv_bv_tools.runtime.reader_image import normalize


def decode(data, options):
    limits = options["limits"]
    start = time.monotonic()
    try:
        image, source_format, notes = normalize(data, limits, options["background"])
    except ImageLimitError as exc:
        result = _failure(str(exc))
        result["search_status"] = "resource_limited"
        return result
    except (ValueError, OSError, SyntaxError, EOFError):
        return _failure("invalid_or_unsupported_image")
    normalized = time.monotonic()
    try:
        import zxingcpp
    except ImportError:
        return _failure("missing_qr_dependency")
    if version("zxing-cpp") != "3.1.1":
        return _failure("unsupported_decoder_version")
    try:
        found = zxingcpp.read_barcodes(
            image, formats=zxingcpp.BarcodeFormat.QRCodeModel2,
            try_rotate=True, try_downscale=True, try_invert=True,
            text_mode=zxingcpp.TextMode.Plain, return_errors=True,
        )
    except Exception:
        return _failure("decoder_error")
    decoded = time.monotonic()
    result = dict(
        search_status="completed", stop_reason="configured_search_completed",
        image_size=[image.shape[1], image.shape[0]], input_format=source_format,
        coordinate_space="exif_normalized_image", coordinate_origin="top_left",
        polygon_order="backend top_left, top_right, bottom_right, bottom_left",
        symbols=[], undecoded=[], warnings=notes,
        backend={"name": "zxing-cpp", "version": version("zxing-cpp"),
                 "pass": "native_multicode", "format_filter": "QRCodeModel2",
                 "try_rotate": True, "try_downscale": True, "try_invert": True},
        payload_semantics="Backend-returned message bytes for one symbol; not original QR bitstream/segments.",
        text_provenance="ZXing TextMode.Plain interpretation; ECI/encoding segments are not exposed by this binding.",
        sequence_semantics="Single symbols only; structured-append metadata is not exposed and messages are not assembled.",
        capability_limitations=["empty_payload_symbols_not_supported", "structured_append_metadata_not_exposed"],
        normalize_seconds=normalized - start, decoder_seconds=decoded - normalized,
    )
    total, chars = 0, 0
    # Binding internally searches up to 255 results; it cannot accept a lower
    # discovery cap. Enforce serialization caps here; deadline bounds the search.
    if len(found) >= 255:
        result.update(search_status="resource_limited", stop_reason="backend_symbol_limit")
    for barcode in found:
        position = barcode.position
        polygon = [[p.x, p.y] for p in (position.top_left, position.top_right,
                                       position.bottom_right, position.bottom_left)]
        if not barcode.valid:
            if len(result["undecoded"]) >= limits["max_symbols"]:
                result.update(search_status="resource_limited", stop_reason="candidate_limit")
                continue
            result["undecoded"].append({"polygon": polygon, "status": "not_decoded"})
            continue
        payload = barcode.bytes
        text = None if barcode.content_type == zxingcpp.ContentType.Binary else barcode.text
        if (len(result["symbols"]) >= limits["max_symbols"] or
                len(payload) > limits["max_payload_bytes"] or
                total + len(payload) > limits["max_total_payload_bytes"] or
                chars + len(text or "") > limits["max_text_chars"]):
            result.update(search_status="resource_limited", stop_reason="result_limit")
            continue
        total += len(payload)
        chars += len(text or "")
        extra = barcode.extra
        result["symbols"].append(dict(
            format="QR Code Model 2", payload_base64=base64.b64encode(payload).decode("ascii"),
            text=text, polygon=polygon, content_type=str(barcode.content_type),
            symbology_identifier=barcode.symbology_identifier,
            qr_version=str(extra.get("Version", "unknown")), ec_level=barcode.ec_level,
            sequence_metadata=None,
        ))
    # Geometric order, not payload order: identical messages remain separate.
    for field in ("symbols", "undecoded"):
        result[field].sort(key=lambda s: (min(p[1] for p in s["polygon"]),
                                         min(p[0] for p in s["polygon"])))
    result["worker_peak_rss_kib"] = peak_rss_kib()
    return result


def main():
    try:
        header = sys.stdin.buffer.readline(4097)
        if len(header) > 4096 or not header.endswith(b"\n"):
            raise ValueError("Invalid header")
        options = json.loads(header)
        QRLimits(**options["limits"]).validate()
        data = sys.stdin.buffer.read(options["limits"]["max_input_bytes"] + 1)
        if not data or len(data) > options["limits"]["max_input_bytes"]:
            raise ValueError("Invalid input size")
        result = decode(data, options)
    except Exception:
        result = _failure("worker_error")
    encoded = json.dumps(result, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_WIRE_BYTES:
        encoded = json.dumps(_failure("worker_output_limit")).encode()
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
