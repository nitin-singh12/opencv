"""Private JSON/stdin protocol. Image parsing and native decoding stay here."""

import base64
from importlib.metadata import version
import json
from .process_support import peak_rss_kib
import sys
import time


from opencv_bv_tools.runtime.worker_process import MAX_WIRE_BYTES
from opencv_bv_tools.runtime.worker_process import _failure
from opencv_bv_tools.barcode_reader import BarcodeLimits
from opencv_bv_tools.barcode_reader import _formats
from opencv_bv_tools.barcode_reader import _itf_lengths


from opencv_bv_tools.runtime.reader_image import ImageLimitError
from opencv_bv_tools.runtime.reader_image import normalize


FORMAT_ENUM = {"EAN13": "EAN13", "EAN8": "EAN8", "UPCA": "UPCA", "UPCE": "UPCE",
               "Code128": "Code128", "Code39": "Code39Std", "Code39Extended": "Code39Ext", "ITF": "ITF"}
FORMAT_NAMES = {"EAN-13": "EAN13", "EAN-8": "EAN8", "UPC-A": "UPCA", "UPC-E": "UPCE",
                "Code 128": "Code128", "Code 39": "Code39", "Code 39 Extended": "Code39Extended", "ITF": "ITF"}


def decode(data, options):
    limits = options["limits"]
    itf_lengths = _itf_lengths(options.get("itf_lengths"), _formats(options["formats"]))
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
        return _failure("missing_barcode_dependency")
    if version("zxing-cpp") != "3.1.1":
        return _failure("unsupported_decoder_version")
    try:
        found = []
        backend_limited = False
        # ZXing 3.1.1 try_invert does not recover inverted 1D fixtures. Search
        # both visible luminance polarities explicitly under the same deadline.
        for polarity in ("visible", "inverted"):
            view = image if polarity == "visible" else 255 - image
            items = zxingcpp.read_barcodes(
                view, formats=[getattr(zxingcpp.BarcodeFormat, FORMAT_ENUM[n])
                               for n in _formats(options["formats"])],
                try_rotate=True, try_downscale=True, try_invert=False,
                text_mode=zxingcpp.TextMode.Plain, return_errors=True,
                ean_add_on_symbol=zxingcpp.EanAddOnSymbol.Ignore,
            )
            backend_limited |= len(items) >= 255
            found.extend((barcode, polarity) for barcode in items)
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
                 "pass": "native_multicode_visible_and_inverted", "format_filter": options["formats"], "ean_add_on_symbol": "Ignore",
                 "try_rotate": True, "try_downscale": True, "try_invert": False,
                 "explicit_polarities": ["visible", "inverted"], "itf_lengths": itf_lengths},
        payload_semantics="ZXing message bytes; UPC-A/E normalize to EAN-13. Not original bars or printed digits.",
        text_provenance="ZXing TextMode.Plain; no GS1 application-identifier parsing or URL following.",
        capability_limitations=["EAN_2_5_supplements_ignored", "Code39_ITF_checksums_not_required",
                               "Code39_interpretation_requires_explicit_selection",
                               "aligned_identical_barcodes_may_merge_into_one_region",
                               "native_regions_are_not_physical_symbol_counts",
                               "ITF_length_filter_is_not_checksum_validation",
                               "cross_polarity_native_regions_are_not_deduplicated"],
        normalize_seconds=normalized - start, decoder_seconds=decoded - normalized,
    )
    total, chars = 0, 0
    # Each binding pass internally searches up to 255 results; no lower
    # discovery cap. Enforce serialization caps here; deadline bounds the search.
    if backend_limited:
        result.update(search_status="resource_limited", stop_reason="backend_symbol_limit")
    for barcode, polarity in found:
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
        if str(barcode.format) == "ITF" and (not payload.isdigit() or len(payload) not in itf_lengths):
            if len(result["undecoded"]) >= limits["max_symbols"]:
                result.update(search_status="resource_limited", stop_reason="candidate_limit")
            else:
                result["undecoded"].append({"polygon": polygon, "status": "rejected_length", "format": "ITF"})
            continue
        text = None if barcode.content_type == zxingcpp.ContentType.Binary else barcode.text
        if (len(result["symbols"]) >= limits["max_symbols"] or
                len(payload) > limits["max_payload_bytes"] or
                total + len(payload) > limits["max_total_payload_bytes"] or
                chars + len(text or "") > limits["max_text_chars"]):
            result.update(search_status="resource_limited", stop_reason="result_limit")
            continue
        total += len(payload)
        chars += len(text or "")
        name = FORMAT_NAMES.get(str(barcode.format))
        if name not in options["formats"]:
            # Never expose an unexpected native format as a requested symbol.
            return _failure("unexpected_decoder_format")
        extra = barcode.extra or {}
        compressed = extra.get("UPCE") if name == "UPCE" else None
        if compressed is not None and (not isinstance(compressed, str) or
                                       len(compressed) != 8 or not compressed.isascii() or
                                       not compressed.isdigit()):
            return _failure("invalid_decoder_metadata")
        result["symbols"].append(dict(
            format=name, format_name=str(barcode.format), polarity=polarity,
            payload_base64=base64.b64encode(payload).decode("ascii"),
            text=text, polygon=polygon, content_type=str(barcode.content_type),
            symbology_identifier=barcode.symbology_identifier,
            upc_e_compressed=compressed,
            checksum_policy=("optional_not_required" if name in ("Code39", "Code39Extended", "ITF")
                             else "native_mandatory_checksum"),
        ))
    # Sort native regions only; ZXing can merge aligned identical barcodes.
    # Do not deduplicate by payload or merge potentially distinct regions.
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
        BarcodeLimits(**options["limits"]).validate()
        options["formats"] = _formats(options["formats"])
        options["itf_lengths"] = _itf_lengths(options.get("itf_lengths"), options["formats"])
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
