# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Convert bounded, single-frame image bytes without filesystem or network access."""

import argparse
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

from .runtime import image_io

from opencv_bv_tools.runtime.file_output import (
    ConversionError,
    _check_destination,
    _publish_bytes,
)

MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 20_000_000
FORMATS = ("PNG", "JPEG", "WEBP")
MIME_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


@dataclass(frozen=True)
class ConversionResult:
    data: bytes
    input_format: str
    output_format: str
    width: int
    height: int
    mime_type: str
    warnings: tuple[str, ...]


def normalize_format(value: str) -> str:
    if not isinstance(value, str):
        raise ConversionError("Output format must be PNG, JPEG (or JPG), or WEBP.")
    value = value.upper()
    if value == "JPG":
        value = "JPEG"
    if value not in FORMATS:
        raise ConversionError("Output format must be PNG, JPEG (or JPG), or WEBP.")
    return value


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ConversionError(f"{name} must be a positive integer.")


def convert_image(
    data: bytes,
    output_format: str,
    *,
    quality: int = 90,
    webp_lossless: bool = True,
    jpeg_background: tuple[int, int, int] | None = None,
    max_input_bytes: int = MAX_INPUT_BYTES,
    max_pixels: int = MAX_PIXELS,
) -> ConversionResult:
    """Return re-encoded PNG/JPEG/WebP bytes; never modifies the input.

    JPEG requires an explicit RGB background if any pixel is nonopaque.
    EXIF orientation is applied, then EXIF/XMP/text metadata is removed.
    RGB ICC profiles are retained; this is not a redaction/anonymization tool.
    Limits bound compressed bytes and pixels, not total process memory or time.
    Raises ConversionError on invalid options, unsupported input, or codec errors.
    """
    target = normalize_format(output_format)
    _positive_integer("max_input_bytes", max_input_bytes)
    _positive_integer("max_pixels", max_pixels)
    if not isinstance(data, bytes) or not data:
        raise ConversionError("Input must be nonempty bytes.")
    if len(data) > max_input_bytes:
        raise ConversionError(f"Input exceeds {max_input_bytes} bytes.")
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ConversionError("quality must be an integer from 1 to 100.")
    if type(webp_lossless) is not bool:
        raise ConversionError("webp_lossless must be a boolean.")
    if jpeg_background is not None and (
        not isinstance(jpeg_background, tuple)
        or len(jpeg_background) != 3
        or any(type(v) is not int or not 0 <= v <= 255 for v in jpeg_background)
    ):
        raise ConversionError(
            "jpeg_background must be a tuple of three integers, 0–255."
        )
    notes = []
    try:
        info = image_io.header(data)
        if target == "WEBP" and max(info.width, info.height) > 16383:
            raise ConversionError("WebP dimensions must not exceed 16383 pixels.")
        image, info, icc = image_io.decode(data, max_pixels=max_pixels)
        if icc and info.grayscale:
            raise ConversionError("ICC-tagged grayscale input is not supported.")
        if target == "JPEG" and image.shape[2] == 4:
            if image[:, :, 3].min() < 255:
                if jpeg_background is None:
                    raise ConversionError("JPEG cannot preserve transparency; provide jpeg_background.")
                image = image_io.opaque(image, jpeg_background)
                notes.append("Transparency was flattened onto the specified background.")
            else:
                image = image[:, :, :3]
        if target == "JPEG":
            notes.append("JPEG encoding is lossy, including same-format conversion.")
        elif target == "WEBP" and not webp_lossless:
            notes.append("Lossy WebP encoding may change pixel values.")
        encoded = image_io.encode(image, target, quality=quality, lossless=webp_lossless, icc=icc)
        return ConversionResult(encoded, info.format, target, image.shape[1], image.shape[0], MIME_TYPES[target], tuple(notes))
    except ConversionError:
        raise
    except image_io.ImageLimitError as exc:
        raise ConversionError(f"Image exceeds {max_pixels} pixels.") from exc
    except (OSError, ValueError, SyntaxError, EOFError) as exc:
        raise ConversionError(f"Image conversion failed: {exc}") from exc


# MCP interface: registration imports MCP only when the server starts.
def register_conversion_tool(
    server, *, file_ops, folders, call_slots, stop_if_cancelled, max_pixels=MAX_PIXELS
):
    """Return the registered function; all filesystem access uses host policy."""
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations
    from pydantic import StrictBool, StrictInt

    if type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError("Conversion pixel limit must be a positive integer.")
    pixel_limit = min(MAX_PIXELS, max_pixels)

    @server.tool(
        description=(
            "Convert a local PNG, JPEG or WebP to another of these formats. "
            "image_path and output_path must be in allowed folders; the output extension "
            "(.png, .jpg, .jpeg, .webp) selects the format. Never changes the input, even "
            "through a hard link. Existing output requires overwrite=true. "
            "For transparent input to JPEG, supply jpeg_background=[R,G,B], each 0–255. "
            "WebP is lossless by default; quality is 1–100 for JPEG/WebP. "
            "Single-frame inputs only, at most 32 MiB and 20 megapixels (or the server's "
            "smaller limit). Applies EXIF orientation; removes EXIF/XMP/text, retains ICC. "
            "Returns metadata only. THIS DOES NOT REDACT EMAILS: converted pixels still "
            "contain original content. Do not preview the input or converted image when "
            "email privacy matters; use redact_emails and view only its redacted output."
        ),
        annotations=ToolAnnotations(
            title="Convert image format",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=False,
        ),
    )
    def convert_image_format(
        image_path: str,
        output_path: str,
        quality: StrictInt = 90,
        webp_lossless: StrictBool = True,
        jpeg_background: list[StrictInt] | None = None,
        overwrite: StrictBool = False,
    ):
        with call_slots:
            stop_if_cancelled()
            try:
                if not image_path or not output_path:
                    raise ConversionError(
                        "image_path and output_path are both required."
                    )
                suffix = Path(output_path).suffix.lower()
                target = EXTENSIONS.get(suffix)
                if target is None:
                    raise ConversionError(
                        "Output extension must be .png, .jpg, .jpeg, or .webp."
                    )
                # Validate options before reading or decoding any image bytes.
                if type(quality) is not int or not 1 <= quality <= 100:
                    raise ConversionError("quality must be an integer from 1 to 100.")
                if type(webp_lossless) is not bool or type(overwrite) is not bool:
                    raise ConversionError(
                        "webp_lossless and overwrite must be booleans."
                    )
                if jpeg_background is not None and (
                    not isinstance(jpeg_background, list)
                    or len(jpeg_background) != 3
                    or any(
                        type(v) is not int or not 0 <= v <= 255 for v in jpeg_background
                    )
                ):
                    raise ConversionError(
                        "jpeg_background must be three integers, 0–255."
                    )
                source, _, identity = file_ops.open_input(image_path, folders)
                protect = {identity}
                with source:
                    if os.fstat(source.fileno()).st_size > MAX_INPUT_BYTES:
                        raise ConversionError(f"Input exceeds {MAX_INPUT_BYTES} bytes.")
                    file_ops.check_output(
                        output_path, folders, protect, overwrite, suffix=suffix
                    )
                    data = source.read(MAX_INPUT_BYTES + 1)
                # A file growing after fstat is still caught by the bounded read.
                if len(data) > MAX_INPUT_BYTES:
                    raise ConversionError(f"Input exceeds {MAX_INPUT_BYTES} bytes.")
                result = convert_image(
                    data,
                    target,
                    quality=quality,
                    webp_lossless=webp_lossless,
                    jpeg_background=tuple(jpeg_background)
                    if jpeg_background is not None
                    else None,
                    max_pixels=pixel_limit,
                )
                stop_if_cancelled()
                written, _ = file_ops.write_file(
                    result.data,
                    output_path,
                    folders,
                    protect,
                    overwrite,
                    suffix=suffix,
                    before_publish=stop_if_cancelled,
                )
                return json.dumps(
                    {
                        "input_format": result.input_format,
                        "output_format": result.output_format,
                        "image_size": [result.width, result.height],
                        "mime_type": result.mime_type,
                        "bytes_written": len(result.data),
                        "output_path": str(written),
                        "warnings": list(result.warnings),
                        "redacted": False,
                    },
                    indent=1,
                )
            except (ConversionError, file_ops.UnsafePath) as exc:
                raise ToolError(str(exc)) from None
            except OSError as exc:
                raise ToolError(f"File error: {exc.strerror or exc}") from None

    return convert_image_format


# Command-line interface: the installed executable calls main().
EXTENSIONS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}


def convert_file(input_path, output_path, *, overwrite=False, **options):
    """Convert a regular local file and atomically publish in a trusted directory.

    Output extension sets format. Requires a filesystem supporting hard links for
    no-overwrite publication. Parent directories must already exist and be trusted;
    this function is not a sandbox for untrusted MCP filesystem paths.
    """
    if type(overwrite) is not bool:
        raise ConversionError("overwrite must be a boolean.")
    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()
    target = EXTENSIONS.get(destination.suffix.lower())
    if target is None:
        raise ConversionError("Output extension must be .png, .jpg, .jpeg, or .webp.")
    supplied_format = options.pop("output_format", target)
    if normalize_format(supplied_format) != target:
        raise ConversionError("Output format and filename extension disagree.")
    # Nonblocking open lets us reject a FIFO without waiting for a writer.
    fd = os.open(source, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    with os.fdopen(fd, "rb") as handle:
        input_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(input_stat.st_mode):
            raise ConversionError("Input must be a regular file.")
        limit = options.get("max_input_bytes", MAX_INPUT_BYTES)
        if type(limit) is not int or limit <= 0:
            raise ConversionError("max_input_bytes must be a positive integer.")
        if input_stat.st_size > limit:
            raise ConversionError(f"Input exceeds {limit} bytes.")
        _check_destination(destination, input_stat, overwrite)
        data = handle.read(limit + 1)
    result = convert_image(data, target, **options)
    _publish_bytes(result.data, destination, input_stat, overwrite)
    return result


def _background(value):
    try:
        channels = tuple(int(v) for v in value.split(","))
        if len(channels) != 3 or any(not 0 <= v <= 255 for v in channels):
            raise ValueError
        return channels
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use three RGB values, e.g. 255,255,255."
        ) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a local PNG, JPEG or WebP image."
    )
    parser.add_argument("input")
    parser.add_argument(
        "output", help="Destination with .png, .jpg, .jpeg or .webp extension"
    )
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--lossy-webp", action="store_true")
    parser.add_argument("--jpeg-background", type=_background, metavar="R,G,B")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = convert_file(
            args.input,
            args.output,
            overwrite=args.overwrite,
            quality=args.quality,
            webp_lossless=not args.lossy_webp,
            jpeg_background=args.jpeg_background,
        )
    except (ConversionError, OSError) as exc:
        parser.exit(2, f"Conversion failed: {exc}\n")
    summary = {key: value for key, value in asdict(result).items() if key != "data"}
    summary.update(
        output=str(Path(args.output).expanduser()), bytes_written=len(result.data)
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
