# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Bounded image admission and OpenCV codecs. Arrays are uint8 RGB/RGBA.

This module imports no native libraries until a codec/array operation is called.
Header checks run before decoding; individual tools choose their accepted formats,
bit depths, size limits and color-profile policy.
"""

from dataclasses import dataclass
import struct
import zlib


class ImageLimitError(ValueError):
    pass


@dataclass(frozen=True)
class Header:
    format: str
    width: int
    height: int
    depth: int = 8
    grayscale: bool = False
    cmyk: bool = False
    animated: bool = False
    gray16: bool = False
    icc: bytes | None = None
    exif: bytes | None = None


def header(data):
    """Read allocation-relevant fields without invoking an image decoder."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 33:
        if data[12:16] != b"IHDR" or int.from_bytes(data[8:12], "big") != 13:
            raise ValueError("invalid_png_header")
        w, h, depth, color = struct.unpack_from(">IIBB", data, 16)
        position, frames, separate_default, seen_idat, first_frame = (
            8,
            0,
            False,
            False,
            False,
        )
        expanded_metadata = 0
        icc = exif = None
        from .png_structure import _metadata_expansion, _inflate

        while position + 12 <= len(data):
            length = int.from_bytes(data[position : position + 4], "big")
            kind = data[position + 4 : position + 8]
            end = position + 12 + length
            if end > len(data):
                raise ValueError("truncated_png")
            value = data[position + 8 : end - 4]
            if zlib.crc32(kind + value) & 0xFFFFFFFF != int.from_bytes(
                data[end - 4 : end], "big"
            ):
                raise ValueError("invalid_png_crc")
            if kind == b"acTL":
                if len(value) != 8 or not int.from_bytes(value[:4], "big"):
                    raise ValueError("invalid_apng")
                frames = int.from_bytes(value[:4], "big")
            if kind == b"fcTL" and not first_frame:
                separate_default = seen_idat
                first_frame = True
            if kind == b"IDAT":
                seen_idat = True
            expanded_metadata += _metadata_expansion(kind, value, 1024 * 1024)
            if expanded_metadata > 64 * 1024 * 1024:
                raise ImageLimitError("metadata_expansion_limit")
            if kind == b"iCCP":
                icc = _inflate(value.partition(b"\0")[2][1:], 1024 * 1024)
            elif kind == b"eXIf":
                exif = value
            position = end
            if kind == b"IEND":
                break
        if position > len(data) or kind != b"IEND":
            raise ValueError("truncated_png")
        return Header(
            "PNG",
            w,
            h,
            depth,
            color in (0, 4),
            animated=frames + int(separate_default) > 1,
            gray16=color == 0,
            icc=icc,
            exif=exif,
        )
    if data.startswith(b"\xff\xd8"):
        position = 2
        dimensions = None
        icc_parts = []
        exif = None
        while position < len(data):
            if data[position] != 255:
                raise ValueError("invalid_jpeg_marker")
            while position < len(data) and data[position] == 255:
                position += 1
            if position >= len(data):
                break
            marker = data[position]
            position += 1
            if marker in (0xD9, 0xDA):
                break
            if marker in range(0xD0, 0xDA) or marker == 1:
                continue
            if position + 2 > len(data):
                break
            length = int.from_bytes(data[position : position + 2], "big")
            if length < 2 or position + length > len(data):
                break
            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                if length < 8:
                    break
                depth, h, w, channels = struct.unpack_from(">BHHB", data, position + 2)
                dimensions = (w, h, depth, channels == 1, channels == 4)
            value = data[position + 2 : position + length]
            if marker == 0xE1 and value.startswith(b"Exif\0\0"):
                exif = value
            if (
                marker == 0xE2
                and value.startswith(b"ICC_PROFILE\0")
                and len(value) >= 14
            ):
                icc_parts.append((value[12], value[13], value[14:]))
            position += length
        if dimensions is None or data.rfind(b"\xff\xd9") < position:
            raise ValueError("invalid_jpeg_header")
        icc = None
        ordered = sorted(icc_parts)
        if ordered and all(
            seq == index + 1 and count == len(ordered)
            for index, (seq, count, _) in enumerate(ordered)
        ):
            icc = b"".join(value for _, _, value in ordered)
        return Header("JPEG", *dimensions, icc=icc, exif=exif)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        if len(data) < 30 or int.from_bytes(data[4:8], "little") + 8 > len(data):
            raise ValueError("truncated_webp")
        icc = exif = None
        position = 12
        end = int.from_bytes(data[4:8], "little") + 8
        while position + 8 <= end:
            length = int.from_bytes(data[position + 4 : position + 8], "little")
            if position + 8 + length > end:
                raise ValueError("truncated_webp_chunk")
            kind = data[position : position + 4]
            value = data[position + 8 : position + 8 + length]
            if kind == b"ICCP":
                icc = value
            elif kind == b"EXIF":
                exif = value
            position += 8 + length + (length % 2)
        kind, value = data[12:16], data[20:]
        if kind == b"VP8X" and len(value) >= 10:
            return Header(
                "WEBP",
                1 + int.from_bytes(value[4:7], "little"),
                1 + int.from_bytes(value[7:10], "little"),
                animated=bool(value[0] & 2),
                icc=icc,
                exif=exif,
            )
        if kind == b"VP8L" and value[0] == 0x2F:
            bits = int.from_bytes(value[1:5], "little")
            return Header(
                "WEBP",
                1 + (bits & 0x3FFF),
                1 + ((bits >> 14) & 0x3FFF),
                icc=icc,
                exif=exif,
            )
        if kind == b"VP8 " and value[3:6] == b"\x9d\x01\x2a":
            w, h = struct.unpack_from("<HH", value, 6)
            return Header("WEBP", w & 0x3FFF, h & 0x3FFF, icc=icc, exif=exif)
        raise ValueError("invalid_webp_header")
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 13:
        w, h = struct.unpack_from("<HH", data, 6)
        return Header("GIF", w, h)
    if data[:2] == b"BM" and len(data) >= 26:
        dib = int.from_bytes(data[14:18], "little")
        if dib == 12:
            w, h = struct.unpack_from("<HH", data, 18)
        elif dib >= 40 and len(data) >= 54:
            w, h = struct.unpack_from("<ii", data, 18)
            h = abs(h)
        else:
            raise ValueError("unsupported_bmp_header")
        return Header("BMP", w, h)
    raise ValueError(
        "Only PNG, JPEG and WEBP inputs are supported (BMP/GIF for email tools)."
    )


def orientation(raw):
    raw = raw.removeprefix(b"Exif\0\0")
    if len(raw) < 8 or raw[:2] not in (b"II", b"MM"):
        raise ValueError("invalid_exif_header")
    order = "<" if raw[:2] == b"II" else ">"
    magic, offset = struct.unpack_from(order + "HI", raw, 2)
    if magic != 42 or offset + 2 > len(raw):
        raise ValueError("invalid_exif_directory")
    count = struct.unpack_from(order + "H", raw, offset)[0]
    if offset + 2 + count * 12 > len(raw):
        raise ValueError("truncated_exif")
    for pos in range(offset + 2, offset + 2 + count * 12, 12):
        tag, kind, length = struct.unpack_from(order + "HHI", raw, pos)
        if tag == 274:
            if kind != 3 or length != 1:
                raise ValueError("invalid_exif_orientation")
            value = struct.unpack_from(order + "H", raw, pos + 8)[0]
            return value if value in range(1, 9) else 1
    return 1


def orient(image, value):
    import numpy as np

    if value == 2:
        image = image[:, ::-1]
    elif value == 3:
        image = image[::-1, ::-1]
    elif value == 4:
        image = image[::-1]
    elif value == 5:
        image = image.swapaxes(0, 1)
    elif value == 6:
        image = np.rot90(image, -1)
    elif value == 7:
        image = image.swapaxes(0, 1)[::-1, ::-1]
    elif value == 8:
        image = np.rot90(image, 1)
    return np.ascontiguousarray(image)


def decode(
    data,
    *,
    max_pixels=20_000_000,
    max_dimension=None,
    formats=("PNG", "JPEG", "WEBP"),
    allow_16=False,
    allow_cmyk=False,
    allow_animation=False,
    apply_orientation=True,
):
    info = header(data)
    if info.format not in formats:
        raise ValueError("unsupported_image_format")
    if info.width <= 0 or info.height <= 0:
        raise ValueError("invalid_image_dimensions")
    if max_pixels and info.width * info.height > max_pixels:
        raise ImageLimitError("input_pixel_limit")
    if max_dimension and max(info.width, info.height) > max_dimension:
        raise ImageLimitError("input_dimension_limit")
    if info.animated and not allow_animation:
        raise ValueError("Animated or multiframe images are not supported.")
    if info.depth > 8 and not allow_16:
        raise ValueError("PNG input must have at most 8 bits per channel.")
    if info.cmyk and not allow_cmyk:
        raise ValueError("Unsupported input color mode: CMYK.")
    import cv2
    import numpy as np

    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ValueError("image_decode_failed") from exc
    if image is None or image.shape[:2] != (info.height, info.width):
        raise ValueError("image_decode_failed")
    if image.dtype == np.uint16:
        # Pillow exposes gray16 at full depth, but decodes RGB16 to its high byte.
        image = (image // (257 if info.gray16 else 256)).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError("unsupported_decoded_depth")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.shape[2] in (3, 4):
        image = image[:, :, [2, 1, 0] + ([3] if image.shape[2] == 4 else [])]
    else:
        raise ValueError("unsupported_decoded_channels")
    value = orientation(info.exif) if info.exif is not None and apply_orientation else 1
    return orient(image, value), info, info.icc


def opaque(image, background=(255, 255, 255)):
    import numpy as np

    if image.shape[2] == 3:
        return image
    alpha = image[:, :, 3:4].astype(np.uint32)
    rgb = image[:, :, :3].astype(np.uint32)
    return (
        (rgb * alpha + np.array(background, dtype=np.uint32) * (255 - alpha) + 127)
        // 255
    ).astype(np.uint8)


def luminance(image):
    import numpy as np

    rgb = image.astype(np.uint32)
    return (
        (rgb[:, :, 0] * 19595 + rgb[:, :, 1] * 38470 + rgb[:, :, 2] * 7471 + 32768)
        >> 16
    ).astype(np.uint8)


def encode(image, format="PNG", *, quality=90, lossless=True, icc=None):
    import cv2
    import numpy as np

    if image.ndim == 3:
        image = image[:, :, [2, 1, 0] + ([3] if image.shape[2] == 4 else [])]
    if not cv2.haveImageWriter("." + format.lower()):
        raise ValueError("Missing image codec: " + format)
    params = []
    if format == "JPEG":
        params = [
            cv2.IMWRITE_JPEG_QUALITY,
            quality,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        ]
    elif format == "WEBP":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
        if lossless:
            params += [
                cv2.IMWRITE_WEBP_LOSSLESS_MODE,
                cv2.IMWRITE_WEBP_LOSSLESS_PRESERVE_COLOR,
            ]
    elif format == "PNG":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    try:
        if icc and format != "PNG":
            ok, result = cv2.imencodeWithMetadata(
                "." + format.lower(),
                image,
                [cv2.IMAGE_METADATA_ICCP],
                [np.frombuffer(icc, np.uint8)],
                params,
            )
        else:
            ok, result = cv2.imencode("." + format.lower(), image, params)
    except cv2.error as exc:
        raise ValueError("image_encode_failed") from exc
    if not ok:
        raise ValueError("image_encode_failed")
    encoded = result.tobytes()
    if icc and format == "PNG":
        # ICC is opaque application metadata. libpng rejects some profiles that
        # the established conversion contract preserves without validating.
        value = b"ICC Profile\0\0" + zlib.compress(icc)
        body = b"iCCP" + value
        block = (
            struct.pack(">I", len(value))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )
        encoded = encoded[:33] + block + encoded[33:]
    return encoded
