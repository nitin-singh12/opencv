"""Independent compatibility gates for the OpenCV migration."""

from io import BytesIO
import hashlib
from pathlib import Path
import struct
import zipfile
import zlib

import numpy as np
import pytest
from PIL import Image

from opencv_bv_tools.runtime import image_io
from opencv_bv_tools.runtime.face_worker import resize_bilinear
from opencv_bv_tools.runtime.png_worker import optimize
from opencv_bv_tools.png_compression import PNGLimits
from dataclasses import asdict


@pytest.mark.parametrize("orientation", range(1, 9))
def test_optional_cmyk_exact_original_pixels(orientation):
    from PIL import ImageOps
    from opencv_bv_tools.runtime.image_redact import open_image

    source = np.random.default_rng(20261111).integers(0, 256, (29, 31, 4), np.uint8)
    exif = Image.Exif()
    exif[274] = orientation
    stream = BytesIO()
    Image.frombytes("CMYK", (31, 29), source.tobytes()).save(stream, "JPEG", exif=exif)
    data = stream.getvalue()
    with Image.open(BytesIO(data)) as original:
        expected = np.asarray(ImageOps.exif_transpose(original).convert("RGB"))
    assert np.array_equal(open_image(BytesIO(data)), expected)


def test_pillow_required_only_for_cmyk_and_limits_checked_first(monkeypatch):
    import builtins
    from opencv_bv_tools.runtime.image_redact import (
        CMYKCompatibilityRequired,
        ImageTooLarge,
        open_image,
    )
    from opencv_bv_tools.runtime.ocr_worker import process
    from opencv_bv_tools.runtime.ocr_process import OCRWorkerError, validate_response

    cmyk = BytesIO()
    Image.new("CMYK", (31, 29)).save(cmyk, "JPEG")
    normal = BytesIO()
    Image.new("RGB", (31, 29), "red").save(normal, "PNG")
    original_import = builtins.__import__
    attempts = []

    def without_pillow(name, *args, **kwargs):
        if name.split(".")[0] == "PIL":
            attempts.append(name)
            raise ImportError("Pillow is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pillow)
    assert open_image(BytesIO(normal.getvalue())).shape == (29, 31, 3)
    with pytest.raises(ImageTooLarge):
        open_image(BytesIO(cmyk.getvalue()), max_pixels=10)
    assert attempts == []
    with pytest.raises(CMYKCompatibilityRequired, match="optional cmyk extra"):
        open_image(BytesIO(cmyk.getvalue()))
    result, payload = process(cmyk.getvalue(), {"max_pixels": 1000}, {})
    assert result["reason"] == "cmyk_dependency_required"
    with pytest.raises(OCRWorkerError, match="optional cmyk extra"):
        validate_response(
            dict(result, request_id=1),
            payload,
            redact=True,
            max_pixels=1000,
            pid=0,
            request_id=1,
        )


def chunk(kind, raw):
    return (
        struct.pack(">I", len(raw))
        + kind
        + raw
        + struct.pack(">I", zlib.crc32(kind + raw) & 0xFFFFFFFF)
    )


@pytest.mark.parametrize(
    "shape", [(1, 1), (16384, 1), (23, 17), (480, 640), (799, 1001), (29, 31)]
)
@pytest.mark.parametrize("size", [(13, 17), (320, 240), (640, 480), (1, 1)])
def test_detector_resize_exact_against_original(shape, size):
    source = np.random.default_rng(42).integers(0, 256, (*shape, 3), np.uint8)
    expected = np.asarray(
        Image.fromarray(source).resize(size, Image.Resampling.BILINEAR)
    )
    assert np.array_equal(resize_bilinear(source, size), expected)


def test_apng_default_plus_one_frame_rejected():
    default = Image.new("RGB", (4, 4), "red")
    frame = Image.new("RGB", (4, 4), "blue")
    buf = BytesIO()
    default.save(
        buf,
        format="PNG",
        save_all=True,
        append_images=[frame],
        default_image=True,
        duration=10,
    )
    assert Image.open(BytesIO(buf.getvalue())).n_frames == 2
    with pytest.raises(ValueError, match="multiframe"):
        image_io.decode(buf.getvalue())


@pytest.mark.parametrize("gray", [0, 255, 256, 257, 511, 512, 65534, 65535])
@pytest.mark.parametrize("alpha", [0, 255, 256, 512, 32768, 65535])
def test_gray_alpha16_matches_original_normalization(gray, alpha):
    raw = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", 1, 1, 16, 4, 0, 0, 0)
    )
    raw += chunk(
        b"IDAT", zlib.compress(b"\0" + struct.pack(">HH", gray, alpha))
    ) + chunk(b"IEND", b"")
    with Image.open(BytesIO(raw)) as im:
        expected = np.asarray(
            Image.alpha_composite(
                Image.new("RGBA", im.size, "white"), im.convert("RGBA")
            ).convert("RGB")
        )
    decoded, _, _ = image_io.decode(raw, allow_16=True)
    assert np.array_equal(image_io.opaque(decoded), expected)


def test_compressed_metadata_bounded_before_native_decoder(monkeypatch):
    import cv2

    buf = BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="PNG")
    raw = buf.getvalue()
    raw = (
        raw[:33]
        + chunk(b"zTXt", b"key\0\0" + zlib.compress(b"x" * (1024 * 1024 + 1)))
        + raw[33:]
    )
    monkeypatch.setattr(
        cv2,
        "imdecode",
        lambda *a: pytest.fail("native decoder must not run"),
    )
    with pytest.raises(ValueError):
        image_io.decode(raw)


@pytest.mark.parametrize(
    "palette",
    [
        [(10, 20, 30, 255), (10, 20, 30, 255)],
        [(10, 20, 30, 0), (10, 20, 30, 255)],
        [(10, 20, 30, 0), (30, 20, 10, 0)],
    ],
)
@pytest.mark.parametrize("interlace", [False, True])
def test_palette_sample_verification_keeps_indices(palette, interlace):
    import png

    stream = BytesIO()
    png.Writer(
        9, 7, palette=palette, bitdepth=1, interlace=interlace, compression=0
    ).write(stream, [[x % 2 for x in range(9)] for _ in range(7)])
    source = stream.getvalue()
    result = optimize(source, {"limits": asdict(PNGLimits())})
    assert result["status"] in ("optimized", "unchanged"), result
    import base64

    out = base64.b64decode(result["data_base64"])
    a = png.Reader(bytes=source).read()
    b = png.Reader(bytes=out).read()
    assert a[:2] == b[:2] and [list(row) for row in a[2]] == [list(row) for row in b[2]]
