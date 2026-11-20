"""Contract, validation, and codec failures; pixel semantics have independent tests."""

from io import BytesIO
import struct
import zlib

from PIL import Image
import pytest

from opencv_bv_tools import ConversionError
from opencv_bv_tools import convert_image
from opencv_bv_tools import image_conversion as module


def encoded(fmt="PNG", mode="RGB", size=(17, 11), **kwargs):
    image = Image.new(mode, size)
    output = BytesIO()
    image.save(output, format=fmt, **kwargs)
    return output.getvalue()


@pytest.mark.parametrize("source", ["PNG", "JPEG", "WEBP"])
@pytest.mark.parametrize("target", ["PNG", "JPEG", "WEBP"])
def test_all_format_pairs(source, target):
    data = encoded(source)
    result = convert_image(data, target)
    assert result.input_format == source
    assert result.output_format == target
    assert (result.width, result.height) == (17, 11)
    assert result.mime_type == f"image/{target.lower()}"
    with Image.open(BytesIO(result.data)) as output:
        output.verify()
        assert output.format == target
    assert data == encoded(source)


@pytest.mark.parametrize("target", ["png", "jpeg", "jpg", "JpG", "webp"])
def test_format_aliases(target):
    assert convert_image(encoded(), target).data


@pytest.mark.parametrize("target", [None, 1, "", "TIFF", " PNG", ".png"])
def test_invalid_format(target):
    with pytest.raises(ConversionError):
        convert_image(encoded(), target)


@pytest.mark.parametrize("data", [b"", b"not an image", "png", None, bytearray(b"png")])
def test_invalid_data(data):
    with pytest.raises(ConversionError):
        convert_image(data, "PNG")


@pytest.mark.parametrize("fmt", ["BMP", "GIF", "TIFF", "ICO"])
def test_unsupported_input_decoders(fmt):
    with pytest.raises(ConversionError):
        convert_image(encoded(fmt, size=(32, 32)), "PNG")


@pytest.mark.parametrize("option,value", [
    ("quality", 0), ("quality", 101), ("quality", True), ("quality", 1.5),
    ("quality", "90"), ("webp_lossless", 1), ("webp_lossless", None),
    ("jpeg_background", [255, 255, 255]), ("jpeg_background", (1, 2)),
    ("jpeg_background", (-1, 2, 3)), ("jpeg_background", (1, 256, 3)),
    ("jpeg_background", (True, 2, 3)), ("max_input_bytes", 0),
    ("max_input_bytes", -1), ("max_pixels", True), ("max_pixels", 1.1),
])
def test_invalid_options(option, value):
    with pytest.raises(ConversionError):
        convert_image(encoded(), "PNG", **{option: value})


@pytest.mark.parametrize("quality", [1, 90, 100])
@pytest.mark.parametrize("target", ["JPEG", "WEBP"])
def test_quality_boundaries(quality, target):
    result = convert_image(encoded(), target, quality=quality, webp_lossless=False)
    assert result.warnings


def test_compressed_byte_boundary():
    data = encoded()
    assert convert_image(data, "PNG", max_input_bytes=len(data)).data
    with pytest.raises(ConversionError, match="bytes"):
        convert_image(data, "PNG", max_input_bytes=len(data) - 1)


def test_pixel_boundary():
    data = encoded()
    assert convert_image(data, "PNG", max_pixels=187).data
    with pytest.raises(ConversionError, match="pixels"):
        convert_image(data, "PNG", max_pixels=186)


def test_size_rejected_before_decode(monkeypatch):
    import cv2
    data = encoded()
    def unexpected_decode(*args):
        pytest.fail("Pixel decoding must not happen past the pixel limit")
    monkeypatch.setattr(cv2, "imdecode", unexpected_decode)
    with pytest.raises(ConversionError, match="pixels"):
        convert_image(data, "PNG", max_pixels=100)


def test_no_global_safety_mutation():
    from PIL import ImageFile
    before = (Image.MAX_IMAGE_PIXELS, ImageFile.LOAD_TRUNCATED_IMAGES)
    convert_image(encoded(), "WEBP")
    with pytest.raises(ConversionError):
        convert_image(b"bad", "PNG")
    assert (Image.MAX_IMAGE_PIXELS, ImageFile.LOAD_TRUNCATED_IMAGES) == before


def test_webp_codec_missing(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, "haveImageWriter", lambda name: False)
    with pytest.raises(ConversionError, match="codec"):
        convert_image(encoded(), "WEBP")


def test_webp_dimension_limit():
    with pytest.raises(ConversionError, match="16383"):
        convert_image(encoded(size=(16384, 1)), "WEBP")


def test_decompression_bomb():
    def chunk(name, payload):
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload))
    # Header only: no large image is ever allocated.
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 100000, 100000, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", b"") + chunk(b"IEND", b"")
    with pytest.raises(ConversionError):
        convert_image(data, "PNG")


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_truncated_image(fmt):
    data = encoded(fmt, size=(64, 32))
    with pytest.raises(ConversionError):
        convert_image(data[:len(data) // 2], "PNG")


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
def test_animation_rejected(fmt):
    first = Image.new("RGB", (8, 8), "red")
    second = Image.new("RGB", (8, 8), "blue")
    output = BytesIO()
    first.save(output, format=fmt, save_all=True, append_images=[second], duration=100)
    with pytest.raises(ConversionError, match="multiframe"):
        convert_image(output.getvalue(), "PNG")


def test_cmyk_rejected():
    with pytest.raises(ConversionError, match="color mode"):
        convert_image(encoded("JPEG", mode="CMYK"), "PNG")


def test_encoder_failure_is_conversion_error(monkeypatch):
    data = encoded()
    def broken(*args, **kwargs):
        raise OSError("injected encoder failure")
    monkeypatch.setattr(module.image_io, "encode", broken)
    with pytest.raises(ConversionError, match="injected encoder"):
        convert_image(data, "JPEG")
