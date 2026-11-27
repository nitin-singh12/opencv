"""Search codecs accept stills without importing Pillow in the worker."""
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
import pytest

from opencv_bv_tools.runtime.search_images import decode_still


@pytest.mark.parametrize("format_name", ["PNG", "JPEG", "WEBP", "BMP", "TIFF", "GIF", "JPEG2000"])
def test_common_still_formats(format_name):
    image = Image.new("RGB", (64, 48), (220, 30, 12))
    raw = BytesIO()
    try:
        image.save(raw, format=format_name)
    except OSError:
        pytest.skip("Test image encoder is unavailable")
    actual = decode_still(raw.getvalue(), max_pixels=10000)
    reference = np.asarray(ImageOps.exif_transpose(Image.open(BytesIO(raw.getvalue()))).convert("RGB"))
    assert actual.shape == reference.shape
    assert np.max(np.abs(actual.astype(int) - reference.astype(int))) <= 2


def test_transparency_composites_white():
    image = Image.new("RGBA", (8, 8), (100, 20, 200, 0))
    raw = BytesIO()
    image.save(raw, format="PNG")
    assert (decode_still(raw.getvalue(), 1000) == 255).all()


@pytest.mark.parametrize("color", [(20, 120, 220, 128), (250, 10, 80, 0), (1, 200, 50, 255)])
def test_transparent_tiff_matches_previous_decoder(color):
    raw = BytesIO()
    Image.new("RGBA", (8, 8), color).save(raw, format="TIFF")
    source = Image.open(BytesIO(raw.getvalue())).convert("RGBA")
    reference = Image.new("RGB", source.size, "white")
    reference.paste(source, mask=source.getchannel("A"))
    actual = decode_still(raw.getvalue(), 1000)
    assert np.max(np.abs(actual.astype(int) - np.asarray(reference).astype(int))) <= 1


@pytest.mark.parametrize("format_name", ["PNG", "TIFF"])
def test_grayscale_16bit_matches_previous_decoder(format_name):
    image = Image.fromarray(np.array([[0, 1, 127, 255, 256, 1024, 65535]], dtype=np.uint16))
    raw = BytesIO()
    image.save(raw, format=format_name)
    expected = np.asarray(Image.open(BytesIO(raw.getvalue())).convert("RGB"))
    assert np.array_equal(decode_still(raw.getvalue(), 1000), expected)


def test_cmyk_jpeg_matches_previous_decoder():
    raw = BytesIO()
    Image.new("CMYK", (8, 8), (20, 70, 130, 40)).save(raw, format="JPEG")
    expected = np.asarray(Image.open(BytesIO(raw.getvalue())).convert("RGB"))
    assert np.max(np.abs(decode_still(raw.getvalue(), 1000).astype(int) - expected.astype(int))) <= 2


def test_animated_gif_rejected():
    raw = BytesIO()
    Image.new("RGB", (8, 8), "red").save(raw, format="GIF", save_all=True,
                                             append_images=[Image.new("RGB", (8, 8), "blue")])
    with pytest.raises(ValueError, match="animated_image"):
        decode_still(raw.getvalue(), 1000)


def test_image_size_limit():
    raw = BytesIO()
    Image.new("RGB", (32, 32)).save(raw, format="PNG")
    with pytest.raises(ValueError, match="image_size_limit"):
        decode_still(raw.getvalue(), 100)
