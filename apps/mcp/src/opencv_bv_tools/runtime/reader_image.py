"""Shared input policy: bounded still images and visible, oriented luminance."""

from .image_io import ImageLimitError, decode, opaque, luminance


def normalize(data, limits, background):
    """Visible pixels only; coordinates after EXIF. No color-profile conversion."""
    image, info, icc = decode(
        data, max_pixels=limits["max_pixels"], max_dimension=limits["max_dimension"]
    )
    notes = []
    if icc:
        notes.append("ICC profile ignored; no color-managed rendering is performed.")
    if image.shape[2] == 4:
        image = opaque(image, background)
        notes.append(
            "Transparency composited onto the specified visible RGB background."
        )
    return luminance(image), info.format, notes
