# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Shared conservative still-image normalization to visible, oriented RGB."""

from .image_io import decode, opaque, ImageLimitError as ImageInputLimitError
from opencv_bv_tools.runtime.png_structure import parse_png

COLOR_CHUNKS = {b"iCCP", b"gAMA", b"cHRM", b"cICP", b"mDCV", b"cLLI"}


def normalize_opaque_image(data, limits):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        structure = parse_png(
            data, {**limits, "max_chunks": 4096, "max_metadata_bytes": 1024 * 1024}
        )
        if any(kind in COLOR_CHUNKS for kind, _, _ in structure.chunks):
            raise ValueError("color_profile_not_supported")
    image, info, icc = decode(
        data, max_pixels=limits["max_pixels"], max_dimension=limits["max_dimension"]
    )
    if icc:
        raise ValueError("color_profile_not_supported")
    return opaque(image), info.format
