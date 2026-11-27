# SPDX-License-Identifier: Apache-2.0
"""Image formats supported by the OpenCV search decoder."""

EXTENSIONS = {".jpg", ".jpeg", ".jpe", ".jfif", ".mpo", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".jp2", ".j2k", ".heic", ".heif", ".hif", ".avif"}


def decode_still(raw, max_pixels):
    from .image_io import ImageLimitError, decode, opaque
    try:
        if raw.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8", b"RIFF")):
            image, info, _ = decode(raw, max_pixels=max_pixels, max_dimension=16384,
                                    allow_16=True, allow_cmyk=True)
            if info.gray16:
                # Pillow's historical RGBA conversion clamps 16-bit grayscale,
                # while the shared image decoder scales it for other features.
                import cv2
                import numpy as np
                from .image_io import orient, orientation
                gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
                gray = np.clip(gray, 0, 255).astype(np.uint8)
                image = np.repeat(gray[:, :, None], 3, axis=2)
                image = orient(image, orientation(info.exif) if info.exif is not None else 1)
            return opaque(image)
        return _decode_other(raw, max_pixels)
    except ImageLimitError as exc:
        raise ValueError("image_size_limit") from exc
    except (ValueError, OSError) as exc:
        if str(exc) in {"unsupported_format", "animated_image", "image_size_limit"}:
            raise
        raise ValueError("invalid_image") from exc


def _decode_other(raw, max_pixels):
    """Decode a single supported still using the configured OpenCV codec build."""
    import cv2
    import numpy as np
    from .image_io import opaque

    iso_image = raw[4:8] == b"ftyp" and any(
        brand in raw[8:64] for brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"avif", b"avis")
    )
    if not (iso_image or raw.startswith((b"BM", b"II*\x00", b"MM\x00*", b"GIF87a", b"GIF89a",
                                         b"\x00\x00\x00\x0cjP  \r\n\x87\n", b"\xffO\xffQ"))):
        raise ValueError("invalid_image")
    try:
        ok, pages = cv2.imdecodemulti(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED, range=(0, 2))
    except cv2.error as exc:
        raise ValueError("invalid_image") from exc
    if not ok or not pages:
        raise ValueError("unsupported_format" if iso_image else "invalid_image")
    if len(pages) != 1:
        raise ValueError("animated_image")
    image = pages[0]
    height, width = image.shape[:2]
    if max(height, width) > 16384 or width * height > max_pixels:
        raise ValueError("image_size_limit")
    if image.dtype == np.uint16:
        image = (np.clip(image, 0, 255) if image.ndim == 2 else image // 256).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError("invalid_image")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        image = image[:, :, [2, 1, 0] + ([3] if image.shape[2] == 4 else [])]
    else:
        raise ValueError("invalid_image")
    if raw.startswith((b"II*\x00", b"MM\x00*")) and image.shape[2] == 4:
        # OpenCV's TIFF decoder supplies premultiplied colour channels.
        rgb = image[:, :, :3].astype(np.uint16)
        alpha = image[:, :, 3:4].astype(np.uint16)
        return np.minimum(rgb + 255 - alpha, 255).astype(np.uint8)
    return opaque(image)
