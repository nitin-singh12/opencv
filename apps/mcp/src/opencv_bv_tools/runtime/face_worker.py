"""Private local CPU pipeline. Never reads arbitrary model paths or downloads."""

import base64
import hashlib
from importlib import resources
from importlib.metadata import version
import json
import math
from .process_support import peak_rss_kib
import sys
import time

from .image_io import encode

from opencv_bv_tools.face_blurring import FaceLimits
from opencv_bv_tools.face_blurring import MODEL_NAME
from opencv_bv_tools.face_blurring import MODEL_SHA256
from opencv_bv_tools.face_blurring import QUALIFICATION
from opencv_bv_tools.face_blurring import _failure
from opencv_bv_tools.runtime.png_structure import PNGResourceError
from opencv_bv_tools.runtime.opaque_image import COLOR_CHUNKS
from opencv_bv_tools.runtime.opaque_image import (
    ImageInputLimitError as FaceResourceError,
)
from opencv_bv_tools.runtime.opaque_image import normalize_opaque_image as normalize

MAX_CANDIDATES = 512
SCALES = (320, 640)


def inverse_box(box, size, rotation, resized):
    """Map a top-left-letterboxed view's box back before rounding/expansion."""
    w, h = size
    rw, rh = (h, w) if rotation % 2 else (w, h)
    x0, y0, x1, y1 = box
    x0, x1 = x0 * rw / resized[0], x1 * rw / resized[0]
    y0, y1 = y0 * rh / resized[1], y1 * rh / resized[1]
    if rotation == 0:
        return (x0, y0, x1, y1)
    if rotation == 1:
        return (w - y1, x0, w - y0, x1)
    if rotation == 2:
        return (w - x1, h - y1, w - x0, h - y0)
    return (y0, h - x1, y1, h - x0)


def expand_box(box, size):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * 0.35, (y1 - y0) * 0.35
    box = [
        max(0, math.floor(x0 - dx)),
        max(0, math.floor(y0 - dy)),
        min(size[0], math.ceil(x1 + dx)),
        min(size[1], math.ceil(y1 + dy)),
    ]
    return box if box[0] < box[2] and box[1] < box[3] else None


def merge_regions(boxes):
    """Deterministic transitive bounding-rectangle union, including edge touch."""
    pending = sorted([list(b) for b in boxes])
    output = []
    while pending:
        box = pending.pop(0)
        while True:
            hits = [
                b
                for b in pending
                if b[0] <= box[2]
                and box[0] <= b[2]
                and b[1] <= box[3]
                and box[1] <= b[3]
            ]
            if not hits:
                break
            box = [
                min(b[0] for b in [box] + hits),
                min(b[1] for b in [box] + hits),
                max(b[2] for b in [box] + hits),
                max(b[3] for b in [box] + hits),
            ]
            pending = [b for b in pending if b not in hits]
        output.append(box)
    # An expanded component can reach one already emitted; repeat to fixed point.
    if len(output) < len(boxes):
        return merge_regions(output)
    return sorted(output, key=lambda b: (b[1], b[0], b[3], b[2]))


def model_bytes():
    data = (
        resources.files("opencv_bv_tools").joinpath("models", MODEL_NAME).read_bytes()
    )
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("model_hash_mismatch")
    return data


def detect_regions(image, cv, np, model):
    detector = cv.FaceDetectorYN.create(
        "onnx",
        np.frombuffer(model, dtype=np.uint8),
        np.empty(0, dtype=np.uint8),
        (320, 320),
        0.6,
        0.3,
        10000,
        cv.dnn.DNN_BACKEND_OPENCV,
        cv.dnn.DNN_TARGET_CPU,
    )
    boxes = []
    for side in SCALES:
        detector.setInputSize((side, side))
        for rotation in range(4):
            view = np.rot90(image, rotation)
            height, width = view.shape[:2]
            scale = side / max(width, height)
            resized = (max(1, round(width * scale)), max(1, round(height * scale)))
            view = resize_bilinear(view, resized)
            canvas = np.full((side, side, 3), 255, np.uint8)
            canvas[: resized[1], : resized[0]] = view
            found = detector.detect(canvas[:, :, ::-1].copy())[1]
            if found is None:
                continue
            if found.ndim != 2 or found.shape[1] != 15 or not np.isfinite(found).all():
                raise ValueError("invalid_detector_result")
            if len(boxes) + len(found) > MAX_CANDIDATES:
                raise FaceResourceError("candidate_limit")
            for row in found:
                x, y, w, h = map(float, row[:4])
                score = float(row[14])
                if w <= 0 or h <= 0 or not 0.6 <= score <= 1:
                    raise ValueError("invalid_detector_result")
                # Clip padding before mapping; do not magnify out-of-image boxes.
                box = (
                    max(0, x),
                    max(0, y),
                    min(resized[0], x + w),
                    min(resized[1], y + h),
                )
                if box[0] >= box[2] or box[1] >= box[3]:
                    continue
                expanded = expand_box(
                    inverse_box(
                        box, (image.shape[1], image.shape[0]), rotation, resized
                    ),
                    (image.shape[1], image.shape[0]),
                )
                if expanded is not None:
                    boxes.append(expanded)
    return merge_regions(boxes), len(boxes)


def resize_bilinear(image, size):
    """Antialiased separable bilinear with the established 22-bit rounding.

    This preserves the model's existing RGB8 inputs, including downsampling.
    cv.resize(INTER_LINEAR) has different downsampling/rounding semantics.
    Coefficients are bounded by the input axis; processing is one output slice
    at a time so no full floating-point image or weight matrix is allocated.
    """
    import numpy as np

    result = image
    for axis, target in ((1, size[0]), (0, size[1])):
        source = result.shape[axis]
        if source == target:
            continue
        scale = source / target
        support = max(1.0, scale)
        shape = list(result.shape)
        shape[axis] = target
        output = np.empty(shape, np.uint8)
        for position in range(target):
            center = (position + 0.5) * scale
            left = max(0, int(center - support + 0.5))
            right = min(source, int(center + support + 0.5))
            weights = [
                max(0.0, 1 - abs((x - center + 0.5) / support))
                for x in range(left, right)
            ]
            total = sum(weights)
            fixed = np.array(
                [int(w / total * (1 << 22) + 0.5) for w in weights], np.int64
            )
            if axis == 1:
                values = np.einsum(
                    "hkc,k->hc", result[:, left:right].astype(np.int64), fixed
                )
                output[:, position] = np.clip(
                    (values + (1 << 21)) >> 22, 0, 255
                ).astype(np.uint8)
            else:
                values = np.einsum(
                    "kwc,k->wc", result[left:right].astype(np.int64), fixed
                )
                output[position] = np.clip((values + (1 << 21)) >> 22, 0, 255).astype(
                    np.uint8
                )
        result = output
    return result


def apply_blur(image, regions):
    import cv2

    output = image.copy()
    for x0, y0, x1, y1 in regions:
        crop = image[y0:y1, x0:x1]
        height, width = crop.shape[:2]
        tiny = cv2.resize(
            crop, (min(8, width), min(8, height)), interpolation=cv2.INTER_AREA
        )
        enlarged = cv2.resize(tiny, (width, height), interpolation=cv2.INTER_LINEAR)
        output[y0:y1, x0:x1] = cv2.GaussianBlur(
            enlarged, (0, 0), max(2, min(width, height) / 8)
        )
    return output


def process(data, options):
    limits = options["limits"]
    start = time.monotonic()
    try:
        image, source_format = normalize(data, limits)
    except ImportError:
        return _failure("missing_face_dependency")
    except (FaceResourceError, PNGResourceError) as exc:
        result = _failure(str(exc))
        result["status"] = "resource_limited"
        return result
    except (ValueError, OSError, SyntaxError, EOFError) as exc:
        return _failure(
            "color_profile_not_supported"
            if str(exc) == "color_profile_not_supported"
            else "invalid_or_unsupported_image"
        )
    try:
        import cv2 as cv
        import numpy as np
    except ImportError:
        return _failure("missing_face_dependency")
    if version("opencv-python-headless") != "5.0.0.93":
        return _failure("unsupported_detector_version")
    try:
        model = model_bytes()
    except (OSError, ValueError):
        return _failure("missing_or_invalid_model")
    try:
        cv.setNumThreads(1)
        cv.ocl.setUseOpenCL(False)
        if cv.dnn.DNN_TARGET_CPU not in cv.dnn.getAvailableTargets(
            cv.dnn.DNN_BACKEND_OPENCV
        ):
            return _failure("cpu_backend_unavailable")
        regions, candidates = detect_regions(image, cv, np, model)
        if len(regions) > limits["max_regions"]:
            raise FaceResourceError("region_limit")
        if not regions:
            result = _failure("no_faces_detected")
            result.update(
                status="no_faces", image_size=[image.shape[1], image.shape[0]]
            )
            return result
        output = apply_blur(image, regions)
        encoded = encode(output)
        if len(encoded) > limits["max_output_bytes"]:
            raise FaceResourceError("output_byte_limit")
    except FaceResourceError as exc:
        result = _failure(str(exc))
        result["status"] = "resource_limited"
        return result
    except Exception:
        return _failure("detector_or_processing_error")
    return dict(
        status="blurred",
        stop_reason="configured_search_completed",
        data_base64=base64.b64encode(encoded).decode(),
        image_size=[image.shape[1], image.shape[0]],
        input_format=source_format,
        regions=regions,
        coordinate_space="exif_normalized_image",
        rectangle_semantics="left_top_right_bottom_half_open",
        region_count=len(regions),
        candidate_count=candidates,
        qualification=QUALIFICATION,
        output_bytes=len(encoded),
        output_sha256=hashlib.sha256(encoded).hexdigest(),
        backend={
            "name": "YuNet",
            "opencv": version("opencv-python-headless"),
            "model_sha256": MODEL_SHA256,
            "scales": list(SCALES),
            "rotations": [0, 90, 180, 270],
            "score_threshold": 0.6,
            "nms_threshold": 0.3,
            "top_k": 10000,
            "target": "CPU",
            "threads": 1,
        },
        worker_seconds=time.monotonic() - start,
        worker_peak_rss_kib=peak_rss_kib(),
    )


def main():
    try:
        header = sys.stdin.buffer.readline(4097)
        if len(header) > 4096 or not header.endswith(b"\n"):
            raise ValueError("Invalid header")
        options = json.loads(header)
        FaceLimits(**options["limits"]).validate()
        data = sys.stdin.buffer.read(options["limits"]["max_input_bytes"] + 1)
        if not data or len(data) > options["limits"]["max_input_bytes"]:
            raise ValueError("Invalid input")
        result = process(data, options)
    except Exception:
        result = _failure("worker_error")
    encoded = json.dumps(
        result, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode()
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
