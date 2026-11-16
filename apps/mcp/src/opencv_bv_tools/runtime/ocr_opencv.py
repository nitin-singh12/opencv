# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""CPU inference for the bundled OCR models; RapidOCR owns pre/postprocessing.

Sessions are private to a resident worker, whose requests are serialized. The
worker/CLI sets OpenCV's process-wide thread limit before constructing OCR.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np


MODELS = {
    "det": ("PP-OCRv6_det_small.onnx", "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"),
    "rec": ("PP-OCRv6_rec_small.onnx", "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
}
CHARACTERS = Path(__file__).resolve().parents[1] / "models/ppocrv6-characters.txt"
CHARACTERS_SHA256 = "f7aa897ca828a4c7c9e2739c30f9161a33306d532f020bcdb91dcfb664a5507e"
CHARACTER_COUNT = 18708


def _verified_bytes(path, digest):
    try:
        data = path.read_bytes()
    except OSError:
        data = None
    if data is None or hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError(f"OCR asset {path.name} is missing or has changed. Reinstall the package.")
    return data


def _model_bytes(role):
    import rapidocr
    name, digest = MODELS[role]
    path = Path(rapidocr.__file__).parent / "models" / name
    return _verified_bytes(path, digest)


def _characters():
    # This matches RapidOCR's original ONNX metadata reader exactly: preserve
    # character order and whitespace. Its CTC decoder adds blank and space.
    characters = _verified_bytes(CHARACTERS, CHARACTERS_SHA256).decode("utf-8").splitlines()
    if len(characters) != CHARACTER_COUNT:
        raise RuntimeError("OCR character dictionary has an unexpected size.")
    return characters


def check_models():
    """Verify the installed weights/dictionary without fetching replacements."""
    for role in MODELS:
        _model_bytes(role)
    _characters()


class OpenCVSession:
    """RapidOCR session interface for pinned PP-OCRv6 small Chinese models."""

    def __init__(self, cfg):
        self.role = cfg.task_type.value
        if (self.role not in MODELS or cfg.ocr_version.value != "PP-OCRv6"
                or cfg.model_type.value != "small" or cfg.lang_type.value != "ch"):
            raise ValueError("OpenCV OCR supports only the bundled v6-small detector and recognizer.")
        self.characters = _characters() if self.role == "rec" else None
        # Python reads Unicode paths consistently; OpenCV receives the same
        # verified bytes directly instead of opening a platform-specific path.
        model = np.frombuffer(_model_bytes(self.role), dtype=np.uint8)
        self.net = cv2.dnn.readNetFromONNX(model)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def __call__(self, tensor):
        self.net.setInput(tensor, "x")
        output = self.net.forward("fetch_name_0")
        if self.role == "rec" and (output.ndim != 3 or output.shape[-1] != CHARACTER_COUNT + 2):
            raise RuntimeError("OCR recognizer output does not match the character dictionary.")
        return output

    def have_key(self, key="character"):
        return key == "character" and self.characters is not None

    def get_character_list(self, key="character"):
        if not self.have_key(key):
            raise KeyError(key)
        return self.characters.copy()
