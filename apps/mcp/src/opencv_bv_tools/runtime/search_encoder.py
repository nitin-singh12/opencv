# SPDX-License-Identifier: Apache-2.0
"""Verified, offline FG-CLIP2 embeddings through OpenCV DNN."""
from hashlib import sha256
import json
import os
from pathlib import Path

from . import safe_files

from ..model_assets import SEARCH_MODELS

ASSETS = {name: (size, digest) for name, (size, digest, _) in SEARCH_MODELS.items()}


def snapshot_assets(source_dir, destination, allowed_dirs):
    """Copy verified bytes to a private worker snapshot before native parsing."""
    source_dir = Path(source_dir).expanduser().resolve(strict=True)
    allowed = safe_files.Folders.from_canonical(allowed_dirs)
    allowed.check(source_dir, "The configured model folder")
    source_root = safe_files.Folders.from_canonical((source_dir,))
    destination = Path(destination) / "model"
    destination.mkdir(mode=0o700)
    (destination / "model").mkdir(mode=0o700)
    for name, (expected_size, expected_hash) in ASSETS.items():
        source, real, _ = safe_files.open_input(source_dir / name, allowed)
        with source:
            source_root.check(real, "The model file")
            if os.fstat(source.fileno()).st_size != expected_size:
                raise ValueError("fgclip2_integrity_error")
            digest = sha256()
            total = 0
            with (destination / name).open("xb") as target:
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > expected_size:
                        raise ValueError("fgclip2_integrity_error")
                    digest.update(chunk)
                    target.write(chunk)
            if total != expected_size or digest.hexdigest() != expected_hash:
                raise ValueError("fgclip2_integrity_error")
    return destination


class Encoder:
    def __init__(self, options):
        import cv2
        import numpy as np
        from .fgclip_opencv import Encoder as OpenCVEncoder

        if options["backend"] != "fgclip2" or options["device"] != "cpu":
            raise ValueError("unsupported_search_backend")
        assets = snapshot_assets(options["model_dir"], options["snapshot_dir"], options["allowed_dirs"])
        self.model = OpenCVEncoder(assets)
        cv2.setNumThreads(options["threads"])
        contract = {"assets": ASSETS, "backend": "fgclip2-opencv-dnn", "precision": "fp32",
                    "opencv": cv2.__version__, "numpy": np.__version__,
                    "image_preprocess": "rgb-white-exif-opencv-v1", "text": "lowercase-bpe-short-v1"}
        self.info = {"fingerprint": sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
                     "backend": "fgclip2-opencv-dnn", "device": "cpu", "precision": "fp32",
                     "model_revision": "430fbc8a912c86fd4de601381b6245a0edab22f0"}

    def encode(self, raw, kind, max_pixels):
        if kind == "text":
            vector, truncated = self.model.encode_text(raw.decode("utf-8"))
        elif kind == "image":
            from .search_images import decode_still
            vector = self.model.encode_rgb(decode_still(raw, max_pixels))
            truncated = False
        else:
            raise ValueError("invalid_request")
        return {"vector": vector.reshape(-1).tolist(), "truncated": truncated, **self.info}
