# SPDX-License-Identifier: Apache-2.0
# OCR preprocessing, detection and CTC decoding adapted from RapidOCR 3.10.0
# by SWHL / RapidOCR authors
# (https://github.com/RapidAI/RapidOCR), derived from PaddleOCR.
# Copyright (c) 2020 PaddlePaddle Authors. All rights reserved.
# Modifications: use local OpenCV DNN inference and verified OCR models.
# See LICENSE and THIRD_PARTY_NOTICES.md for licence and attribution.
"""PP-OCRv6 image preparation and decoding with OpenCV DNN.

Keep the detector's polygon expansion and the recognizer's character columns
compatible with the earlier OCR pipeline; redaction uses both for coverage.
"""
from __future__ import annotations
import math
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple
import cv2
import numpy as np
from opencv_bv_tools import _clipper_native

from . import ocr_opencv as assets

class DetPreProcess:
    def __init__(
        self, limit_side_len: int = 736, limit_type: str = "min", mean=None, std=None
    ):
        if mean is None:
            mean = [0.5, 0.5, 0.5]

        if std is None:
            std = [0.5, 0.5, 0.5]

        self.mean = np.array(mean)
        self.std = np.array(std)
        self.scale = 1 / 255.0

        self.limit_side_len = limit_side_len
        self.limit_type = limit_type

    def __call__(self, img: np.ndarray) -> Optional[np.ndarray]:
        resized_img = self.resize(img)
        if resized_img is None:
            return None

        img = self.normalize(resized_img)
        img = self.permute(img)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def normalize(self, img: np.ndarray) -> np.ndarray:
        return (img.astype("float32") * self.scale - self.mean) / self.std

    def permute(self, img: np.ndarray) -> np.ndarray:
        return img.transpose((2, 0, 1))

    def resize(self, img: np.ndarray) -> Optional[np.ndarray]:
        """resize image to a size multiple of 32 which is required by the network"""
        h, w = img.shape[:2]

        if self.limit_type == "max":
            if max(h, w) > self.limit_side_len:
                if h > w:
                    ratio = float(self.limit_side_len) / h
                else:
                    ratio = float(self.limit_side_len) / w
            else:
                ratio = 1.0
        else:
            if min(h, w) < self.limit_side_len:
                if h < w:
                    ratio = float(self.limit_side_len) / h
                else:
                    ratio = float(self.limit_side_len) / w
            else:
                ratio = 1.0

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = int(round(resize_h / 32) * 32)
        resize_w = int(round(resize_w / 32) * 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except Exception as exc:
            raise ResizeImgError from exc

        return img

class ResizeImgError(Exception):
    pass

class DBPostProcess:
    """The post process for Differentiable Binarization (DB)."""

    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.7,
        max_candidates: int = 1000,
        unclip_ratio: float = 2.0,
        score_mode: str = "fast",
        use_dilation: bool = False,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode

        self.dilation_kernel = None
        if use_dilation:
            self.dilation_kernel = np.array([[1, 1], [1, 1]])

    def __call__(
        self, pred: np.ndarray, ori_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[float]]:
        src_h, src_w = ori_shape
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        mask = segmentation[0]
        if self.dilation_kernel is not None:
            mask = cv2.dilate(
                np.array(segmentation[0]).astype(np.uint8), self.dilation_kernel
            )
        boxes, scores = self.boxes_from_bitmap(pred[0], mask, src_w, src_h)
        boxes, scores = self.filter_det_res(boxes, scores, src_h, src_w)
        return boxes, scores

    def boxes_from_bitmap(
        self, pred: np.ndarray, bitmap: np.ndarray, dest_width: int, dest_height: int
    ) -> Tuple[np.ndarray, List[float]]:
        """
        bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        """

        height, width = bitmap.shape

        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes, scores = [], []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue

            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)

            if self.box_thresh > score:
                continue

            box = self.unclip(points)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            boxes.append(box.astype(np.int32))
            scores.append(score)
        return np.array(boxes, dtype=np.int32), scores

    def get_mini_boxes(self, contour: np.ndarray) -> Tuple[np.ndarray, float]:
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0

        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = np.array(
            [points[index_1], points[index_2], points[index_3], points[index_4]]
        )
        return box, min(bounding_box[1])

    @staticmethod
    def box_score_fast(bitmap: np.ndarray, _box: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def box_score_slow(self, bitmap: np.ndarray, contour: np.ndarray) -> float:
        """use polyon mean score as the mean score"""
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def unclip(self, box: np.ndarray) -> np.ndarray:
        vertices = [(float(x), float(y)) for x, y in box]
        if 0 < len(vertices) < 3:
            raise ValueError("A linearring requires at least 4 coordinates.")
        if vertices and vertices[-1] != vertices[0]:
            vertices.append(vertices[0])

        twice_area = 0.0
        if vertices:
            origin_x = vertices[0][0]
            for before, vertex, after in zip(vertices, vertices[1:], vertices[2:]):
                twice_area += (vertex[0] - origin_x) * (before[1] - after[1])
        area = abs(twice_area / 2.0)
        perimeter = 0.0
        for start, stop in zip(vertices, vertices[1:]):
            dx, dy = stop[0] - start[0], stop[1] - start[1]
            perimeter += math.sqrt(dx * dx + dy * dy)
        distance = area * self.unclip_ratio / perimeter

        expanded = np.array(_clipper_native.offset(box, distance)).reshape((-1, 1, 2))
        return expanded

    def filter_det_res(
        self, dt_boxes: np.ndarray, scores: List[float], img_height: int, img_width: int
    ) -> Tuple[np.ndarray, List[float]]:
        dt_boxes_new, new_scores = [], []
        for box, score in zip(dt_boxes, scores):
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)

            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue

            dt_boxes_new.append(box)
            new_scores.append(score)
        return np.array(dt_boxes_new), new_scores

    def order_points_clockwise(self, pts: np.ndarray) -> np.ndarray:
        """
        reference from:
        https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(
        self, points: np.ndarray, img_height: int, img_width: int
    ) -> np.ndarray:
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

class Model:
    def __init__(self, model_path):
        path = Path(model_path)
        digest = next((digest for name, digest in assets.MODELS.values() if name == path.name), None)
        if digest is None:
            raise ValueError("Unknown OCR model")
        data = assets._verified_bytes(path, digest)
        model = np.frombuffer(data, dtype=np.uint8)
        self.net = cv2.dnn.readNetFromONNX(model)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    def __call__(self, data):
        self.net.setInput(data, 'x')
        return self.net.forward('fetch_name_0')

class Detector:
    """Locate text regions using the bundled detector and polygon postprocessing."""
    def __init__(self, path):
        self.session = Model(path)
        self.postprocess_op = DBPostProcess(thresh=.3, box_thresh=.3, unclip_ratio=1.6, use_dilation=True)
    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        tensor = DetPreProcess(736, 'min', [.5]*3, [.5]*3)(img)
        if tensor is None:
            return None
        boxes, _ = self.postprocess_op(self.session(tensor), (img_h, img_w))
        if not len(boxes):
            return None
        y = boxes[:, 0, 1]
        order = np.argsort(y, kind='stable')
        boxes = boxes[order]
        groups = np.concatenate([[0], np.cumsum((np.diff(y[order]) >= 10).astype(np.int32))])
        return boxes[np.lexsort((boxes[:, 0, 0], groups))]

class Recognizer:
    """Decode one text crop and retain the model columns for each character."""
    def __init__(self, path, characters):
        data = assets._verified_bytes(Path(characters), assets.CHARACTERS_SHA256)
        dictionary = data.decode('utf-8').splitlines()
        if len(dictionary) != assets.CHARACTER_COUNT:
            raise RuntimeError('OCR character dictionary has an unexpected size.')
        self.characters = ['blank', *dictionary, ' ']
        self.session = Model(path)
    def __call__(self, image):
        h, w = image.shape[:2]
        ratio = w / float(h)
        max_ratio = max(320 / 48, ratio)
        target_w = int(48 * max_ratio)
        resized_w = min(target_w, int(math.ceil(48 * ratio)))
        image = cv2.resize(image, (resized_w, 48)).astype('float32')
        image = image.transpose((2, 0, 1)) / 255
        image -= .5
        image /= .5
        tensor = np.zeros((1, 3, 48, target_w), dtype=np.float32)
        tensor[0, :, :, :resized_w] = image
        pred = self.session(tensor)
        if pred.ndim != 3 or pred.shape[-1] != len(self.characters):
            raise RuntimeError('OCR recognizer output does not match the character dictionary.')
        ids = pred.argmax(axis=2)[0]
        select = np.ones(len(ids), dtype=bool)
        select[1:] = ids[1:] != ids[:-1]
        select &= ids != 0
        chars = [self.characters[i] for i in ids[select]]
        cols = np.where(select)[0]
        words = [[int(col) for char, col in zip(chars, cols) if not char.isspace()]]
        return SimpleNamespace(txts=[''.join(chars)], word_results=[SimpleNamespace(
            word_cols=words, line_txt_len=len(ids) * ratio / max_ratio)])

class NativeOCR:
    """Load detection and recognition models only when each is first needed."""
    def __init__(self, models, characters):
        self.models = Path(models) if models is not None else None
        self.characters = Path(characters)
        self.text_det = None
        self.text_rec = None

    def model_path(self, name):
        if self.models is not None:
            return self.models / name
        from opencv_bv_tools.model_assets import model_path
        return model_path(name)
    @staticmethod
    def preprocess(image):
        h, w = image.shape[:2]
        rh = rw = 1.0
        if max(h, w) > 100000:
            ratio = 100000 / max(h, w)
            nh, nw = int(round(int(h * ratio) / 32) * 32), int(round(int(w * ratio) / 32) * 32)
            if nh <= 0 or nw <= 0:
                raise ValueError('invalid preprocessing dimensions')
            image = cv2.resize(image, (nw, nh))
            rh, rw = h / nh, w / nw
        h, w = image.shape[:2]
        if min(h, w) < 30:
            ratio = 30 / min(h, w)
            nh, nw = int(round(int(h * ratio) / 32) * 32), int(round(int(w * ratio) / 32) * 32)
            if nh <= 0 or nw <= 0:
                raise ValueError('invalid preprocessing dimensions')
            image = cv2.resize(image, (nw, nh))
            rh, rw = h / nh, w / nw
        return image, rh, rw
    def __call__(self, image, *, use_det, use_cls, use_rec, return_word_box=False):
        if use_cls or use_det == use_rec:
            raise ValueError("Only detection or recognition is supported; classification is unavailable.")
        original_h, original_w = image.shape[:2]
        image, rh, rw = self.preprocess(image)
        if use_det:
            if self.text_det is None:
                self.text_det = Detector(self.model_path('PP-OCRv6_det_small.onnx'))
            h, w = image.shape[:2]
            pad = 0
            if h <= 30 or w / h > 8:
                pad = int(abs(max(int(w / 8), 30) * 2 - h) / 2)
                image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            boxes = self.text_det(image)
            if boxes is not None:
                boxes[:, :, 1] -= pad
                boxes[:, :, 0] *= rw
                boxes[:, :, 1] *= rh
                boxes = np.maximum(boxes, 0)
                boxes[..., 0] = np.minimum(boxes[..., 0], original_w)
                boxes[..., 1] = np.minimum(boxes[..., 1], original_h)
            return SimpleNamespace(boxes=boxes)
        if use_rec:
            if self.text_rec is None:
                self.text_rec = Recognizer(self.model_path('PP-OCRv6_rec_small.onnx'), self.characters)
            return self.text_rec(image)
        raise ValueError('expected exactly one OCR task')
