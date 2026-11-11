#!/usr/bin/env python3
"""Find email addresses in a screenshot and cover them with solid boxes.

The pipeline has three steps:
  1. Find the text lines with RapidOCR's detector (small ONNX model, CPU
     only), then read every line twice from padded crops: once in colour and
     once normalised to grey. Each reading gives rough character positions.
  2. Look for valid addresses in each visual row of text, in both readings,
     including addresses that wrap onto the next line.
  3. Snap each address box to the real ink pixels, so anti-aliased glyph edges
     are covered and neighbouring characters are left alone.

Usage:
  python -m opencv_bv_tools.runtime.image_redact input.png output.png [--mode accurate|fast] [--overlay check.png] [--overwrite]
The input is never changed, and an existing output is only replaced with --overwrite.
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from . import image_io

# --------------------------------------------------------------------------
# Address pattern
# --------------------------------------------------------------------------

# Dot-atom local part, then a domain made of hostname labels and a letter TLD.
# The RFC also allows ! # $ & ' * / = ? ^ ` { | } ~ in the local part, but
# real addresses almost never use them, and allowing them lets an address
# swallow text OCR ran into it ("/8369100100/rohit.magar@ril.com").
LOCAL = r"[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+)*"
LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
EMAIL_RE = re.compile(rf"(?<![A-Za-z0-9.@_%+-])({LOCAL})@((?:{LABEL}\.)+[A-Za-z]{{2,24}})(?![A-Za-z0-9@_-])")
FULL_EMAIL_RE = re.compile(rf"{LOCAL}@(?:{LABEL}\.)+[A-Za-z]{{2,24}}")

# Padding around each box, as a fraction of the OCR line height.
PAD = 0.25

# Characters that can sit inside an address. Used to find the pieces of an
# address that wraps from the end of one row to the start of the next.
ADDR_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_%+-.@")


def find_emails(text):
    """Return (start, end) character spans of valid addresses in text."""
    spans = []
    for m in EMAIL_RE.finditer(text):
        start, end = m.span()
        # "user@example.test.": a sentence full stop is fine, but a dot that
        # runs straight into more address characters means the match is cut.
        if end < len(text) and text[end] == "." and end + 1 < len(text) and text[end + 1] in ADDR_CHARS:
            continue
        spans.append((start, end))
    return spans


# A space OCR put next to the "@" ("anna @example.com"), as it does on some
# PDF fonts where the "@" has wide margins.
_SPACE_AT = re.compile(r"(?<=[A-Za-z0-9_%+.-]) (?=@)|(?<=@) (?=[A-Za-z0-9])")
# Several names sharing one domain, as research papers write their authors:
# "{anna, bob.lee}@example.org". OCR may read the braces as ( ) or [ ].
GROUPED_RE = re.compile(rf"[{{(\[]\s*{LOCAL}(?:\s*[,;|]\s*{LOCAL})*\s*[}})\]]\s?@(?:{LABEL}\.)+[A-Za-z]{{2,24}}"
                        rf"(?![A-Za-z0-9@_-])")


def find_emails_pdf(text):
    """find_emails() that also accepts what PDFs show: one space on either
    side of the "@", and grouped addresses ("{anna, bob}@example.org").
    Spans index the original text."""
    drop = {m.start() for m in _SPACE_AT.finditer(text)}
    keep = [i for i in range(len(text)) if i not in drop]
    joined = "".join(text[i] for i in keep)
    spans = {(keep[s], keep[e - 1] + 1) for s, e in find_emails(joined)}
    spans |= {m.span() for m in GROUPED_RE.finditer(text)}
    return sorted(spans)


def normalise_ocr_text(text):
    """Undo OCR substitutions that only ever happen around addresses."""
    return text.replace("＠", "@").replace("．", ".")


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------

@dataclass
class Char:
    ch: str
    x0: float
    x1: float


@dataclass
class Line:
    """One OCR text line: its box and the characters inside it."""
    x0: float
    y0: float
    x1: float
    y1: float
    chars: list = field(default_factory=list)
    aligned: bool = False
    low_contrast: bool = False  # found only by the faint-text pass
    src: object = field(default=None, repr=False, compare=False)  # detected line a reading came from

    @property
    def height(self):
        return self.y1 - self.y0


# Model sets RapidOCR can load: (detector, recogniser) as (version, size, language).
MODEL_SETS = {
    "v6-small": (("PP-OCRv6", "small", "ch"), ("PP-OCRv6", "small", "ch")),
    "v6-tiny": (("PP-OCRv6", "tiny", "ch"), ("PP-OCRv6", "tiny", "ch")),
    "v5-mobile": (("PP-OCRv5", "mobile", "ch"), ("PP-OCRv5", "mobile", "ch")),
    "v4-mobile-en": (("PP-OCRv4", "mobile", "ch"), ("PP-OCRv4", "mobile", "en")),
}


# The default models ship inside the rapidocr 3.10.0 wheel. They are checked
# before loading, so RapidOCR never fetches replacements at run time.
MODEL_SHA256 = {
    "PP-OCRv6_det_small.onnx": "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    "PP-OCRv6_rec_small.onnx": "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
}


def check_models():
    """Raise RuntimeError if the v6-small model files are missing or changed."""
    import rapidocr
    folder = Path(rapidocr.__file__).parent / "models"
    for name, digest in MODEL_SHA256.items():
        try:
            actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        except OSError:
            actual = None
        if actual != digest:
            raise RuntimeError(f"The OCR model {folder / name} is missing or has changed. "
                               "Run install.sh again to restore it.")


def _normalise(crop):
    """Grey, dark text on a light background, full contrast range. Gives the
    recogniser a second view that ignores colour and theme."""
    grey = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    values, counts = np.unique(grey, return_counts=True)
    if values[counts.argmax()] < 128:
        grey = 255 - grey
    lo, hi = int(grey.min()), int(grey.max())
    if hi > lo:
        grey = ((grey.astype(np.float32) - lo) * (255.0 / (hi - lo))).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)


def local_contrast(rgb, clip=3.0, tile=128):
    """CLAHE on lightness only, with tiles of about `tile` px. Lifts faint
    text (light orange on grey, a blue link on navy) enough for the detector
    to find it, and keeps the colours."""
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    grid = (max(1, round(w / tile)), max(1, round(h / tile)))
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# Faint-text pass. Two extra views of the screenshot, each searched for text
# lines the normal pass missed:
#  - local_contrast (CLAHE on lightness) for text slightly lighter or darker
#    than its background, such as a blue link on navy;
#  - channel_difference for text that differs mostly in colour, such as pale
#    orange on grey: the largest per-channel distance from the local
#    background, drawn dark on white.
# Boxes for lines found this way are fitted to the channel difference, because
# faint glyphs fall below the normal brightness ink threshold.
CH_WINDOW = 31    # median window for the local background, px
CH_FLOOR = 6.0    # differences below this are treated as JPEG noise
CH_GAIN = 4.0     # how dark a given difference looks in the view
CH_INK_MIN = 12   # lowest channel difference counted as ink when boxing


def channel_difference(rgb):
    """Returns (view, diff): diff is, per pixel, the largest R, G or B distance
    from the local background (median of a CH_WINDOW square); view shows it
    as dark text on white."""
    bg = np.stack([cv2.medianBlur(np.ascontiguousarray(rgb[..., k]), CH_WINDOW) for k in range(3)], axis=-1)
    diff = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(axis=2).astype(np.float32)
    grey = (255 - np.clip((diff - CH_FLOOR) * CH_GAIN, 0, 255)).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=2), diff


def merged_view(rgb, kind):
    """One image standing in for both faint-text views, so the detector runs
    once instead of twice. "clahe-diff": channel difference of the CLAHE copy.
    "stack": CLAHE lightness, channel difference and original grey as the
    three colour channels of one image."""
    clahe = local_contrast(rgb)
    if kind == "clahe-diff":
        return channel_difference(clahe)[0]
    if kind == "stack":
        light = cv2.cvtColor(clahe, cv2.COLOR_RGB2LAB)[..., 0]
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return np.ascontiguousarray(np.dstack([light, channel_difference(rgb)[0][..., 0], grey]))
    raise ValueError(kind)


# A first reading "looks address-like" when it has an @ or a character OCR
# confuses with one, a domain-like word, or ends in a hyphen (an address can
# wrap at a hyphen). Only such lines are read a second time when
# gate_second_read is on.
_AT_LIKE = re.compile(r"[@＠©®]|\(at\)|\[at\]", re.IGNORECASE)
_DOMAIN_LIKE = re.compile(r"[A-Za-z0-9-]{2,}\.[A-Za-z]{2,}")


def address_like(text):
    t = text.rstrip()
    return bool(_AT_LIKE.search(t) or _DOMAIN_LIKE.search(t) or t.endswith("-"))


def _text(line):
    return "".join(" " if c is None else c.ch for c in line.chars)


def _already_found(line, lines):
    """True when an existing line covers at least half of this one."""
    area = max(1.0, (line.x1 - line.x0) * (line.y1 - line.y0))
    for other in lines:
        iw = min(line.x1, other.x1) - max(line.x0, other.x0)
        ih = min(line.y1, other.y1) - max(line.y0, other.y0)
        if iw > 0 and ih > 0 and iw * ih >= 0.5 * area:
            return True
    return False


class Ocr:
    """RapidOCR finds the text lines; each line is then read twice from a
    padded crop, once in colour and once normalised to grey. Two independent
    readings catch most single misreads (9 px text, odd colours).

    contrast_pass=True runs the detector a second time on a contrast-enhanced
    copy (local_contrast) and adds the lines only that pass finds. Reading
    and box snapping still use the original pixels.

    size_guard=True keeps every detector input in the size range the
    benchmark covers and bounded in memory: strips wider than TILE_ABOVE are
    read in overlapping tiles (on wide, mostly empty 4K and 5K screenshots
    the detector otherwise cuts lines short), thin strips are padded rather
    than blown up by RapidOCR, and a strip dense enough to reach the
    detector's limit of 1000 candidate lines is read again without it.
    Off by default, so benchmark runs stay reproducible; the server and the
    command line turn it on.

    pdf_forms=True also accepts an address OCR read with a space next to the
    "@" ("anna @example.com"), which happens on some PDF fonts, and grouped
    addresses ("{anna, bob}@example.org") as research papers write them.
    Off by default, like size_guard; pdf_redact.py turns it on for PDF pages."""

    TILE_ABOVE = 2200      # wider strips are tiled; the benchmark goes up to 2160 px
    TILE = 1920
    TILE_OVERLAP = 640
    UPSCALE_BUDGET = 8e6   # detector pixels; RapidOCR scales the short side up to 736
    DENSE = 500            # this many lines in one strip: it may have hit the limit

    def __init__(self, models="v6-small", strip_height=1600, strip_overlap=200, contrast_pass=False,
                 device="cpu", threads=None, faint_pass=False, merged_view=None, gate_second_read=False,
                 size_guard=False, pdf_forms=False):
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import LangDet, LangRec, ModelType, OCRVersion
        (dv, ds, dl), (rv, rs, rl) = MODEL_SETS[models]
        self.models = models
        extra = {}
        if threads:
            extra["EngineConfig.onnxruntime.intra_op_num_threads"] = threads
        if device == "cuda":
            # Needs onnxruntime-gpu. preload_dlls() finds the CUDA and cuDNN
            # libraries installed from pip.
            import onnxruntime
            onnxruntime.preload_dlls()
            extra["EngineConfig.onnxruntime.use_cuda"] = True
            # RapidOCR's default makes cuDNN benchmark every new input shape,
            # and every text line has a new width; pick algorithms by heuristic.
            extra["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] = "HEURISTIC"
        self.engine = RapidOCR(params={**extra,
            # Our own strips keep tall screenshots at full resolution, so the
            # library must not shrink them again.
            "Global.max_side_len": 100000,
            "Global.log_level": "error",
            # A lower box threshold than the default 0.5 keeps faint or
            # tightly spaced lines (a blue underlined link under a long line,
            # for one). Extra boxes cost a little time; text that is not an
            # address is ignored anyway.
            "Det.box_thresh": 0.3,
            "Det.ocr_version": OCRVersion(dv), "Det.model_type": ModelType(ds), "Det.lang_type": LangDet(dl),
            "Rec.ocr_version": OCRVersion(rv), "Rec.model_type": ModelType(rs), "Rec.lang_type": LangRec(rl),
        })
        self.strip_height = strip_height
        self.strip_overlap = strip_overlap
        self.contrast_pass = contrast_pass
        self.faint_pass = faint_pass
        self.merged_view = merged_view            # None, "clahe-diff" or "stack"
        self.gate_second_read = gate_second_read
        self.size_guard = size_guard
        self.pdf_forms = pdf_forms

    def providers(self):
        """Execution providers each model actually runs on (after first use)."""
        out = {}
        for name in ("text_det", "text_rec"):
            model = getattr(self.engine, name, None)
            if model is not None:
                out[name] = model.session.session.get_providers()
        return out

    def lines(self, rgb):
        """Detected text lines (boxes only, no text yet)."""
        found = self._detect_strips(rgb)
        if self.contrast_pass:
            # Lines the first pass already has are skipped, so they are not
            # read twice.
            found += [l for l in self._detect_strips(local_contrast(rgb)) if not _already_found(l, found)]
        return found

    def _detect_strips(self, rgb):
        h = rgb.shape[0]
        if h <= self.strip_height:
            return self._detect_wide(rgb, 0)
        found = []
        step = self.strip_height - self.strip_overlap
        top = 0
        while True:
            bottom = min(h, top + self.strip_height)
            # Keep a line only in the strip where it sits clear of the cut,
            # so each line is found once and never from a cropped copy.
            keep_from = top + (self.strip_overlap // 2 if top > 0 else 0)
            keep_to = bottom - (self.strip_overlap // 2 if bottom < h else 0)
            for line in self._detect_wide(rgb[top:bottom], top):
                if keep_from <= (line.y0 + line.y1) / 2 < keep_to:
                    found.append(line)
            if bottom >= h:
                break
            top += step
        return found

    def _detect_wide(self, rgb, y_offset):
        """_detect, except that with size_guard a wide strip is read in
        overlapping tiles and the pieces of each line are joined again."""
        w = rgb.shape[1]
        if not self.size_guard or w <= self.TILE_ABOVE:
            return self._detect(rgb, y_offset)
        lines = []
        for x0 in list(range(0, w - self.TILE, self.TILE - self.TILE_OVERLAP)) + [w - self.TILE]:
            for line in self._detect(rgb[:, x0:x0 + self.TILE], y_offset):
                line.x0 += x0
                line.x1 += x0
                lines.append(line)
        return _join_pieces(lines)

    def _detect(self, rgb, y_offset):
        h, w = rgb.shape[:2]
        if self.size_guard and min(h, w) < 736 and h * w * (736 / min(h, w)) ** 2 > self.UPSCALE_BUDGET:
            # RapidOCR would scale a thin strip up until its short side is
            # 736 px, which for a 1x800 image means gigabytes. Pad it to that
            # size with its border colour instead; the text keeps its scale.
            rgb = _pad_to(rgb, max(h, 736), max(w, 736))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        result = self.engine(bgr, use_det=True, use_cls=False, use_rec=False)
        post = getattr(getattr(self.engine, "text_det", None), "postprocess_op", None)
        if self.size_guard and result.boxes is not None and len(result.boxes) >= self.DENSE and post is not None:
            # RapidOCR keeps at most 1000 candidate regions per call, so rows
            # of a dense table can be dropped. Read this strip again without
            # that limit. (Raising it everywhere changes results on noisy
            # low-contrast images, so it is only lifted where it matters.)
            cap = post.max_candidates
            post.max_candidates = 20 * cap
            try:
                result = self.engine(bgr, use_det=True, use_cls=False, use_rec=False)
            finally:
                post.max_candidates = cap
        if result.boxes is None:
            return []
        lines = []
        for box in result.boxes:
            box = np.asarray(box, dtype=float)
            x0, y0, x1, y1 = box[:, 0].min(), box[:, 1].min(), min(box[:, 0].max(), w), min(box[:, 1].max(), h)
            if x0 < x1 and y0 < y1:
                lines.append(Line(x0, y0 + y_offset, x1, y1 + y_offset))
        return lines

    def read(self, rgb, lines, normalise=False):
        """Recognise each line from a padded crop scaled to the model's input
        height. Returns new Line objects with characters and x positions."""
        H, W = rgb.shape[:2]
        out = []
        for line in lines:
            h = line.height
            cx0, cx1 = int(max(0, line.x0 - 0.25 * h)), int(min(W, line.x1 + 0.25 * h))
            cy0, cy1 = int(max(0, line.y0 - 0.15 * h)), int(min(H, line.y1 + 0.15 * h))
            crop = rgb[cy0:cy1, cx0:cx1]
            if crop.shape[0] < 4 or crop.shape[1] < 4:
                continue
            if normalise:
                crop = _normalise(crop)
            scale = min(3.0, max(1.0, 48.0 / crop.shape[0]))
            if scale > 1:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            r = self.engine(np.ascontiguousarray(crop[:, :, ::-1]), use_det=False, use_cls=False, use_rec=True, return_word_box=True)
            if not r.txts or not r.word_results or not r.txts[0]:
                continue
            text, info = r.txts[0], r.word_results[0]
            cols = [c for word in info.word_cols for c in word]
            if not info.line_txt_len or len(cols) != len(text.replace(" ", "")):
                continue
            # Recogniser output column -> image x. The positions are only
            # rough here; align_chars() corrects them against the glyphs.
            col_w = crop.shape[1] / info.line_txt_len / scale
            step = (cols[-1] - cols[0]) * col_w / max(1, len(cols) - 1) if len(cols) > 1 else 2 * col_w
            half = 0.6 * max(step, col_w)
            new = Line(line.x0, line.y0, line.x1, line.y1, low_contrast=line.low_contrast, src=line)
            it = iter(cols)
            for c in text:
                if c == " ":
                    new.chars.append(None)
                    continue
                centre = cx0 + (next(it) + 0.5) * col_w
                new.chars.append(Char(c, centre - half, centre + half))
            out.append(new)
        return out


def _pad_to(rgb, h, w):
    """rgb in the top-left corner of an h x w canvas filled with its border colour."""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    out = np.empty((h, w, 3), rgb.dtype)
    out[:] = np.median(border, axis=0).astype(rgb.dtype)
    out[:rgb.shape[0], :rgb.shape[1]] = rgb
    return out


def _join_pieces(lines):
    """Lines from neighbouring tiles: copies of one line, and pieces of a line
    cut at a tile edge, overlap side by side on the same row. Each such group
    becomes one line covering all of it."""
    lines = sorted(lines, key=lambda l: l.y0)
    parent = list(range(len(lines)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(lines):
        for j in range(i + 1, len(lines)):
            b = lines[j]
            if b.y0 >= a.y1:
                break
            same_row = min(a.y1, b.y1) - max(a.y0, b.y0) >= 0.5 * min(a.height, b.height)
            if same_row and min(a.x1, b.x1) > max(a.x0, b.x0):
                parent[root(j)] = root(i)
    groups = {}
    for i, line in enumerate(lines):
        groups.setdefault(root(i), []).append(line)
    return [Line(min(l.x0 for l in g), min(l.y0 for l in g), max(l.x1 for l in g), max(l.y1 for l in g))
            for g in groups.values()]


# --------------------------------------------------------------------------
# Rows: OCR lines that share a baseline, joined left to right
# --------------------------------------------------------------------------

@dataclass
class Row:
    lines: list
    text: str = ""
    # For every character of text: (line index, char index) or None for a
    # space that joins two separate OCR lines.
    index: list = field(default_factory=list)

    @property
    def y0(self):
        return min(l.y0 for l in self.lines)

    @property
    def y1(self):
        return max(l.y1 for l in self.lines)

    @property
    def x0(self):
        return min(l.x0 for l in self.lines)

    @property
    def x1(self):
        return max(l.x1 for l in self.lines)

    @property
    def height(self):
        return float(np.median([l.height for l in self.lines]))


def build_rows(lines):
    rows = []
    for line in sorted(lines, key=lambda l: (l.y0 + l.y1) / 2):
        centre = (line.y0 + line.y1) / 2
        for row in rows:
            ref = row.lines[0]
            if abs(centre - (ref.y0 + ref.y1) / 2) < 0.5 * min(ref.height, line.height):
                row.lines.append(line)
                break
        else:
            rows.append(Row([line]))
    for row in rows:
        row.lines.sort(key=lambda l: l.x0)
        parts, index = [], []
        for li, line in enumerate(row.lines):
            if li:
                parts.append(" ")
                index.append(None)
            for ci, c in enumerate(line.chars):
                parts.append(" " if c is None else c.ch)
                index.append(None if c is None else (li, ci))
        row.text = normalise_ocr_text("".join(parts))
        row.index = index
    rows.sort(key=lambda r: r.y0)
    return rows


# --------------------------------------------------------------------------
# Pixel snapping
# --------------------------------------------------------------------------

def background_colour(patch):
    """Most common exact colour in an (h, w, 3) patch."""
    flat = patch.reshape(-1, 3).astype(np.int32)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    values, counts = np.unique(packed, return_counts=True)
    v = int(values[counts.argmax()])
    return np.array([(v >> 16) & 255, (v >> 8) & 255, v & 255], dtype=np.int16)


def noise_threshold(diff):
    """Ink threshold for one window: 0 for a lossless render, where every
    pixel away from the letters is exactly a background colour, and 40 when
    compression speckle surrounds the text (JPEG screenshots)."""
    strong = (diff > 40).astype(np.uint8)
    far = cv2.dilate(strong, np.ones((7, 7), np.uint8)) == 0
    if far.sum() < 100:
        return 40
    speckle = far & (diff > 0) & (diff <= 24)
    return 0 if speckle.sum() <= 0.01 * far.sum() else 40


def ink_mask(rgb, bx0, by0, bx1, by1, threshold=None, de=None):
    """Pixels in the window that differ from both the window's background
    colour and their own row's background colour (so coloured strips and
    panels are not ink). threshold=None picks it from the window's noise.

    With de (a channel_difference map) ink is instead every pixel whose
    channel difference is above an Otsu split of the window, and at least
    CH_INK_MIN: that finds text close to its background in brightness."""
    if de is not None:
        win = de[by0:by1, bx0:bx1]
        otsu, _ = cv2.threshold(np.clip(win, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return win > max(CH_INK_MIN, otsu)
    H, W = rgb.shape[:2]
    patch = rgb[by0:by1, bx0:bx1].astype(np.int16)
    bg = background_colour(patch)
    diff = np.abs(patch - bg).max(axis=2)
    # Row backgrounds come from a wider window so a long address cannot
    # outvote the real background in its own rows.
    margin = 4 * (by1 - by0)
    wide = rgb[by0:by1, max(0, bx0 - margin):min(W, bx1 + margin)].astype(np.int16)
    for i in range(wide.shape[0]):
        row_bg = background_colour(wide[i:i + 1])
        if np.any(row_bg != bg):
            diff[i] = np.minimum(diff[i], np.abs(patch[i] - row_bg).max(axis=1))
    if threshold is None:
        threshold = noise_threshold(diff)
    return diff > threshold


def snap_box(rgb, threshold, y0, y1, left_limit, right_limit, wraps_left=False, wraps_right=False, de=None):
    """Box every glyph of one address fragment, pixel-exact.

    y0..y1 is the OCR line box. left_limit/right_limit are x positions that
    separate the fragment's first and last characters from their neighbours
    (midpoints between OCR character centres). A glyph is a connected blob of
    ink; it belongs to the fragment when it crosses the middle of the text
    line and its centre lies between the limits. Returns (x0, y0, x1, y1),
    end-exclusive, or None when no glyph qualifies.
    """
    H, W = rgb.shape[:2]
    h = max(4.0, y1 - y0)
    by0, by1 = int(max(0, y0 - 0.45 * h)), int(min(H, y1 + 0.45 * h))
    bx0 = 0 if left_limit < 3 * h else int(left_limit - 1.5 * h)
    bx1 = W if W - right_limit < 3 * h else int(right_limit + 1.5 * h)
    ink = ink_mask(rgb, bx0, by0, bx1, by1, threshold, de)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)

    # Middle band of the line, safely inside the glyph bodies.
    mid0, mid1 = y0 + 0.28 * h - by0, y1 - 0.28 * h - by0
    line0, line1 = y0 - by0, y1 - by0
    lo, hi = left_limit - bx0, right_limit - bx0
    glyphs = []  # (x0, x1, y0, y1) of blobs on this text line
    for k in range(1, n):
        x, y, w, hh = stats[k, :4]
        crosses_middle = y < mid1 and y + hh > mid0
        inside_line = y >= line0 - 1 and y + hh <= line1 + 1
        if not (crosses_middle or inside_line):
            continue
        if hh < 0.2 * h and w > 3 * h:
            continue  # underline or table rule, not a glyph
        glyphs.append((x, x + w, y, y + hh))
    chosen = [g for g in glyphs if lo <= (g[0] + g[1]) / 2 < hi]
    if not chosen:
        return None
    left = min(g[0] for g in chosen)
    right = max(g[1] for g in chosen)
    top = min(g[2] for g in chosen)
    bottom = max(g[3] for g in chosen)

    # Text cut off by the screenshot edge: tightly spaced glyphs between the
    # fragment and the edge are its clipped first or last characters.
    gap = max(2, int(0.35 * h))
    others = sorted(g for g in glyphs if g not in chosen)
    if bx0 == 0 and not wraps_left and left <= 1.5 * h:
        pos, extra = left, []
        for g in reversed([g for g in others if g[1] <= left]):
            if pos - g[1] > gap:
                break
            pos = g[0]
            extra.append(g)
        if pos <= gap:
            left = 0
            top = min([top] + [g[2] for g in extra])
            bottom = max([bottom] + [g[3] for g in extra])
    if bx1 == W and not wraps_right and ink.shape[1] - right <= 1.5 * h:
        pos, extra = right, []
        for g in [g for g in others if g[0] >= right]:
            if g[0] - pos > gap:
                break
            pos = g[1]
            extra.append(g)
        if ink.shape[1] - pos <= gap:
            right = ink.shape[1]
            top = min([top] + [g[2] for g in extra])
            bottom = max([bottom] + [g[3] for g in extra])

    # Pad into empty space only, staying one pixel clear of any other ink.
    def clear_col(c):
        return 0 <= c < ink.shape[1] and not ink[top:bottom, c].any()

    def clear_row(r):
        return 0 <= r < ink.shape[0] and not ink[r, left:right].any()

    # Generous padding covers JPEG ringing and matches line-height boxes; the
    # clearance checks keep it from touching any neighbouring text.
    pad = max(2, int(round(PAD * h)))
    for _ in range(pad):
        if left > 0 and clear_col(left - 1) and (left - 2 < 0 or clear_col(left - 2)):
            left -= 1
        if right < ink.shape[1] and clear_col(right) and (right + 1 >= ink.shape[1] or clear_col(right + 1)):
            right += 1
    for _ in range(pad):
        if top > 0 and clear_row(top - 1) and (top - 2 < 0 or clear_row(top - 2)):
            top -= 1
        if bottom < ink.shape[0] and clear_row(bottom) and (bottom + 1 >= ink.shape[0] or clear_row(bottom + 1)):
            bottom += 1
    return bx0 + left, by0 + top, bx0 + right, by0 + bottom


def glyph_centres(rgb, threshold, line, de=None):
    """x centres of the glyphs on one OCR line, left to right. Blobs that
    overlap horizontally (an i and its dot) count as one glyph."""
    H, W = rgb.shape[:2]
    h = max(4.0, line.height)
    by0, by1 = int(max(0, line.y0 - 0.2 * h)), int(min(H, line.y1 + 0.2 * h))
    bx0, bx1 = int(max(0, line.x0 - h)), int(min(W, line.x1 + h))
    if by1 - by0 < 2 or bx1 - bx0 < 2:
        return np.zeros(0)
    ink = ink_mask(rgb, bx0, by0, bx1, by1, threshold, de)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    mid0, mid1 = line.y0 + 0.28 * h - by0, line.y1 - 0.28 * h - by0
    spans = sorted((x, x + w) for x, y, w, hh, _ in stats[1:]
                   if y < mid1 and y + hh > mid0 and not (hh < 0.2 * h and w > 3 * h))
    merged = []
    for a, b in spans:
        if merged and a < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return np.array([bx0 + (a + b) / 2 for a, b in merged])


def align_chars(rgb, threshold, line, de=None):
    """Correct OCR character positions against the real glyphs.

    The recogniser places each character a little after its glyph, and the
    offset grows along the line. Matching every character to its nearest glyph
    and fitting x' = a*x + b (dropping poor matches) removes that drift, so the
    split between an address and a touching bracket lands in the right gap.
    """
    if line.aligned:
        return
    line.aligned = True
    chars = [c for c in line.chars if c is not None]
    glyphs = glyph_centres(rgb, threshold, line, de if line.low_contrast else None)
    if len(chars) < 3 or len(glyphs) < 3:
        return
    c = np.array([(ch.x0 + ch.x1) / 2 for ch in chars])
    pitch = max(1.0, float(np.median(np.diff(c))))
    a, b = 1.0, 0.0
    for _ in range(4):
        t = a * c + b
        idx = np.clip(np.searchsorted(glyphs, t), 1, len(glyphs) - 1)
        nearest = np.where(np.abs(glyphs[idx - 1] - t) <= np.abs(glyphs[idx] - t), glyphs[idx - 1], glyphs[idx])
        ok = np.abs(nearest - t) < 0.45 * pitch
        if ok.sum() < 3:
            return
        a_new, b_new = np.polyfit(c[ok], nearest[ok], 1)
        if not 0.9 < a_new < 1.1:
            a_new = 1.0
            b_new = float(np.median(nearest[ok] - c[ok]))
        a, b = a_new, b_new
    for ch in chars:
        ch.x0, ch.x1 = a * ch.x0 + b, a * ch.x1 + b


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

@dataclass
class Hit:
    text: str
    boxes: list  # [x, y, w, h] per visual fragment


def _span_chars(row, start, end):
    """Character objects (with their line) for text[start:end], skipping spaces."""
    out = []
    for k in range(max(0, start), min(end, len(row.index))):
        ref = row.index[k]
        if ref is not None:
            li, ci = ref
            out.append((row.lines[li], row.lines[li].chars[ci], li, ci))
    return out


def _centre(c):
    return (c.x0 + c.x1) / 2


def _fragment_box(rgb, threshold, row, start, end, wraps_left=False, wraps_right=False, de=None):
    for line in {id(l): l for l in row.lines}.values():
        align_chars(rgb, threshold, line, de)
    chars = _span_chars(row, start, end)
    if not chars:
        return None
    first, last = chars[0][1], chars[-1][1]
    pitch = (_centre(last) - _centre(first)) / (len(chars) - 1) if len(chars) > 1 else max(2.0, first.x1 - first.x0)
    # Split from a touching neighbour at the midpoint between character
    # centres; after a space the gap is wide, so one pitch is enough.
    before = _span_chars(row, start - 1, start) if start > 0 else []
    after = _span_chars(row, end, end + 1)
    left = (_centre(before[0][1]) + _centre(first)) / 2 if before else _centre(first) - pitch
    right = (_centre(last) + _centre(after[0][1])) / 2 if after else _centre(last) + pitch
    y0 = min(l.y0 for l, _, _, _ in chars)
    y1 = max(l.y1 for l, _, _, _ in chars)
    low = de is not None and any(l.low_contrast for l, _, _, _ in chars)
    box = snap_box(rgb, threshold, y0, y1, left, right, wraps_left, wraps_right, de if low else None)
    ocr_box = (max(0, min(c.x0 for _, c, _, _ in chars)), max(0, y0),
               min(rgb.shape[1], max(c.x1 for _, c, _, _ in chars)), min(rgb.shape[0], y1))
    if box is None:
        # No ink found (very faint text): fall back to OCR's own estimate.
        box = ocr_box
    elif low:
        # Faint text can lose its weakest glyphs to the ink threshold, so a
        # low-contrast box never stops short of where OCR read the address.
        box = (min(box[0], ocr_box[0]), min(box[1], ocr_box[1]), max(box[2], ocr_box[2]), max(box[3], ocr_box[3]))
    gx0, gy0, gx1, gy1 = box
    return [int(gx0), int(gy0), int(gx1 - gx0), int(gy1 - gy0)]


def _tail_token(text):
    """Start index of the trailing run of address characters."""
    k = len(text)
    while k > 0 and text[k - 1] in ADDR_CHARS:
        k -= 1
    return k


def _head_token(text):
    k = 0
    while k < len(text) and text[k] in ADDR_CHARS:
        k += 1
    return k


def _line_row(line):
    """A one-line Row, so line-level matching can reuse the row helpers."""
    return build_rows([line])[0]


def _next_line(line, lines):
    """The OCR line a wrapped word would continue on: directly below, in the
    same column, and leftmost if several sit on that row."""
    h = line.height
    best = None
    for other in lines:
        if other is line:
            continue
        gap = other.y0 - line.y1
        ratio = line.height / max(1.0, other.height)
        overlap = min(line.x1, other.x1) - max(line.x0, other.x0)
        if -0.4 * h <= gap <= 1.3 * h and 0.6 < ratio < 1.7 and overlap > 0:
            key = (round(other.y0 / max(1.0, 0.5 * h)), other.x0)
            if best is None or key < best[0]:
                best = (key, other)
    return best[1] if best else None


def _find_wraps(rgb, threshold, lines, de=None):
    """Addresses that start at the end of one line and finish at the start of
    the next, e.g. "ced31@" / "cedar.test" or "gbtp-" / "exed@iima.ac.in"."""
    hits = []
    for line in lines:
        row = _line_row(line)
        t0 = _tail_token(row.text)
        tail = row.text[t0:]
        if not tail or any(s <= t0 < e for s, e in find_emails(row.text)):
            continue
        # Browsers break after "@" or at a hyphen, never just before "@".
        if not ("@" in tail or tail.endswith("-")):
            continue
        nxt = _next_line(line, lines)
        if nxt is None:
            continue
        nrow = _line_row(nxt)
        head = nrow.text[:_head_token(nrow.text)].rstrip(".")
        if not head or ("@" in tail) == ("@" in head):
            continue  # exactly one half must hold the "@"
        if "." not in head:
            continue  # the domain's dot is always in the second half
                      # ("hell@ilumi.c" + "Website" is a misread, not a wrap)
        options = [tail + head]
        if tail.endswith("-"):
            options.append(tail[:-1] + head)  # line-break hyphen
        for cand in options:
            if FULL_EMAIL_RE.fullmatch(cand):
                b1 = _fragment_box(rgb, threshold, row, t0, len(row.text), wraps_right=True, de=de)
                b2 = _fragment_box(rgb, threshold, nrow, 0, len(head), wraps_left=True, de=de)
                if b1 and b2:
                    hits.append(Hit(cand, [b1, b2]))
                break
    return hits


def _find_in_lines(rgb, threshold, lines, de=None, pdf_forms=False):
    hits = []
    find = find_emails_pdf if pdf_forms else find_emails
    for row in build_rows(lines):
        for start, end in find(row.text):
            box = _fragment_box(rgb, threshold, row, start, end, de=de)
            if box:
                hits.append(Hit(row.text[start:end], [box]))
    return hits + _find_wraps(rgb, threshold, lines, de)


def _merge_boxes(boxes):
    """Union boxes that mostly overlap (the same address found twice)."""
    boxes = [list(b) for b in boxes]
    if len(boxes) > 64:
        return _merge_many(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ax, ay, aw, ah = boxes[i]
                bx, by, bw, bh = boxes[j]
                iw = min(ax + aw, bx + bw) - max(ax, bx)
                ih = min(ay + ah, by + bh) - max(ay, by)
                if iw > 0 and ih > 0 and iw * ih >= 0.5 * min(aw * ah, bw * bh):
                    x0, y0 = min(ax, bx), min(ay, by)
                    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def _merge_many(boxes):
    """_merge_boxes for pages with many boxes (comparing every pair after
    every merge takes minutes for a few hundred). Boxes are swept in order of
    their top edge, so only boxes that share some rows are compared. Every
    union still covers all the boxes it replaces."""
    merged = True
    while merged:
        merged = False
        boxes.sort(key=lambda b: b[1])
        alive = [True] * len(boxes)
        for i, a in enumerate(boxes):
            if not alive[i]:
                continue
            j = i + 1
            while j < len(boxes) and boxes[j][1] < a[1] + a[3]:
                b = boxes[j]
                if alive[j]:
                    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                    if iw > 0 and ih > 0 and iw * ih >= 0.5 * min(a[2] * a[3], b[2] * b[3]):
                        x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                        x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                        a[:] = [x0, y0, x1 - x0, y1 - y0]
                        alive[j] = False
                        merged = True
                j += 1
        boxes = [b for b, keep in zip(boxes, alive) if keep]
    return boxes


@dataclass
class Result:
    addresses: list  # address strings as read (one per hit, both readings)
    boxes: list      # final [x, y, w, h] boxes to fill
    sideways: list = field(default_factory=list)  # [x, y, w, h] of text lines much taller than wide


def detect(rgb, ocr):
    """Find every address in an RGB array. Returns a Result."""
    threshold = None  # chosen per window from the local noise
    lines = ocr.lines(rgb)
    de, extra = None, []
    if getattr(ocr, "faint_pass", False):
        # Look for lines the normal pass missed on two extra views. Each new
        # line is also read from the view that found it, and its box is fitted
        # to the channel difference instead of brightness.
        channel_view, de = channel_difference(rgb)
        kind = getattr(ocr, "merged_view", None)
        views = (merged_view(rgb, kind),) if kind else (local_contrast(rgb), channel_view)
        for view in views:
            new = [l for l in ocr._detect_strips(view) if not _already_found(l, lines + [l for _, l in extra])]
            for line in new:
                line.low_contrast = True
            extra += [(view, l) for l in new]
        lines = lines + [l for _, l in extra]
    spaced = getattr(ocr, "pdf_forms", False)
    first = ocr.read(rgb, lines, False)
    hits = _find_in_lines(rgb, threshold, first, de, spaced)
    second = lines
    if getattr(ocr, "gate_second_read", False):
        # Read again only lines that might hold an address: those whose first
        # reading looks address-like, those whose first reading failed, and
        # every faint line.
        read_ok = {id(l.src) for l in first}
        likely = {id(l.src) for l in first if address_like(_text(l))}
        second = [l for l in lines if l.low_contrast or id(l) not in read_ok or id(l) in likely]
    hits += _find_in_lines(rgb, threshold, ocr.read(rgb, second, True), de, spaced)
    for view in {id(v): v for v, _ in extra}.values():
        own = [l for v, l in extra if v is view]
        hits += _find_in_lines(rgb, threshold, ocr.read(view, own), de, spaced)
    addresses = sorted({h.text for h in hits})
    # Lines of sideways text (a rotated page, a margin note) cannot be read
    # as they are; pdf_redact.py turns that part of the page and looks again.
    sideways = [[int(l.x0), int(l.y0), int(l.x1 - l.x0), int(l.y1 - l.y0)] for l in lines
                if l.y1 - l.y0 > max(40, 3 * (l.x1 - l.x0))]
    return Result(addresses, _merge_boxes([b for h in hits for b in h.boxes]), sideways)


# The formats a screenshot comes in. Anything else (EPS, PSD, ...) is refused,
# so no rarely used decoder sees the file and EPS never reaches Ghostscript.
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "BMP", "GIF")


class ImageTooLarge(ValueError):
    pass


class CMYKCompatibilityRequired(ValueError):
    pass


def open_image(fileobj, max_pixels=None):
    """Decode to visible RGB, with the same EXIF and alpha normalization as OCR."""
    data = fileobj.read()
    limit = max_pixels or 89_478_485
    info = image_io.header(data)
    if info.width * info.height > limit:
        raise ImageTooLarge(f"Image exceeds the limit of {limit} pixels.")
    if info.cmyk:
        # CMYK conversion differs between codecs. Preserve the original OCR
        # pixels only for these files, without loading Pillow for normal images.
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise CMYKCompatibilityRequired(
                "CMYK JPEG requires the optional cmyk extra (Pillow). "
                "See docs/lightweight-branch.md for installation."
            ) from exc
        with Image.open(io.BytesIO(data), formats=("JPEG",)) as image:
            if image.width * image.height > limit:
                raise ImageTooLarge(f"Image exceeds the limit of {limit} pixels.")
            image.load()
            return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
    try:
        image, _, _ = image_io.decode(data, max_pixels=limit,
                                      formats=IMAGE_FORMATS, allow_16=True,
                                      allow_animation=True)
    except image_io.ImageLimitError as exc:
        raise ImageTooLarge(str(exc)) from exc
    return image_io.opaque(image)


def detect_image(image, ocr):
    """Run unchanged OCR on a contiguous RGB array from open_image()."""
    return detect(np.ascontiguousarray(image), ocr)


def png_bytes(image):
    """Encode pixels only; no text, EXIF or ICC metadata survives."""
    return image_io.encode(image)


def redact(image, boxes, fill=(0, 0, 0, 255)):
    arr = np.concatenate((image[:, :, :3], np.full((*image.shape[:2], 1), 255, np.uint8)), axis=2)
    for x, y, w, h in boxes:
        arr[y:y + h, x:x + w] = fill
    return arr


def overlay(image, boxes, colour=(230, 30, 30), alpha=0.35):
    """Private diagnostic overlay; the text beneath the boxes stays readable."""
    output = image[:, :, :3].copy()
    mask = np.zeros(image.shape[:2], np.uint8)
    for x, y, w, h in boxes:
        cv2.rectangle(mask, (x,y), (x+w-1,y+h-1), int(255*alpha), -1)
        # Inset two-pixel outline, matching the diagnostic rectangle semantics.
        mask[y:y+min(2,h),x:x+w]=255; mask[y+max(0,h-2):y+h,x:x+w]=255
        mask[y:y+h,x:x+min(2,w)]=255; mask[y:y+h,x+max(0,w-2):x+w]=255
    a=mask[:,:,None].astype(np.uint32)
    output=((np.array(colour,dtype=np.uint32)*a+output.astype(np.uint32)*(255-a)+127)//255).astype(np.uint8)
    return np.concatenate((output,np.full((*output.shape[:2],1),255,np.uint8)),axis=2)


def main():
    if __package__:
        from . import safe_files as sf
    else:
        import safe_files as sf
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path, help="where to save the redacted image (.png)")
    ap.add_argument("--overlay", type=Path, metavar="PNG",
                    help="also save the original with see-through boxes, to check what was hidden")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace the output (and --overlay) if it already exists; the input is never replaced")
    ap.add_argument("--json", action="store_true", help="print what was found as JSON")
    ap.add_argument("--mode", choices=["accurate", "fast"], default="accurate",
                    help="accurate also finds faint or colour-on-colour text; fast is about twice as fast on CPU")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads to use (default 4)")
    ap.add_argument("--models", choices=sorted(MODEL_SETS), default="v6-small",
                    help="OCR models; sets other than v6-small are downloaded from ModelScope on first use")
    args = ap.parse_args()
    anywhere = sf.Folders(None)
    try:
        if args.overlay and (Path(os.path.realpath(args.overlay.expanduser().parent)) / args.overlay.name
                             == Path(os.path.realpath(args.output.expanduser().parent)) / args.output.name):
            raise sf.UnsafePath("--overlay must be a different file from the output.")
        source, _, ident = sf.open_input(args.input, anywhere)
        with source:
            try:
                image = open_image(source)
            except (OSError, ValueError, SyntaxError) as exc:
                raise sf.UnsafePath(f"{args.input} is not an image that can be read "
                                    f"({', '.join(IMAGE_FORMATS)}): {exc}") from None
        protect = {ident}
        sf.check_output(args.output, anywhere, protect, args.overwrite)
        if args.overlay:
            sf.check_output(args.overlay, anywhere, protect, args.overwrite)
        if args.models == "v6-small":
            check_models()
        cv2.setNumThreads(args.threads)
        ocr = Ocr(args.models, threads=args.threads, faint_pass=args.mode == "accurate", gate_second_read=True,
                  size_guard=True)
        started = time.perf_counter()
        result = detect_image(image, ocr)
        elapsed = (time.perf_counter() - started) * 1000
        written, out_ident = sf.write_file(png_bytes(redact(image, result.boxes)), args.output, anywhere,
                                           protect, args.overwrite)
        if args.overlay:
            sf.write_file(png_bytes(overlay(image, result.boxes)), args.overlay, anywhere,
                          protect | {out_ident}, args.overwrite)
    except (sf.UnsafePath, OSError, RuntimeError) as exc:
        sys.exit(f"image_redact: {exc}")
    if args.json:
        print(json.dumps({"addresses": result.addresses, "boxes": result.boxes, "ms": round(elapsed)}))
    else:
        print(f"{len(result.boxes)} box(es) over {len(result.addresses)} address(es) in {elapsed:.0f} ms -> {written}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
