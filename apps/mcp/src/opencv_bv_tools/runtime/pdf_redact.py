#!/usr/bin/env python3
"""Find email addresses in a PDF and cover them with solid boxes.

Every page is drawn as a picture, the way a PDF viewer shows it, and the
addresses are found twice: by the same OCR detection used for screenshots,
and in the page's own text, when it has any. Each address is covered with a
black box in the picture, and the pictures are written as a new PDF.

The new PDF holds nothing but those pictures, and an invisible text layer
read by OCR from the redacted pictures, so its text can be selected, copied
and searched. The text, links, form fields, attachments, metadata and
earlier revisions of the original are not carried over, so nothing under a
box can be copied out or recovered, and nothing hidden in the original
reaches the copy: the text layer only holds what can be read on the redacted
page, and no word with an "@" at all.

Usage:
  python pdf_redact.py input.pdf output.pdf [--mode accurate|fast] [--overwrite]
The input is never changed, and an existing output is only replaced with --overwrite.
"""
import argparse
import contextlib
import ctypes
import json
import math
import os
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import dataclass, field

import cv2
import numpy as np
from .image_io import encode as encode_image

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

if __package__:
    from . import image_redact as sr
else:
    import image_redact as sr

DPI = 150       # pages are drawn, checked and saved at this resolution
MAX_PAGES = 100
MAX_OUTPUT = 1 << 30  # bytes; a larger redacted PDF is refused
PAD = 2         # px around a box found in the page's text
JPEG_QUALITY = 90
MAX_SKEW = 3.0  # degrees; scans are straightened up to this much
MIN_SKEW = 0.15  # degrees; below this a page is left as it is
# OCR places letters only roughly on PDF-sized text, so a box can stop a
# letter short, most often next to a bracket. A box from OCR is extended
# through a letter it cuts, and over one more letter when only a gap
# narrower than a space separates it (a space is judged from the gaps
# between the letters inside the box). At most EXTEND times its height.
EXTEND = 0.6

# Pages are drawn in a separate process (see Pages) with these limits.
DRAW_MEMORY = 2 << 30  # bytes of data memory
DRAW_SECONDS = 120     # to open the PDF, or to draw one page

# PDFium must never be used by two threads at once, not even for different
# documents, so every call into it holds this lock. (The server only uses it
# from the drawing process, which has one thread.)
PDFIUM = threading.Lock()


class PdfError(ValueError):
    """A PDF that cannot be redacted; the message is meant for the user."""


@dataclass
class PdfResult:
    engine: str = "picture"  # how the PDF was redacted: "picture" or "mupdf"
    fallback: str = ""       # why MuPDF was not used, when it was asked for
    pages: int = 0
    addresses: set = field(default_factory=set)
    boxes: list = field(default_factory=list)  # [x, y, w, h] boxes of each page, in pixels at DPI
    sizes: list = field(default_factory=list)  # (width, height) of each page in pixels


def open_pdf(fileobj):
    """Open a PDF from a file opened for binary reading. The file must stay
    open until the document is closed."""
    with PDFIUM:
        try:
            pdf = pdfium.PdfDocument(fileobj)
        except pdfium.PdfiumError as exc:
            if getattr(exc, "err_code", None) == pdfium_c.FPDF_ERR_PASSWORD:
                raise PdfError("The PDF is protected by a password; remove the password first.") from None
            raise PdfError("Not a PDF that can be read; check that the file is complete.") from None
        try:
            pdf.init_forms()  # so filled-in form fields are drawn as well
        except pdfium.PdfiumError:
            pass
        return pdf


def close_pdf(pdf):
    with PDFIUM:
        pdf.close()


def page_count(pdf):
    with PDFIUM:
        return len(pdf)


def _page_text(textpage):
    """The characters of the page's own text, and for each whether PDFium
    added it (a space or line break the PDF itself does not contain)."""
    chars, added = [], []
    for i in range(max(0, pdfium_c.FPDFText_CountChars(textpage))):
        code = pdfium_c.FPDFText_GetUnicode(textpage, i)
        chars.append(chr(code) if 0 < code < 0x110000 and not 0xD800 <= code < 0xE000 else "\ufffd")
        added.append(pdfium_c.FPDFText_IsGenerated(textpage, i) == 1)
    return sr.normalise_ocr_text("".join(chars)), added


HYPHEN = "\x02"  # how PDFium marks a word hyphenated at the end of a line


def _joined(text, added):
    """The text without the spaces and line breaks PDFium added inside an
    address, so an address it split is found whole: a space next to "@" or
    ".", a line break after "@", "." or "-" or before "@" or ".", a line
    break after a comma in a list of names, and a word hyphenated at a line
    break. Returns the joined text and, for each of its
    characters, its index in text."""
    keep, i = [], 0
    while i < len(text):
        if text[i] == HYPHEN:
            if not (0 < i < len(text) - 1 and text[i - 1] in sr.ADDR_CHARS and text[i + 1] in sr.ADDR_CHARS):
                keep.append(i)
            i += 1
            continue
        if not (added[i] and text[i].isspace()):
            keep.append(i)
            i += 1
            continue
        end = i
        while end < len(text) and added[end] and text[end].isspace():
            end += 1
        before, after = text[i - 1] if i else "", text[end] if end < len(text) else ""
        breaks = "\n" in text[i:end] or "\r" in text[i:end]
        inside = before in sr.ADDR_CHARS and after in sr.ADDR_CHARS
        joins = inside and (before in "@.-" or after in "@.") and (breaks or before != "-")
        joins |= breaks and before == "," and after in sr.ADDR_CHARS  # a list of names: "{anna,\nbob}@x.org"
        if not joins:
            keep.extend(range(i, end))
        i = end
    return "".join(text[k] for k in keep), keep


def _text_hits(textpage, to_pixels, width, height):
    """(address, box) for every address in the page's own text, with boxes
    in pixels of the drawn page; one box per line an address spans."""
    text, added = _page_text(textpage)
    if not text:
        return []
    joined, index = _joined(text, added)
    hits = []
    loose = pdfium_c.FS_RECTF()
    l, r, b, t = (ctypes.c_double() for _ in range(4))
    for start, end in sr.find_emails_pdf(joined):
        lines = [[]]  # character indices of the address, split where a line break was joined
        for k in range(start, end):
            if k > start and any(c in "\r\n" + HYPHEN for c in text[index[k - 1] + 1:index[k]]):
                lines.append([])
            lines[-1].append(index[k])
        for chars in lines:
            xs, ys = [], []
            for i in chars:
                if added[i]:
                    continue  # a space PDFium added; it has no box of its own
                # The glyph's own outline. The font's full height (the loose
                # box) is only a fallback: in some fonts it spans two lines,
                # and a box that tall would also cover the next line.
                rects = []
                if pdfium_c.FPDFText_GetCharBox(textpage, i, l, r, b, t) and r.value > l.value and t.value > b.value:
                    rects.append((l.value, b.value, r.value, t.value))
                elif pdfium_c.FPDFText_GetLooseCharBox(textpage, i, loose):
                    rects.append((loose.left, loose.bottom, loose.right, loose.top))
                for left, bottom, right, top in rects:
                    for px in (left, right):  # all four corners, so rotated pages and text are covered
                        for py in (bottom, top):
                            x, y = to_pixels(px, py)
                            xs.append(x)
                            ys.append(y)
            if not xs:
                continue
            x0, y0 = max(0, min(xs) - PAD), max(0, min(ys) - PAD)
            x1, y1 = min(width, max(xs) + PAD + 1), min(height, max(ys) + PAD + 1)
            if x1 > x0 and y1 > y0:
                hits.append((joined[start:end], [x0, y0, x1 - x0, y1 - y0]))
    return hits


def draw_page(pdf, index, dpi=DPI, max_pixels=None):
    """Draw one page as an RGB array, as a viewer shows it (rotation, crop
    box, annotations and form fields included). Returns the array, the page
    size in points and the addresses found in the page's own text."""
    scale = dpi / 72
    with PDFIUM:
        try:
            page = pdf[index]
        except pdfium.PdfiumError:
            raise PdfError(f"Page {index + 1} of the PDF cannot be read.") from None
        try:
            w_pt, h_pt = page.get_size()
            width, height = math.ceil(w_pt * scale), math.ceil(h_pt * scale)  # as render() sizes it
            if width < 1 or height < 1:
                raise PdfError(f"Page {index + 1} has no size.")
            if max_pixels and width * height > max_pixels:
                raise PdfError(f"Page {index + 1} would be {width}x{height} pixels at {dpi:g} dpi; "
                               f"the limit is {max_pixels / 1e6:g} megapixels.")
            bitmap = page.render(scale=scale, rev_byteorder=True, may_draw_forms=True, draw_annots=True)
            try:
                rgb = bitmap.to_numpy()[..., :3].copy()
                to_pixels = bitmap.get_posconv(page).to_bitmap
                textpage = page.get_textpage()
                try:
                    hits = _text_hits(textpage, to_pixels, rgb.shape[1], rgb.shape[0])
                finally:
                    textpage.close()
            finally:
                bitmap.close()
        finally:
            page.close()
    return np.ascontiguousarray(rgb), (w_pt, h_pt), hits


def _turned(rgb, detect, sideways):
    """Check sideways text again: the part of the page that holds it is
    turned a quarter both ways (the text may run up or down) and checked like
    a normal page. Returns the addresses and their boxes on the page."""
    h, w = rgb.shape[:2]
    pad = 2 * int(np.median([min(bw, bh) for _, _, bw, bh in sideways]))
    x0 = max(0, min(b[0] for b in sideways) - pad)
    y0 = max(0, min(b[1] for b in sideways) - pad)
    x1 = min(w, max(b[0] + b[2] for b in sideways) + pad)
    y1 = min(h, max(b[1] + b[3] for b in sideways) + pad)
    crop = rgb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    addresses, boxes = set(), []
    for k in (1, 3):  # a quarter turn anticlockwise, then clockwise
        more, found, _ = _found(np.ascontiguousarray(np.rot90(crop, k)), detect)
        addresses |= more
        for bx, by, bw, bh in found:
            x, y = (cw - by - bh, bx) if k == 1 else (by, ch - bx - bw)
            boxes.append([x0 + x, y0 + y, bh, bw])
    return addresses, boxes


def skew(rgb):
    """How many degrees the text lines of a scanned page are turned
    anticlockwise, between -MAX_SKEW and MAX_SKEW; 0 for a straight page or
    one without clear lines of text. The angle is the one at which the rows
    of ink line up most sharply."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    step = max(1, grey.shape[1] // 1000)
    small = grey[::step, ::step]
    ys, xs = np.nonzero(small < np.median(small) - 50)
    if len(xs) < 500:
        return 0.0
    if len(xs) > 200_000:
        keep = np.random.default_rng(0).choice(len(xs), 200_000, replace=False)
        ys, xs = ys[keep], xs[keep]
    ys, xs = ys.astype(np.float64), xs.astype(np.float64)

    def sharpness(angle):
        a = np.radians(angle)
        rows = np.round(ys * np.cos(a) + xs * np.sin(a)).astype(np.int64)
        return float(np.sum(np.bincount(rows - rows.min()).astype(np.float64) ** 2))

    coarse = max(np.arange(-MAX_SKEW, MAX_SKEW + 0.01, 0.1), key=sharpness)
    best = max(np.arange(coarse - 0.1, coarse + 0.101, 0.02), key=sharpness)
    # A page of pictures has no clear rows at any angle; only trust an angle
    # that lines the rows up clearly better than leaving the page as it is.
    return float(best) if sharpness(best) > 1.05 * sharpness(0.0) else 0.0


def _straightened(rgb, angle):
    """The page turned back by angle degrees on a canvas large enough to keep
    every corner, and the matrix that maps its pixels back to the page."""
    h, w = rgb.shape[:2]
    a = np.radians(abs(angle))
    nw, nh = int(np.ceil(w * np.cos(a) + h * np.sin(a))), int(np.ceil(h * np.cos(a) + w * np.sin(a)))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    m[0, 2] += (nw - w) / 2
    m[1, 2] += (nh - h) / 2
    out = cv2.warpAffine(rgb, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return out, cv2.invertAffineTransform(m)


def _mapped(boxes, back, w, h):
    """Boxes on the straightened page as boxes on the page: the box around
    each turned box, so it always covers all of it."""
    out = []
    for x, y, bw, bh in boxes:
        corners = np.array([[x, y, 1], [x + bw, y, 1], [x, y + bh, 1], [x + bw, y + bh, 1]], np.float64)
        px, py = (corners @ back.T).T
        x0, y0 = max(0, int(np.floor(px.min()))), max(0, int(np.floor(py.min())))
        x1, y1 = min(w, int(np.ceil(px.max()))), min(h, int(np.ceil(py.max())))
        if x1 > x0 and y1 > y0:
            out.append([x0, y0, x1 - x0, y1 - y0])
    return out


def _runs(flags):
    """(start, end, value) for each run of equal values."""
    runs, start = [], 0
    for i in range(1, len(flags) + 1):
        if i == len(flags) or flags[i] != flags[start]:
            runs.append((start, i, flags[start]))
            start = i
    return runs


def _extend(inked, pos, step, stop, space):
    """From pos, moving by step: through the ink of a letter the box cuts,
    then over one narrow gap and the letter after it. Returns the new edge."""
    def through_ink(p):
        while p != stop and inked[p]:
            p += step
        return p
    pos = through_ink(pos)
    gap = pos
    while gap != stop and not inked[gap] and abs(gap - pos) < space:
        gap += step
    if gap != stop and inked[gap]:  # a narrow gap, then another letter
        pos = through_ink(gap)
    return pos


def _underline(rgb, x, y, bw, bh):
    """The columns (start, end) of a link's underline under a box, or None.
    A link is underlined exactly as wide as its text, so the box is widened
    to it. A line reaching more than a box width past either end (a rule
    across the page or a table) is not taken for an underline."""
    x0, x1 = max(0, x - bw), min(rgb.shape[1], x + 2 * bw)
    rows = rgb[max(0, y + bh // 2):min(rgb.shape[0], y + bh + bh // 4), x0:x1].mean(axis=2)
    if rows.size == 0:
        return None
    a, b = x - x0, x + bw - x0  # the box within the window
    for row in rows < np.median(rows) - 60:
        if b <= a or row[a:b].mean() < 0.9:
            continue  # not a line under (nearly) the whole box
        left, right = a, b
        while left > 0 and row[left - 1]:
            left -= 1
        while right < len(row) and row[right]:
            right += 1
        if left > 0 and right < len(row):
            return x0 + left, x0 + right
    return None


def _completed(rgb, boxes):
    """Boxes from OCR extended sideways so no letter is left partly or
    wholly showing at either end (see EXTEND). Only the middle of the box's
    height is looked at, and rows inked nearly all the way across (a link's
    underline) are left out, so they do not join letters."""
    h, w = rgb.shape[:2]
    out = []
    for x, y, bw, bh in boxes:
        reach = int(EXTEND * bh)
        top, bottom = y + int(0.15 * bh), y + max(int(0.85 * bh), int(0.15 * bh) + 1)
        x0, x1 = max(0, x - reach), min(w, x + bw + reach)
        strip = rgb[max(0, top):min(h, bottom), x0:x1].mean(axis=2)
        if strip.size == 0:
            out.append([x, y, bw, bh])
            continue
        ink = strip < np.median(strip) - 60
        ink = ink[ink.mean(axis=1) < 0.6]  # rows inked nearly across are an underline or a rule, not letters
        inked = list(ink.any(axis=0)) if len(ink) else [False] * strip.shape[1]  # columns with ink
        a, b = x - x0, x + bw - x0  # the box within the strip
        inner = [e - s for s, e, v in _runs(inked[a:b]) if not v and s > 0 and e < b - a]
        space = max(3, 2 * int(np.percentile(inner, 75))) if inner else max(3, bh // 8)
        left = _extend(inked, a - 1, -1, -1, space) + 1
        right = _extend(inked, b, 1, len(inked), space)
        line = _underline(rgb, x, y, bw, bh)
        if line:
            left, right = min(left, line[0] - x0), max(right, line[1] - x0)
        out.append([x0 + left, y, right - left, bh])
    return out


def _found(rgb, detect):
    """detect() on one picture, with its boxes completed in that picture, so
    they are extended along its lines of text. Returns (addresses, boxes,
    sideways lines)."""
    found = detect(rgb)
    return set(found.addresses), _completed(rgb, found.boxes), found.sideways


def ocr_page(rgb, detect):
    """Addresses OCR finds on a drawn page. A slightly turned page (a scan)
    is straightened first, and sideways text is checked again turned.
    detect(rgb) returns an sr.Result. Returns (addresses, boxes)."""
    h, w = rgb.shape[:2]
    angle = skew(rgb)
    straight, back = _straightened(rgb, angle) if abs(angle) >= MIN_SKEW else (rgb, None)
    addresses, boxes, sideways = _found(straight, detect)
    if sideways:
        more, extra = _turned(straight, detect, sideways)
        addresses |= more
        boxes += extra
    if back is not None:
        boxes = _mapped(boxes, back, w, h)
    return addresses, boxes


def _encode(image):
    """A page picture as (data, filter, colour space). JPEG for pages with
    photos, where it is far smaller; lossless for the rest, which keeps text
    sharp and the boxes exactly black. Grey pages are stored as grey."""
    arr = np.asarray(image)
    grey = bool((arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all())
    if grey:
        image, arr = arr[..., 0], arr[..., 0]
    space = b"/DeviceGray" if grey else b"/DeviceRGB"
    flate = zlib.compress(np.ascontiguousarray(arr).tobytes(), 6)
    jpeg = encode_image(image, "JPEG", quality=JPEG_QUALITY)
    if len(jpeg) * 2 < len(flate):
        return jpeg, b"/DCTDecode", space
    return flate, b"/FlateDecode", space


def text_lines(rgb, ocr):
    """The lines of text OCR reads on a redacted page, as (text, x0, y0, x1,
    y1) in pixels: what the invisible text layer is made from."""
    lines = ocr.read(rgb, ocr.lines(rgb), False)
    return [(sr._text(l), l.x0, l.y0, l.x1, l.y1) for l in lines]


def _no_addresses(text):
    """The line without any word holding an "@" (or a full-width one), so no
    address, even one OCR missed on the page, can be copied or searched."""
    return " ".join(w for w in text.split() if "@" not in w and "\uff20" not in w)


def text_layer(lines, scale, h_pt, max_bytes=MAX_OUTPUT):
    """Content-stream operators that write lines invisibly (text rendering
    mode 3) over the page picture, each stretched to the width OCR found, so
    selecting text highlights the right place. scale is points per pixel.
    Courier is used because every letter is the same width, which makes the
    stretch exact; characters outside Windows-1252 become "?"."""
    ops, total = [], 0
    for text, x0, y0, x1, y1 in sorted(lines, key=lambda l: (round(l[2] / max(1.0, l[4] - l[2])), l[1])):
        text = _no_addresses(text)
        width, height = (x1 - x0) * scale, (y1 - y0) * scale
        if not text or width <= 0 or height <= 0 or height > 2 * width:
            continue  # nothing left, or sideways text that OCR cannot read in place
        size = 0.75 * height
        stretch = 100 * width / (0.6 * size * len(text))
        raw = text.encode("cp1252", "replace").replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
        baseline = h_pt - y1 * scale + 0.22 * height
        op = (b"BT 3 Tr /F1 %.2f Tf %.2f Tz 1 0 0 1 %.2f %.2f Tm (%s) Tj ET"
              % (size, stretch, x0 * scale, baseline, raw))
        total += len(op) + bool(ops)
        if total > max_bytes:
            raise PdfError("The redacted PDF exceeds the configured output byte limit.")
        ops.append(op)
    return b"\n".join(ops)


def pdf_chunks(pages, max_bytes=MAX_OUTPUT):
    from .pdf_writer import pdf_chunks as write_chunks, PdfWriteError
    try:
        return write_chunks(pages, max_bytes)
    except PdfWriteError as exc:
        raise PdfError(str(exc)) from exc


class Pages:
    """The pages of a PDF, drawn in a separate process. PDFium has to expand a
    page before it can be measured, and a PDF of a few kilobytes can be built
    so that this needs hundreds of gigabytes or hours (drawings nested inside
    drawings, each drawn several times). The drawing process has a memory and
    a time limit (DRAW_MEMORY, DRAW_SECONDS), so such a PDF ends only that
    process and is refused. waiting() is called while a page is being drawn,
    so a cancelled call stops at once."""

    def __init__(self, fileobj, dpi=DPI, max_pixels=None, waiting=None):
        self.max_pixels = max_pixels
        fd = fileobj.fileno()
        single = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1")
        from .process_support import start_worker
        self.proc = start_worker(
            ([sys.executable, "-m", __package__ + ".pdf_redact"] if __package__ else
             [sys.executable, os.path.abspath(__file__)]) + ["--draw-pages", str(fd), str(dpi), str(int(max_pixels or 0))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, pass_fds=(fd,), memory_limit=DRAW_MEMORY,
            env={**os.environ, **single})
        self.waiting = waiting
        self.buf = bytearray()
        try:
            self.count = self._reply(None)["pages"]
        except BaseException:
            self.close()
            raise

    def page(self, index):
        """(RGB array, page size in points, addresses in the page's text)."""
        self.proc.stdin.write(b"%d\n" % index)
        self.proc.stdin.flush()
        msg = self._reply(index)
        w, h = msg["width"], msg["height"]
        rgb = np.frombuffer(self._take(w * h * 3, index), np.uint8).reshape(h, w, 3).copy()
        return rgb, tuple(msg["size_pt"]), [(address, box) for address, box in msg["hits"]]

    def close(self):
        from .process_support import stop_worker
        stop_worker(self.proc)
        for pipe in (self.proc.stdin, self.proc.stdout):
            with contextlib.suppress(OSError):
                pipe.close()

    def _reply(self, index):
        deadline = time.monotonic() + DRAW_SECONDS
        while b"\n" not in self.buf:
            if len(self.buf) > 1 << 20:
                raise PdfError("The PDF renderer returned an oversized response header.")
            self._more(index, deadline)
        line, _, rest = bytes(self.buf).partition(b"\n")
        if len(line) > 1 << 20:
            raise PdfError("The PDF renderer returned an oversized response header.")
        self.buf = bytearray(rest)
        try:
            msg = json.loads(line)
            if not isinstance(msg, dict):
                raise ValueError()
            if "error" in msg:
                if type(msg["error"]) is not str or len(msg["error"]) > 8192:
                    raise ValueError()
            elif index is None:
                if type(msg.get("pages")) is not int or msg["pages"] < 0:
                    raise ValueError()
            else:
                w, h = msg["width"], msg["height"]
                if (type(w) is not int or type(h) is not int or w <= 0 or h <= 0
                        or (self.max_pixels and w * h > self.max_pixels)):
                    raise ValueError()
                size, hits = msg["size_pt"], msg["hits"]
                if (not isinstance(size, list) or len(size) != 2
                        or any(type(n) not in (int, float) or not math.isfinite(n) or n <= 0 for n in size)
                        or not isinstance(hits, list) or len(hits) > 100_000):
                    raise ValueError()
                for address, box in hits:
                    if (type(address) is not str or len(address) > 8192 or len(box) != 4
                            or any(type(n) is not int for n in box)
                            or box[0] < 0 or box[1] < 0 or box[2] <= 0 or box[3] <= 0
                            or box[0] + box[2] > w or box[1] + box[3] > h):
                        raise ValueError()
        except (ValueError, KeyError, TypeError, RecursionError):
            raise PdfError("The PDF renderer returned an invalid response.") from None
        if "error" in msg:
            raise PdfError(msg["error"])
        return msg

    def _take(self, size, index):
        deadline = time.monotonic() + DRAW_SECONDS
        while len(self.buf) < size:
            self._more(index, deadline)
        data = bytes(self.buf[:size])
        del self.buf[:size]
        return data

    def _more(self, index, deadline):
        what = "Opening the PDF" if index is None else f"Drawing page {index + 1}"
        out = self.proc.stdout.fileno()
        while True:
            if time.monotonic() > deadline:
                raise PdfError(f"{what} takes longer than the {DRAW_SECONDS} s allowed, so it was not redacted.")
            if self.waiting:
                self.waiting()
            from .process_support import read_pipe
            chunk = read_pipe(out, 1 << 20, 0.25)
            if chunk is not None:
                if not chunk:
                    raise PdfError(f"{what} needs more than the {DRAW_MEMORY >> 30} GB of memory allowed (the PDF "
                                   "may be built to exhaust memory), so it was not redacted.")
                self.buf += chunk
                return
            if time.monotonic() > deadline:
                raise PdfError(f"{what} takes longer than the {DRAW_SECONDS} s allowed (the PDF may be built to "
                               "take forever), so it was not redacted.")


def _draw_pages(fd, dpi, max_pixels):
    """The drawing process (see Pages). Answers each page number on stdin
    with a JSON line, followed by the page's RGB pixels."""
    from .process_support import limit_renderer_memory
    limit_renderer_memory(DRAW_MEMORY)
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything a library prints cannot reach the answers

    def send(msg, data=b""):
        out.write(json.dumps(msg).encode() + b"\n")
        out.write(data)
        out.flush()
    try:
        pdf = open_pdf(os.fdopen(fd, "rb"))
    except PdfError as exc:
        send({"error": str(exc)})
        return
    send({"pages": len(pdf)})
    for line in sys.stdin.buffer:
        index = int(line)
        try:
            rgb, size, hits = draw_page(pdf, index, dpi, max_pixels or None)
        except PdfError as exc:
            send({"error": str(exc)})
            continue
        except MemoryError:
            send({"error": f"Drawing page {index + 1} needs more than the {DRAW_MEMORY >> 30} GB of memory allowed, "
                           "so it was not redacted."})
            continue
        send({"width": rgb.shape[1], "height": rgb.shape[0], "size_pt": size, "hits": hits}, rgb.tobytes())


def redact_pdf(fileobj, detect, max_pages=MAX_PAGES, max_pixels=None, dpi=DPI, before_page=None, waiting=None,
               read_text=None, engine="picture", max_output=MAX_OUTPUT):
    """Redact every page. detect(rgb) returns an sr.Result for one page.
    read_text(rgb), if given, returns the lines on a redacted page (see
    text_lines) for the invisible text layer. before_page(index, count) runs
    before each page, for progress reports and for stopping early; waiting()
    runs often while a page is drawn. engine="mupdf" removes the addresses
    from inside the original PDF instead (see pdf_mupdf.py), and falls back
    to pictures when that cannot be done safely. Returns (PDF as a list of
    byte strings, PdfResult)."""
    if engine == "mupdf":
        return _redact_with_mupdf(fileobj, detect, max_pages, max_pixels, dpi, before_page, waiting, read_text, max_output)
    pages = Pages(fileobj, dpi, max_pixels, waiting)
    try:
        count = pages.count
        if count == 0:
            raise PdfError("The PDF has no pages.")
        if max_pages and count > max_pages:
            raise PdfError(f"The PDF has {count} pages; the limit is {max_pages}.")
        result, encoded, size = PdfResult(pages=count), [], 0
        for index in range(count):
            if before_page:
                before_page(index, count)
            rgb, (w_pt, h_pt), hits = pages.page(index)
            addresses, boxes = ocr_page(rgb, detect)
            boxes = sr._merge_boxes(boxes + [box for _, box in hits])
            result.addresses |= addresses
            result.addresses.update(address for address, _ in hits)
            result.boxes.append(boxes)
            result.sizes.append((rgb.shape[1], rgb.shape[0]))
            for x, y, w, h in boxes:
                rgb[y:y + h, x:x + w] = 0
            text = text_layer(read_text(rgb), 72 / dpi, h_pt, max_output - size) if read_text else b""
            data, filt, space = _encode(rgb)
            size += len(data) + len(text)
            if size > max_output:
                raise PdfError("The redacted PDF exceeds the configured output byte limit.")
            encoded.append((data, filt, space, rgb.shape[1], rgb.shape[0], w_pt, h_pt, text))
            del rgb
    finally:
        pages.close()
    return pdf_chunks(encoded, max_output), result


def find_addresses(fileobj, detect, max_pages=MAX_PAGES, max_pixels=None, dpi=DPI, before_page=None, waiting=None):
    """Only find the addresses: a PdfResult with the boxes of each page, and
    the boxes in points from the top-left of each page as displayed."""
    pages = Pages(fileobj, dpi, max_pixels, waiting)
    try:
        count = pages.count
        if count == 0:
            raise PdfError("The PDF has no pages.")
        if max_pages and count > max_pages:
            raise PdfError(f"The PDF has {count} pages; the limit is {max_pages}.")
        result, points = PdfResult(pages=count), []
        for index in range(count):
            if before_page:
                before_page(index, count)
            rgb, _, hits = pages.page(index)
            addresses, boxes = ocr_page(rgb, detect)
            boxes = sr._merge_boxes(boxes + [box for _, box in hits])
            result.addresses |= addresses
            result.addresses.update(address for address, _ in hits)
            result.boxes.append(boxes)
            result.sizes.append((rgb.shape[1], rgb.shape[0]))
            s = 72 / dpi
            points.append([[x * s, y * s, (x + w) * s, (y + h) * s] for x, y, w, h in boxes])
    finally:
        pages.close()
    return result, points


def _redact_with_mupdf(fileobj, detect, max_pages, max_pixels, dpi, before_page, waiting, read_text, max_output):
    if __package__:
        from . import pdf_mupdf
    else:
        import pdf_mupdf
    result, points = find_addresses(fileobj, detect, max_pages, max_pixels, dpi, before_page, waiting)
    try:
        data = pdf_mupdf.redact(fileobj, points, result.addresses, dpi, waiting, max_output=max_output)
    except pdf_mupdf.Fallback as exc:
        # Pages are found again while drawn as pictures; the boxes are the same.
        chunks, again = redact_pdf(fileobj, detect, max_pages, max_pixels, dpi, before_page, waiting, read_text,
                                   max_output=max_output)
        again.fallback = str(exc)
        return chunks, again
    if len(data) > max_output:
        raise PdfError("The redacted PDF exceeds the configured output byte limit.")
    result.engine = "mupdf"
    return [data], result


def main():
    if __package__:
        from . import safe_files as sf
    else:
        import safe_files as sf
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output", help="where to save the redacted PDF (.pdf)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace the output if it already exists; the input is never replaced")
    ap.add_argument("--json", action="store_true", help="print what was found as JSON")
    ap.add_argument("--mode", choices=["accurate", "fast"], default="accurate",
                    help="accurate also finds faint or colour-on-colour text; fast is about twice as fast on CPU")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads to use (default 4)")
    ap.add_argument("--max-pages", type=int, default=MAX_PAGES, help=f"refuse longer PDFs (default {MAX_PAGES})")
    ap.add_argument("--engine", choices=["picture", "mupdf"], default="picture",
                    help="mupdf keeps the original PDF and removes only the addresses (needs PyMuPDF, AGPL; "
                         "see pdf_mupdf.py); picture rebuilds every page as a picture (default)")
    ap.add_argument("--no-text", action="store_true",
                    help="leave out the invisible text layer (the text can then not be selected or searched)")
    ap.add_argument("--max-megapixels", type=float, default=40.0,
                    help=f"refuse pages larger than this at {DPI} dpi (default 40)")
    args = ap.parse_args()
    if args.engine == "mupdf":
        if __package__:
            from . import pdf_mupdf
        else:
            import pdf_mupdf
        if not pdf_mupdf.available():
            sys.exit("pdf_redact: --engine mupdf needs PyMuPDF; install it with install.sh --mupdf.")
    anywhere = sf.Folders(None)
    started = time.perf_counter()
    try:
        source, _, ident = sf.open_input(args.input, anywhere)
        protect = {ident}
        with source:
            sf.check_output(args.output, anywhere, protect, args.overwrite, suffix=".pdf")
            sr.check_models()
            cv2.setNumThreads(args.threads)
            ocr = sr.Ocr(threads=args.threads, faint_pass=args.mode == "accurate", gate_second_read=True,
                         size_guard=True, pdf_forms=True)
            chunks, result = redact_pdf(source, lambda rgb: sr.detect(rgb, ocr), args.max_pages,
                                        args.max_megapixels * 1e6,
                                        before_page=lambda i, n: print(f"page {i + 1} of {n}", file=sys.stderr),
                                        read_text=None if args.no_text else lambda rgb: text_lines(rgb, ocr),
                                        engine=args.engine)
        written, _ = sf.write_file(chunks, args.output, anywhere, protect, args.overwrite, suffix=".pdf")
    except (sf.UnsafePath, PdfError, OSError, RuntimeError) as exc:
        sys.exit(f"pdf_redact: {exc}")
    elapsed = time.perf_counter() - started
    boxes = sum(len(b) for b in result.boxes)
    if args.json:
        print(json.dumps({"pages": result.pages, "addresses": sorted(result.addresses),
                          "boxes": result.boxes, "seconds": round(elapsed, 2)}))
    else:
        if result.fallback:
            print(f"MuPDF was not used ({result.fallback}); the pages were redacted as pictures instead.",
                  file=sys.stderr)
        print(f"{boxes} box(es) over {len(result.addresses)} address(es) on {result.pages} page(s) "
              f"in {elapsed:.1f} s ({result.engine}) -> {written}", file=sys.stderr)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--draw-pages"]:
        _draw_pages(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    else:
        main()
