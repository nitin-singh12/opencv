#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""True redaction of PDFs with MuPDF: an optional engine for pdf_redact.py.

With this engine the redacted PDF stays the original PDF: same text, fonts,
vector drawings, links and layout, so its text can be selected, searched and
edited exactly as before. Only the addresses are removed from inside it: the
characters themselves, the pixels of any image under a box, and drawn lines
that lie entirely under a box. A black box is drawn where each address was.

Licence: MuPDF and PyMuPDF are available under the GNU AGPL 3.0 or under a
commercial licence from Artifex. They are not part of the default install;
`install.sh --mupdf` adds them, and the server only uses them when started
with `--pdf-engine mupdf`. Check that your use fits one of those licences.

The addresses are found exactly as for the picture method (pdf_redact.py:
the page's own text and OCR). MuPDF then works in a separate process with
the same memory and time limits as the drawing process, and checks its own
result before anything is returned:

  - under every box, nothing of the original remains: the page is redacted a
    second time without the black fill and drawn, and no dark pixel of the
    original may be left there;
  - no text remains under a box, and no address is left in the page text;
  - no address, and nothing of the addresses found, is left anywhere in the
    file: metadata, bookmarks, links, form data, attachments, scripts and
    every other object are searched.

If any check fails, or the PDF has layers that are hidden now but could be
switched on later, the caller is told why and falls back to the picture
method, so a PDF is never returned with an address still recoverable.
"""
import contextlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

if __package__:
    from . import image_redact as sr
else:
    import image_redact as sr

# Pages are redacted in a separate process with these limits (as for drawing).
MEMORY = 2 << 30
SECONDS = 600        # for the whole document
LEFTOVER = 0.02      # share of an address's dark pixels allowed to remain under its box (noise)
MARGIN = 1.0         # points added around each box for MuPDF, so image pixels the edge cuts are cleared too
CORE = 0.3           # share of a box's height along which its text is removed (see _apply)

ANYWHERE = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}")


class Fallback(Exception):
    """MuPDF could not redact this PDF safely; the message says why."""


def available():
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        return False
    return True


def redact(fileobj, boxes, addresses, dpi, waiting=None, max_output=1 << 30):
    """Redact the PDF open in fileobj. boxes: for each page, [x0, y0, x1,
    y1] in points from the top-left of the page as displayed; addresses: the
    addresses found. Returns the redacted PDF as bytes, or raises Fallback."""
    fd = fileobj.fileno()
    os.lseek(fd, 0, os.SEEK_SET)
    single = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1")
    from .process_support import start_worker, stop_worker, read_pipe
    proc = start_worker(([sys.executable, "-m", __package__ + ".pdf_mupdf"] if __package__ else
                             [sys.executable, os.path.abspath(__file__)]) + ["--redact", str(fd)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            pass_fds=(fd,), memory_limit=MEMORY, env={**os.environ, **single})
    try:
        proc.stdin.write(json.dumps({"boxes": boxes, "addresses": sorted(addresses), "dpi": dpi,
                                     "max_output": max_output}).encode())
        proc.stdin.close()
        out, buf, deadline = proc.stdout.fileno(), bytearray(), time.monotonic() + SECONDS
        while True:
            if time.monotonic() > deadline:
                raise Fallback(f"MuPDF took longer than {SECONDS} s")
            if waiting:
                waiting()
            chunk = read_pipe(out, 1 << 20, 0.25)
            if chunk is not None:
                if not chunk:
                    break
                buf += chunk
                if len(buf) > max_output + 4096:
                    raise Fallback("MuPDF exceeded the configured output byte limit")
            elif time.monotonic() > deadline:
                raise Fallback(f"MuPDF took longer than {SECONDS} s")
        line, _, data = bytes(buf).partition(b"\n")
        try:
            msg = json.loads(line)
        except ValueError:
            raise Fallback("MuPDF stopped unexpectedly (the PDF may need more memory than allowed)") from None
        if not msg.get("ok"):
            raise Fallback(msg.get("reason", "unknown reason"))
        if len(data) != msg["size"]:
            raise Fallback("MuPDF's answer was incomplete")
        return data
    finally:
        stop_worker(proc)
        with contextlib.suppress(OSError):
            proc.stdout.close()


# --------------------------------------------------------------------------
# The redacting process
# --------------------------------------------------------------------------

def _glyphs(page):
    """Each letter on the page with its tight outline (not the font's much
    taller box), as rectangles on the page as displayed."""
    import pymupdf
    flags = pymupdf.TEXT_ACCURATE_BBOXES | pymupdf.TEXT_PRESERVE_LIGATURES | pymupdf.TEXT_MEDIABOX_CLIP
    back = page.rotation_matrix
    return [(pymupdf.Rect(ch["bbox"]) * back, ch["c"])
            for block in page.get_text("rawdict", flags=flags)["blocks"] for line in block.get("lines", [])
            for span in line["spans"] for ch in span["chars"] if ch["c"].strip()]


def _core(box):
    """The band through the middle of a box (as displayed) along which text is
    removed: CORE of its height, or of its width for sideways text."""
    import pymupdf
    x0, y0, x1, y1 = box
    if y1 - y0 <= x1 - x0:
        mid, half = (y0 + y1) / 2, CORE * (y1 - y0) / 2
        return pymupdf.Rect(x0 - MARGIN, mid - half, x1 + MARGIN, mid + half)
    mid, half = (x0 + x1) / 2, CORE * (x1 - x0) / 2
    return pymupdf.Rect(mid - half, y0 - MARGIN, mid + half, y1 + MARGIN)


def _apply(doc, boxes, fill):
    """Redact every box; with fill=False nothing is drawn in their place.
    MuPDF removes a letter whenever its outline touches a redaction area,
    and a box reaches a little into the lines above and below, so text is
    removed only along the middle band of each box (_core): every letter of
    the address crosses it, letters of the next line do not. Where the box
    holds no picture, the black bar is drawn over the removed letters only,
    so the lines above and below stay fully visible; over a picture (a scan)
    its pixels are cleared, and the bar drawn, over the whole box."""
    import pymupdf
    for index, page_boxes in enumerate(boxes):
        if not page_boxes:
            continue
        page = doc[index]
        turn, back = page.derotation_matrix, page.rotation_matrix
        pictures = [r * back for img in page.get_images(full=True) for r in page.get_image_rects(img[0])]
        letters = [r for r, _ in _glyphs(page)]
        cores = [_core(b) for b in page_boxes]
        bars = []
        for box, core in zip(page_boxes, cores):
            full = pymupdf.Rect(box) + (-MARGIN, -MARGIN, MARGIN, MARGIN)
            removed = [r for r in letters if r.intersects(core)]
            if removed and not any(full.intersects(p) for p in pictures):
                bar = pymupdf.Rect(removed[0])
                for r in removed[1:]:
                    bar |= r
                bars.append(bar + (-MARGIN, -MARGIN, MARGIN, MARGIN))
            else:
                bars.append(full)
            page.add_redact_annot(core * turn, fill=False, cross_out=False)
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                              text=pymupdf.PDF_REDACT_TEXT_REMOVE)
        for bar in bars:
            page.add_redact_annot(bar * turn, fill=(0, 0, 0) if fill else False, cross_out=False)
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS,
                              graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                              text=pymupdf.PDF_REDACT_TEXT_NONE)
        rects = [pymupdf.Rect(b) * turn for b in page_boxes]
        for link in page.get_links():
            target = " ".join(str(v) for v in link.values() if isinstance(v, str))
            if "@" in target or "mailto:" in target.lower() or any(link["from"].intersects(r) for r in rects):
                page.delete_link(link)


def _jpeg_places(doc, boxes):
    """Where each page with a box shows a JPEG picture: {page: [rect, ...]}."""
    places = {}
    for index, page_boxes in enumerate(boxes):
        if page_boxes:
            places[index] = [i["bbox"] for i in doc[index].get_image_info(xrefs=True)
                             if i.get("xref") and doc.xref_get_key(i["xref"], "Filter")[1] == "/DCTDecode"]
    return places


def _as_jpeg(doc, places, quality=90):
    """MuPDF stores a picture it redacted losslessly, which can make a scan
    many times larger; a picture that was a JPEG is stored as a JPEG again.
    Its redacted pixels are already gone, so this cannot bring them back."""
    import pymupdf
    for index, rects in places.items():
        page = doc[index]
        for info in page.get_image_info(xrefs=True):
            xref = info.get("xref")
            if not xref or doc.xref_get_key(xref, "Filter")[1] == "/DCTDecode":
                continue
            if not any(pymupdf.Rect(info["bbox"]).intersects(pymupdf.Rect(r)) for r in rects):
                continue
            pix = pymupdf.Pixmap(doc, xref)
            if pix.alpha or pix.n not in (1, 3):
                continue
            page.replace_image(xref, stream=pix.tobytes("jpeg", jpg_quality=quality))


def _clean(doc):
    """Remove addresses from the rest of the document."""
    meta = {k: v for k, v in (doc.metadata or {}).items()
            if k in ("title", "author", "subject", "keywords", "creator", "producer", "creationDate", "modDate")}
    doc.set_metadata({k: ("" if v and "@" in v else v or "") for k, v in meta.items()})
    toc = doc.get_toc(simple=False)
    if any("@" in entry[1] for entry in toc):
        for entry in toc:
            entry[1] = sr.EMAIL_RE.sub("[redacted]", entry[1]).replace("@", " ")
        doc.set_toc(toc)
    doc.scrub(attached_files=True, clean_pages=True, embedded_files=True, hidden_text=False, javascript=True,
              metadata=False, redactions=False, redact_images=0, remove_links=False, reset_fields=True,
              reset_responses=True, thumbnails=True, xml_metadata=True)


def _dark(pix):
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].mean(axis=2)
    return arr


def _may_stay(page, box):
    """What may still be drawn under a box after redaction, as rectangles on
    the page as displayed: any letter still there (the letters are checked
    on their own by _check_text, which makes sure none of them is part of an
    address), and straight
    lines and single rectangles that run on beyond the box (underlines,
    table rules, backgrounds, frames). Anything else, such as a shape made of
    curves or of many strokes or rectangles, is not in the list: that is how
    text turned into outlines would be drawn."""
    import pymupdf
    turn = page.rotation_matrix
    rects = [r for r, _ in _glyphs(page) if r.intersects(box)]
    for d in page.get_drawings():
        r = d["rect"] * turn
        if not r.intersects(box) or box.contains(r):
            continue
        kinds = {item[0] for item in d["items"]}
        straight = all(item[0] != "l" or abs(item[1].x - item[2].x) < 0.01 or abs(item[1].y - item[2].y) < 0.01
                       for item in d["items"])
        single_rect = ((kinds <= {"re", "qu"} and len(d["items"]) <= 2)  # a background or a frame,
                       or (kinds == {"l"} and straight and len(d["items"]) <= 5))  # or one drawn as four edges
        thin_line = kinds <= {"l", "re", "qu"} and straight and min(r.width, r.height) <= 3  # an underline or rule
        if single_rect or thin_line:
            rects.append(r)
    return rects


def _check_pixels(original, probe, boxes, dpi):
    """Draw each page with boxes from the original and from the probe (the
    same redaction without the black fill). Dark pixels of the original under
    a box must be gone in the probe, apart from what may stay (_may_stay).
    Returns a problem or None."""
    import pymupdf
    s = dpi / 72
    matrix = pymupdf.Matrix(s, s)
    for index, page_boxes in enumerate(boxes):
        if not page_boxes:
            continue
        before = _dark(original[index].get_pixmap(matrix=matrix, alpha=False))
        after = _dark(probe[index].get_pixmap(matrix=matrix, alpha=False))
        paper = np.median(before)
        for x0, y0, x1, y1 in page_boxes:
            area = (slice(max(0, int(y0 * s)), int(y1 * s) + 1), slice(max(0, int(x0 * s)), int(x1 * s) + 1))
            b, a = before[area], after[area]
            if b.size == 0 or a.shape != b.shape:
                continue
            left = (b < paper - 60) & (a < paper - 60)
            for r in _may_stay(probe[index], pymupdf.Rect(x0, y0, x1, y1)):
                left[max(0, int((r.y0 - y0) * s)):max(0, int((r.y1 - y0) * s) + 1),
                     max(0, int((r.x0 - x0) * s)):max(0, int((r.x1 - x0) * s) + 1)] = False
            if _letter_like(left, 3 * s) > LEFTOVER * max(1, (b < paper - 60).sum()):
                return f"something under a box on page {index + 1} could not be removed"
    return None


def _letter_like(left, thin):
    """Pixels of what is left under a box that could be (part of) a letter.
    A piece that runs out of the box and is a thin bar or a solid rectangle
    is an underline, a rule or a background, and does not count; a letter, or
    a dot of a dot-matrix letter, stays inside the box or is not solid."""
    import cv2
    count, _, stats, _ = cv2.connectedComponentsWithStats(left.astype(np.uint8), connectivity=8)
    h, w = left.shape
    total = 0
    for x, y, bw, bh, area in stats[1:count]:
        leaves = x == 0 or y == 0 or x + bw == w or y + bh == h
        if leaves and (min(bw, bh) <= thin or area >= 0.9 * bw * bh):
            continue
        total += area
    return total


def _check_text(doc, boxes, addresses):
    """No address in the text of any page, no "@" left under a box, and
    nothing left on the line of an address: every letter crossing the middle
    band of a box (_core) should have been removed."""
    import pymupdf
    for index, page in enumerate(doc):
        text = page.get_text()
        if sr.find_emails_pdf(sr.normalise_ocr_text(text)) or any(a in text for a in addresses):
            return f"an address is still in the text of page {index + 1}"
        glyphs = _glyphs(page)
        for b in boxes[index] if index < len(boxes) else []:
            box, core = pymupdf.Rect(b), _core(b) + (MARGIN, 0, -MARGIN, 0)
            sideways = core.height > core.width
            for rect, ch in glyphs:
                if not rect.intersects(box):
                    continue
                if ch in "@\uff20":
                    return f"an @ is still under a box on page {index + 1}"
                part = rect & core
                across = part.width >= 0.5 * core.width if sideways else part.height >= 0.5 * core.height
                if part.width > 0.5 and part.height > 0.5 and across:
                    return f"text is still on the line of an address on page {index + 1}"
    return None


def _check_file(data, addresses):
    """No address anywhere in the saved file: every object and every stream
    that is not a font or an image, unpacked."""
    import pymupdf
    doc = pymupdf.open(stream=data, filetype="pdf")
    needles = [a.encode("latin-1", "ignore") for a in addresses]
    try:
        # Font programs often carry their maker's address in a copyright
        # notice; they hold glyph shapes, not page text, so they are skipped.
        fonts = set()
        for xref in range(1, doc.xref_length()):
            fonts |= {int(n) for n in re.findall(r"/FontFile\d?\s+(\d+)\s+0\s+R", doc.xref_object(xref, compressed=False))}
        for xref in range(1, doc.xref_length()):
            source = doc.xref_object(xref, compressed=False).encode("latin-1", "replace")
            chunks = [source]
            if doc.xref_is_stream(xref) and xref not in fonts and not doc.xref_is_image(xref):
                with contextlib.suppress(Exception):
                    chunks.append(doc.xref_stream(xref) or b"")
            for chunk in chunks:
                if ANYWHERE.search(chunk) or any(n and n in chunk for n in needles):
                    return f"an address is still in the file (object {xref})"
    finally:
        doc.close()
    return None


def _hidden_layers(doc):
    """True when the PDF has optional content (layers) that is not on screen
    now but could be shown later: a layer that is off, other layer settings
    to switch to, visibility that depends on zoom, language or user, a state
    set to OFF for viewing, printing or export, or content shown only while
    a layer is off. Layers that are all on with none of that, like
    Acrobat's "Watermark" and "Headers/Footers", hide nothing: switching
    them off can only hide more."""
    layers = doc.get_ocgs()
    if not layers:
        return False
    if any(not v["on"] for v in layers.values()) or doc.get_layers():
        return True
    for xref in range(1, doc.xref_length()):
        o = doc.xref_object(xref, compressed=False)
        if re.search(r"/Type\s*/OCMD", o) and (re.search(r"/P\s*/(AllOff|AnyOff)", o) or "/VE" in o):
            return True  # shown only while a layer is off
        if re.search(r"/Type\s*/OCG\b", o) and (re.search(r"State\s*/OFF", o) or re.search(r"/(Zoom|Language|User)\b", o)):
            return True  # hidden for some use, zoom, language or user
        if "/OCProperties" in o or re.search(r"/OCGs\s*\[", o):
            if "/Configs" in o or re.search(r"/OFF\s*\[\s*\d", o) or re.search(r"/BaseState\s*/(OFF|Unchanged)", o):
                return True  # other layer settings, or layers off by default
    return False


def _redact_process(fd):
    from .process_support import limit_renderer_memory
    limit_renderer_memory(MEMORY)
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything a library prints cannot reach the answer

    def send(msg, data=b""):
        out.write(json.dumps(msg).encode() + b"\n" + data)
        out.flush()
    try:
        import pymupdf
        pymupdf.TOOLS.mupdf_display_errors(False)
        job = json.loads(sys.stdin.buffer.read())
        boxes, addresses, dpi = job["boxes"], job["addresses"], job["dpi"]
        with os.fdopen(fd, "rb") as f:
            source = f.read()
        doc = pymupdf.open(stream=source, filetype="pdf")
        if doc.needs_pass:
            return send({"ok": False, "reason": "the PDF is protected by a password"})
        if len(doc) != len(boxes):
            return send({"ok": False, "reason": "MuPDF sees a different number of pages than PDFium"})
        if _hidden_layers(doc):
            return send({"ok": False, "reason": "the PDF has layers that are hidden now but can be switched on later"})
        doc.bake(annots=True, widgets=True)  # comments and form fields become part of the page, so they are redacted too
        original = pymupdf.open(stream=doc.tobytes(), filetype="pdf")
        probe = pymupdf.open(stream=doc.tobytes(), filetype="pdf")
        _apply(probe, boxes, fill=False)
        problem = _check_pixels(original, probe, boxes, dpi)
        if problem:
            return send({"ok": False, "reason": problem})
        jpegs = _jpeg_places(doc, boxes)
        _apply(doc, boxes, fill=True)
        _as_jpeg(doc, jpegs)
        _clean(doc)
        data = doc.tobytes(garbage=4, clean=True, deflate=True)
        if len(data) > job.get("max_output", 1 << 30):
            return send({"ok": False, "reason": "MuPDF exceeded the configured output byte limit"})
        result = pymupdf.open(stream=data, filetype="pdf")
        problem = _check_text(result, boxes, addresses) or _check_file(data, addresses)
        if problem:
            return send({"ok": False, "reason": problem})
        send({"ok": True, "size": len(data)}, data)
    except MemoryError:
        send({"ok": False, "reason": "MuPDF needed more memory than allowed"})
    except Exception as exc:  # any MuPDF failure means: use the picture method
        send({"ok": False, "reason": f"MuPDF could not process the PDF ({type(exc).__name__})"})


if __name__ == "__main__":
    if sys.argv[1:2] == ["--redact"]:
        _redact_process(int(sys.argv[2]))
