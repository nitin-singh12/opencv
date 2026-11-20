# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Small, bounded PDF serializer for reconstructed page images."""


class PdfWriteError(ValueError):
    pass


def pdf_chunks(pages, max_bytes, *, placements=None, creator=False):
    """A PDF with one picture per page, and optionally an invisible text
    layer, as a list of byte strings. pages: (data, filter, colour space,
    width px, height px, width pt, height pt, text operators) for each page.
    Nothing else is written: no metadata, links or document ID."""
    chunks = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets = []
    pos = len(chunks[0])

    def add(body, stream=None):
        nonlocal pos
        offsets.append(pos)
        head = b"%d 0 obj\n" % len(offsets) + body
        parts = (
            [head, b"\nendobj\n"]
            if stream is None
            else [head, b"\nstream\n", stream, b"\nendstream\nendobj\n"]
        )
        if pos + sum(map(len, parts)) > max_bytes:
            raise PdfWriteError(
                "The redacted PDF exceeds the configured output byte limit."
            )
        chunks.extend(parts)
        pos += sum(len(p) for p in parts)

    n = len(pages)
    kids = b" ".join(b"%d 0 R" % (4 + 3 * i) for i in range(n))
    add(b"<< /Type /Catalog /Pages 2 0 R >>")
    add(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, n))
    add(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>"
    )  # object 3
    for i, (data, filt, space, w_px, h_px, w_pt, h_pt, text) in enumerate(pages):
        page, content, image = 4 + 3 * i, 5 + 3 * i, 6 + 3 * i
        size = b"%.4f %.4f" % (w_pt, h_pt)
        add(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %s] /Resources << /XObject << /Im0 %d 0 R >> "
            b"/Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % (size, image, content)
        )
        draw = b"q %.4f 0 0 %.4f 0 0 cm /Im0 Do Q" % (w_pt, h_pt)
        if placements is not None:
            matrix, display = placements[i]
            draw = (
                b"1 1 1 rg 0 0 %.4f %.4f re f\nq " % (w_pt, h_pt)
                + b" ".join(("%.6f" % value).encode() for value in matrix)
                + b" cm "
                + b"%.6f 0 0 %.6f 0 0 cm /Im0 Do Q" % tuple(display)
            )
        if text:
            draw += b"\n" + text
        add(b"<< /Length %d >>" % len(draw), draw)
        add(
            b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace %s /BitsPerComponent 8 "
            b"/Filter %s /Length %d >>" % (w_px, h_px, space, filt, len(data)),
            data,
        )
    if creator:
        add(b"<< /Creator (opencv-bv-tools) /Title (Rotated image) >>")
    xref = pos
    table = [b"xref\n0 %d\n0000000000 65535 f \n" % (len(offsets) + 1)]
    table += [b"%010d 00000 n \n" % off for off in offsets]
    table.append(
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(offsets) + 1, xref)
    )
    if creator:
        table[-1] = table[-1].replace(
            b"/Root 1 0 R", b"/Root 1 0 R /Info %d 0 R" % len(offsets)
        )
    if pos + sum(map(len, table)) > max_bytes:
        raise PdfWriteError(
            "The redacted PDF exceeds the configured output byte limit."
        )
    return chunks + table
