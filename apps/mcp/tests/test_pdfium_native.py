"""Regression checks for the private PDFium binding and its resource cleanup."""
import ctypes
import io

import pytest
from reportlab.pdfgen import canvas

from opencv_bv_tools.runtime import pdfium_native as native


def form_pdf():
    output = io.BytesIO()
    document = canvas.Canvas(output, pagesize=(300, 200))
    document.acroForm.textfield(name="mail", value="form@example.com", x=10, y=100, width=200, height=30)
    document.showPage()
    document.save()
    output.seek(0)
    return output


def test_form_layout_matches_reference_binding():
    import pypdfium2.raw as reference
    assert ctypes.sizeof(native.FormInfo) == ctypes.sizeof(reference.FPDF_FORMFILLINFO)
    assert native.FormInfo.callbacks.offset == reference.FPDF_FORMFILLINFO.Release.offset
    assert native.FormInfo.callbacks.offset + 15 * ctypes.sizeof(ctypes.c_void_p) == reference.FPDF_FORMFILLINFO.m_pJsPlatform.offset
    assert native.FormInfo.xfa_disabled.offset == reference.FPDF_FORMFILLINFO.xfa_disabled.offset
    assert native.FormInfo.extra.offset == reference.FPDF_FORMFILLINFO.FFI_DisplayCaret.offset


def test_form_init_and_resource_closes_are_idempotent():
    document = native.PdfDocument(form_pdf())
    document.init_forms()
    first_form = document.form
    assert first_form
    document.init_forms()
    assert document.form == first_form
    page = document[0]
    bitmap = page.render(scale=1, rev_byteorder=True, may_draw_forms=True, draw_annots=True)
    text = page.get_textpage()
    assert bitmap.to_numpy().shape == (200, 300, 3)
    text.close()
    text.close()
    bitmap.close()
    bitmap.close()
    page.close()
    page.close()
    document.close()
    document.close()


def test_bitmap_is_released_when_fill_fails(monkeypatch):
    document = native.PdfDocument(form_pdf())
    page = document[0]
    destroyed = []
    original = native.lib.FPDFBitmap_Destroy
    monkeypatch.setattr(native.lib, "FPDFBitmap_FillRect", lambda *args: 0)
    def destroy(handle):
        destroyed.append(handle)
        original(handle)
    monkeypatch.setattr(native.lib, "FPDFBitmap_Destroy", destroy)
    try:
        with pytest.raises(native.PdfiumError, match="fill"):
            page.render(scale=1, rev_byteorder=True, may_draw_forms=True, draw_annots=True)
        assert len(destroyed) == 1
    finally:
        page.close()
        document.close()


def test_invalid_pdf_is_rejected():
    with pytest.raises(native.PdfiumError):
        native.PdfDocument(io.BytesIO(b"not a PDF"))
