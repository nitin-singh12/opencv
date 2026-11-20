# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Private minimal binding for the pinned PDFium binary."""
import ctypes as c
import math
import sys
from pathlib import Path
import numpy as np
_library_name = {'linux': 'libpdfium.so', 'darwin': 'libpdfium.dylib', 'win32': 'pdfium.dll'}.get(sys.platform)
if not _library_name:
    raise ImportError('Bundled PDFium is not available for this platform')
try:
    lib = c.CDLL(str(Path(__file__).resolve().parents[1] / '_native' / _library_name))
except OSError as error:
    raise ImportError('Bundled PDFium library is missing or could not be loaded') from error
# C types from the pinned PDFium public headers.
P, I, U, D = c.c_void_p, c.c_int, c.c_ulong, c.c_double
BITMAP_BGR = 2
DRAW_ANNOTATIONS = 1
REVERSE_BYTE_ORDER = 16
OPAQUE_WHITE = 0xffffffff

class FS_RECTF(c.Structure):
    _fields_ = [('left', c.c_float), ('top', c.c_float), ('right', c.c_float), ('bottom', c.c_float)]
CALLBACK = c.CFUNCTYPE(I, P, U, c.POINTER(c.c_ubyte), U)

class Access(c.Structure):
    _fields_ = [('length', U), ('read', CALLBACK), ('param', P)]

# PDFium 156.0.8076.0, fpdf_formfill.h: all callbacks are NULL. The two
# arrays preserve the header's callback ordering around xfa_disabled.
class FormInfo(c.Structure):
    _fields_ = [('version', I), ('callbacks', P * 16), ('xfa_disabled', I), ('extra', P * 17)]

def bind(name, result, *args):
    fn = getattr(lib, name)
    fn.restype = result
    fn.argtypes = args
    return fn
bind('FPDF_InitLibrary', None)()
bind('FPDF_LoadCustomDocument', P, c.POINTER(Access), c.c_char_p)
bind('FPDF_GetLastError', U)
bind('FPDF_GetPageCount', I, P)
bind('FPDF_GetFormType', I, P)
bind('FPDFDOC_InitFormFillEnvironment', P, P, c.POINTER(FormInfo))
bind('FPDFDOC_ExitFormFillEnvironment', None, P)
bind('FPDF_LoadPage', P, P, I)
bind('FPDF_ClosePage', None, P)
bind('FPDF_CloseDocument', None, P)
bind('FORM_OnAfterLoadPage', None, P, P)
bind('FORM_OnBeforeClosePage', None, P, P)
bind('FPDF_GetPageWidthF', c.c_float, P)
bind('FPDF_GetPageHeightF', c.c_float, P)
bind('FPDFBitmap_CreateEx', P, I, I, I, P, I)
bind('FPDFBitmap_Destroy', None, P)
bind('FPDFBitmap_FillRect', I, P, I, I, I, I, U)
bind('FPDFBitmap_GetBuffer', P, P)
bind('FPDFBitmap_GetStride', I, P)
bind('FPDF_RenderPageBitmap', None, P, P, I, I, I, I, I, I)
bind('FPDF_FFLDraw', None, P, P, P, I, I, I, I, I, I)
bind('FPDF_PageToDevice', I, P, I, I, I, I, I, D, D, c.POINTER(I), c.POINTER(I))
bind('FPDFText_LoadPage', P, P)
bind('FPDFText_ClosePage', None, P)
FPDFText_CountChars = bind('FPDFText_CountChars', I, P)
FPDFText_GetUnicode = bind('FPDFText_GetUnicode', c.c_uint, P, I)
FPDFText_IsGenerated = bind('FPDFText_IsGenerated', I, P, I)
FPDFText_GetCharBox = bind('FPDFText_GetCharBox', I, P, I, c.POINTER(D), c.POINTER(D), c.POINTER(D), c.POINTER(D))
FPDFText_GetLooseCharBox = bind('FPDFText_GetLooseCharBox', I, P, I, c.POINTER(FS_RECTF))
FPDF_ERR_PASSWORD = 4

class PdfiumError(Exception):

    def __init__(self, message, err_code=None):
        super().__init__(message)
        self.err_code = err_code

class PdfDocument:

    def __init__(self, file):
        self.file = file
        current = file.tell()
        file.seek(0, 2)
        size = file.tell()
        file.seek(current)

        @CALLBACK
        def read(_param, position, buffer, length):
            try:
                file.seek(position)
                data = file.read(length)
                if len(data) != length:
                    return 0
                c.memmove(buffer, data, length)
                return 1
            except Exception:
                return 0
        self.callback = read
        self.access = Access(size, self.callback, None)
        self.raw = lib.FPDF_LoadCustomDocument(c.byref(self.access), None)
        if not self.raw or lib.FPDF_GetPageCount(self.raw) < 1:
            error = lib.FPDF_GetLastError()
            if self.raw:
                lib.FPDF_CloseDocument(self.raw)
            raise PdfiumError('PDF cannot be loaded', error)
        self.form = None
        self.form_info = None

    def init_forms(self):
        if self.form:
            return
        if lib.FPDF_GetFormType(self.raw):
            self.form_info = FormInfo()
            self.form_info.version = 2
            self.form = lib.FPDFDOC_InitFormFillEnvironment(self.raw, c.byref(self.form_info))
            if not self.form:
                raise PdfiumError('Cannot initialize forms')

    def __len__(self):
        return lib.FPDF_GetPageCount(self.raw)

    def __getitem__(self, index):
        page = lib.FPDF_LoadPage(self.raw, index)
        if not page:
            raise PdfiumError('Cannot load page')
        if self.form:
            lib.FORM_OnAfterLoadPage(page, self.form)
        return Page(page, self)

    def close(self):
        if self.form:
            lib.FPDFDOC_ExitFormFillEnvironment(self.form)
            self.form = None
        if self.raw:
            lib.FPDF_CloseDocument(self.raw)
            self.raw = None

class Page:

    def __init__(self, raw, document):
        self.raw = raw
        self.document = document

    def get_size(self):
        return (lib.FPDF_GetPageWidthF(self.raw), lib.FPDF_GetPageHeightF(self.raw))

    def render(self, scale, rev_byteorder, may_draw_forms, draw_annots):
        (width, height) = self.get_size()
        (w, h) = (math.ceil(width * scale), math.ceil(height * scale))
        raw = lib.FPDFBitmap_CreateEx(w, h, BITMAP_BGR, None, 0)
        if not raw:
            raise MemoryError('PDF bitmap allocation failed')
        try:
            flags = (REVERSE_BYTE_ORDER if rev_byteorder else 0) | (DRAW_ANNOTATIONS if draw_annots else 0)
            if not lib.FPDFBitmap_FillRect(raw, 0, 0, w, h, OPAQUE_WHITE):
                raise PdfiumError('Cannot fill PDF bitmap')
            lib.FPDF_RenderPageBitmap(raw, self.raw, 0, 0, w, h, 0, flags)
            if self.document.form and may_draw_forms:
                lib.FPDF_FFLDraw(self.document.form, raw, self.raw, 0, 0, w, h, 0, flags)
            return Bitmap(raw, self, w, h)
        except BaseException:
            lib.FPDFBitmap_Destroy(raw)
            raise

    def get_textpage(self):
        raw = lib.FPDFText_LoadPage(self.raw)
        if not raw:
            raise PdfiumError('Cannot load page text')
        return TextPage(raw)

    def close(self):
        if self.raw:
            if self.document.form:
                lib.FORM_OnBeforeClosePage(self.raw, self.document.form)
            lib.FPDF_ClosePage(self.raw)
            self.raw = None

class TextPage:

    def __init__(self, raw):
        self._as_parameter_ = raw

    def close(self):
        if self._as_parameter_:
            lib.FPDFText_ClosePage(self._as_parameter_)
            self._as_parameter_ = None

class Bitmap:

    def __init__(self, raw, page, width, height):
        self.raw = raw
        self.page = page
        self.width = width
        self.height = height

    def to_numpy(self):
        stride = lib.FPDFBitmap_GetStride(self.raw)
        ptr = lib.FPDFBitmap_GetBuffer(self.raw)
        data = np.ctypeslib.as_array((c.c_ubyte * (stride * self.height)).from_address(ptr))
        return data.reshape(self.height, stride)[:, :self.width * 3].reshape(self.height, self.width, 3)

    def get_posconv(self, page):
        return self

    def to_bitmap(self, x, y):
        (bx, by) = (I(), I())
        if not lib.FPDF_PageToDevice(self.page.raw, 0, 0, self.width, self.height, 0, x, y, c.byref(bx), c.byref(by)):
            raise PdfiumError('Cannot translate page coordinates')
        return (bx.value, by.value)

    def close(self):
        if self.raw:
            lib.FPDFBitmap_Destroy(self.raw)
            self.raw = None
