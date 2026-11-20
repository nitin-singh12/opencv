# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""PDF job orchestration inside the resident OCR child; no parent imports."""
import hashlib
import os
import tempfile
import time

from opencv_bv_tools.runtime.ocr_process import MAX_PDF_BYTES


def _error(reason):
    return {"status": "error", "reason": reason, "pdf_bytes": 0}, []


def process_pdf(data, options, state, get_engine, progress):
    try:
        from opencv_bv_tools.runtime import image_redact as sr
        from opencv_bv_tools.runtime import pdf_redact as pr
        if options["engine"] == "mupdf":
            import pymupdf  # noqa: F401; explicit dependency, never silently enabled
    except ImportError:
        return _error("missing_pdf_dependency")
    try:
        engine = get_engine(sr, options, state)
    except RuntimeError:
        return _error("model_verification_failed")
    except Exception:
        return _error("ocr_initialization_failed")
    started = time.perf_counter()
    try:
        # Unlinked/anonymous input cannot leave source bytes behind on SIGKILL.
        with tempfile.TemporaryFile() as source:
            source.write(data)
            source.seek(0)
            chunks, result = pr.redact_pdf(source, lambda rgb: sr.detect(rgb, engine),
                max_pages=options["max_pages"], max_pixels=options["max_pixels"],
                before_page=progress, read_text=lambda rgb: pr.text_lines(rgb, engine),
                engine=options["engine"], max_output=MAX_PDF_BYTES)
        addresses = sorted(result.addresses)
        if len(addresses) > 4096 or any(len(address) > 8192 for address in addresses):
            return _error("pdf_output")
        size, digest = 0, hashlib.sha256()
        for chunk in chunks:
            size += len(chunk)
            if size > MAX_PDF_BYTES:
                return _error("pdf_output")
            digest.update(chunk)
        return dict(status="ok", pdf_bytes=size, pdf_sha256=digest.hexdigest(), pages=result.pages,
                    boxes_per_page=[len(boxes) for boxes in result.boxes], addresses=addresses,
                    seconds=time.perf_counter() - started, pid=os.getpid(), engine=result.engine,
                    fallback=bool(result.fallback)), chunks
    except pr.PdfError as exc:
        message = str(exc).lower()
        reason = ("pdf_password" if "password" in message else
                  "pdf_pages" if "no pages" in message or "pages; the limit" in message else
                  "pdf_pixels" if "megapixels" in message else
                  "pdf_output" if "output byte limit" in message else
                  "pdf_memory" if "memory" in message else
                  "pdf_timeout" if "longer than" in message else "pdf_invalid")
        return _error(reason)
    except Exception:
        return _error("pdf_failed")
    finally:
        engine.pdf_forms = False
