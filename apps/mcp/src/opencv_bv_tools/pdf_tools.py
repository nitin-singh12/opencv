"""PDF email redaction adapter sharing the image tools' resident OCR service."""

import contextlib
import json
from pathlib import Path

import anyio.from_thread
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import StrictBool

from opencv_bv_tools.email_tools import _readings
from opencv_bv_tools.runtime.context import annotations
from opencv_bv_tools.runtime.email_modes import MODES


def register_tools(server, context, service):
    @server.tool(
        annotations=annotations("Redact email addresses in a PDF", writes=True)
    )
    def redact_pdf(
        pdf_path: str,
        output_path: str = "",
        mode: str = context.config.mode,
        overwrite: StrictBool = False,
        reveal_addresses: StrictBool = False,
        ctx: Context = None,
    ):
        """Redact email addresses in scanned or text PDFs into a separate PDF.

        Checks page images with OCR and the PDF's own text. The default picture
        engine rebuilds pages with black boxes and a searchable text layer read
        from the redacted pictures; original metadata, links, attachments and
        revisions are discarded. Optional server --pdf-engine mupdf preserves
        more original content, with verification and a picture fallback.
        Only view the redacted output and review it for detection misses.
        Output defaults to <name>.redacted.pdf; overwriting needs overwrite=true.
        Input is limited to 32 MiB, output to 128 MiB, and pages to the server's
        --max-pages setting. Addresses stay masked unless server --allow-reveal
        and reveal_addresses=true are both enabled. Long PDFs may take minutes.
        """
        with context.call():
            if mode not in MODES:
                raise ToolError(f"mode must be one of {sorted(MODES)}")
            context.require(
                "pdf", "rapidocr", "onnxruntime", "cv2", "numpy", "pypdfium2"
            )
            if context.config.pdf_engine == "mupdf":
                context.require("mupdf", "pymupdf")
            data, _, protect = context.read(pdf_path)
            source = Path(pdf_path).expanduser()
            out = output_path or str(source.with_name(f"{source.stem}.redacted.pdf"))
            context.file_ops.check_output(
                out, context.folders, protect, overwrite, suffix=".pdf"
            )

            def progress(index, count):
                context.stop_if_cancelled()
                if ctx is not None:
                    with contextlib.suppress(Exception):
                        anyio.from_thread.run(
                            ctx.report_progress,
                            index,
                            count,
                            f"checking page {index + 1} of {count}",
                        )

            result, payload = service.run_pdf(data, mode, progress)
            with payload:

                def chunks():
                    while True:
                        context.stop_if_cancelled()
                        chunk = payload.read(65536)
                        if not chunk:
                            break
                        yield chunk

                written = context.write(chunks(), out, protect, overwrite, ".pdf")
            addresses, note = _readings(
                result["addresses"],
                reveal_addresses,
                context.config.allow_reveal,
                "OCR or from the PDF's text",
            )
            how = (
                "Pages were rebuilt from redacted pictures with a text layer read from those pictures. "
                "Original metadata, links, form data, attachments and revisions are not carried over."
            )
            if result["engine"] == "mupdf":
                how = "MuPDF removed detected addresses from the original PDF and verified the result."
            elif result["fallback"]:
                how += " MuPDF could not verify native redaction, so the picture engine was used."
            return json.dumps(
                dict(
                    output_path=written,
                    pages=result["pages"],
                    boxes_redacted=sum(result["boxes_per_page"]),
                    addresses=addresses,
                    addresses_note=note,
                    boxes_per_page={
                        str(i + 1): n
                        for i, n in enumerate(result["boxes_per_page"])
                        if n
                    },
                    mode=mode,
                    seconds=round(result["seconds"], 2),
                    engine=result["engine"],
                    note=how,
                ),
                indent=1,
            )
