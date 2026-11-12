# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Email tools with on-demand OCR, reuse between calls and idle retirement."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace

from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError
from .runtime import image_io
from pydantic import StrictBool

from opencv_bv_tools.runtime.context import MAX_INPUT_BYTES, annotations
from opencv_bv_tools.runtime.email_modes import MODES
from opencv_bv_tools.runtime.ocr_process import OCRProcess, OCRWorkerError


class EmailService:
    """One private worker, reused while active and reaped after inactivity."""

    def __init__(self, context):
        self.context = context
        self.worker = OCRProcess(context.config)

    def close(self):
        self.worker.close()

    def run(self, data, mode, *, redact=False):
        if mode not in MODES:
            raise ToolError(f"mode must be one of {sorted(MODES)}")
        try:
            return self.worker.request(
                data,
                mode=mode,
                redact=redact,
                check_cancelled=self.context.stop_if_cancelled,
            )
        except OCRWorkerError as exc:
            raise ToolError(f"Email processing failed: {exc}.") from None

    def run_pdf(self, data, mode, progress=None):
        if mode not in MODES:
            raise ToolError(f"mode must be one of {sorted(MODES)}")
        try:
            return self.worker.request(
                data,
                mode=mode,
                redact=True,
                pdf=True,
                progress=progress,
                check_cancelled=self.context.stop_if_cancelled,
            )
        except OCRWorkerError as exc:
            raise ToolError(f"PDF processing failed: {exc}") from None

    def load(self, image_path, image_base64):
        if bool(image_path) == bool(image_base64):
            raise ToolError("Give exactly one of image_path or image_base64.")
        self.context.require("ocr", "rapidocr", "onnxruntime", "cv2", "numpy")
        real, protect = None, set()
        if image_path:
            data, real, protect = self.context.read(image_path)
        else:
            if len(image_base64) > 4 * ((MAX_INPUT_BYTES + 2) // 3) + 1024:
                raise ToolError("Input exceeds 32 MiB.")
            try:
                data = base64.b64decode(image_base64.split(",", 1)[-1], validate=False)
            except ValueError as exc:
                raise ToolError("image_base64 is not valid base64.") from exc
            if len(data) > MAX_INPUT_BYTES:
                raise ToolError("Input exceeds 32 MiB.")
        return real, protect, data

    def summary(self, result, mode, reveal):
        return _summary(
            SimpleNamespace(addresses=result["addresses"], boxes=result["boxes"]),
            SimpleNamespace(
                width=result["image_size"][0], height=result["image_size"][1]
            ),
            mode,
            result["seconds"],
            reveal,
            self.context.config.allow_reveal,
        )


def _mask(address):
    local, _, domain = address.partition("@")
    end = domain.rsplit(".", 1)[-1] if "." in domain else ""
    end = end if end.isalpha() and len(end) <= 6 else "***"
    return f"{local[:1]}***@{domain[:1]}***.{end}"


def _readings(addresses, reveal, allow_reveal, source="OCR"):
    full = reveal and allow_reveal
    readings = sorted(set(addresses)) if full else sorted({_mask(a) for a in addresses})
    note = f"as read by {source}, possibly with small misreadings; one address can appear in more than one spelling"
    if not full:
        note += "; masked"
    if len(readings) > 50:
        note += f"; only the first 50 of {len(readings)} are listed"
        readings = readings[:50]
    if reveal and not allow_reveal:
        note += " (full addresses are switched off on this server; the user can start it with --allow-reveal)"
    return readings, note


def _summary(result, image, mode, seconds, reveal, allow_reveal):
    readings, note = _readings(result.addresses, reveal, allow_reveal)
    return dict(
        boxes_redacted=len(result.boxes),
        addresses=readings,
        addresses_note=note,
        boxes_xywh=result.boxes,
        image_size=[image.width, image.height],
        mode=mode,
        seconds=round(seconds, 2),
    )


def register_tools(server, context):
    service = EmailService(context)

    @server.tool(annotations=annotations("Redact email addresses", writes=True))
    def redact_emails(
        image_path: str = "",
        image_base64: str = "",
        output_path: str = "",
        mode: str = context.config.mode,
        overwrite: StrictBool = False,
        return_image: StrictBool = False,
        reveal_addresses: StrictBool = False,
    ):
        """Cover detected email addresses with solid black boxes and save a separate PNG.

        Give exactly one of image_path or image_base64. Local paths must be in
        allowed folders. Default output: <name>.redacted.png beside the input.
        Existing outputs require overwrite=true; input is never changed.
        Base64 input without output_path returns the redacted image. Addresses
        are masked unless both server --allow-reveal and reveal_addresses are
        enabled. View only the redacted result; OCR can still miss addresses.
        """
        with context.call():
            if mode not in MODES:
                raise ToolError(f"mode must be one of {sorted(MODES)}")
            real, protect, data = service.load(image_path, image_base64)
            if output_path:
                out = output_path
            elif real is not None:
                given = Path(image_path).expanduser()
                out = str(given.with_name(f"{given.stem}.redacted.png"))
            else:
                out, return_image = None, True
            if out is not None:
                context.file_ops.check_output(out, context.folders, protect, overwrite)
            result, png = service.run(data, mode, redact=True)
            summary = service.summary(result, mode, reveal_addresses)
            context.stop_if_cancelled()
            if out is not None:
                summary["output_path"] = context.write(
                    png, out, protect, overwrite, ".png"
                )
            if not return_image:
                return json.dumps(summary, indent=1)
            # Preserve the original RGB MCP-image behavior independently of
            # the file's PNG mode (which can be opaque RGBA).
            redacted, _, _ = image_io.decode(png, max_pixels=context.max_pixels)
            preview = image_io.encode(redacted[:, :, :3])
            context.stop_if_cancelled()
            return [
                json.dumps(summary, indent=1),
                Image(data=preview, format="png"),
            ]

    @server.tool(annotations=annotations("Find email addresses"))
    def find_emails(
        image_path: str = "",
        image_base64: str = "",
        mode: str = context.config.mode,
        reveal_addresses: StrictBool = False,
    ):
        """Find email-address boxes without writing an image. Give image_path or
        image_base64. Returns masked addresses and [x,y,width,height] boxes.
        Full addresses require server --allow-reveal and reveal_addresses=true.
        """
        with context.call():
            if mode not in MODES:
                raise ToolError(f"mode must be one of {sorted(MODES)}")
            _, _, data = service.load(image_path, image_base64)
            result, _ = service.run(data, mode)
            return json.dumps(service.summary(result, mode, reveal_addresses), indent=1)

    return service
