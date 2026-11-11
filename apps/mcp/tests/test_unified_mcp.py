"""Unified registration, optional dependencies, policy and lazy OCR lifecycle."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from PIL import Image
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from opencv_bv_tools.runtime.context import ServerConfig
from opencv_bv_tools.runtime.context import ToolContext
from opencv_bv_tools import email_tools as mcp_email

FEATURE_MODULES = {"qr": "qr_reader", "barcode": "barcode_reader", "png": "png_compression",
                   "face": "face_blurring", "scanner": "document_scanner", "email": "email_tools"}

TOOL_NAMES = {"convert_image_format", "read_qr", "read_barcodes", "compress_png",
              "blur_faces", "scan_document", "redact_emails", "find_emails", "redact_pdf"}


class Registry:
    def __init__(self):
        self.tools = {}
    def tool(self, **metadata):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn
        return register


@pytest.fixture
def context(tmp_path):
    return ToolContext(ServerConfig(allow_dirs=(tmp_path,)))


def test_all_tools_registered_without_heavy_imports(tmp_path):
    script = """
import asyncio, json, sys
from pathlib import Path
from opencv_bv_tools.server import create_server, ServerConfig
server = create_server(ServerConfig(allow_dirs=(Path(sys.argv[1]),)))
print(json.dumps({'tools': [t.name for t in asyncio.run(server.list_tools())],
                  'heavy': sorted(set(sys.modules) & {'cv2','numpy','onnxruntime','rapidocr','pypdfium2','pymupdf'})}))
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert set(data["tools"]) == TOOL_NAMES and data["heavy"] == []


@pytest.mark.parametrize("options", [{}, {"allow_any_path": True, "threads": 0},
    {"allow_any_path": True, "max_megapixels": float("nan")},
    {"allow_any_path": True, "max_megapixels": float("inf")},
    {"allow_any_path": True, "max_megapixels": 0}])
def test_invalid_server_configuration(options):
    with pytest.raises(ValueError):
        ServerConfig(**options).validate()


@pytest.mark.parametrize("adapter,name,extra,args", [
    ("qr", "read_qr", "qr", {}), ("barcode", "read_barcodes", "barcode", {}),
    ("png", "compress_png", "png", {"output_path": "out.png"}),
    ("face", "blur_faces", "face", {"output_path": "out.png"}),
    ("scanner", "scan_document", "scanner", {"output_path": "out.pdf"}),
    ("email", "find_emails", "ocr", {}), ("email", "redact_emails", "ocr", {}),
])
def test_missing_optional_dependencies_are_actionable(context, monkeypatch, adapter, name, extra, args):
    registry = Registry()
    importlib.import_module(f"opencv_bv_tools.{FEATURE_MODULES[adapter]}").register_tools(registry, context)
    monkeypatch.setattr(importlib.util, "find_spec", lambda module: None)
    with pytest.raises(ToolError, match=rf"opencv-bv-tools\[{extra}\]"):
        registry.tools[name](image_path="input.png", **args)


@pytest.mark.parametrize("adapter,name,operation,status,suffix", [
    ("png", "compress_png", "compress_png", "unchanged", ".png"),
    ("face", "blur_faces", "blur_faces", "blurred", ".png"),
    ("scanner", "scan_document", "scan_document", "completed", ".pdf"),
])
@pytest.mark.parametrize("alias", ["same", "hardlink", "outside", "existing", "cancel_after_worker"])
def test_transform_adapters_protect_files(context, tmp_path, monkeypatch, adapter, name, operation, status, suffix, alias):
    source = tmp_path / ("source" + suffix)
    source.write_bytes(b"original source")
    output = tmp_path / ("output" + suffix)
    module = importlib.import_module(f"opencv_bv_tools.{FEATURE_MODULES[adapter]}")
    registry = Registry()
    module.register_tools(registry, context)
    monkeypatch.setattr(context, "require", lambda *args: None)
    calls = []
    def work(*args, **kwargs):
        calls.append(True)
        if alias == "cancel_after_worker":
            def cancel(): raise RuntimeError("cancelled")
            context.stop_if_cancelled = cancel
        return {"status": status, "data": b"new output"}
    monkeypatch.setattr(module, operation, work)
    if alias == "same": output = source
    elif alias == "hardlink": output.hardlink_to(source)
    elif alias == "outside": output = tmp_path.parent / ("outside" + suffix)
    else: output.write_bytes(b"existing output")
    error = RuntimeError if alias == "cancel_after_worker" else ToolError
    with pytest.raises(error):
        registry.tools[name](image_path=str(source), output_path=str(output), overwrite=alias != "existing")
    assert source.read_bytes() == b"original source"
    if alias in {"existing", "cancel_after_worker"}: assert output.read_bytes() == b"existing output"
    assert bool(calls) == (alias == "cancel_after_worker")
    assert not list(tmp_path.glob(".email-redact-*.tmp"))


def test_email_summary_preserves_masking_and_reveal_gate():
    result = SimpleNamespace(addresses=["jane.doe@example.com"], boxes=[[1, 2, 3, 4]])
    image = Image.new("RGB", (20, 10))
    masked = mcp_email._summary(result, image, "accurate", 1.234, True, False)
    assert masked["addresses"] == ["j***@e***.com"] and "switched off" in masked["addresses_note"]
    assert masked["boxes_xywh"] == result.boxes and masked["seconds"] == 1.23
    assert mcp_email._summary(result, image, "fast", 1, True, True)["addresses"] == result.addresses
