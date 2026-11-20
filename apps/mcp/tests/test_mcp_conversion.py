"""Adapter policy, cancellation, schema and real stdio failures (server deps required)."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp", reason="Install the adapter/test extras for integration tests")
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.mcpserver.exceptions import ToolError

from opencv_bv_tools import image_conversion as adapter

ROOT = Path(__file__).resolve().parents[1]
from opencv_bv_tools.runtime import safe_files as sf


class Registry:
    def tool(self, **metadata):
        self.metadata = metadata
        return lambda fn: fn


@pytest.fixture
def setup(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "input.png"
    source.write_bytes((ROOT / "tests/fixtures/smoke/email.png").read_bytes())
    settings = dict(file_ops=sf, folders=sf.Folders([allowed]),
                    call_slots=threading.BoundedSemaphore(2), stop_if_cancelled=lambda: None)
    registry = Registry()
    convert = adapter.register_conversion_tool(registry, **settings)
    return SimpleNamespace(source=source, out=allowed / "output.webp", convert=convert,
                           settings=settings, registry=registry, root=tmp_path)


@pytest.mark.parametrize("options", [
    {"quality": True}, {"quality": 1.5}, {"quality": 0}, {"quality": 101},
    {"quality": None}, {"overwrite": "false"}, {"webp_lossless": 1},
    {"jpeg_background": [1, 2]}, {"jpeg_background": [1, 2, 3, 4]},
    {"jpeg_background": [True, 2, 3]}, {"jpeg_background": [1.5, 2, 3]},
    {"jpeg_background": [1, -1, 3]}, {"jpeg_background": [1, 256, 3]},
    {"jpeg_background": "white"}, {"jpeg_background": (1, 2, 3)},
])
def test_options_rejected_before_read(setup, monkeypatch, options):
    monkeypatch.setattr(sf, "open_input", lambda *a: pytest.fail("invalid options must fail before input read"))
    with pytest.raises(ToolError):
        setup.convert(str(setup.source), str(setup.out), **options)


@pytest.mark.parametrize("input_name,output_name", [("", "output.webp"), ("input.png", ""), ("input.png", "x.gif")])
def test_required_paths_and_suffix(setup, input_name, output_name):
    with pytest.raises(ToolError):
        setup.convert(str(setup.source.parent / input_name) if input_name else "",
                      str(setup.source.parent / output_name) if output_name else "")


def test_metadata_only_and_annotations(setup):
    original = setup.source.read_bytes()
    summary = json.loads(setup.convert(str(setup.source), str(setup.out)))
    assert set(summary) == {"input_format", "output_format", "image_size", "mime_type",
                            "bytes_written", "output_path", "warnings", "redacted"}
    assert summary["redacted"] is False
    assert summary["bytes_written"] == setup.out.stat().st_size
    assert setup.source.read_bytes() == original
    annotations = setup.registry.metadata["annotations"]
    assert annotations.destructive_hint and not annotations.read_only_hint
    assert not annotations.open_world_hint and not annotations.idempotent_hint


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("alias", ["same", "hardlink", "symlink"])
def test_input_aliases(setup, overwrite, alias):
    before = setup.source.read_bytes(), setup.source.stat().st_ino
    target = setup.source.parent / "alias.png"
    if alias == "same":
        target = setup.source
    elif alias == "hardlink":
        os.link(setup.source, target)
    else:
        target.symlink_to(setup.source)
    with pytest.raises(ToolError):
        setup.convert(str(setup.source), str(target), overwrite=overwrite)
    assert (setup.source.read_bytes(), setup.source.stat().st_ino) == before


def test_outside_folders_read_and_write(setup):
    outside = setup.root / "outside.png"
    outside.write_bytes(setup.source.read_bytes())
    for source, target in [(outside, setup.out), (setup.source, setup.root / "outside.webp")]:
        with pytest.raises(ToolError, match="outside"):
            setup.convert(str(source), str(target))
    linked = setup.source.parent / "outside-link.png"
    linked.symlink_to(outside)
    with pytest.raises(ToolError, match="outside"):
        setup.convert(str(linked), str(setup.out))


def test_early_size_limit(setup, monkeypatch):
    monkeypatch.setattr(adapter, "MAX_INPUT_BYTES", 10)
    with pytest.raises(ToolError, match="bytes"):
        setup.convert(str(setup.source), str(setup.out))


def test_growth_after_fstat_is_bounded(setup, monkeypatch):
    real_open = sf.open_input
    def open_then_grow(*args):
        source, real, identity = real_open(*args)
        class GrowingReader:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                source.close()
            def fileno(self):
                return source.fileno()
            def read(self, n):
                assert n == adapter.MAX_INPUT_BYTES + 1
                return b"x" * n
        return GrowingReader(), real, identity
    monkeypatch.setattr(adapter, "MAX_INPUT_BYTES", 5000)
    monkeypatch.setattr(sf, "open_input", open_then_grow)
    # The core default is independently bounded at 32 MiB; the adapter must also
    # reject its read boundary before sending bytes to the decoder.
    monkeypatch.setattr(adapter, "convert_image", lambda *a, **kw: pytest.fail("grown input reached decoder"))
    with pytest.raises(ToolError, match="bytes"):
        setup.convert(str(setup.source), str(setup.out))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_registration_rejects_invalid_limit(setup, limit):
    with pytest.raises(ValueError):
        adapter.register_conversion_tool(Registry(), **setup.settings, max_pixels=limit)


def test_server_pixel_limit_applies(setup):
    convert = adapter.register_conversion_tool(Registry(), **setup.settings, max_pixels=100)
    with pytest.raises(ToolError, match="pixels"):
        convert(str(setup.source), str(setup.out))


@pytest.mark.parametrize("when", [1, 2])
def test_cancel_before_work_or_before_save(setup, when):
    count = 0
    def cancelled():
        nonlocal count
        count += 1
        if count == when:
            raise asyncio.CancelledError
    convert = adapter.register_conversion_tool(Registry(), **{**setup.settings, "stop_if_cancelled": cancelled})
    with pytest.raises(asyncio.CancelledError):
        convert(str(setup.source), str(setup.out))
    assert not setup.out.exists()
    assert not list(setup.source.parent.glob(".email-redact-*.tmp"))


def test_queued_cancellation_happens_before_read(setup, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    class Queue:
        def __enter__(self):
            entered.set()
            assert release.wait(5)
        def __exit__(self, *args):
            pass
    def cancelled():
        raise asyncio.CancelledError
    monkeypatch.setattr(sf, "open_input", lambda *a: pytest.fail("queued cancelled request read input"))
    convert = adapter.register_conversion_tool(Registry(), **{
        **setup.settings, "call_slots": Queue(), "stop_if_cancelled": cancelled,
    })
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(convert, str(setup.source), str(setup.out))
        assert entered.wait(5)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            pending.result(5)


def test_publication_error_preserves_files(setup, monkeypatch):
    setup.out.write_bytes(b"existing")
    original = setup.source.read_bytes()
    def failure(*args, **kwargs):
        raise OSError("injected write failure")
    monkeypatch.setattr(sf, "_write_all", failure)
    with pytest.raises(ToolError, match="File error"):
        setup.convert(str(setup.source), str(setup.out), overwrite=True)
    assert setup.out.read_bytes() == b"existing" and setup.source.read_bytes() == original
    assert not list(setup.source.parent.glob(".email-redact-*.tmp"))


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1", "0.0000001"])
def test_invalid_server_megapixels_exit_cleanly(tmp_path, value):
    result = subprocess.run([sys.executable, "-m", "opencv_bv_tools.server",
                             "--allow-dir", str(tmp_path), "--max-megapixels", value],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0 and "max_megapixels" in result.stderr
    assert "Traceback" not in result.stderr


def test_real_protocol_schema_errors_without_ocr(setup):
    # Conversion may load base OpenCV, but must not initialize either OCR engine.
    # Registration without OpenCV/NumPy is checked separately in test_unified_mcp.
    launcher = setup.root / "failed_ocr_server.py"
    launcher.write_text(
        "import builtins, runpy, sys\n"
        "original_import = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in {'image_redact','rapidocr','onnxruntime'}:\n"
        "        raise RuntimeError('OCR dependency imported by combined server')\n"
        "    return original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n"
        "runpy.run_module('opencv_bv_tools.server', run_name='__main__')\n"
    )
    async def exercise():
        process = StdioServerParameters(command=sys.executable,
            args=[str(launcher), "--allow-dir", str(setup.source.parent)])
        async with stdio_client(process) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                assert "convert_image_format" in tools and len(tools) == 9
                wire = tools["convert_image_format"].model_dump(by_alias=True)
                schema = wire["inputSchema"]
                assert set(schema["required"]) == {"image_path", "output_path"}
                assert schema["properties"]["overwrite"]["type"] == "boolean"
                assert wire["annotations"]["destructiveHint"] is True
                args = {"image_path": str(setup.source), "output_path": str(setup.out)}
                result = await session.call_tool("convert_image_format", args)
                assert not result.is_error
                assert all(block.type == "text" for block in result.content)
                assert json.loads(result.content[0].text)["redacted"] is False
                for changes in [
                    {}, {"quality": True}, {"quality": 1.5}, {"overwrite": "false"},
                    {"jpeg_background": [True, 0, 0]}, {"jpeg_background": [1.5, 0, 0]},
                    {"jpeg_background": [1, 2]}, {"output_path": str(setup.root / "outside.webp")},
                    {"output_path": str(setup.source)}, {"image_path": str(setup.root / "missing.png")},
                ]:
                    failed = await session.call_tool("convert_image_format", {**args, **changes})
                    assert failed.is_error, changes
                second = await session.call_tool("convert_image_format", {**args, "overwrite": True})
                assert not second.is_error
    asyncio.run(exercise())


def test_real_protocol_cancel_during_encoding_then_recover(setup):
    launcher = setup.root / "slow_conversion_server.py"
    entered = setup.root / "encoding-started"
    launcher.write_text(
        "import runpy, sys, time\n"
        "from pathlib import Path\n"
        "from opencv_bv_tools import image_conversion as adapter\n"
        "original = adapter.convert_image\n"
        "def slow(*args, **kwargs):\n"
        f"    Path({str(entered)!r}).write_text('started')\n"
        "    time.sleep(0.5)\n"
        "    return original(*args, **kwargs)\n"
        "adapter.convert_image = slow\n"
        "runpy.run_module('opencv_bv_tools.server', run_name='__main__')\n"
    )
    async def exercise():
        process = StdioServerParameters(command=sys.executable,
            args=[str(launcher), "--allow-dir", str(setup.source.parent)])
        async with stdio_client(process) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                task = asyncio.create_task(session.call_tool("convert_image_format", {
                    "image_path": str(setup.source), "output_path": str(setup.out),
                }))
                deadline = time.monotonic() + 5
                while not entered.exists() and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                assert entered.exists(), "encoding did not start"
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                await asyncio.sleep(0.7)
                assert not setup.out.exists()
                recovery = setup.source.parent / "recovery.webp"
                result = await session.call_tool("convert_image_format", {
                    "image_path": str(setup.source), "output_path": str(recovery),
                })
                assert not result.is_error and recovery.is_file()
                assert not list(setup.source.parent.glob(".email-redact-*.tmp"))
    asyncio.run(exercise())
