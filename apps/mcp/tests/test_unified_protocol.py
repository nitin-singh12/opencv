"""Real MCP cancellation must reap the worker and leave outputs untouched."""
import asyncio
import json
from pathlib import Path
import sys
import time

from PIL import Image
import psutil
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_cancel_native_worker_and_recover(tmp_path):
    source, output = tmp_path / "source.png", tmp_path / "out.png"
    Image.new("RGB", (16, 16), "white").save(source)
    output.write_bytes(b"keep original output")
    pidfile = tmp_path / "child.pid"
    launcher = tmp_path / "slow_server.py"
    child = f"import os,time;from pathlib import Path;Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(20)"
    launcher.write_text(
        "import sys\nfrom dataclasses import asdict\n"
        "from opencv_bv_tools import png_compression as mcp_png, png_compression\n"
        "def slow(data, *, limits, cancelled):\n"
        f"    return png_compression._run_worker(data, {{'limits':asdict(limits)}}, timeout=limits.timeout_seconds, cancelled=cancelled, command=[sys.executable, '-c', {child!r}])\n"
        "mcp_png.compress_png = slow\nfrom opencv_bv_tools.server import main\nmain()\n"
    )
    async def exercise():
        process = StdioServerParameters(command=sys.executable, args=[str(launcher), "--allow-dir", str(tmp_path)])
        async with stdio_client(process) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                pending = asyncio.create_task(session.call_tool("compress_png", {
                    "image_path": str(source), "output_path": str(output), "overwrite": True, "timeout_seconds": 10.0}))
                deadline = time.monotonic() + 5
                while not pidfile.exists() and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                assert pidfile.exists(), "worker did not start"
                pid = int(pidfile.read_text())
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                deadline = time.monotonic() + 3
                while psutil.pid_exists(pid) and time.monotonic() < deadline:
                    await asyncio.sleep(0.02)
                assert not psutil.pid_exists(pid), "cancelled worker was not reaped"
                assert output.read_bytes() == b"keep original output"
                assert not list(tmp_path.glob(".email-redact-*.tmp"))
                response = await session.call_tool("convert_image_format", {
                    "image_path": str(source), "output_path": str(tmp_path / "recovered.webp")})
                assert not response.is_error
    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["render", "native_ocr"])
def test_cancel_pdf_tree_then_recover(tmp_path, phase):
    from reportlab.pdfgen import canvas
    source, output = tmp_path / "source.pdf", tmp_path / "output.pdf"
    document = canvas.Canvas(str(source), pagesize=(360,180))
    document.drawString(20,100,"Contact support@example.com")
    document.save()
    output.write_bytes(b"keep existing output")
    marker = tmp_path / "pdf-pids.json"
    child = tmp_path / "pdf_child.py"
    child.write_text(
        "import ctypes,json,os,runpy,signal,tempfile\nfrom pathlib import Path\n"
        "from opencv_bv_tools.runtime import pdf_worker as _pdf_worker\noriginal=_pdf_worker.process_pdf\n"
        f"marker=Path({str(marker)!r})\n"
        "def controlled(data,options,state,get_engine,progress):\n"
        "    if marker.exists(): return original(data,options,state,get_engine,progress)\n"
        "    from opencv_bv_tools.runtime import pdf_redact as pdf\n"
        "    with tempfile.TemporaryFile() as f:\n"
        "        f.write(data);f.seek(0);pages=pdf.Pages(f,max_pixels=40000000)\n"
        "        try:\n"
        + ("            os.kill(pages.proc.pid,signal.SIGSTOP)\n" if phase == "render" else
           "            signal.signal(signal.SIGTERM,signal.SIG_IGN)\n")
        + "            marker.write_text(json.dumps([os.getpid(),pages.proc.pid]))\n"
        + ("            pages.page(0)\n" if phase == "render" else "            ctypes.CDLL(None).pause()\n")
        + "        finally: pages.close()\n"
        "_pdf_worker.process_pdf=controlled\nrunpy.run_module('opencv_bv_tools.runtime.ocr_worker',run_name='__main__')\n"
    )
    launcher = tmp_path / "pdf_server.py"
    launcher.write_text(
        "import sys\nfrom opencv_bv_tools import email_tools as mcp_email\nOriginal=mcp_email.OCRProcess\n"
        "class TestProcess(Original):\n"
        f"    def __init__(self,config): super().__init__(config,command=[sys.executable,{str(child)!r}])\n"
        "mcp_email.OCRProcess=TestProcess\nfrom opencv_bv_tools.server import main\nmain()\n"
    )
    async def exercise():
        params=StdioServerParameters(command=sys.executable,args=[str(launcher),'--allow-dir',str(tmp_path)])
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize()
                task=asyncio.create_task(session.call_tool('redact_pdf',{'pdf_path':str(source),'output_path':str(output),'overwrite':True}))
                deadline=time.monotonic()+10
                while not marker.exists() and time.monotonic()<deadline: await asyncio.sleep(0.01)
                assert marker.exists()
                worker,helper=json.loads(marker.read_text())
                task.cancel()
                with pytest.raises(asyncio.CancelledError): await task
                deadline=time.monotonic()+4
                def stopped(pid):
                    return not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE
                while (psutil.pid_exists(worker) or not stopped(helper)) and time.monotonic()<deadline:
                    await asyncio.sleep(0.02)
                assert not psutil.pid_exists(worker) and stopped(helper)
                assert output.read_bytes()==b'keep existing output'
                assert not list(tmp_path.glob('.email-redact-*.tmp'))
                result=await session.call_tool('redact_pdf',{'pdf_path':str(source),'output_path':str(output),'overwrite':True})
                assert not result.is_error, result.content
                assert output.read_bytes().startswith(b'%PDF-')
    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["startup", "processing"])
def test_cancel_ocr_worker_then_recover(tmp_path, phase):
    source = tmp_path / "source.png"
    Image.new("RGB", (80, 60), "white").save(source)
    pidfile = tmp_path / "ocr-child.pid"
    child = tmp_path / "ocr_child.py"
    child.write_text(
        "import json,os,runpy,sys,time\nfrom pathlib import Path\n"
        f"marker=Path({str(pidfile)!r})\n"
        "if not marker.exists():\n"
        + ("    options=json.loads(sys.stdin.buffer.readline())\n    sys.stdin.buffer.read(options['input_bytes'])\n" if phase == "processing" else "")
        + "    marker.write_text(str(os.getpid()))\n    time.sleep(30)\n"
        "else:\n    runpy.run_module('opencv_bv_tools.runtime.ocr_worker',run_name='__main__')\n"
    )
    launcher = tmp_path / "ocr_server.py"
    launcher.write_text(
        "import sys\nfrom opencv_bv_tools import email_tools as mcp_email\nOriginal=mcp_email.OCRProcess\n"
        "class TestProcess(Original):\n"
        f"    def __init__(self,config): super().__init__(config,command=[sys.executable,{str(child)!r}])\n"
        "mcp_email.OCRProcess=TestProcess\nfrom opencv_bv_tools.server import main\nmain()\n"
    )
    async def exercise():
        params = StdioServerParameters(command=sys.executable, args=[str(launcher), "--allow-dir", str(tmp_path)])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                task = asyncio.create_task(session.call_tool("find_emails", {"image_path": str(source)}))
                deadline = time.monotonic() + 5
                while not pidfile.exists() and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                assert pidfile.exists()
                pid = int(pidfile.read_text())
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                deadline = time.monotonic() + 3
                while psutil.pid_exists(pid) and time.monotonic() < deadline:
                    await asyncio.sleep(0.02)
                assert not psutil.pid_exists(pid)
                response = await session.call_tool("find_emails", {"image_path": str(source)})
                assert not response.is_error, response.content
    asyncio.run(exercise())
