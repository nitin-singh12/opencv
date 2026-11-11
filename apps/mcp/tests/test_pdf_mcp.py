"""PDF integration limits, publication policy, settings and protocol validation."""
from dataclasses import replace
import hashlib
import importlib.util
import io
import tempfile
from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageFont
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from opencv_bv_tools import pdf_tools as mcp_pdf
from opencv_bv_tools.runtime.ocr_process import MAX_PDF_BYTES
from opencv_bv_tools.runtime.ocr_process import OCRProcess
from opencv_bv_tools.runtime.ocr_process import OCRWorkerError
from opencv_bv_tools.runtime.ocr_process import validate_pdf_response
from opencv_bv_tools.runtime.context import ServerConfig
from opencv_bv_tools.runtime.context import ToolContext
from opencv_bv_tools.email_tools import EmailService
from opencv_bv_tools.email_tools import _readings


def make_pdf():
    from reportlab.pdfgen import canvas
    output = io.BytesIO()
    document = canvas.Canvas(output, pagesize=(360, 180))
    document.drawString(20, 110, "Contact support@example.com")
    document.drawString(20, 65, "Public information stays searchable")
    document.save()
    return output.getvalue()


@pytest.mark.parametrize("options", [{"max_pages":0}, {"max_pages":True}, {"max_pages":101},
    {"pdf_engine":"auto"}, {"pdf_timeout":0}, {"pdf_timeout":float("inf")}, {"pdf_timeout":3601}])
def test_pdf_config_limits(options):
    with pytest.raises(ValueError):
        ServerConfig(allow_any_path=True, **options).validate()


def test_pdf_image_pdf_reuses_worker_and_resets_settings(tmp_path):
    from pypdf import PdfReader
    service = EmailService(ToolContext(ServerConfig(allow_dirs=(tmp_path,), threads=2)))
    image = Image.new("RGB", (640, 120), "white")
    ImageDraw.Draw(image).text((20, 40), "Contact support@example.com", font=ImageFont.load_default(size=22), fill="black")
    png = io.BytesIO()
    image.save(png, format="PNG")
    try:
        first, payload = service.run_pdf(make_pdf(), "accurate")
        with payload:
            text = PdfReader(payload).pages[0].extract_text()
            assert "@" not in text and "Public" in text
        middle, _ = service.run(png.getvalue(), "fast")
        last, payload = service.run_pdf(make_pdf(), "accurate")
        payload.close()
        assert first["pid"] == middle["pid"] == last["pid"]
        assert first["boxes_per_page"] == last["boxes_per_page"] == [1]
        assert middle["addresses"] == ["support@example.com"]
        with pytest.raises(ToolError, match="cannot be read"):
            service.run_pdf(b"not a PDF", "fast")
        recovered, _ = service.run(png.getvalue(), "accurate")
        assert recovered["pid"] != last["pid"] and recovered["boxes"] == middle["boxes"]
    finally:
        service.close()


class Registry:
    def tool(self, **kwargs):
        def register(function):
            self.call = function
            return function
        return register


@pytest.mark.parametrize("case", ["same", "hardlink", "outside", "existing", "cancel_before_publish"])
def test_pdf_publication_protects_source_and_existing_output(tmp_path, monkeypatch, case):
    context = ToolContext(ServerConfig(allow_dirs=(tmp_path,)))
    monkeypatch.setattr(context, "require", lambda *args: None)
    source = tmp_path / "original.pdf"
    source.write_bytes(b"original")
    output = tmp_path / "out.pdf"
    if case == "same": output = source
    elif case == "hardlink": output.hardlink_to(source)
    elif case == "outside": output = tmp_path.parent / "outside.pdf"
    else: output.write_bytes(b"keep")
    calls, payloads = [], []
    def run(*args):
        calls.append(True)
        payload = io.BytesIO(b"%PDF-new")
        payloads.append(payload)
        if case == "cancel_before_publish":
            write = context.file_ops._write_all
            def after_write(fd, data):
                write(fd, data)
                def cancel(): raise RuntimeError("cancelled")
                context.stop_if_cancelled = cancel
            monkeypatch.setattr(context.file_ops, "_write_all", after_write)
        return {}, payload
    registry = Registry()
    mcp_pdf.register_tools(registry, context, SimpleNamespace(run_pdf=run))
    with pytest.raises((ToolError, RuntimeError)):
        registry.call(str(source), str(output), overwrite=case != "existing")
    assert source.read_bytes() == b"original"
    assert bool(calls) == (case == "cancel_before_publish")
    assert all(payload.closed for payload in payloads)
    if case in {"existing", "cancel_before_publish"}: assert output.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".email-redact-*.tmp"))


def test_pdf_missing_dependencies_and_mask_before_limit(tmp_path, monkeypatch):
    registry = Registry()
    mcp_pdf.register_tools(registry, ToolContext(ServerConfig(allow_dirs=(tmp_path,))), None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ToolError, match=r"opencv-bv-tools\[pdf\]"):
        registry.call("input.pdf")
    addresses = [f"person{i}@example.com" for i in range(60)] + ["another@different.org"]
    assert len(_readings(addresses, False, False)[0]) == 2
    assert len(_readings(addresses, True, True)[0]) == 50


@pytest.mark.parametrize("case", ["length", "hash", "pages", "counts", "address", "pid", "stale", "engine", "nan", "eof"])
def test_invalid_pdf_terminal_frames(case):
    pdf = b"%PDF-1.4\n%%EOF\n"
    response = dict(status="ok", pdf_bytes=len(pdf), pdf_sha256=hashlib.sha256(pdf).hexdigest(),
        pages=1, boxes_per_page=[1], addresses=["private@example.com"], seconds=1.0, pid=10,
        request_id=2, engine="picture", fallback=False)
    mutations = {"length":("pdf_bytes", MAX_PDF_BYTES+1), "hash":("pdf_sha256", "x"),
        "pages":("pages",True), "counts":("boxes_per_page",[True]), "address":("addresses",["x"*8193]),
        "pid":("pid",11), "stale":("request_id",1), "engine":("engine","other"), "nan":("seconds",float("nan"))}
    if case == "eof":
        pdf = b"%PDF-1.4\nmissing\n"
        response.update(pdf_bytes=len(pdf),pdf_sha256=hashlib.sha256(pdf).hexdigest())
    else:
        key, value = mutations[case]
        response[key] = value
    with pytest.raises(ValueError):
        validate_pdf_response(response, io.BytesIO(pdf), pid=10, request_id=2, max_pages=100)


def test_complete_pdf_budget_includes_text_and_structure():
    from opencv_bv_tools.runtime import pdf_redact as pdf
    pages = [(b"x"*200, b"/FlateDecode", b"/DeviceGray", 20, 20, 10, 10, b"test text"*60)] * 3
    chunks = pdf.pdf_chunks(pages)
    exact = sum(map(len, chunks))
    assert pdf.pdf_chunks(pages, max_bytes=exact) == chunks
    with pytest.raises(pdf.PdfError, match="output byte limit"):
        pdf.pdf_chunks(pages, max_bytes=exact-1)
    with pytest.raises(pdf.PdfError, match="output byte limit"):
        pdf.text_layer([("visible text "*1000, 0, 0, 100, 20)], 1, 100, max_bytes=100)


@pytest.mark.parametrize("message", [b'{"pages":true}\n', b'x' * ((1 << 20) + 1) + b'\n'],
                         ids=["boolean_pages", "oversized"])
def test_invalid_renderer_headers_are_bounded_and_refused(tmp_path, monkeypatch, message):
    from opencv_bv_tools.runtime import pdf_redact as pdf
    real = subprocess.Popen
    fake = tmp_path / 'renderer.py'
    fake.write_text('import sys\nsys.stdout.buffer.write('+repr(message)+');sys.stdout.buffer.flush()\n')
    monkeypatch.setattr(pdf.subprocess,'Popen',lambda *args, **kwargs: real([sys.executable,str(fake)],**kwargs))
    with tempfile.TemporaryFile() as source, pytest.raises(pdf.PdfError,match='response'):
        pdf.Pages(source,max_pixels=4)


@pytest.mark.parametrize("phase", ["leader_exits", "native_hang"])
def test_stops_pdf_helpers_even_if_leader_exited_or_ignores_term(tmp_path, phase):
    import psutil
    pidfile = tmp_path / "helper.pid"
    script = tmp_path / "worker.py"
    helper = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"
    script.write_text(
        "import json,os,signal,subprocess,sys,time\nfrom pathlib import Path\n"
        "header=json.loads(sys.stdin.buffer.readline());sys.stdin.buffer.read(header['input_bytes'])\n"
        f"child=subprocess.Popen([sys.executable,'-c',{helper!r}])\n"
        f"Path({str(pidfile)!r}).write_text(str(child.pid))\n"
        + ("sys.exit(0)\n" if phase == "leader_exits" else
           "signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)\n"))
    worker = OCRProcess(ServerConfig(allow_dirs=(tmp_path,), pdf_timeout=0.2), command=[sys.executable,str(script)])
    try:
        with pytest.raises(OCRWorkerError):
            worker.request(b"pdf",mode="accurate",redact=True,pdf=True,check_cancelled=lambda:None)
        pid = int(pidfile.read_text())
        deadline = time.monotonic()+3
        def stopped():
            return not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        while not stopped() and time.monotonic() < deadline: time.sleep(0.02)
        assert stopped(), "orphan PDF helper is still running"
        assert worker.pid is None
    finally:
        worker.close()


@pytest.mark.parametrize("case", ["success", "progress_flood", "progress_bad_id", "truncated", "cancel_transfer"])
def test_pdf_streaming_protocol(tmp_path, case):
    script = tmp_path / "worker.py"
    script.write_text('''import hashlib,json,os,sys,time
header=json.loads(sys.stdin.buffer.readline());sys.stdin.buffer.read(header['input_bytes'])
case=%r
progress=dict(status='progress',request_id=header['request_id'],index=0,pages=1)
if case=='progress_bad_id':progress['request_id']+=1
for i in range(3 if case=='progress_flood' else 1):
 print(json.dumps(progress),flush=True)
pdf=b'%%PDF-1.4\\n'+b'x'*200000+b'\\n%%%%EOF\\n'
result=dict(status='ok',request_id=header['request_id'],pdf_bytes=len(pdf),pdf_sha256=hashlib.sha256(pdf).hexdigest(),
pages=1,boxes_per_page=[0],addresses=[],seconds=0,pid=os.getpid(),engine='picture',fallback=False)
print(json.dumps(result),flush=True)
if case=='cancel_transfer':
 sys.stdout.buffer.write(pdf[:1024]);sys.stdout.buffer.flush();time.sleep(30)
else:
 sys.stdout.buffer.write(pdf[:-100] if case=='truncated' else pdf);sys.stdout.buffer.flush()
''' % case)
    worker=OCRProcess(ServerConfig(allow_dirs=(tmp_path,),max_pages=1,pdf_timeout=3),command=[sys.executable,str(script)])
    seen=[]
    started=time.monotonic()
    def cancel():
        if case=='cancel_transfer' and seen and time.monotonic()-started>0.2: raise RuntimeError('cancelled')
    try:
        if case=='success':
            result,payload=worker.request(b'pdf',mode='accurate',redact=True,pdf=True,check_cancelled=cancel,
                                         progress=lambda *args:seen.append(args))
            with payload:
                assert payload.read().startswith(b'%PDF-') and seen==[(0,1)]
        else:
            with pytest.raises((OCRWorkerError,RuntimeError)):
                worker.request(b'pdf',mode='accurate',redact=True,pdf=True,check_cancelled=cancel,
                               progress=lambda *args:seen.append(args))
            assert worker.pid is None
    finally:
        worker.close()
