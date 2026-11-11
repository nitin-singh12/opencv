"""Independent encoder, geometry, protocol and lifecycle regression tests."""

import base64
from dataclasses import asdict, replace
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image, ImageOps
import pytest
segno = pytest.importorskip("segno", reason="Install the qr-test extra for QR fixtures")

from opencv_bv_tools.qr_reader import QRLimits
from opencv_bv_tools.qr_reader import QRReadError
from opencv_bv_tools.qr_reader import read_qr
from opencv_bv_tools.qr_reader import _run_worker


def encoded(image, kind="PNG", **options):
    stream = BytesIO()
    image.save(stream, format=kind, **options)
    return stream.getvalue()


def qr(payload="QR test", **options):
    stream = BytesIO()
    segno.make(payload, micro=False, boost_error=False, **options).save(stream, kind="png", scale=5)
    return Image.open(BytesIO(stream.getvalue())).convert("RGB")


def payloads(result):
    assert result["search_status"] == "completed", result
    return [base64.b64decode(x["payload_base64"]) for x in result["symbols"]]


@pytest.mark.parametrize("version", range(1, 41))
@pytest.mark.parametrize("error", ["L", "M", "Q", "H"])
def test_versions_and_error_correction(version, error):
    text = f"V{version}{error}"
    result = read_qr(encoded(qr(text, version=version, error=error)))
    assert payloads(result) == [text.encode()]
    assert result["symbols"][0]["qr_version"] == str(version)
    assert result["symbols"][0]["ec_level"] == error


@pytest.mark.parametrize("content,options,expected,text", [
    (b"\x00\x01\xff\x80\nEND", {"mode": "byte"}, b"\x00\x01\xff\x80\nEND", None),
    ("café 日本語", {"encoding": "utf-8", "eci": True}, "café 日本語".encode(), "café 日本語"),
    ("漢字", {"mode": "kanji"}, "漢字".encode("shift_jis"), "漢字"),
    ("123456789012345", {"mode": "numeric"}, b"123456789012345", "123456789012345"),
    ("HELLO 123:$", {"mode": "alphanumeric"}, b"HELLO 123:$", "HELLO 123:$"),
    ("A" * 2900, {"mode": "byte", "error": "L"}, b"A" * 2900, "A" * 2900),
])
def test_payload_modes(content, options, expected, text):
    result = read_qr(encoded(qr(content, **options)))
    assert payloads(result) == [expected]
    assert result["symbols"][0]["text"] == text


def test_empty_payload_known_backend_limitation():
    result=read_qr(encoded(qr("",mode="byte")))
    assert payloads(result) == []
    assert "empty_payload_symbols_not_supported" in result["capability_limitations"]


def test_independent_encoder_mixed_segments_and_duplicates():
    import qrcode
    from qrcode.util import QRData, MODE_NUMBER, MODE_8BIT_BYTE
    generator = qrcode.QRCode(box_size=5, border=4)
    generator.add_data(QRData(b"123456789", mode=MODE_NUMBER))
    generator.add_data(QRData(b"abc\x00z", mode=MODE_8BIT_BYTE))
    image = generator.make_image().convert("RGB")
    canvas = Image.new("RGB", (image.width * 2 + 100, image.height + 100), "white")
    canvas.paste(image, (20, 30)); canvas.paste(image, (image.width + 70, 30))
    result = read_qr(encoded(canvas))
    assert payloads(result) == [b"123456789abc\x00z"] * 2
    assert result["symbols"][0]["polygon"] != result["symbols"][1]["polygon"]


@pytest.mark.parametrize("kind", ["PNG", "JPEG", "WEBP"])
@pytest.mark.parametrize("mode", ["1", "L", "P", "RGB", "RGBA", "LA"])
def test_format_modes(kind, mode):
    image = qr().convert(mode)
    if kind == "JPEG" and mode not in ("L", "RGB"):
        image = image.convert("RGB")
    data = encoded(image, kind)
    original = bytes(data)
    assert payloads(read_qr(data)) == [b"QR test"]
    assert original == data


def exif_point(x, y, width, height, orientation):
    # Independently specified EXIF coordinate mapping (pixel edge convention).
    return {1:(x,y), 2:(width-x,y), 3:(width-x,height-y), 4:(x,height-y),
            5:(y,x), 6:(height-y,x), 7:(height-y,width-x), 8:(y,width-x)}[orientation]


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_geometry(orientation):
    canvas = Image.new("RGB", (700, 500), "white")
    tile = qr()
    canvas.paste(tile, (50, 120))
    exif = Image.Exif(); exif[274] = orientation
    result = read_qr(encoded(canvas, exif=exif))
    assert payloads(result) == [b"QR test"]
    assert result["image_size"] == ([700,500] if orientation < 5 else [500,700])
    expected = [exif_point(x,y,700,500,orientation) for x,y in
                [(70,140),(50+tile.width-20,140),(50+tile.width-20,120+tile.height-20),(70,120+tile.height-20)]]
    actual = result["symbols"][0]["polygon"]
    for x,y in expected:
        assert min((x-a)**2 + (y-b)**2 for a,b in actual) <= 9


@pytest.mark.parametrize("transform", ["invert", "mirror", "rotate", "contrast"])
def test_transformations(transform):
    image = qr("transformation", error="H")
    if transform == "invert": image = ImageOps.invert(image)
    if transform == "mirror": image = ImageOps.mirror(image)
    if transform == "rotate": image = image.rotate(31, expand=True, fillcolor="white")
    if transform == "contrast": image = image.convert("L").point(lambda p: 80 + p * 0.4)
    assert payloads(read_qr(encoded(image))) == [b"transformation"]


def test_hidden_rgb_and_palette_transparency():
    image = qr().convert("RGBA"); image.putalpha(0)
    assert payloads(read_qr(encoded(image))) == []
    palette = qr().convert("P", palette=Image.Palette.ADAPTIVE, colors=2)
    # All pixels transparent, regardless of which palette entry is white.
    assert payloads(read_qr(encoded(palette, transparency=bytes([0,0])))) == []
    image.putalpha(255)
    assert payloads(read_qr(encoded(image))) == [b"QR test"]


def test_micro_qr_excluded():
    b=BytesIO(); segno.make('123',micro=True).save(b,kind="png",scale=8)
    assert payloads(read_qr(b.getvalue())) == []


def test_structured_append_not_claimed_as_complete_message():
    sequence=segno.make_sequence("ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 3,symbol_count=2)
    b=BytesIO();sequence[0].save(b,kind="png",scale=6)
    result=read_qr(b.getvalue())
    assert len(payloads(result)) == 1
    assert "not assembled" in result["sequence_semantics"]
    assert result["symbols"][0]["sequence_metadata"] is None


@pytest.mark.parametrize("data", [b"not image", b"\x89PNG\r\n\x1a\n", encoded(qr())[:80],
                                   encoded(qr(),"GIF"), encoded(Image.new("I",(10,10)))])
def test_bad_or_unsupported_images(data):
    result=read_qr(data)
    assert result["search_status"] == "failed"
    assert result["stop_reason"] == "invalid_or_unsupported_image"


@pytest.mark.parametrize("kind", ["PNG", "WEBP"])
def test_animation_rejected(kind):
    data=encoded(qr(),kind,save_all=True,append_images=[qr("other")],duration=50,loop=0)
    assert read_qr(data)["search_status"] == "failed"


@pytest.mark.parametrize("field,value", [("max_pixels",True),("max_pixels",0),
    ("max_pixels",20_000_001),("max_symbols",129),("timeout_seconds",float("nan")),
    ("timeout_seconds",float("inf")),("timeout_seconds",True),("timeout_seconds",0),
    ("max_dimension",-1),("max_total_payload_bytes",1.5)])
def test_invalid_limits(field,value):
    with pytest.raises(QRReadError): read_qr(b"data",limits=replace(QRLimits(),**{field:value}))


@pytest.mark.parametrize("data", [b"", "not bytes", bytearray(b"x")])
def test_invalid_bytes(data):
    with pytest.raises(QRReadError): read_qr(data)


def test_pixel_dimension_byte_and_output_limits():
    image=qr("too many bytes")
    assert read_qr(encoded(image),limits=QRLimits(max_pixels=10))["stop_reason"]=="input_pixel_limit"
    assert read_qr(encoded(Image.new("L",(17000,1))))["stop_reason"]=="input_dimension_limit"
    with pytest.raises(QRReadError): read_qr(b"123",limits=QRLimits(max_input_bytes=2))
    result=read_qr(encoded(image),limits=QRLimits(max_payload_bytes=2))
    assert result["search_status"]=="resource_limited" and not result["symbols"]
    canvas=Image.new("RGB",(image.width*2+80,image.height+40),"white")
    canvas.paste(image,(0,0));canvas.paste(image,(image.width+50,0))
    result=read_qr(encoded(canvas),limits=QRLimits(max_symbols=1))
    assert result["search_status"]=="resource_limited" and len(result["symbols"])==1


def fake_worker(code, timeout=0.25, data=b"x", cancelled=None):
    return _run_worker(data,{"limits":asdict(QRLimits()),"background":[255]*3},
                       timeout=timeout,cancelled=cancelled,command=[sys.executable,"-c",code])


@pytest.mark.parametrize("code,reason", [
    ("import time;time.sleep(10)","deadline_exceeded"),
    ("import sys,time;sys.stdout.write('{');sys.stdout.flush();time.sleep(10)","deadline_exceeded"),
    ("import sys;sys.stdout.buffer.write(b'x'*2000000)","worker_output_limit"),
    ("import sys;sys.stderr.buffer.write(b'x'*10000)","worker_diagnostic_limit"),
    ("import os;os._exit(7)","worker_failed"),
    ("print('{}')","invalid_worker_response"),
])
def test_worker_protocol_failures_and_reaping(code,reason):
    import psutil
    before={p.pid for p in psutil.Process().children()}
    started=time.monotonic()
    result=fake_worker(code,data=b"x"*1_000_000)
    assert result["stop_reason"]==reason
    assert time.monotonic()-started < 2
    assert {p.pid for p in psutil.Process().children()}==before


def test_valid_output_then_hung_worker_is_timeout():
    valid=dict(search_status="completed",stop_reason="done",symbols=[],undecoded=[],image_size=[1,1])
    code=f"import sys,time;print({json.dumps(valid)!r},flush=True);time.sleep(10)"
    assert fake_worker(code)["stop_reason"]=="deadline_exceeded"


def test_cancel_before_during_and_exception_cleanup():
    import psutil
    assert read_qr(encoded(qr()),cancelled=lambda:True)["stop_reason"]=="cancelled"
    before={p.pid for p in psutil.Process().children()}
    started=time.monotonic()
    assert fake_worker("import time;time.sleep(10)",cancelled=lambda:time.monotonic()-started>.05)["stop_reason"]=="cancelled"
    def interrupt():
        if time.monotonic()-started>.05: raise KeyboardInterrupt
        return False
    started=time.monotonic()
    with pytest.raises(KeyboardInterrupt): fake_worker("import time;time.sleep(10)",cancelled=interrupt)
    assert {p.pid for p in psutil.Process().children()}==before
    assert payloads(read_qr(encoded(qr()))) == [b"QR test"]


def test_repeated_requests_do_not_leak_fds_or_children():
    import psutil
    current=psutil.Process()
    before=(current.num_fds(),{p.pid for p in current.children()})
    for _ in range(8): assert payloads(read_qr(encoded(qr()))) == [b"QR test"]
    assert (current.num_fds(),{p.pid for p in current.children()})==before


def test_cli_json_and_original_unchanged(tmp_path):
    path=tmp_path/"input.png";data=encoded(qr(b"\x1b[31m\x00",mode="byte"));path.write_bytes(data)
    run=subprocess.run([sys.executable,"-m","opencv_bv_tools.qr_reader",str(path)],capture_output=True,timeout=5)
    assert run.returncode==0 and b"\x1b" not in run.stdout
    assert payloads(json.loads(run.stdout)) == [b"\x1b[31m\x00"]
    assert path.read_bytes()==data


def test_termination_escalates_and_reaps():
    code="import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(10)"
    start=time.monotonic()
    assert fake_worker(code)["stop_reason"]=="deadline_exceeded"
    assert time.monotonic()-start < 1.5


def test_unavailable_worker_executable():
    result=_run_worker(b"x",{"limits":asdict(QRLimits()),"background":[255]*3},timeout=1,
                       command=["/nonexistent-qr-test-python"])
    assert result["stop_reason"]=="worker_start_failed"


@pytest.mark.parametrize("background", [(256,0,0),[0,0,0],(True,0,0),(0,0),None])
def test_invalid_background(background):
    with pytest.raises(QRReadError):read_qr(b"x",background=background)


@pytest.mark.parametrize("change", [
    {"symbols": [{"polygon":[[0,0]]*4,"payload_base64":"!!!","text":None}]},
    {"symbols": [{"polygon":[[999,0]]*4,"payload_base64":"eA==","text":"x"}]},
    {"image_size": [True,1]},
    {"search_status": "invented"},
    {"undecoded": "not a list"},
])
def test_invalid_worker_response_fields(change):
    result=dict(search_status="completed",stop_reason="done",symbols=[],undecoded=[],image_size=[1,1])
    result.update(change)
    assert fake_worker(f"print({json.dumps(result)!r})")["stop_reason"]=="invalid_worker_response"


def test_mutated_inputs_finish_without_child_leaks():
    import random,psutil
    generator=random.Random(20261103)
    data=encoded(qr())
    before={p.pid for p in psutil.Process().children()}
    for _ in range(16):
        damaged=bytearray(data)
        for _ in range(4):damaged[generator.randrange(len(damaged))]=generator.randrange(256)
        result=read_qr(bytes(damaged),limits=QRLimits(timeout_seconds=.3))
        assert result["search_status"] in ("completed","failed","budget_exhausted")
    assert {p.pid for p in psutil.Process().children()}==before


def test_icc_ignored_and_bad_exif_rejected():
    result=read_qr(encoded(qr(),icc_profile=b"invalid profile"))
    assert payloads(result)==[b"QR test"]
    assert any("ICC profile ignored" in x for x in result["warnings"])
    result=read_qr(encoded(qr(),exif=b"not-exif"))
    assert result["stop_reason"]=="invalid_or_unsupported_image"


def test_aggregate_limits_and_negative_no_results():
    for limits in [QRLimits(max_total_payload_bytes=2),QRLimits(max_text_chars=2)]:
        result=read_qr(encoded(qr()),limits=limits)
        assert result["search_status"]=="resource_limited" and result["symbols"]==[]
    assert payloads(read_qr(encoded(Image.new("L",(40,40),255))))==[]


def test_large_visible_alpha_with_orientation_under_default_limit():
    canvas=Image.new("RGBA",(3200,2400),(255,255,255,200));canvas.paste(qr(),(80,200))
    exif=Image.Exif();exif[274]=6
    result=read_qr(encoded(canvas,exif=exif))
    assert payloads(result)==[b"QR test"]
    assert result["image_size"]==[2400,3200]
    # This measures this Linux worker, not an enforced process memory limit.
    assert result["worker_peak_rss_kib"] < 512*1024
