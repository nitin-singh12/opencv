"""Exercise all nine tools through MCP stdio using synthetic/local fixtures.

Run with the all and qr-test extras installed. --fixtures defaults to this
checkout's tests/fixtures; pass its absolute path when testing an installed
wheel from a different working directory. All outputs use a temporary folder.
"""
import argparse
import asyncio
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from opencv_bv_tools.runtime.image_io import decode, encode
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

NAMES = {"convert_image_format", "read_qr", "read_barcodes", "compress_png",
         "blur_faces", "scan_document", "redact_emails", "find_emails", "redact_pdf"}
LAUNCH = """
import socket
def offline(*args, **kwargs):
    raise RuntimeError('External network is disabled during the MCP smoke test')
connect = socket.socket.connect
def offline_connect(self, address):
    # Windows implements asyncio's socketpair through numeric loopback TCP.
    if self.family in (socket.AF_INET, socket.AF_INET6) and address[0] in ('127.0.0.1', '::1'):
        return connect(self, address)
    return offline()
socket.socket.connect = offline_connect
socket.create_connection = offline
from opencv_bv_tools.server import main
main()
"""


async def exercise(work, fixtures, pdf_engine="picture"):
    qr = work / "qr.png"
    qr.write_bytes((fixtures / "smoke/qr.png").read_bytes())
    barcode = work / "barcode.png"
    barcode.write_bytes((fixtures / "barcode/upce_0123456.png").read_bytes())
    face = work / "face.png"
    face.write_bytes(encode(decode((fixtures / "face/grace_hopper.jpg").read_bytes())[0]))
    email = work / "email.png"
    email.write_bytes((fixtures / "smoke/email.png").read_bytes())
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (qr, barcode, face, email)}
    server = StdioServerParameters(command=sys.executable, args=["-c", LAUNCH, "--allow-dir", str(work),
                                                               "--pdf-engine", pdf_engine])
    observations = {}
    started = time.perf_counter()
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            observations["handshake_seconds"] = round(time.perf_counter() - started, 3)
            names = {tool.name for tool in (await session.list_tools()).tools}
            assert names == NAMES, names
            observations["tools"] = sorted(names)
            async def call(name, **arguments):
                response = await session.call_tool(name, arguments)
                assert not response.is_error, (name, response.content)
                text = next(block.text for block in response.content if block.type == "text")
                result = json.loads(text)
                observations[name] = {k: result[k] for k in ("status", "search_status", "boxes_redacted", "output_format") if k in result}
                if "output_path" in result:
                    paths = {"returned": Path(result["output_path"])}
                    if arguments.get("output_path"):
                        paths["requested"] = Path(arguments["output_path"])
                    files = {}
                    for label, path in paths.items():
                        files[label] = {"path": str(path), "exists": path.is_file(),
                                        "entries": sorted(p.name for p in path.parent.iterdir())}
                        if path.is_file():
                            files[label]["bytes"] = path.stat().st_size
                    print(json.dumps({"tool": name, "outputs": files}), file=sys.stderr, flush=True)
                    assert all(item["exists"] for item in files.values()), files
                return result, response
            converted, _ = await call("convert_image_format", image_path=str(qr), output_path=str(work / "qr.webp"))
            assert converted["output_format"] == "WEBP"
            decoded, _ = await call("read_qr", image_path=str(qr))
            assert decoded["search_status"] == "completed" and any(base64.b64decode(s["payload_base64"]) == b"MCP smoke" for s in decoded["symbols"])
            decoded, _ = await call("read_barcodes", image_path=str(barcode), formats=["UPCE"])
            assert decoded["search_status"] == "completed" and decoded["symbols"]
            compressed, _ = await call("compress_png", image_path=str(qr), output_path=str(work / "small.png"))
            assert compressed["status"] in {"optimized", "unchanged"}
            assert np.array_equal(decode(qr.read_bytes())[0], decode((work / "small.png").read_bytes())[0])
            blurred, _ = await call("blur_faces", image_path=str(face), output_path=str(work / "blurred.png"))
            assert blurred["status"] == "blurred" and blurred["regions"]
            scanned, _ = await call("scan_document", image_path=str(email), output_path=str(work / "document.pdf"), rotation=15.0)
            assert scanned["status"] == "completed" and (work / "document.pdf").read_bytes().startswith(b"%PDF")
            pdf_source = work / "email.pdf"
            pdf_source.write_bytes((fixtures / "smoke/email.pdf").read_bytes())
            pdf, _ = await call("redact_pdf", pdf_path=str(pdf_source))
            assert pdf["pages"] == 1 and pdf["boxes_redacted"] >= 2 and pdf["engine"] == pdf_engine
            assert Path(pdf["output_path"]).read_bytes().startswith(b"%PDF")
            found, _ = await call("find_emails", image_path=str(email), reveal_addresses=True)
            assert found["boxes_redacted"] >= 2 and all("***" in address for address in found["addresses"])
            redacted, _ = await call("redact_emails", image_path=str(email), output_path=str(work / "redacted.png"))
            assert redacted["boxes_xywh"] == found["boxes_xywh"]
            output=decode((work / "redacted.png").read_bytes())[0]
            original=decode(email.read_bytes())[0]
            for x,y,w,h in redacted["boxes_xywh"]:
                assert not output[y:y+h,x:x+w,:3].any()
            assert np.array_equal(output[120:170,:,:3],original[120:170,:,:3])
            inline, response = await call("redact_emails", image_base64=base64.b64encode(email.read_bytes()).decode())
            assert "output_path" not in inline and any(block.type == "image" for block in response.content)
            for alpha in (255, 120):
                alpha_source = work / f"alpha-{alpha}.png"
                original=decode(email.read_bytes())[0]
                rgba=np.concatenate((original,np.full((*original.shape[:2],1),alpha,np.uint8)),axis=2)
                alpha_source.write_bytes(encode(rgba))
                output = work / f"alpha-{alpha}-redacted.png"
                _, response = await call("redact_emails", image_path=str(alpha_source),
                                         output_path=str(output), return_image=True)
                returned = next(block for block in response.content if block.type == "image")
                saved=decode(output.read_bytes())[0]
                preview=decode(base64.b64decode(returned.data))[0]
                assert saved.shape[2]==4 and (saved[:,:,3]==255).all()
                assert preview.shape[2]==3 and np.array_equal(saved[:,:,:3],preview)
            observations["alpha_file_and_returned_image_parity"] = "passed"
            for name, args in [("compress_png", {"image_path": str(qr), "output_path": str(qr), "overwrite": True}),
                               ("read_qr", {"image_path": str(work.parent / "outside.png")}),
                               ("scan_document", {"image_path": str(email), "output_path": str(work / "bad.pdf"), "dpi": True})]:
                response = await session.call_tool(name, args)
                assert response.is_error, name
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in originals.items())
    observations.update(network="server external socket connections blocked; numeric loopback allowed; subprocess network not isolated",
                        pdf_engine=pdf_engine, lookalike="unchanged", original_inputs="unchanged",
                        elapsed_seconds=round(time.perf_counter()-started, 3))
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures")
    parser.add_argument("--pdf-engine", choices=["picture", "mupdf"], default="picture")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="opencv-bv-smoke-") as directory:
        print(json.dumps(asyncio.run(exercise(Path(directory), args.fixtures.resolve(), args.pdf_engine)), indent=2))


if __name__ == "__main__":
    main()
