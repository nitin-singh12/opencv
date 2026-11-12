"""Collect native-platform diagnostics without installing or registering anything.

Uses only Python's standard library. The selected interpreter reports its own
architecture, dependencies and server registration result. No inference worker,
MCP client, network request or paid model call is started. A successful check is
registration-only evidence, not nine-tool or operating-system acceptance.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


PROBE = r'''
import asyncio
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

packages = ["opencv-bv-tools", "Pillow", "mcp", "zxing-cpp", "pyoxipng",
            "reportlab", "requests", "charset-normalizer", "opencv-python-headless", "numpy",
            "rapidocr", "onnxruntime", "pypdfium2", "segno"]
versions = {}
for name in packages:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
kernel = platform.release()
runtime = "linux-wsl" if sys.platform == "linux" and "microsoft" in kernel.lower() else sys.platform
result = dict(python=sys.version, executable=sys.executable, os=platform.system(),
              runtime=runtime, kernel=kernel, architecture=platform.machine(),
              packages=versions,
              capabilities=dict(fcntl=importlib.util.find_spec("fcntl") is not None,
                                resource=importlib.util.find_spec("resource") is not None,
                                killpg=hasattr(os, "killpg"),
                                O_DIRECTORY=hasattr(os, "O_DIRECTORY"),
                                O_NOFOLLOW=hasattr(os, "O_NOFOLLOW"),
                                dir_fd_operations=sorted(f.__name__ for f in os.supports_dir_fd)),
              capability_note="API presence only; native file-safety and process behavior are not validated",
              installer_platform_gate=(platform.machine() in {"linux": {"x86_64"},
                  "darwin": {"arm64"}, "win32": {"AMD64", "x86_64"}}.get(sys.platform, set())
                  and sys.version_info[:2] == (3, 12)),
              tool_calls="not_run", client_registration="not_changed")
try:
    from opencv_bv_tools.server import create_server, ServerConfig
    with tempfile.TemporaryDirectory(prefix="opencv-bv-platform-") as directory:
        server = create_server(ServerConfig(allow_dirs=(Path(directory),)))
        names = sorted(tool.name for tool in asyncio.run(server.list_tools()))
    expected = {"convert_image_format", "read_qr", "read_barcodes", "compress_png",
                "blur_faces", "scan_document", "find_emails", "redact_emails", "redact_pdf"}
    heavy = sorted(set(sys.modules) & {"cv2", "numpy", "rapidocr", "onnxruntime", "pypdfium2", "pymupdf"})
    result.update(tools=names, heavy_imports=heavy)
    optional = {"Pillow", "reportlab", "requests", "charset-normalizer", "segno"}
    missing = [name for name, version in versions.items() if version is None and name not in optional]
    result["missing_runtime_dependencies"] = missing
    result["status"] = "registration_only_passed" if set(names) == expected and not heavy and not missing else "blocked"
except Exception as error:
    result.update(status="blocked", startup_error={"type": type(error).__name__, "message": str(error)})
print(json.dumps(result, ensure_ascii=True, allow_nan=False))
'''


def run_probe(python, output, timeout):
    """Run only the registration probe; it never launches inference children."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    started = time.perf_counter()
    report = dict(started=datetime.now(timezone.utc).isoformat(),
                  requested_python=str(python), scope="dependency inventory and in-process registration only",
                  probe_sha256=hashlib.sha256(PROBE.encode()).hexdigest(),
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  native_tool_acceptance="not_run", provider_cost_usd=0)
    code = 2
    try:
        # File-backed logs avoid unbounded output in memory. No shell is used.
        with (output / "stdout.txt").open("wb") as stdout, (output / "stderr.txt").open("wb") as stderr:
            process = subprocess.run([str(python), "-c", PROBE], stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, cwd=output,
                                     env=env, timeout=timeout, check=False)
        report["process_exit_code"] = process.returncode
        if process.returncode:
            report.update(status="probe_failed")
        else:
            raw = (output / "stdout.txt").read_bytes()
            if len(raw) > 1024 * 1024:
                raise ValueError("Probe response exceeds 1 MiB")
            result = json.loads(raw.decode("utf-8"))
            if not isinstance(result, dict) or result.get("status") not in {"blocked", "registration_only_passed"}:
                raise ValueError("Invalid probe response")
            report.update(status=result["status"], target=result)
            code = 0 if result["status"] == "registration_only_passed" else 2
    except subprocess.TimeoutExpired:
        # subprocess.run kills/reaps the direct probe. The probe intentionally
        # performs no inference and creates no subprocess descendants.
        report.update(status="timeout", timeout_seconds=timeout)
        code = 3
    except (OSError, ValueError) as error:
        report.update(status="probe_failed", driver_error=dict(type=type(error).__name__, message=str(error)))
    report.update(runtime_seconds=time.perf_counter()-started,
                  finished=datetime.now(timezone.utc).isoformat())
    (output / "platform-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Absolute path to the target environment's Python")
    parser.add_argument("--output", type=Path, required=True, help="New directory for report and raw logs")
    parser.add_argument("--timeout", type=float, default=30, help="Probe timeout in seconds, 0.1–120")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0.1 <= args.timeout <= 120:
        parser.error("--timeout must be finite and between 0.1 and 120 seconds")
    if not args.python.is_absolute():
        parser.error("--python must be an absolute path")
    os.umask(0o077)
    try:
        report, code = run_probe(args.python, args.output, args.timeout)
    except OSError as error:
        parser.error(f"Cannot create a fresh report directory: {error}")
    print(f"{report['status']}: {args.output.resolve() / 'platform-report.json'}")
    print("Native tool acceptance was not run. This does not certify platform support.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
