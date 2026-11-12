# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Build, install and exercise a clean ZIP on one native OS. No cloud services."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = []


def run(command, cwd=ROOT, timeout=900, env=None):
    command = [str(value) for value in command]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    environment.update(env or {})
    record = {"command": command, "cwd": str(cwd)}
    COMMANDS.append(record)
    print(json.dumps(record), flush=True)
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=cwd, env=environment, timeout=timeout)
        record["exit_code"] = result.returncode
        result.check_returncode()
    finally:
        record["seconds"] = round(time.monotonic() - started, 3)


def python_in(environment):
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def content_digest(archive):
    """Compare extracted payloads, independently of ZIP compression/attributes."""
    entries = {}
    with zipfile.ZipFile(archive) as bundle:
        for name in sorted(bundle.namelist()):
            data = bundle.read(name)
            if name.endswith("/bundle.json"):
                # Archive hashes differ across zlib builds; validate them separately.
                continue
            if name.endswith(".whl"):
                with zipfile.ZipFile(io.BytesIO(data)) as wheel:
                    for member in sorted(wheel.namelist()):
                        if member.endswith(".dist-info/RECORD"):
                            continue
                        entries[name + "/" + member] = hashlib.sha256(wheel.read(member)).hexdigest()
            else:
                entries[name] = hashlib.sha256(data).hexdigest()
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(), entries


def verify_bundle(archive, commit):
    with zipfile.ZipFile(archive) as bundle:
        prefix = "opencv-bv-server-mcp/"
        manifest = json.loads(bundle.read(prefix + "bundle.json"))
        assert manifest["package_source_commit"] == commit
        assert len(bundle.namelist()) == len(set(bundle.namelist()))
        assert set(bundle.namelist()) == {prefix + name for name in manifest["files_sha256"]} | {prefix + "bundle.json"}
        for name, expected in manifest["files_sha256"].items():
            assert ".." not in Path(name).parts and not name.startswith(("/", "\\"))
            assert hashlib.sha256(bundle.read(prefix + name)).hexdigest() == expected, name
        assert manifest["wheel_sha256"] == manifest["files_sha256"][manifest["wheel"]]
        return manifest


def validate(work, prepare_only=False):
    work.mkdir(parents=True, exist_ok=False)
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError("Commit tracked changes before validating a versioned package")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    epoch = subprocess.check_output(["git", "show", "-s", "--format=%ct", "HEAD"], cwd=ROOT, text=True).strip()
    report = {"commit": commit, "os": platform.platform(), "architecture": platform.machine(),
              "python": sys.version, "runner": os.environ.get("RUNNER_NAME"),
              "runner_image": os.environ.get("ImageOS"), "image_version": os.environ.get("ImageVersion"),
              "commands": COMMANDS, "status": "running"}
    started = time.monotonic()
    try:
        build_env = work / "build-env"
        run([sys.executable, "-m", "venv", build_env])
        builder = python_in(build_env)
        run([builder, "-m", "pip", "install", "build==1.6.1", "setuptools==84.2.1", "wheel==0.45.1"])
        upstream = work / "upstream"
        run([builder, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
             "rapidocr==3.10.0", "-d", upstream])
        companion = work / "companion"
        run([builder, ROOT / "scripts/build_rapidocr.py", upstream / "rapidocr-3.10.0-py3-none-any.whl",
             "--output", companion])
        dist = work / "dist"
        run([builder, "-m", "build", "--wheel", "--no-isolation", "--outdir", dist], env={"SOURCE_DATE_EPOCH": epoch})
        wheel, = dist.glob("opencv_bv_tools-*.whl")
        ocr_wheel, = companion.glob("*.whl")
        archive = dist / "opencv-bv-server-mcp.zip"
        run([builder, ROOT / "scripts/build_mcp_zip.py", "--wheel", wheel, "--rapidocr-wheel", ocr_wheel,
             "--source-commit", commit, "--output", archive])
        manifest = verify_bundle(archive, commit)
        digest, entries = content_digest(archive)
        report.update(zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), payload_sha256=digest,
                      wheel_sha256=manifest["wheel_sha256"], package_version=manifest["package_version"])
        (work / "payload-manifest.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
        expected = os.environ.get("EXPECTED_PAYLOAD_SHA256")
        report["same_payload_as_linux"] = digest == expected if expected else None
        if expected:
            # Keep functional validation separate from reproducibility.
            print(f"Payload matches Linux: {digest == expected}; sha256={digest}", flush=True)
            if digest != expected:
                print("PAYLOAD MANIFEST " + json.dumps(entries, sort_keys=True), flush=True)
        output = os.environ.get("GITHUB_OUTPUT")
        if output:
            with open(output, "a", encoding="utf-8") as stream:
                stream.write(f"payload_sha256={digest}\n")
        extracted = work / "package with spaces café" / "opencv-bv-server-mcp"
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted.parent)
        run([sys.executable, extracted / "scripts/install_mcp.py", "--install-only"], cwd=extracted, timeout=1200)
        state = json.loads((extracted / ".install-state.json").read_text())
        environment = (extracted / state["environment"]).resolve()
        installed = python_in(environment)
        report.update(environment=str(environment), extracted=str(extracted), installed_python=str(installed),
                      nine_tool_smoke="passed during clean ZIP installation")
        run([installed, "-I", "-c", "from pathlib import Path; import opencv_bv_tools,sys; "
             "p=Path(opencv_bv_tools.__file__).resolve(); print(p); "
             "assert p.is_relative_to(Path(sys.prefix).resolve())"], cwd=work)
        run([installed, extracted / "dev/check_platform.py", "--output", work / "platform"], cwd=work)
        platform_report = json.loads((work / "platform/platform-report.json").read_text())
        report["default_install"] = platform_report["target"]
        assert all(report["default_install"]["packages"][name] is None
                   for name in ("Pillow", "reportlab", "requests", "charset-normalizer"))
        print("DEFAULT INSTALL " + json.dumps(report["default_install"], sort_keys=True), flush=True)
        run([installed, "-m", "ensurepip", "--upgrade"], cwd=work)
        if not prepare_only:
            # Independent TEST oracles, absent during the preceding runtime checks.
            run([installed, "-m", "pip", "install", "pytest==9.0.2", "psutil==7.2.2",
                 "Pillow==12.3.0", "reportlab==4.4.9"], cwd=work)
            shutil.copytree(ROOT / "tests", extracted / "tests", dirs_exist_ok=True)
            shutil.copyfile(ROOT / "scripts/build_mcp_zip.py", extracted / "scripts/build_mcp_zip.py")
            shutil.copyfile(ROOT / "pyproject.toml", extracted / "pyproject.toml")
            run([installed, "-m", "pytest", "-q", "-ra", "tests/test_install_mcp.py",
                 "tests/test_native_runtime.py", "tests/test_ocr_process.py", "tests/test_unified_protocol.py",
                 "tests/test_face_blurring.py", "--junitxml", work / "tests.xml"], cwd=extracted, timeout=900)
            report["native_contracts"] = "passed; consult pytest log for platform-specific skips"
        report["status"] = "passed"
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = round(time.monotonic() - started, 3)
        (work / "package-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("PACKAGE REPORT " + json.dumps(report, sort_keys=True), flush=True)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(f"### Package on {report['os']}\n\nStatus: **{report['status']}**; commit `{commit}`.\n\n")
                stream.write(f"Elapsed: {report['seconds']} seconds. See job logs for commands, dependencies and failures.\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="New directory outside the checkout")
    parser.add_argument("--prepare-only", action="store_true", help="Build/install/smoke without extra unit tests")
    args = parser.parse_args()
    validate(args.work.resolve(), args.prepare_only)
