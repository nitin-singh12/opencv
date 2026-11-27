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
COMMAND_LOG_DIR = None


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
        if COMMAND_LOG_DIR is None:
            result = subprocess.run(command, cwd=cwd, env=environment, timeout=timeout)
        else:
            log = COMMAND_LOG_DIR / f"{len(COMMANDS):02d}.log"
            record["log"] = str(log)
            with log.open("wb") as output:
                result = subprocess.run(command, cwd=cwd, env=environment, timeout=timeout,
                                        stdout=output, stderr=subprocess.STDOUT)
            print(log.read_text(encoding="utf-8", errors="replace"), flush=True)
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


def verify_bundle(archive, commit, unpublished=False):
    with zipfile.ZipFile(archive) as bundle:
        prefix = "opencv-bv-server-mcp/"
        manifest = json.loads(bundle.read(prefix + "bundle.json"))
        assert manifest["package_source_commit"] is None
        assert manifest["claimed_source_commit"] == (None if unpublished else commit)
        assert manifest["base_commit"] == (commit if unpublished else None)
        assert manifest["source_state"] == ("local-unpublished" if unpublished else "source-claimed")
        assert len(bundle.namelist()) == len(set(bundle.namelist()))
        assert set(bundle.namelist()) == {prefix + name for name in manifest["files_sha256"]} | {prefix + "bundle.json"}
        for name, expected in manifest["files_sha256"].items():
            assert ".." not in Path(name).parts and not name.startswith(("/", "\\"))
            assert hashlib.sha256(bundle.read(prefix + name)).hexdigest() == expected, name
        assert manifest["wheel_sha256"] == manifest["files_sha256"][manifest["wheel"]]
        return manifest


def validate(work, prepare_only=False, unpublished=False, feature_tests=False, search_tests=False):
    global COMMAND_LOG_DIR
    work.mkdir(parents=True, exist_ok=False)
    COMMAND_LOG_DIR = work / "logs"
    COMMAND_LOG_DIR.mkdir()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    # Setuptools and the ZIP builder read working-tree files, including new
    # modules/assets. They must belong to the claimed commit in committed mode.
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", "."],
        cwd=ROOT, text=True,
    ).rstrip("\0").split("\0")
    untracked = [name for name in untracked if name]
    if (dirty or untracked) and not unpublished:
        raise RuntimeError("Commit tracked changes and untracked package files before validating a versioned package, or use --unpublished")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    epoch = subprocess.check_output(["git", "show", "-s", "--format=%ct", "HEAD"], cwd=ROOT, text=True).strip()
    report = {"commit": None if unpublished else commit,
              "base_commit": commit if unpublished else None,
              "source_state": "local-unpublished" if unpublished else "committed",
              "tracked_changes": dirty.splitlines(),
              "untracked_files": untracked,
              "os": platform.platform(), "architecture": platform.machine(),
              "python": sys.version, "runner": os.environ.get("RUNNER_NAME"),
              "runner_image": os.environ.get("ImageOS"), "image_version": os.environ.get("ImageVersion"),
              "commands": COMMANDS, "status": "running"}
    started = time.monotonic()
    try:
        build_env = work / "build-env"
        run([sys.executable, "-m", "venv", build_env])
        builder = python_in(build_env)
        run([builder, "-m", "pip", "install", "build==1.6.1", "setuptools==84.2.1", "wheel==0.45.1",
             "nanobind==2.11.0", "cmake>=3.21,<4", "pypdfium2==5.14.0"])
        dist = work / "dist"
        run([builder, "-m", "build", "--wheel", "--no-isolation", "--outdir", dist], env={"SOURCE_DATE_EPOCH": epoch})
        wheel, = dist.glob("opencv_bv_tools-*.whl")
        archive = dist / "opencv-bv-server-mcp.zip"
        run([builder, ROOT / "scripts/build_mcp_zip.py", "--wheel", wheel,
             "--source-commit", commit, "--output", archive, *(["--unpublished"] if unpublished else [])])
        manifest = verify_bundle(archive, commit, unpublished)
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
                   for name in ("pypdfium2", "zxing-cpp", "Pillow", "pyoxipng", "pyclipper", "reportlab", "requests", "charset-normalizer", "rapidocr", "onnxruntime", "onnxruntime-gpu", "Shapely"))
        assert report["default_install"]["pypdfium2_absent"], "Standalone pypdfium2 must not be installed or importable"
        assert report["default_install"]["pdfium_binding_present"], "Bundled PDFium binding is missing"
        assert report["default_install"]["zxing_distribution_absent"], "Standalone ZXing must not be installed or importable"
        assert report["default_install"]["zxing_extension_present"], "Bundled ZXing extension is missing"
        assert report["default_install"]["pyclipper_absent"], "pyclipper must not be installed or importable"
        assert report["default_install"]["clipper_extension_present"], "The native Clipper extension is missing"
        assert report["default_install"]["rapidocr_absent"], "RapidOCR must not be installed or importable"
        assert report["default_install"]["onnxruntime_absent"], "ONNX Runtime must not be installed or importable"
        assert report["default_install"]["shapely_absent"], "Shapely must not be installed or importable"
        print("DEFAULT INSTALL " + json.dumps(report["default_install"], sort_keys=True), flush=True)
        run([installed, "-m", "ensurepip", "--upgrade"], cwd=work)
        if search_tests:
            # Exercise production dependencies before adding test-only oracles.
            # Public, checksummed assets; inference itself remains offline/local.
            run([installed, "-I", "-m", "opencv_bv_tools.model_assets", "--search"], cwd=work, timeout=1800)
            run([installed, "-I", ROOT / "dev/smoke_search.py", "--work", work / "search"],
                cwd=work, timeout=2400)
            report["real_model_search"] = json.loads((work / "search/search-report.json").read_text(encoding="utf-8"))
            assert report["real_model_search"]["status"] == "passed"
        if not prepare_only:
            # Independent TEST oracles, absent during the preceding runtime checks.
            test_dependencies = ["pytest==9.0.2", "psutil==7.2.2", "Pillow==12.3.0", "reportlab==4.4.9"]
            run([installed, "-m", "pip", "install", *test_dependencies], cwd=work)
            run([installed, extracted / "dev/check_platform.py", "--output", work / "platform-with-tests"], cwd=work)
            test_platform = json.loads((work / "platform-with-tests/platform-report.json").read_text())
            assert test_platform["target"]["onnxruntime_absent"], "Test dependencies must not install ONNX Runtime"
            assert test_platform["target"]["shapely_absent"], "Test dependencies must not install Shapely"
            report["test_install_onnxruntime_absent"] = True
            report["test_install_shapely_absent"] = True
            if feature_tests:
                # Install independent test oracles only after checking the runtime environment.
                run([installed, "-m", "pip", "install", "pypdfium2==5.14.0", "pypdf==6.10.0",
                     "pypng==0.20220715.0", "python-barcode==0.16.1", "segno==1.6.6", "qrcode==8.2"], cwd=work)
                run([installed, "-c", "import segno, qrcode"], cwd=work)
            shutil.copytree(ROOT / "tests", extracted / "tests", dirs_exist_ok=True)
            shutil.copyfile(ROOT / "scripts/build_mcp_zip.py", extracted / "scripts/build_mcp_zip.py")
            shutil.copyfile(ROOT / "dev/validate_package.py", extracted / "dev/validate_package.py")
            shutil.copyfile(ROOT / "pyproject.toml", extracted / "pyproject.toml")
            shutil.copyfile(ROOT / "setup.py", extracted / "setup.py")
            (extracted / "dist").mkdir()
            shutil.copyfile(wheel, extracted / "dist" / wheel.name)
            test_targets = ["tests/test_install_mcp.py", "tests/test_client_setup.py", "tests/test_native_wheel_tags.py",
                 "tests/test_native_runtime.py", "tests/test_ocr_process.py", "tests/test_ocr_opencv.py",
                 "tests/test_ocr_model_loading.py", "tests/test_model_assets.py",
                 "tests/test_unified_protocol.py",
                 "tests/test_face_blurring.py"]
            test_targets += ["tests/test_semantic_search.py", "tests/test_search_pagination.py",
                             "tests/test_search_encoder.py", "tests/test_search_decoding.py",
                             "tests/test_search_files.py", "tests/test_search_process.py"]
            if feature_tests:
                test_targets += ["tests/test_conversion.py", "tests/test_independent_conversion.py",
                                 "tests/test_barcode_reader.py", "tests/test_pdfium_native.py", "tests/test_qr_reader.py"]
                if sys.platform == "win32":
                    # Ordinary Windows output overwrite is intentionally refused;
                    # these portable feature cases supplement the nine-tool smoke
                    # and Windows-specific file/process contracts above.
                    test_targets += [
                        "tests/test_document_scanner.py::test_exif_then_clockwise_rotation_preserves_samples",
                        "tests/test_document_scanner.py::test_all_modes_alpha_hidden_samples_and_metadata",
                        "tests/test_document_scanner.py::test_deterministic_repeat_and_equivalent_half_turns",
                        "tests/test_document_scanner.py::test_scanner_supervision_cancellation_timeout_and_protocol",
                        "tests/test_png_compression.py::test_pillow_modes_and_non_growing_repeat",
                        "tests/test_png_compression.py::test_independent_encoder_depths_adam7_empty_passes",
                        "tests/test_png_compression.py::test_orientation_and_rendering_metadata_preserved",
                        "tests/test_png_compression.py::test_protocol_faults_reap_without_reader_keys",
                        "tests/test_pdf_mcp.py::test_pdf_image_pdf_reuses_worker_and_resets_settings",
                        "tests/test_pdf_mcp.py::test_pdf_redaction_covers_rendered_address_and_preserves_page_size",
                    ]
                    report["feature_scope"] = "portable Windows codec/PDF cases; POSIX overwrite/FIFO cases excluded"
                else:
                    test_targets += ["tests/test_files_cli.py", "tests/test_document_scanner.py",
                                     "tests/test_png_compression.py", "tests/test_pdf_mcp.py",
                                     "tests/test_integration_regressions.py", "tests/test_mcp_conversion.py"]
                    report["feature_scope"] = "selected feature modules including POSIX file publication cases"
            report["pytest_targets"] = test_targets
            run([installed, "-m", "pytest", "-q", "-ra", *test_targets, "--junitxml", work / "tests.xml"], cwd=extracted, timeout=900)
            report["native_contracts"] = ("selected tests plus feature regressions" if feature_tests else "selected tests") + "; passed; consult pytest log for skips"
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
                identity = f"unpublished working tree based on `{commit}`" if unpublished else f"commit `{commit}`"
                stream.write(f"### Package on {report['os']}\n\nStatus: **{report['status']}**; {identity}.\n\n")
                stream.write(f"Elapsed: {report['seconds']} seconds. See job logs for commands, dependencies and failures.\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="New directory outside the checkout")
    parser.add_argument("--prepare-only", action="store_true", help="Build/install/smoke without extra unit tests")
    parser.add_argument("--unpublished", action="store_true", help="Validate local changes and record HEAD only as the base commit")
    parser.add_argument("--feature-tests", action="store_true", help="Run additional selected feature regressions using test-only dependencies")
    parser.add_argument("--search-tests", action="store_true", help="Download verified search assets and run real installed-wheel CPU search acceptance")
    args = parser.parse_args()
    validate(args.work.resolve(), args.prepare_only, args.unpublished, args.feature_tests, args.search_tests)
