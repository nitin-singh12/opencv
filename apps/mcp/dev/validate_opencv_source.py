# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Compile the checkout's default CPU modules and test MCP with that cv2 on Linux."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from validate_package import COMMANDS, ROOT, run


# Installed only in the disposable validation environment. No import substitution.
AUDIT = r'''
import builtins, hashlib, json, os, sys
from pathlib import Path
_original_import = builtins.__import__
_recorded = False
_busy = False
_config = json.loads(Path(__file__).with_suffix('.json').read_text())

def _audit_import(name, *args, **kwargs):
    global _recorded, _busy
    result = _original_import(name, *args, **kwargs)
    module = sys.modules.get('cv2')
    if not _recorded and not _busy and module is not None and hasattr(module, 'getBuildInformation'):
        _busy = True
        try:
            maps = Path('/proc/self/maps').read_text().splitlines()
            native = sorted({line.split()[-1] for line in maps
                             if '.so' in line and ('/cv2' in line or '/libopencv_' in line)})
            record = dict(pid=os.getpid(), parent_pid=os.getppid(), argv=sys.argv,
                          version=module.__version__, module_file=module.__file__,
                          build_sha256=hashlib.sha256(module.getBuildInformation().encode()).hexdigest(),
                          native_libraries=native)
            target = Path(_config['audit']) / (str(os.getpid()) + '.json')
            target.write_text(json.dumps(record, indent=2) + '\n')
            _recorded = True
        finally:
            _busy = False
    return result

builtins.__import__ = _audit_import
'''

CPP = r'''
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
int main() {
    cv::Mat input(16, 16, CV_8UC3, cv::Scalar(10, 50, 100)), blurred, decoded;
    cv::GaussianBlur(input, blurred, cv::Size(3, 3), 0);
    std::vector<uchar> png;
    if (!cv::imencode(".png", blurred, png)) return 1;
    decoded = cv::imdecode(png, cv::IMREAD_COLOR);
    if (cv::norm(input, decoded, cv::NORM_INF) != 0) return 2;
    std::cout << "C++ compile/link/PNG roundtrip passed: OpenCV " << CV_VERSION << std::endl;
    return 0;
}
'''


def validate(work):
    if sys.platform != "linux":
        raise RuntimeError("This source-build validation targets native Linux")
    package = json.loads((work / "package-report.json").read_text())
    assert package["status"] == "passed"
    environment = Path(package["environment"])
    python = Path(package["installed_python"])
    extracted = Path(package["extracted"])
    repo = ROOT.parents[1]
    build = work / "opencv-build"
    audit = work / "source-imports"
    audit.mkdir(exist_ok=False)
    report = {"commit": package["commit"], "scope": "Linux CPU/default available modules; not every optional backend",
              "commands": COMMANDS, "status": "running", "opencv_unit_suite": "not run"}
    started = time.monotonic()
    try:
        run([python, "-m", "pip", "uninstall", "-y", "opencv-python-headless"], cwd=work)
        run([python, "-I", "-c", "import importlib.util; assert importlib.util.find_spec('cv2') is None"], cwd=work)
        config = json.loads(subprocess.check_output([str(python), "-I", "-c",
            "import json,sysconfig; from pathlib import Path; print(json.dumps(dict("
            "site=sysconfig.get_path('platlib'), include=sysconfig.get_path('include'), "
            "library=str(Path(sysconfig.get_config_var('LIBDIR')) / sysconfig.get_config_var('LDLIBRARY')))))"], text=True))
        site = Path(config['site'])
        assert Path(config['include']).is_dir() and Path(config['library']).is_file(), config
        options = ["-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={environment}",
                   f"-DCMAKE_INSTALL_RPATH={environment / 'lib'}", "-DBUILD_TESTS=OFF", "-DBUILD_PERF_TESTS=OFF",
                   "-DBUILD_EXAMPLES=OFF", "-DBUILD_JAVA=OFF", "-DBUILD_opencv_python3=ON",
                   "-DWITH_CUDA=OFF", "-DWITH_OPENCL=OFF", "-DWITH_GTK=OFF", "-DWITH_QT=OFF",
                   "-DWITH_FFMPEG=OFF", "-DWITH_GSTREAMER=OFF", "-DWITH_V4L=OFF",
                   f"-DPYTHON3_EXECUTABLE={python}", f"-DPYTHON_DEFAULT_EXECUTABLE={python}",
                   f"-DPYTHON3_INCLUDE_DIR={config['include']}", f"-DPYTHON3_LIBRARY={config['library']}",
                   f"-DOPENCV_PYTHON3_INSTALL_PATH={site}"]
        report["cmake_options"] = options
        # No BUILD_LIST: compile every module enabled by this configuration.
        run(["cmake", "-S", repo, "-B", build, "-G", "Ninja", *options], cwd=work, timeout=900)
        cache = (build / "CMakeCache.txt").read_text()
        modules = re.search(r"(?m)^OPENCV_MODULES_BUILD:INTERNAL=(.+)$", cache).group(1).split(';')
        report['modules'] = modules
        assert {'opencv_python3', 'opencv_dnn', 'opencv_imgcodecs', 'opencv_objdetect'} <= set(modules), modules
        run(["cmake", "--build", build, "--parallel", "2"], cwd=work, timeout=7200)
        run(["cmake", "--install", build], cwd=work, timeout=600)
        cpp = work / "source-smoke.cpp"
        cpp.write_text(CPP)
        header, = (environment / "include").glob("*/opencv2/core.hpp")
        run(["c++", "-std=c++17", cpp, "-I" + str(header.parents[1]),
             "-L" + str(environment / "lib"), "-Wl,-rpath," + str(environment / "lib"),
             "-lopencv_imgcodecs", "-lopencv_imgproc", "-lopencv_core", "-o", work / "source-smoke"], cwd=work)
        run([work / "source-smoke"], cwd=work)
        report["cpp_smoke"] = "compile/link/PNG roundtrip passed"
        (site / "opencv_source_audit.py").write_text(AUDIT)
        (site / "opencv_source_audit.json").write_text(json.dumps({"audit": str(audit)}))
        (site / "opencv_source_audit.pth").write_text("import opencv_source_audit\n")
        run([python, "-I", "-c", "import cv2; print(cv2.__file__); print(cv2.getBuildInformation())"], cwd=work)
        # CMake installs a module, not an opencv-python-headless distribution.
        # Retain this explicit packaging difference rather than inventing metadata.
        check = subprocess.run([str(python), "-m", "pip", "check"], cwd=work, text=True, capture_output=True)
        report["pip_check_after_source_install"] = {"exit_code": check.returncode, "output": check.stdout + check.stderr}
        assert check.returncode == 1
        assert all("requires opencv-python-headless, which is not installed" in line
                   for line in check.stdout.splitlines() if line.strip()), check.stdout
        source_env = {"OPENCV_BV_SOURCE_BUILD": str(environment)}
        run([python, extracted / "scripts/smoke_mcp.py", "--fixtures", extracted / "tests/fixtures"],
            cwd=work, env=source_env, timeout=300)
        records = [json.loads(p.read_text()) for p in sorted(audit.glob("*.json"))]
        assert records
        hashes = {record["build_sha256"] for record in records}
        assert len(hashes) == 1, records
        for record in records:
            assert record["version"].startswith("5."), record
            assert Path(record["module_file"]).resolve().is_relative_to(environment), record
            assert record["native_libraries"], record
            assert all(Path(path).resolve().is_relative_to(environment) for path in record["native_libraries"]), record
        for worker in ("face_worker.py", "ocr_worker.py"):
            assert any(any(str(arg).endswith(worker) for arg in record["argv"]) for record in records), (worker, records)
        report.update(status="passed", nine_tool_smoke="passed with source-built OpenCV",
                      source_imports=records, build_sha256=next(iter(hashes)))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = round(time.monotonic() - started, 3)
        if build.exists():
            cache = build / "CMakeCache.txt"
            if cache.exists():
                report["cmake_cache_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
        report["source_imports"] = [json.loads(p.read_text()) for p in sorted(audit.glob("*.json"))]
        (work / "source-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("SOURCE REPORT " + json.dumps(report, sort_keys=True), flush=True)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(f"### OpenCV source build (Linux)\n\nStatus: **{report['status']}**; commit `{report['commit']}`.\n\n")
                stream.write(f"{report['scope']}. Elapsed: {report['seconds']} seconds. OpenCV's own unit suite was not run.\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="Directory from validate_package.py --prepare-only")
    validate(parser.parse_args().work.resolve())
