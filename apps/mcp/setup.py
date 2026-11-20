"""Build the bundled Clipper extension from verified upstream source."""
import hashlib
import json
from importlib import metadata
import os
import platform
import re
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel

root = Path(__file__).resolve().parent
source = root.parent.parent / "3rdparty" / "clipper"
staged = root / ".clipper-vendor"
expected = {
    "clipper.cpp": "5c642a3668311701f72572443aa42c1a981edb037298efc015166d9d90be0755",
    "clipper.hpp": "734eba9dc9d399089b2b467017074bd24728a1b9e64c7429e827806ed10e54cc",
    "LICENSE.txt": "c9bff75738922193e67fa726fa225535870d2aa1059f91452c411736284ad566",
}


def verify(path, digest):
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"Missing or modified Clipper source: {path}")


# A checkout stages the tracked source. An extracted sdist already contains it.
if source.is_dir():
    for name, digest in expected.items():
        verify(source / name, digest)
    staged.mkdir(exist_ok=True)
    for name, digest in expected.items():
        destination = staged / name
        if destination.exists():
            verify(destination, digest)
        else:
            shutil.copyfile(source / name, destination)
for name, digest in expected.items():
    verify(staged / name, digest)

zxing_source = root.parent.parent / "3rdparty" / "zxing"
zxing_staged = root / ".zxing-vendor"
zxing_manifest = json.loads((root / "native" / "zxing-source-manifest.json").read_text())
for name, digest in zxing_manifest["files"].items():
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Invalid ZXing source path")
    destination = zxing_staged / relative
    if zxing_source.is_dir():
        origin = zxing_source / relative
        verify(origin, digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(origin, destination)
    verify(destination, digest)


class PackageBuild(build_py):
    def run(self):
        super().run()
        # Share the installer's existing registration and access helpers with
        # the installed setup command, without maintaining a second copy.
        destination = Path(self.build_lib) / "opencv_bv_tools"
        destination.mkdir(parents=True, exist_ok=True)
        self.copy_file(str(root / "scripts" / "install_mcp.py"), str(destination / "_installation.py"))


class NativeBuild(build_ext):
    def run(self):
        super().run()
        self.bundle_pdfium()
        if sys.platform == "darwin":
            self.verify_macos_binaries()

    def verify_macos_binaries(self):
        wheel = self.distribution.get_command_obj("bdist_wheel")
        wheel.ensure_finalized()
        tag = wheel.get_tag()[2]
        match = re.fullmatch(r"macosx_(\d+)_(\d+)_arm64", tag)
        if not match:
            raise RuntimeError(f"Unsupported macOS wheel platform: {tag}")
        wheel_minimum = tuple(map(int, match.groups()))
        native_dir = ((root / "src") if self.inplace else Path(self.build_lib)) / "opencv_bv_tools" / "_native"
        binaries = [Path(self.get_ext_fullpath(ext.name)) for ext in self.extensions]
        binaries.append(native_dir / "libpdfium.dylib")
        for binary in binaries:
            subprocess.run(["/usr/bin/lipo", str(binary), "-verify_arch", "arm64"], check=True)
            output = subprocess.check_output(["/usr/bin/otool", "-arch", "arm64", "-l", str(binary)], text=True)
            minimums = []
            for block in re.split(r"(?m)^\s*Load command \d+\s*$", output):
                if "cmd LC_BUILD_VERSION" in block:
                    minimum = re.search(r"(?m)^\s*minos\s+(\d+)\.(\d+)", block)
                elif "cmd LC_VERSION_MIN_MACOSX" in block:
                    minimum = re.search(r"(?m)^\s*version\s+(\d+)\.(\d+)", block)
                else:
                    continue
                if minimum:
                    minimums.append(tuple(map(int, minimum.groups())))
            if not minimums or max(minimums) > wheel_minimum:
                raise RuntimeError(f"macOS deployment version of {binary.name} is incompatible with {tag}")

    def bundle_pdfium(self):
        manifest = json.loads((root / "native" / "pdfium-manifest.json").read_text())
        key = f"{sys.platform}-{platform.machine().lower()}"
        if key not in manifest:
            raise RuntimeError(f"No verified PDFium binary for build platform: {key}")
        if self.plat_name and self.plat_name != sysconfig.get_platform():
            raise RuntimeError("Cross-compiling the bundled PDFium binary is not supported")
        source_library = os.environ.get("OPENCV_MCP_PDFIUM_SOURCE_LIBRARY")
        if source_library:
            source_manifest = json.loads((root / "native" / "pdfium-source-manifest.json").read_text())
            if key not in source_manifest:
                raise RuntimeError(f"No validated source-built PDFium binary for: {key}")
            spec = source_manifest[key]
            library = Path(source_library)
            notices = root / "native" / "pdfium-source-notices"
            files = {p.name: p for p in notices.iterdir() if p.is_file()}
        else:
            spec = manifest[key]
            dist = metadata.distribution("pypdfium2")
            if dist.version != "5.14.0":
                raise RuntimeError("The build requires pypdfium2 5.14.0")
            library = Path(dist.locate_file("pypdfium2_raw/" + spec["binary"]))
            files = {}
            for file in dist.files or []:
                parts = Path(file).parts
                if len(parts) >= 3 and parts[0].endswith(".dist-info") and parts[1] == "licenses":
                    relative = Path(*parts[2:])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise RuntimeError("Unsafe PDFium notice path")
                    files[relative.as_posix()] = Path(dist.locate_file(file))
        if not library.is_file() or hashlib.sha256(library.read_bytes()).hexdigest() != spec["binary_sha256"]:
            raise RuntimeError("PDFium binary failed its pinned SHA-256 check")
        output = ((root / "src") if self.inplace else Path(self.build_lib)) / "opencv_bv_tools" / "_native"
        output.mkdir(parents=True, exist_ok=True)
        for old in ("libpdfium.so", "libpdfium.dylib", "pdfium.dll"):
            if old != spec["binary"]:
                (output / old).unlink(missing_ok=True)
        shutil.rmtree(output / "pdfium_notices", ignore_errors=True)
        destination = output / spec["binary"]
        shutil.copyfile(library, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != spec["binary_sha256"]:
            raise RuntimeError("Copied PDFium binary failed its pinned SHA-256 check")
        if set(files) != set(spec["licenses"]):
            raise RuntimeError("PDFium build notices do not match the pinned manifest")
        for name, digest in spec["licenses"].items():
            data = files[name].read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError(f"Modified PDFium build notice: {name}")
            target = output / "pdfium_notices" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def build_extension(self, ext):
        if ext.name != "opencv_bv_tools._native.zxingcpp":
            return super().build_extension(ext)
        import nanobind
        output = Path(self.get_ext_fullpath(ext.name)).resolve()
        build_dir = Path(self.build_temp).resolve() / "zxing"
        build_dir.mkdir(parents=True, exist_ok=True)
        config = "Debug" if self.debug else "Release"
        cmake_args = [
            "cmake", "-S", str(root / "native"), "-B", str(build_dir),
            f"-DZXING_SOURCE_DIR={zxing_staged}",
            f"-DCMAKE_BUILD_TYPE={config}", "-DCMAKE_CXX_STANDARD=20",
            "-DBUILD_SHARED_LIBS=OFF", "-DZXING_PYTHON_MODULE=ON",
            "-DZXING_READERS=ON", "-DZXING_WRITERS=OFF", "-DZXING_C_API=OFF",
            "-DZXING_EXAMPLES=OFF", "-DZXING_DEPENDENCIES=LOCAL",
            f"-Dnanobind_DIR={nanobind.cmake_dir()}", f"-DPython_EXECUTABLE={sys.executable}",
        ]
        if sys.platform == "darwin":
            wheel = self.distribution.get_command_obj("bdist_wheel")
            wheel.ensure_finalized()
            match = re.fullmatch(r"macosx_(\d+)_(\d+)_arm64", wheel.get_tag()[2])
            if not match:
                raise RuntimeError("Unsupported macOS wheel platform")
            cmake_args.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={match.group(1)}.{match.group(2)}")
        subprocess.run(cmake_args, check=True)
        subprocess.run(["cmake", "--build", str(build_dir), "--config", config,
                        "--target", "zxingcpp", "--parallel", "4"], check=True)
        candidates = [p for p in (build_dir / "bindings").rglob("zxingcpp*")
                      if p.suffix in {".so", ".pyd"}]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one ZXing Python extension, found {candidates}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidates[0], output)


class NativeWheel(bdist_wheel):
    def get_tag(self):
        interpreter, abi, wheel_platform = super().get_tag()
        if sys.platform == "darwin" and platform.machine().lower() == "arm64":
            match = re.fullmatch(r"macosx_(\d+)_(\d+)_(?:arm64|universal2)", wheel_platform)
            if match:
                # The pinned PDFium provider wheel requires macOS 13 or later.
                major, minor = max(tuple(map(int, match.groups())), (13, 0))
                wheel_platform = f"macosx_{major}_{minor}_arm64"
        return interpreter, abi, wheel_platform

    def run(self):
        _, _, wheel_platform = self.get_tag()
        machine = platform.machine().lower()
        expected = {
            ("linux", "x86_64"): lambda value: value == "linux_x86_64" or (value.startswith("manylinux") and value.endswith("_x86_64")),
            ("darwin", "arm64"): lambda value: value.startswith("macosx_") and value.endswith("_arm64"),
            ("win32", "amd64"): lambda value: value == "win_amd64",
        }.get((sys.platform, machine))
        if not expected or not expected(wheel_platform):
            raise RuntimeError(f"PDFium wheel target {wheel_platform!r} does not match the native build host")
        super().run()


setup(cmdclass={"build_py": PackageBuild, "build_ext": NativeBuild, "bdist_wheel": NativeWheel}, ext_modules=[
    Extension("opencv_bv_tools._clipper_native",
              sources=["native/clipper_module.cpp", ".clipper-vendor/clipper.cpp"],
              include_dirs=[".clipper-vendor"], language="c++"),
    Extension("opencv_bv_tools._native.zxingcpp", sources=[]),
])
