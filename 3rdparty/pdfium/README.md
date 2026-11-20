# Experimental PDFium source-build target

This optional target builds PDFium 156.0.8076.0 for local Linux x86-64 testing.
PDFium uses GN and Ninja; CMake invokes those tools without changing OpenCV's
compiler settings. `BUILD_MCP_PDFIUM` is OFF by default. Normal OpenCV builds
do not download PDFium, require its tools, or build this target.

The target expects a separately prepared PDFium checkout at commit
`8ca5b735df4263f43c830479b78213854a392c22`, with the pinned Chromium `build`
checkout at `2cd94c0abeada712aeea6906d99021913c9fc28d`. It checks the
upstream release patches, pinned Git dependency revisions, untracked files, the
tested GN binary, and the contents of the Clang and sysroot trees. It also
checks that each third-party target in PDFium's GN dependency graph has a
mapped notice, checks the LLVM compiler runtime link input, and records the
system shared libraries. Remaining tools and the full release licence review
still require verification. Do not use its output as a release artifact yet.

In particular, the upstream release's licence-collection script reports
`harfbuzz` and `dragonbox` as unrecognized for this checkout. Their notices,
and the C++ runtime notices identified from the build graph, are included in
the experimental source-wheel notices. Review the complete set before release.

The patches in `patches/` are from the public `chromium/8076` release of
[pdfium-binaries](https://github.com/bblanchon/pdfium-binaries/tree/chromium/8076)
at commit `f2e9a1c45bb17b85b540abf1af30146ef65416ac`; their licence is included
beside them. They produce a shared library and exported public APIs. The target
uses the release's Linux settings with V8 and XFA disabled and adds an explicit
`use_remoteexec=false`. It does not download source, run setup hooks, install
system packages, or apply patches automatically.

For a fresh Linux checkout, `bash 3rdparty/pdfium/prepare-linux.sh
/absolute/path/to/new-work-directory` downloads the pinned tools and source,
installs the sysroot, and applies the patches. It refuses to reuse an existing
directory. Alternatively prepare the checkout separately using the upstream
GN/depot_tools instructions and `checkout_configuration=minimal`, then apply:

```bash
git -C /absolute/path/to/pdfium checkout 8ca5b735df4263f43c830479b78213854a392c22
git -C /absolute/path/to/pdfium apply /absolute/path/to/opencv/3rdparty/pdfium/patches/shared_library.patch
git -C /absolute/path/to/pdfium apply /absolute/path/to/opencv/3rdparty/pdfium/patches/public_headers.patch
git -C /absolute/path/to/pdfium/build apply /absolute/path/to/opencv/3rdparty/pdfium/patches/clang_rt.patch
```

Syncing the exact DEPS revisions and obtaining the matching Clang, GN, Ninja and
Linux sysroot are required before applying the patches. Source preparation can
use substantial disk space. The build needs CMake, Git, GN, Ninja, Python and
PDFium's pinned Clang/sysroot. Linux system build prerequisites may also be
required, including `readelf` from binutils for the link check. Follow the
[PDFium source instructions](https://pdfium.googlesource.com/pdfium/+/refs/heads/chromium/8076/README.md).

From the OpenCV checkout, configure and invoke the explicit target:

```bash
cmake -S . -B /absolute/path/to/opencv-build \
  -DBUILD_MCP_PDFIUM=ON \
  -DMCP_PDFIUM_SOURCE_DIR=/absolute/path/to/pdfium \
  -DMCP_PDFIUM_GN_BUILD_DIR=/absolute/path/to/pdfium-build
cmake --build /absolute/path/to/opencv-build --target opencv_mcp_pdfium
```

`MCP_PDFIUM_BUILD_JOBS` controls parallel compilation (default 4). The output
is `libpdfium.so` in the configured GN build directory. It is not installed or
linked into OpenCV. To select the Linux source-built binary when building an
MCP wheel, set `OPENCV_MCP_PDFIUM_SOURCE_LIBRARY` to its absolute path. The
wheel build accepts only the specific tested SHA-256 in the source manifest;
different builds fail closed. For example, with the wheel build requirements
already installed:

```bash
cd apps/mcp
OPENCV_MCP_PDFIUM_SOURCE_LIBRARY=/absolute/path/to/pdfium-build/libpdfium.so \
  python -m build --wheel --no-isolation
python dev/verify_source_pdfium_wheel.py dist/*.whl
```

The default prebuilt route declares `pypdfium2` as a build requirement. The
source route does not require it. The source route and its notices remain
experimental. macOS and Windows source builds are not supported by this target
yet.
