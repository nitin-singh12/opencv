# Lightweight branch notes

This application uses OpenCV/NumPy for image I/O and a small bounded writer for
image-only PDF creation. Default runtime no longer needs Pillow, ReportLab,
requests or charset-normalizer. PNG optimization uses Python's zlib; PDFium is
still needed for current PDF rendering/redaction. Replacing a PDF
renderer is a separate compatibility task, not a dependency-free OpenCV call.

The zlib PNG backend was compared on 34 available original development inputs
on Linux. All 34 completed and independently matched decoded samples and
non-IDAT metadata; output never grew relative to the input. Compared with the
prior optimizer, output files were smaller in 20 cases, larger in 10 and equal
in four. The earlier corpus contained 40 cases; this is not a held-out test or
a claim that compression is always better. Native Windows/macOS testing of
this change remains pending.

The experimental no-RapidOCR candidate includes the two verified PP-OCRv6
models and recognition dictionary directly in the application wheel. Its OCR
processing uses OpenCV, NumPy and the bundled Clipper C++ extension; it does not install or import
RapidOCR. The detector code retains attribution to RapidOCR/PaddleOCR and the
models retain their original notices. Follow the [README](../README.md) for
installation. Current comparison evidence is maintained separately; the
measurements below describe historical candidates and do not by themselves
prove the new candidate's accuracy.

The Python wheel is platform-specific because it contains a Clipper extension.
The native code is compiled from the pinned source in `3rdparty/clipper`; the
optional `BUILD_MCP_CLIPPER` CMake target is off by default and is separate
from the wheel build. A wheel or ZIP built on Linux is not a Windows or macOS
package. Each target requires its own build and test.

On Linux, a RapidOCR-free prototype matched the preceding implementation on
all 280 original images and 49 original PDFs (91 pages), including detected
results and rendered redaction pixels. The integrated branch wheel separately
matched nine images and three PDFs against the same baseline. The integrated
branch's full Linux test suite passed 1,461 tests and 51 subtests, with 26
Windows-only skips. These comparisons show parity on the tested cases, not
perfect detection on every input. Native Windows and macOS validation remains
pending for this change.

For the Shapely-free candidate, a fresh Linux installation passed all nine
tools with Shapely and ONNX Runtime absent. The full application suite passed
1,460 tests and 51 subtests, with 26 Windows-only skips. All 280 original image
cases matched the previous version's boxes, readings and decoded output pixels.
All 49 original PDFs (91 pages) matched its input hashes, boxes, address values,
scoring, structure and rendered pixels. A separate nine-image/four-PDF sample
also matched after switching the model loader to a verified byte buffer. Native
Windows/macOS checks are separate; these Linux results do not establish
acceptance on those platforms. An unpinned PDF comparator crashed on this host;
the comparison passed on previously tested CPU cores, without repeating the
completed corpus processing.

## OpenCV DNN OCR backend

The previous application supplied OpenCV DNN sessions for CPU inference with the bundled
PP-OCRv6 small detector and recognizer. It does not install or import the Python
`onnxruntime` package or require a separate Python `onnx` parser. OpenCV loads the
ONNX model files directly. The recognition dictionary in
`models/ppocrv6-characters.txt` comes from the exact bundled recognizer's metadata;
its source model and hash are recorded in
[OCR provenance](../src/opencv_bv_tools/models/ocr-provenance.json). Model downloads,
arbitrary replacement models and CUDA OCR are outside this supported configuration.

The existing lazy persistent worker is retained: registration does not load OCR,
the first OCR call initializes it, later calls reuse it, and the configured idle
timeout releases it. Moving inference into OpenCV does not establish a memory or
speed improvement by itself.

A preliminary experiment was followed by a production-path comparison using an
installed `0.10.0.dev2` candidate with companion `3.10.0+bv.2`. Across nine fixed
inputs, 20 executions and 983 inferences, the compared redaction results, output
pixels, OCR text/boxes/columns and preprocessing tensors matched exactly. OCR
confidence values were not identical: their maximum absolute difference was
`0.00022`. This is finite backend-parity evidence, not a labelled accuracy
benchmark or acceptance on other platforms. The fresh-package and full Linux
suite checks subsequently passed; original-corpus coverage is recorded below.

## CMYK JPEG compatibility

Default email-image processing rejects CMYK JPEGs explicitly because converting
these differently can change OCR input pixels. To preserve the earlier behavior,
install the optional `cmyk` extra into the same environment as the server. After
an installation from source on Linux/macOS:

```bash
uv pip install --python .venv/bin/python -c install-constraints.txt '.[cmyk]'
```

For an extracted ZIP use its application wheel instead:

```bash
uv pip install --python .venv/bin/python -c install-constraints.txt 'wheels/opencv_bv_tools-0.10.0.dev4-py3-none-any.whl[cmyk]'
```

If the environment has pip, `python -m pip install` can be used instead of
`uv pip install --python ...`. On Windows use the installed Python path printed
by the installer. This explicitly installs Pillow for CMYK compatibility only.

## Historical evidence and remaining work

The results in this section predate the OpenCV DNN OCR backend. They describe
earlier application revisions and are not acceptance results for `0.10.0.dev2`.

Prior source-branch measurements compared 20 OCR regression runs containing 983
engine calls, with matching compared OCR results. A separate 22-case regression
had no errors and matching email/face regions. These are finite test results,
not a 99% accuracy claim. Face blur rendering intentionally differs inside the
detected masks; small-face misses and partial email coverage remain known limits.

A prior broad non-QR suite recorded 1,145 passes, one PDF cancellation/recovery
failure, 26 Windows skips and 51 passing subtests. The isolated cancellation
tests passed, so the intermittent issue remains unresolved. QR fixture
generation also failed intermittently before decoding; frozen QR samples passed.
Neither issue is resolved merely by moving the package into this fork.

The relocation check on 11 November 2026 passed 202 targeted tests with 26
Windows-only skips. A first extracted-ZIP run crashed during MCP initialization
with `free(): invalid pointer`; repeating the nine-tool smoke check in that same
installed environment passed. The cause of the first crash remains unresolved.
These results support local evaluation, not a stable-release claim.

The application uses Apache-2.0, with third-party terms retained separately.
The newer checks below distinguish reproduced environment failures from package
defects. Earlier failed-run evidence is retained rather than removed.

## Native and source validation

### OpenCV DNN backend: validation before Shapely removal

Measured on 12 November 2026 using a local unpublished `0.10.0.dev2` candidate,
based on `b8b9b7f984b754ba72d28138eb13b001390854e7`. The candidate used Linux
x86-64, Python 3.12, OpenCV wheel `5.0.0.93`, NumPy `2.2.6` and RapidOCR
`3.10.0+bv.2`. The reference used ONNX Runtime `1.23.2` with companion
`3.10.0+bv.1`. The candidate environment contained neither the Python
`onnxruntime` nor `onnx` distribution. Both paths used CPU inference with two
threads, OpenCL disabled, seed `20261112` and the same detector/recognizer bytes.

| Check | Measured result / remaining work |
| --- | --- |
| Focused OCR/PDF tests | 49 passed: 14 OCR and 35 PDF tests, including actual redaction pixels, geometry and default-factory fallthrough |
| Installed production image OCR parity | 20 executions on 9 fixed inputs; 983 inferences; all compared results, pixels, preprocessing tensors and OCR text/boxes/columns exact; maximum confidence difference `0.00022` |
| Focused CLI tests | 40 passed after correcting a stale CLI test |
| Full application suite | 1,439 passed, 26 Windows-only skips, 51 passing subtests; 174.95 seconds on CPU cores 24–25, after the PDF fix |
| Clean ZIP nine-tool MCP smoke | Passed from an extracted path containing spaces and Unicode; default install contains neither ONNX Runtime nor Pillow/requests/charset-normalizer/ReportLab |
| Original MCP integration regression | 22 cases passed across all nine tools; email boxes and pixels, PDF/scanner pixels, QR/barcode results and image alpha matched the recorded baseline |
| Installer portability | 66 Linux cases passed after narrowing the POSIX-only skip; 23 cases are now Windows-eligible, subject to native CI |
| OpenCV source build with this backend | Passed locally on Linux: C++ compile/link and PNG roundtrip, plus all nine MCP tools using the source-built Python binding. OpenCV's own unit suite was not run |
| Windows x64 / macOS arm64 with this backend | Pending; historical results below do not cover this change |

Raw production comparison evidence is retained locally under
`mcp-branch-review/evidence/ocr-opencv-production-20261112/`, including
`pipeline-comparison-expanded.log` and `pipeline-opencv-expanded.json`. The JSON
records input/model hashes, configuration, candidate/reference versions and
runtime. `metadata_json_sha256` and `packaged_dictionary_sha256` identify the
captured model metadata and packaged dictionary separately. The latter matches
[OCR provenance](../src/opencv_bv_tools/models/ocr-provenance.json).

The protocol initialization stall was reproduced as a sandbox restriction on
local socket-pair writes. Both unchanged protocol tests passed outside that
sandbox. A separate, pure Python Segno QR probe reproduced crashes without
importing this application, OpenCV, NumPy or Pillow: CPU 8 failed three runs,
while nine runs on five other selected cores passed. The broader checks above
therefore use explicit CPU affinity. This isolates an execution-environment
problem; it does not diagnose a hardware/firmware cause or establish unrestricted
host stability. PDF cancellation/recovery passed in the complete suite without
changing production cleanup. The earlier startup crash remains preserved as a
historical observation, not a claimed production fix.

The clean ZIP check initially exposed a zero-context insertion error in the
companion-patch builder. The parser was fixed, patched Python files are compiled
before packaging, and the corrected ZIP passed a new installation and all nine
tool calls. A stale CLI test now patches the actual file-output helper. The
macOS timeout test observes workers it launches, avoiding sandbox-restricted
process enumeration while retaining process, pipe and thread leak assertions.

Current full-suite, original-corpus, package-identity and failed-run evidence is
retained under `mcp-branch-review/evidence/full-validation-20261112/` outside the
repository. Runs use the installed candidate wheels, preserve input files and
record hashes, splits, commands, CPU affinity and raw failures. No paid service
was used. All 280 original email images produced the same boxes as the recorded
historical baseline. One real image reported different OCR text against an older
baseline; a matched comparison against the immediately preceding ONNX Runtime
pipeline returned identical text, box and output pixels for both backends. Both
missed the exact annotated spelling, so this is not a claim of perfect OCR.
On the low-contrast set, 103 of 111 annotated addresses were fully hidden,
8 were partially hidden and one false box was recorded; these results match
the historical boxes. On the synthetic set, all 190 annotated addresses were
fully covered, while three held-out images failed the original strict criterion
because non-email pixels were also covered.

The 49 annotated PDF cases ran across 91 pages. Before the sideways-text fix,
893 of 894 annotated addresses were fully hidden; the missed vertical address
was also missed by the preceding ONNX Runtime version. With the fix, all 894
were fully hidden, no target coverage decreased and none of the 188 lookalikes
was covered over half. On four pages, changed boxes also covered 516 additional
non-target ink pixels. Total processing time across this corpus rose from
2,195.6 to 4,242.7 seconds in these local runs; individual documents ran in
parallel, so this sum is not wall-clock duration. The extra original-page OCR
check can make skewed scanned PDFs considerably slower. The separate 43-page
real-PDF set has no independent accuracy annotations and supports structural
checks only. These finite tests do not guarantee all private information is
detected; review redacted documents before sharing.

| Original non-OCR corpus | Measured coverage |
| --- | --- |
| Face detection/blur | All 50 original images; 101/104 faces fully covered, no false positives on 5 negatives; the same 3 small-face misses remain |
| Barcodes | All 100 original images; 79/88 instances decoded, no wrong payloads or false positives on 20 negatives |
| QR codes | All 100 original images; 84/92 symbols decoded, no wrong payloads or false positives on 14 negatives |
| Conversion | 12 original cases plus the EXIF case passed; alpha preserved |
| PNG optimization | 34 available cases passed sample/alpha/metadata and repeated-optimization checks; 33 inputs match original file hashes, one level-zero screenshot was reconstructed with matching scene pixels |
| Manual scanner pilot | 38 original cases passed, with exact embedded pixels and all 152 rendered markers preserved |
| Original scanner experiments | 2,163 SmartDoc/Mendeley images plus 206 synthetic images, 4,738 calls: 3,580 pixel-exact PDFs, 1,157 expected input-pixel-limit rejections and 1 expected output-size-limit rejection |
| Kaggle reader samples | 6/12 original images recovered with exact hashes and exercised through both readers; no independent payload ground truth, so this is execution coverage |

Scanner experiments check the current manual rotation/PDF feature, not automatic
corner detection or perspective correction. One original harness assertion
incorrectly assumed that an image within the input-pixel limit must produce a
PDF within the separate 32 MiB output limit. Eight targeted calls reproduced the
correct rejection; independent encoding confirmed the compressed image alone
exceeded that cap. The raw failed assertion is retained alongside the corrected
acceptance report. Production limits and scanner code were not changed.

Six original Kaggle images, and their derived PNG cases, could not be recovered
from the searched local locations or archives. The recorded public download now
returns HTTP 404. They are untested coverage gaps, not passes or new replacement
samples. Detection totals above are finite corpus measurements and do not imply
99% accuracy or complete redaction on arbitrary user images.

### Historical validation before the backend change

Measured on 12 November 2026 with Python 3.12. The recorded run is
[43205908479](https://github.com/nitin-singh12/opencv/actions/runs/43205908479)
at commit `b8b9b7f984b754ba72d28138eb13b001390854e7`.

| Environment | Nine-tool MCP smoke | Targeted tests / build | Commit |
| --- | --- | --- | --- |
| Ubuntu 24.04, x86-64 | Passed | 251 passed, 26 skipped | `b8b9b7f` |
| Windows Server 2022, x64 | Passed | 213 passed, 64 skipped | `b8b9b7f` |
| macOS 15, arm64 | Passed in preceding run; retry was queued at recording | 192 passed, 81 skipped in preceding run | `944ce3f` |
| Ubuntu 24.04, OpenCV source build | Passed | C++ libraries, Python binding and C++ PNG roundtrip passed | `b8b9b7f` |

The preceding run is
[43204093556](https://github.com/nitin-singh12/opencv/actions/runs/43204093556).
The source job at `b8b9b7f` built OpenCV `5.2.0-dev` in about 21 minutes and
verified ten process import records, including the actual face and OCR workers.
Every recorded OpenCV import and loaded library belonged to that source install.
The macOS attempt was cancelled before any test started because GitHub
could not assign a runner; only that job was queued for retry at recording.
That code commit added portable cancellation/file tests, enabled the POSIX
installer tests on macOS, and preserved LF checkout line endings on Windows;
it did not change
image algorithms or model files.

All nine tools were called through MCP during a fresh ZIP installation, before
adding test-only packages. The additional tests covered installation, native file
and process behavior, OCR worker lifetime, cancellation/recovery and face tools.
This is a targeted suite, not every application test or an accuracy benchmark.
Windows skips include POSIX executable fakes, signals, FIFO paths and the current
existing-output restriction. Use a new output filename on Windows, with local
NTFS paths shorter than 240 characters and no links or junctions.

The historical Linux and Windows package payloads had 172 identical entries.
The only remaining content difference was LF versus CRLF in generated wheel
`METADATA`; model and application source bytes matched. Raw ZIP hashes were
recorded separately and are
not claimed to be identical across operating systems.

The fork has a dedicated `MCP native and source validation` workflow. It uses
only standard runners in the public repository, with no artifact or cache
uploads. Raw results are printed to job logs and summarized per job. Linux,
Windows x64 and macOS arm64 each build and install a fresh ZIP; archive checksums
and extracted-payload hashes are recorded separately. Other platforms and Python
versions are outside this check.

## Reproduce native and source validation

From `apps/mcp`, using native Python 3.12 and a new directory outside the checkout:

```bash
python dev/validate_package.py --work /absolute/path/to/new-validation
```

On Windows use `py -3.12` and a Windows absolute path. The driver installs the
ZIP in a fresh environment, runs all nine tools before adding any test-only
dependencies, verifies lazy registration, then runs installer, native file/process,
OCR lifetime, cancellation/recovery and face tests. Platform-specific skips are
reported. Pillow and ReportLab are test-only oracles in this driver.

For the additional Linux source check, CMake, Ninja and a C++ compiler are needed:

```bash
python dev/validate_package.py --prepare-only --work /absolute/path/to/source-validation
python dev/validate_opencv_source.py --work /absolute/path/to/source-validation
```

This removes the OpenCV wheel from that disposable environment, compiles every
module enabled by the recorded CPU/headless configuration (no reduced `BUILD_LIST`),
and installs the resulting C++ libraries and Python binding there. It checks a
C++ compile/link/PNG roundtrip and all nine MCP tools. Import instrumentation
records the actual OpenCV paths and build identity inside face and OCR workers.
The upstream OpenCV unit suite and optional GPU/GUI/video backends are not covered.

`OPENCV_BV_SOURCE_BUILD` is an explicit experimental opt-in used by that driver.
It names the absolute installation prefix and is accepted for face detection
only when the pinned wheel distribution is absent and the imported binding is
OpenCV 5.x inside that prefix. Normal installations retain the exact wheel pin.
CMake does not create PyPI distribution metadata, so `pip check` will report the
missing wheel after this deliberate substitution; the driver records and checks
that expected difference without inventing package metadata.
