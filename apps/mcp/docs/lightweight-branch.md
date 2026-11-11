# Lightweight branch notes

This application uses OpenCV/NumPy for image I/O and a small bounded writer for
image-only PDF creation. Default runtime no longer needs Pillow, ReportLab,
requests or charset-normalizer. It still needs pyoxipng for current PNG
optimization and PDFium for current PDF rendering/redaction. Replacing a PDF
renderer is a separate compatibility task, not a dependency-free OpenCV call.

Follow the [README](../README.md) to build the required RapidOCR companion wheel,
install the package and test all nine tools. The companion preserves upstream
model bytes and array OCR code; it makes Pillow and requests helpers optional.

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
uv pip install --python .venv/bin/python -c install-constraints.txt 'wheels/opencv_bv_tools-0.10.0.dev1-py3-none-any.whl[cmyk]'
```

If the environment has pip, `python -m pip install` can be used instead of
`uv pip install --python ...`. On Windows use the installed Python path printed
by the installer. This explicitly installs Pillow for CMYK compatibility only.

## Evidence and remaining work

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

Native Windows/macOS tests must be rerun for this version. Complete the
application licence decision and verify exact release artifacts before public
distribution. GitHub Actions is disabled on the personal fork; no CI workflow
is added by this subproject.
