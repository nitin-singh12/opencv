# Licensing

The application source, project documentation and project-authored tests in this
subproject are licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE)
for the full terms. This follows the licence used for contributions to OpenCV.

Third-party code, dependencies, model weights and externally sourced fixtures
retain their own licences and attribution. They are not relicensed by the
application's Apache-2.0 declaration. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The application wheel includes the MIT-licensed YuNet model, so its distribution
metadata declares `Apache-2.0 AND MIT`. This describes the combined archive;
it does not offer a choice of licence for each component. The full YuNet MIT
notice remains beside the model and in the distribution's licence files.

The companion RapidOCR wheel remains a separately identified modified upstream
distribution, `3.10.0+bv.3`. Its Apache-2.0 licence, model attribution and
modification notice are preserved. The application also includes a recognition
dictionary extracted from the bundled PP-OCRv6 model metadata; it retains the
model's Apache-2.0 terms and attribution, with its
extraction provenance recorded at `opencv_bv_tools/models/ocr-provenance.json`
inside the application wheel. The unchanged upstream
[model notice](docs/licenses/RAPIDOCR-MODEL-NOTICES.txt) is included in both the
application wheel and installer ZIP. Its `PP-OCRv6_rec_small.onnx` entry applies
to this dictionary; the dictionary's character data is not altered by notices.
Changing inference to OpenCV DNN does not relicense those resources or remove
OpenCV's applicable native component notices. Other dependencies retain their
wheel-specific notices; historical notices remain relevant to earlier builds.

## What this distribution contains

The application wheel contains application code, the MIT-licensed YuNet model,
and the Apache-2.0 OCR dictionary. The installer ZIP adds a modified RapidOCR
wheel, installer scripts, documentation and smoke-test fixtures. RapidOCR's
three bundled models, including its unused angle classifier, retain their full
upstream model notice. Every modified RapidOCR Python file carries a change
notice, in addition to the central `BV-MODIFICATIONS.txt` summary.

The ZIP does not include OpenCV, NumPy, MCP, PDFium, ZXing, Oxipng, GEOS or their
installed environments. The installer downloads their packages separately;
it does not strip their licence files or rebuild their native libraries.
No font files are included in the application or RapidOCR wheels. These are
statements about the archive contents, not a completed review of every native
dependency downloaded on every operating system.

An offline bundle or executable that includes those downloaded dependencies is
a different distribution and requires review of those exact artifacts, including
any LGPL source/relinking or MPL source-availability requirements. Such a bundle
is not produced by the current ZIP builder.

`PROVENANCE.json` records the historical subproject import. The ZIP's
`bundle.json` records its current version, source state and file checksums. For a
ZIP built directly, `claimed_source_commit` is supplied by the builder's caller;
the ZIP builder does not verify that its wheels came from that commit. The
separate package-validation report records the checked checkout and build. A
local unpublished ZIP instead records its base commit and makes no commit claim.
Local unpublished candidates are not represented as tagged releases.

This is an experimental application in a personal OpenCV fork. These licence
files do not imply OpenCV endorsement, upstream acceptance or a stable release.
