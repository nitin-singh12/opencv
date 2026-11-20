# Licensing

The application source, project documentation and project-authored tests in this
subproject are licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE)
for the full terms. This follows the licence used for contributions to OpenCV.

Third-party code, dependencies, model weights and externally sourced fixtures
retain their own licences and attribution. They are not relicensed by the
application's Apache-2.0 declaration. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The application wheel includes third-party code and models, so its distribution
metadata declares `Apache-2.0 AND MIT AND BSL-1.0 AND BSD-3-Clause AND LicenseRef-PDFium-Bundled-Notices`. This describes the combined archive;
it does not offer a choice of licence for each component. The full YuNet MIT
notice remains beside the model and in the distribution's licence files.

The bundled PP-OCRv6 model weights and recognition dictionary retain their
Apache-2.0 terms and attribution, with extraction provenance recorded at
`opencv_bv_tools/models/ocr-provenance.json`. The unchanged upstream
[model notice](docs/licenses/RAPIDOCR-MODEL-NOTICES.txt) is included in the
application wheel and installer ZIP. The adapted OCR detector code retains the
upstream attribution and a modification notice. OpenCV and other downloaded
runtime dependencies retain their own notices.

## What this distribution contains

The application wheel includes the Clipper and ZXing native extensions, the PDFium native library, nanobind
binding code, the YuNet and PP-OCRv6 models, and their required notices. The ZIP
adds installer scripts, documentation and smoke-test fixtures. Depending on how
it is built, it may also include unmodified dependency wheels such as OpenCV,
NumPy and MCP. Each included wheel retains its own notices.

The exact contents of any offline bundle require a separate redistribution
review, including any applicable LGPL source/relinking or MPL source-availability
requirements. No font files are included in the application wheel. This document
does not assert a completed licence review for every platform-specific wheel.

`PROVENANCE.json` records the historical subproject import. The ZIP's
`bundle.json` records its current version, source state and file checksums. For a
ZIP built directly, `claimed_source_commit` is supplied by the builder's caller;
the ZIP builder does not verify that its wheels came from that commit. The
separate package-validation report records the checked checkout and build. A
local unpublished ZIP instead records its base commit and makes no commit claim.
Local unpublished candidates are not represented as tagged releases.

This is an experimental application in a personal OpenCV fork. These licence
files do not imply OpenCV endorsement, upstream acceptance or a stable release.
