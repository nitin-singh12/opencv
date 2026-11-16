# Third-party notices

This index describes dependencies and bundled assets for `0.10.0.dev2`. The
application uses Apache-2.0; see [LICENSING.md](LICENSING.md). Preserve
the full notices in each installed distribution, including native components.
Headline licence names alone are not a complete redistribution review.

Only the application and modified RapidOCR wheels are included in the installer
ZIP. The other runtime packages below are downloaded separately during
installation. Their appearance in this dependency index does not mean their
binaries are inside our ZIP; see [distribution contents](LICENSING.md#what-this-distribution-contains).

## Runtime dependencies

| Component | Licence/notice location and source |
| --- | --- |
| OpenCV headless 5.0.0.93 | OpenCV core Apache-2.0; Python wrapper/build scripts MIT; additional native component terms apply. Preserve `cv2/LICENSE.txt` and `cv2/LICENSE-3RD-PARTY.txt`. [Upstream licensing](https://github.com/opencv/opencv-python#licensing). |
| NumPy 2.2.6 | BSD-3-Clause plus included native-library notices. Preserve the distribution's `LICENSE.txt`. [Source](https://github.com/numpy/numpy). |
| MCP 2.3.0 | MIT. Preserve the distribution licence and notices of transitive packages. [Source](https://github.com/modelcontextprotocol/python-sdk). |
| zxing-cpp 3.1.1 | Apache-2.0. Preserve the wheel's licence. [Source](https://github.com/zxing-cpp/zxing-cpp). |
| pyoxipng 9.1.1 | MIT. Preserve the wheel's licence. [Source](https://github.com/nfrasser/pyoxipng). |
| RapidOCR 3.10.0+bv.2 | Apache-2.0 code; separate `MODEL_LICENSES.md` contains weight and conversion attribution. This is a modified wheel, not an official RapidOCR release. [Upstream source](https://github.com/RapidAI/RapidOCR). |
| pypdfium2 5.14.0 | BSD-3-Clause / Apache-2.0 components, plus PDFium and bundled third-party notices. Preserve the full wheel `LICENSES` and build-specific notice directories. [Licensing](https://pypdfium2.readthedocs.io/en/stable/readme.html#licensing). |

This is the direct-dependency list. Transitive and native components have their
own terms; examples include LGPL material in FFmpeg/GEOS and MPL material in
some Python dependencies. An offline dependency bundle or executable requires
review of its exact contents and applicable source/relinking obligations.
No paid licence is purchased by this package.

OCR inference now uses OpenCV DNN on CPU. The Python `onnxruntime` distribution
is no longer a default dependency, and no separate Python `onnx` parser is
required. This does not remove the notices of native components within OpenCV.
For example, OpenCV's source tree includes MIT-licensed MLAS code derived from
ONNX Runtime; retain the notices applicable to the OpenCV build being distributed.
See [OpenCV MLAS provenance](https://github.com/opencv/opencv/blob/5.x/3rdparty/mlas/README.md).

Earlier packages used ONNX Runtime 1.23.2, under MIT plus the native component
notices in `onnxruntime/ThirdPartyNotices.txt`. Those notices still apply when
redistributing such earlier environments or any build that includes it.
[Upstream source](https://github.com/microsoft/onnxruntime).

The retained [FlatBuffers Apache-2.0 licence](docs/licenses/FLATBUFFERS-LICENSE.txt)
comes from [v25.12.19](https://github.com/google/flatbuffers/blob/v25.12.19/LICENSE).
It supplements the previously inspected wheel that omitted its licence file and
is retained as historical licensing material; its presence does not declare
FlatBuffers a direct dependency of this version.

## Bundled model and fixtures

- YuNet `face_detection_yunet_2026may.onnx`: MIT, copyright 2020 Shiqi Yu.
  The full [licence](docs/licenses/YUNET-LICENSE.txt) is also inside the application
  wheel at `opencv_bv_tools/models/YUNET-LICENSE`. The
  `opencv_bv_tools/models/yunet-provenance.json` in the application wheel records its
  OpenCV Zoo source and hash. Model SHA-256:
  `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`.
- The recognition dictionary at `opencv_bv_tools/models/ppocrv6-characters.txt`
  is extracted from metadata in the exact bundled PP-OCRv6 small recognizer.
  It retains the Apache-2.0 model attribution to Baidu and/or the applicable
  PaddleOCR rights holders. Its source model and hash are recorded in the
  `opencv_bv_tools/models/ocr-provenance.json` in the application wheel.
  The unchanged [model notice](docs/licenses/RAPIDOCR-MODEL-NOTICES.txt) accompanies
  the dictionary even when the application wheel is distributed independently.
  Its `PP-OCRv6_rec_small.onnx` entry identifies the source model.
  [Upstream model notice](https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/).
- The installer ZIP includes only `grace_hopper.jpg` from the public face
  fixtures: Matplotlib's derivative of the US Navy photograph by James S. Davis,
  identified as public domain. The ZIP includes its source and hash in
  `tests/fixtures/face/README.md` and `tests/fixtures/face/provenance.json`.
  These fixture files are not included in the application wheel.
- Smoke PNG/PDF/QR files use synthetic content. The ZIP includes fixture notes
  at `tests/fixtures/smoke/README.md`. No font program is embedded in
  the smoke PDF. Barcode regression PNGs were generated with bwip-js/BWIPP;
  externally sourced fixture attribution remains in the fixture records.

## Modified RapidOCR companion

The ZIP contains `rapidocr-3.10.0+bv.2-py3-none-any.whl`, built from the upstream
3.10.0 wheel with SHA-256
`2fc34e26cd0f48514804a94f8832283039c7334076bff60768b1fed9ad068050`.
The builder retains the full licence, `MODEL_LICENSES.md`, model bytes and all
other package data. All seven changed Python files carry a modification notice,
and `BV-MODIFICATIONS.txt` identifies the changes. The patch
and builder are included so the companion can be reproduced.

The image-loading, visualization and download helpers defer Pillow/requests
imports. The companion also accepts a per-instance inference-session factory,
which this application uses to supply OpenCV DNN CPU sessions. Without that
factory, upstream inference-session behavior is retained. OCR preprocessing,
postprocessing and model files are unchanged.
The companion's optional `images` and `downloads` extras restore those helpers.
No model is fetched during normal MCP tool processing.

## Optional and development packages

Pillow (`cmyk` extra) uses MIT-CMU and additional codec notices. It is not in the
default runtime. PyMuPDF/MuPDF is optional, under AGPL or a separate Artifex
commercial agreement; see [Artifex licensing](https://artifex.com/licensing/).
It is excluded from `all` and is not a dependency-reduction substitute by default.

Development tests may install Pillow, ReportLab, Segno and other independent
oracles. Test environments are not shipped in the wheel or installer ZIP.
ReportLab font files are not bundled with this application's default runtime.
