# Third-party notices

This index describes dependencies and bundled assets for `0.10.0.dev1`. The
application uses Apache-2.0; see [LICENSING.md](LICENSING.md). Preserve
the full notices in each installed distribution, including native components.
Headline licence names alone are not a complete redistribution review.

## Runtime dependencies

| Component | Licence/notice location and source |
| --- | --- |
| OpenCV headless 5.0.0.93 | OpenCV core Apache-2.0; wrapper and native component terms are distinct. Preserve `cv2/LICENSE.txt` and `cv2/LICENSE-3RD-PARTY.txt`. [Upstream licensing](https://github.com/opencv/opencv-python#licensing). |
| NumPy 2.2.6 | BSD-3-Clause plus included native-library notices. Preserve the distribution's `LICENSE.txt`. [Source](https://github.com/numpy/numpy). |
| MCP 2.3.0 | MIT. Preserve the distribution licence and notices of transitive packages. [Source](https://github.com/modelcontextprotocol/python-sdk). |
| zxing-cpp 3.1.1 | Apache-2.0. Preserve the wheel's licence. [Source](https://github.com/zxing-cpp/zxing-cpp). |
| pyoxipng 9.1.1 | MIT. Preserve the wheel's licence. [Source](https://github.com/nfrasser/pyoxipng). |
| RapidOCR 3.10.0+bv.1 | Apache-2.0 code; separate `MODEL_LICENSES.md` contains weight and conversion attribution. This is a modified wheel, not an official RapidOCR release. [Upstream source](https://github.com/RapidAI/RapidOCR). |
| ONNX Runtime 1.23.2 | MIT plus native component notices in `onnxruntime/ThirdPartyNotices.txt`. [Source](https://github.com/microsoft/onnxruntime). |
| pypdfium2 5.14.0 | BSD-3-Clause / Apache-2.0 components, plus PDFium and bundled third-party notices. Preserve the full wheel `LICENSES` and build-specific notice directories. [Licensing](https://pypdfium2.readthedocs.io/en/stable/readme.html#licensing). |

This is the direct-dependency list. Transitive and native components have their
own terms; examples include LGPL material in FFmpeg/GEOS and MPL material in
some Python dependencies. An offline dependency bundle or executable requires
review of its exact contents and applicable source/relinking obligations.
No paid licence is purchased by this package.

The retained [FlatBuffers Apache-2.0 licence](docs/licenses/FLATBUFFERS-LICENSE.txt)
comes from [v25.12.19](https://github.com/google/flatbuffers/blob/v25.12.19/LICENSE).
It supplements the previously inspected wheel that omitted its licence file.

## Bundled model and fixtures

- YuNet `face_detection_yunet_2026may.onnx`: MIT, copyright 2020 Shiqi Yu.
  The full [licence](docs/licenses/YUNET-LICENSE.txt) is also inside the application
  wheel at `opencv_bv_tools/models/YUNET-LICENSE`. The
  [model provenance](src/opencv_bv_tools/models/yunet-provenance.json) records its
  OpenCV Zoo source and hash. Model SHA-256:
  `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`.
- Public face photographs retain their [attribution](tests/fixtures/face/README.md)
  and [provenance](tests/fixtures/face/provenance.json). The installer ZIP includes
  only `grace_hopper.jpg` from those photographs.
- Smoke PNG/PDF/QR files use synthetic content. See
  [fixture notes](tests/fixtures/smoke/README.md). No font program is embedded in
  the smoke PDF. Barcode regression PNGs were generated with bwip-js/BWIPP;
  externally sourced fixture attribution remains in the fixture records.

## Modified RapidOCR companion

The ZIP contains `rapidocr-3.10.0+bv.1-py3-none-any.whl`, built from the upstream
3.10.0 wheel with SHA-256
`2fc34e26cd0f48514804a94f8832283039c7334076bff60768b1fed9ad068050`.
The builder retains the full licence, `MODEL_LICENSES.md`, model bytes and all
other package data. `BV-MODIFICATIONS.txt` identifies changed helpers. The patch
and builder are included so the companion can be reproduced.

Only the image-loading, visualization and download helpers are changed to defer
Pillow/requests imports. Array OCR algorithms and model files are unchanged.
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
