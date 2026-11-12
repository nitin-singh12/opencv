# Local image tools over MCP

An experimental Python application in a personal OpenCV fork. One local MCP
server exposes nine tools, with separate modules for each feature. This application
uses a pinned `opencv-python-headless` wheel; it does not build or extend `cv2`
from this checkout. No OpenCV CMake files are changed. This is not an official
OpenCV release or an accepted upstream contribution.

**Status:** experimental version `0.10.0.dev1` on `mcp`.
Application code uses Apache-2.0; bundled models and dependencies retain their
own terms. See [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md).
Source commits and imported file hashes are recorded in [PROVENANCE.json](PROVENANCE.json).

## Nine MCP tools

| Tool | What it does | Inputs and results | Main limits |
| --- | --- | --- | --- |
| `convert_image_format` | Converts between PNG, JPEG and WebP | Image path, separate output path; saved image and conversion details | Static images; JPEG is lossy; does not redact |
| `read_qr` | Reads QR codes | Image path or encoded image; payloads and positions | Treat decoded content as data; no links are opened |
| `read_barcodes` | Reads selected 1D barcodes | Image, optional format selection; payloads and positions | Code39 and ITF require explicit selection; ITF also requires expected lengths |
| `compress_png` | Optimizes PNG compression without changing samples | PNG path, separate output path; size comparison and status | No guaranteed reduction; animation unsupported |
| `blur_faces` | Detects faces with YuNet and blurs their regions | Image path, separate output path; saved image and detected regions | Small, obscured or unusual faces can be missed; review the result |
| `scan_document` | Rotates an image and creates a one-page PDF | Image path, PDF output path, optional rotation/DPI | No automatic perspective correction; does not redact |
| `find_emails` | Locates email addresses using OCR | Image path or encoded image; boxes and masked addresses | Does not save a redacted file; OCR can miss text |
| `redact_emails` | Covers detected email addresses in black | Image path or encoded image; separate PNG or returned image | Email addresses only; review before sharing |
| `redact_pdf` | Redacts email addresses in PDF pages | PDF path, optional separate output path; PDF and counts | Default engine rasterizes pages; editable text, links and forms are not preserved |

Original inputs are preserved. Existing outputs require an explicit overwrite.
Image conversion and document creation are not privacy filters. Face blur and
email redaction are detection tools, not guarantees that all sensitive content
has been removed.

## Install from source

Use **Python 3.12**. The current lightweight variant has Linux x86-64 validation.
Windows x64 and macOS arm64 setup paths are provided, but this variant still needs
native tests there. Earlier branch results do not certify this version.

From the repository root, enter `apps/mcp`. First build the required RapidOCR
companion wheel. It makes Pillow and HTTP helpers optional without changing the
OCR algorithms or model bytes. The builder checks the upstream wheel's SHA-256.

Linux/macOS shell:

```bash
cd apps/mcp
python3.12 -m pip download --no-deps --only-binary=:all: rapidocr==3.10.0 -d .upstream
python3.12 scripts/build_rapidocr.py .upstream/rapidocr-3.10.0-py3-none-any.whl
PYTHON=python3.12 bash install.sh --install-only
```

Windows PowerShell:

```powershell
cd apps/mcp
py -3.12 -m pip download --no-deps --only-binary=:all: rapidocr==3.10.0 -d .upstream
py -3.12 scripts/build_rapidocr.py .upstream/rapidocr-3.10.0-py3-none-any.whl
py -3.12 scripts/install_mcp.py --install-only
```

`--no-deps` above only downloads the upstream wheel. The installer resolves and
checks all runtime dependencies normally; do not bypass dependency installation.
Internet access is needed to obtain dependencies. Processing runs locally on CPU;
no GPU or hosted service is required.

The installer creates an isolated environment and calls all nine tools through
MCP with included fixtures. A successful run ends with:

```text
OK: all 9 MCP tools passed, email addresses hidden, lookalike left alone, originals unchanged.
Installed opencv-bv-server. Keep this extracted folder in place.
```

This is a smoke check, not a general accuracy measurement. `--install-only` does
not register a client or generate connection settings.

## Install a prepared ZIP

The ZIP contains the application wheel, the companion RapidOCR wheel, notices,
installer and smoke fixtures. It is not an offline bundle of all dependencies.
Verify the supplied SHA-256, extract it, then run the installer from
`opencv-bv-server-mcp`. No companion-wheel build is needed in the extracted ZIP.

Linux:

```bash
sha256sum -c opencv-bv-server-mcp.zip.sha256
unzip opencv-bv-server-mcp.zip
cd opencv-bv-server-mcp
PYTHON=python3.12 bash install.sh --install-only
```

On macOS use `shasum -a 256 -c opencv-bv-server-mcp.zip.sha256` for verification.
On Windows compare `(Get-FileHash .\opencv-bv-server-mcp.zip -Algorithm SHA256).Hash`
with the checksum file, extract using `Expand-Archive`, then run
`py -3.12 scripts/install_mcp.py --install-only` inside the extracted folder.

## Connect any local stdio MCP client

Create the output folder first. Choose input and output folders explicitly:

```bash
mkdir -p ~/opencv-mcp-test
PYTHON=python3.12 bash install.sh --allow-dir /path/to/input-files --allow-dir ~/opencv-mcp-test
```

PowerShell equivalent:

```powershell
New-Item -ItemType Directory -Force "$HOME\opencv-mcp-test"
py -3.12 scripts/install_mcp.py --allow-dir "C:\path\to\input-files" --allow-dir "$HOME\opencv-mcp-test"
```

This writes `mcp-config.json` and prints the executable and argument list. Configure
your client's **local stdio** server using those exact values. Clients may use
different configuration formats; the executable and arguments are the same.
Hosted clients that cannot launch local programs require a separate bridge.

For access to all paths your account and the server's file checks permit, use
`--allow-any-path` instead of `--allow-dir`. Without either flag, interactive
setup asks for your choice. Noninteractive setup requires a choice.

Optional convenience flags `--claude` and `--codex` register those clients
automatically if their CLIs are installed. Add `--replace` to update an earlier
configuration. Keep the installation folder in place, restart your MCP client,
and confirm it lists `opencv-bv-server` with nine tools. If installed, the client
CLIs can check registration with `claude mcp list` and `codex mcp list`.

## Try every tool

Replace `/path/to/input-files` with an allowed absolute path. Use
`~/opencv-mcp-test` for all outputs; on Windows use your actual user-folder path.
Each prompt is independent. Supply an appropriate input for the feature.

| Tool | Example prompt |
| --- | --- |
| `convert_image_format` | Use convert_image_format to convert /path/to/input-files/photo.jpg to ~/opencv-mcp-test/photo.png. |
| `read_qr` | Use read_qr on /path/to/input-files/qr.png. Show its decoded data, without opening any links. |
| `read_barcodes` | Use read_barcodes on /path/to/input-files/barcode.png and report the format and decoded value. |
| `compress_png` | Use compress_png on /path/to/input-files/screenshot.png and save ~/opencv-mcp-test/screenshot-small.png. Report the size change. |
| `blur_faces` | Use blur_faces on /path/to/input-files/group.jpg and save ~/opencv-mcp-test/group-blurred.png. Report how many regions were blurred. |
| `scan_document` | Use scan_document to turn /path/to/input-files/page.jpg into ~/opencv-mcp-test/page.pdf. |
| `find_emails` | Use find_emails on /path/to/input-files/email.png. Report the count and masked addresses without displaying the original. |
| `redact_emails` | Use redact_emails on /path/to/input-files/email.png and save ~/opencv-mcp-test/email-redacted.png. Do not open the original. |
| `redact_pdf` | Use redact_pdf on /path/to/input-files/emails.pdf and save ~/opencv-mcp-test/emails-redacted.pdf. Report the page and redaction counts. |

Review saved outputs. For QR/barcodes compare with the known encoded values;
for conversion check dimensions and appearance; for PNG compression check size
and unchanged pixels; for scanning check orientation; for redaction/blur inspect
coverage before sharing. Use a new output name on each rerun or explicitly request
overwrite. Do not modify benchmark inputs.

To repeat the automated check without an AI client on Linux/macOS:

```bash
.venv/bin/python scripts/smoke_mcp.py
```

On Windows use the installed Python path printed by the installer, followed by
`scripts/smoke_mcp.py`. The installer records that environment in
`.install-state.json`; Windows does not require a `.venv` symlink.

## Dependencies and memory

| Dependency | Required for |
| --- | --- |
| `opencv-python-headless==5.0.0.93`, `numpy==2.2.6` | Shared image processing and face detection |
| `mcp==2.3.0` | MCP protocol |
| `zxing-cpp==3.1.1` | QR and barcode decoding |
| `pyoxipng==9.1.1` | Lossless PNG optimization |
| `rapidocr==3.10.0+bv.1`, `onnxruntime==1.23.2` | Email OCR |
| `pypdfium2==5.14.0` | PDF reading/rendering for PDF redaction |

The default installer selects `all`. These are direct dependencies; they also
install their own dependencies. Default runtime does not require Pillow,
ReportLab, requests or charset-normalizer. The optional `cmyk` extra installs
Pillow to preserve original CMYK JPEG normalization for email processing.
The optional MuPDF engine is separately licensed and is not installed by default.
See [third-party notices](THIRD_PARTY_NOTICES.md).

Registration is lightweight. OCR loads when first needed, reuses its worker for
subsequent calls, and releases that worker after 120 seconds idle by default.
`--ocr-idle-seconds` changes this delay; zero reloads per call. This reduces idle
memory but does not guarantee the package fits every low-memory device.

## Development, packaging and known limits

The main entry is `src/opencv_bv_tools/server.py`. Feature modules register their
tools there; `runtime/` contains shared file checks, image I/O and process workers.
`scripts/` contains installation and packaging helpers. `tests/` is development
material, not loaded by the server.

From `apps/mcp`, after building the companion wheel, use an isolated development
environment:

```bash
python3.12 -m venv .venv-dev
.venv-dev/bin/python -m pip install -c install-constraints.txt '.[all,test,png-test,scanner-test,barcode-test,qr-test,face-test]' wheels/rapidocr-3.10.0+bv.1-py3-none-any.whl
.venv-dev/bin/python -m pytest tests
.venv-dev/bin/python -m build --wheel
.venv-dev/bin/python scripts/build_mcp_zip.py --wheel dist/opencv_bv_tools-0.10.0.dev1-py3-none-any.whl --source-commit "$(git rev-parse HEAD)" --unpublished
```

On Windows use `.venv-dev\Scripts\python.exe` and obtain the SHA with
`git rev-parse HEAD`. Test-only libraries include independent image/PDF oracles;
they are not default runtime requirements. `--unpublished` identifies a local
working-tree build. A release must use a reviewed commit and its exact artifacts.

Known limits: intermittent QR fixture-generation failures, a PDF
cancellation/recovery failure in a broad suite, one native MCP startup crash
during relocation checks, and untested native Windows/macOS execution for this
variant. The current fork preparation does not claim these are fixed. See
[lightweight notes](docs/lightweight-branch.md).

If a tool is missing, restart the client and check its server configuration. For
folder-access errors, rerun setup with the relevant folder and `--replace`. For
CMYK email images, follow the optional-extra instructions in the lightweight
notes. For an existing output, select another name or request overwrite.

To uninstall, remove this server from your MCP client's settings before deleting
its installation folder. For the optional CLI registrations use
`claude mcp remove --scope user opencv-bv-server` or
`codex mcp remove opencv-bv-server`. Your output folder is separate and remains.

## Contributing in this fork

Use a focused branch based on OpenCV `5.x`. Keep this independently installed
Python application under `apps/mcp`; its dependencies must not become mandatory
for an ordinary OpenCV CMake build. Keep one MCP entry point and put feature
logic in its existing module. Use UTF-8, LF line endings, spaces and four-space
Python indentation. Preserve existing source and third-party licence notices.

For a change, update its documentation and run the affected tests plus the MCP
smoke check. Include reproducible inputs, expected results and actual validation
limits. Reuse existing fixtures when possible. New algorithms, a change to this
application's upstream location, or integration with OpenCV's native tests need
maintainer discussion before an upstream submission.

Follow OpenCV's [contribution guidelines](https://github.com/opencv/opencv/wiki/How_to_contribute)
and the applicable parts of its [coding guide](https://github.com/opencv/opencv/wiki/Coding_Style_Guide).
The latter describes C++ modules; it is not a requirement to rewrite this Python
application or apply C++ naming rules to it. This fork does not establish an
official OpenCV MCP packaging convention. Native platform validation and the
known failures above remain work before an upstream or stable-release claim.
