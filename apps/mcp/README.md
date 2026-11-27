# Local image tools over MCP

An experimental Python application in a personal OpenCV fork. One local MCP
server exposes nine core tools and an optional semantic image-search feature
with four MCP operations, with separate modules for each feature. This application
uses a pinned `opencv-python-headless` wheel by default. An additional validation
job builds OpenCV from this checkout and tests the application against that build.
The OpenCV build includes optional, disabled-by-default native targets for the
bundled dependencies. This is not an official OpenCV release or an accepted
upstream contribution.

**Status:** experimental version `0.10.0.dev18`. The `mcp-prebuilt-wheels` and
`mcp-source-install` branches support wheel and application-source testing.
The current model-download and optional image-search work is on
[`codex/mcp-upstream-readiness`](https://github.com/nitin-singh12/opencv/tree/codex/mcp-upstream-readiness/apps/mcp).
The older approach branches at `bbba05442` (`0.10.0.dev14`) bundle models and
have only nine tools; use the current branch for the search setup below.
The RapidOCR-free and bundled PDFium implementations are experimental and are
not a public release. An earlier Linux wheel passed 1,477 tests, with 26
Windows-only skips. A comparison against the previous implementation covered
280 real images and 49 PDFs (91 pages) with matching results using the same
OpenCV CPU configuration. Different OpenCV build configurations can change OCR
text, so these results are not a guarantee for every build or image.

On commit `418828a6bc`, clean package installation, nine-tool smoke tests and
native contract checks passed on Linux x86-64, Windows x64 and macOS 15 arm64
([CI results](https://github.com/nitin-singh12/opencv/actions/runs/44686125514)).
The Linux PDFium source build through OpenCV CMake and its installed wheel check
also passed ([CI results](https://github.com/nitin-singh12/opencv/actions/runs/44686125798)).
Windows and macOS PDFium source builds remain unverified. See the
[validation notes](docs/lightweight-branch.md#native-and-source-validation)
for test scope and historical results.
The Linux CI is configured to run selected feature regressions, including the full QR
test suite, and the nine-tool smoke test.
The face regression tests check detected blur regions against independently
annotated face areas on three sample images. This is not a measurement of
accuracy across all images or a guarantee of anonymity.
Application code uses Apache-2.0; models and dependencies retain their
own terms. See [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md).
Source commits and imported file hashes are recorded in [PROVENANCE.json](PROVENANCE.json).

## Optional image search

After installing either the wheel or source package, run `opencv-bv-setup --search`
in the activated environment. Setup lets you choose a client, downloads the
optional 1.54 GB FG-CLIP2 assets with SHA-256 verification, and creates a private
local index. The default nine-tool setup remains unchanged. See
[image search setup, supported formats and test prompts](docs/image-search.md).

## Tools

| Tool | What it does | Inputs and results | Main limits |
| --- | --- | --- | --- |
| `convert_image_format` | Converts between PNG, JPEG and WebP | Image path, separate output path; saved image and conversion details | Static images; JPEG is lossy; does not redact |
| `read_qr` | Reads QR codes | Image path or encoded image; payloads and positions | Treat decoded content as data; no links are opened |
| `read_barcodes` | Reads selected 1D barcodes | Image, optional format selection; payloads and positions | Code39 and ITF require explicit selection; ITF also requires expected lengths |
| `compress_png` | Optimizes PNG compression without changing samples | PNG path, separate output path; size comparison and status | No guaranteed reduction; animation unsupported |
| `blur_faces` | Detects faces with YuNet and blurs their regions | Image path, separate output path; saved image and detected regions | Small, obscured or unusual faces can be missed; review the result |
| `scan_document` | Rotates an image and creates a one-page PDF | Image path, PDF output path, optional rotation/DPI | No automatic crop, deskew or perspective correction; does not redact |
| `find_emails` | Locates email addresses using OCR | Image path or encoded image; boxes and masked addresses | Does not save a redacted file; OCR can miss text |
| `redact_emails` | Covers detected email addresses in black | Image path or encoded image; separate PNG or returned image | Email addresses only; review before sharing |
| `redact_pdf` | Redacts email addresses in PDF pages | PDF path, optional separate output path; PDF and counts | Default engine rasterizes pages; editable text, links and forms are not preserved |
| Image search | Finds indexed local photos matching a text query | `index_images`, `search_images`, `image_search_status`, `cancel_image_index` | Opt-in, requires separate large model assets; [setup and limits](docs/image-search.md) |

Original inputs are preserved. On Linux/macOS, replacing an existing output
requires an explicit overwrite. The Windows candidate currently requires a new
output filename, local NTFS paths shorter than 240 characters, and no links or
junctions. These restrictions remain in place while Windows file-safety support
is qualified.
Image conversion and document creation are not privacy filters. Face blur and
email redaction are detection tools, not guarantees that all sensitive content
has been removed.

## Choosing an OpenCV installation

The base package includes the MCP library, so the server can start and list its
tools. Image operations also need OpenCV. The installer selects the `wheel`
extra automatically; manual wheel users can install `opencv-bv-tools[all,wheel]`. The `all` extra selects the feature
dependencies; it does not select an OpenCV provider.

To use OpenCV built from source instead, install its Python `cv2` module in the
same environment and install `opencv-bv-tools[all]`.
Set `OPENCV_BV_SOURCE_BUILD` to the absolute installation prefix for the face
worker. Source builds must be validated for the target platform; the Linux
source-build smoke check is described in the validation notes.

For example, after installing source-built `cv2` in an active Python environment, run from
`apps/mcp` on Linux or macOS:

```bash
python -m pip install -c install-constraints.txt '.[all]'
export OPENCV_BV_SOURCE_BUILD=/absolute/path/to/opencv/install-prefix
python -m pip check
```

On Windows PowerShell, use the same `python -m pip install` command and set
`$env:OPENCV_BV_SOURCE_BUILD = 'C:\absolute\path\to\opencv\install-prefix'`.
Keep this environment variable when launching the MCP server. Source builds on
Windows and macOS have not yet been validated with this packaging change.

After extracting a prepared ZIP, users who prefer pip can install the local
application wheel and resolve the other dependencies from their configured index:

```bash
python -m pip install -c install-constraints.txt --find-links wheels 'opencv-bv-tools[all,wheel]==0.10.0.dev18'
```

Use Python 3.12 in an isolated environment. This command installs the package;
the provided installer additionally runs the nine-tool smoke check and can
prepare client connection settings.

## Set up an installed wheel

Earlier experimental Python 3.12 test wheels were built for Linux x86-64 (tested
on Ubuntu 22.04), Windows x64 (tested on Windows Server 2022), and Apple Silicon
macOS (wheel requires macOS 13 or later; tested on macOS 15). These are historical
results for earlier commits, not validation of this checkout's `0.10.0.dev18`
model-download changes on Windows or macOS. Draft release artifacts may also be
older than this checkout. Repository maintainers can inspect the draft under
[GitHub releases](https://github.com/nitin-singh12/opencv/releases); draft
downloads are not publicly available. Verify the matching `.sha256` file.

Use a wheel built for your Python version, OS and CPU architecture. Its
checksum file and wheel must come from a trusted distribution source. Put both
in the same folder, then replace the example filename below with the actual
wheel name. On Linux:

```bash
sha256sum -c YOUR_WHEEL.whl.sha256
```

On macOS:

```bash
shasum -a 256 -c YOUR_WHEEL.whl.sha256
```

On Windows PowerShell, calculate the hash and compare it with the value in the
trusted `.sha256` file. Continue only when they match:

```powershell
Get-FileHash "C:\path\to\YOUR_WHEEL.whl" -Algorithm SHA256
Get-Content "C:\path\to\YOUR_WHEEL.whl.sha256"
```

Use a separate Python 3.12 virtual environment for each installation. For
example, Linux and macOS users can create and activate one with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

In Windows PowerShell, create the environment and use its executables directly:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: "C:\path\to\YOUR_WHEEL.whl[wheel]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\opencv-bv-setup.exe
```

For a downloaded wheel, this example uses the Linux filename from its download
directory. Replace the name with the actual wheel for your platform:

```bash
python -m pip install --only-binary=:all: './opencv_bv_tools-0.10.0.dev18-cp312-cp312-linux_x86_64.whl[wheel]'
python -m pip check
opencv-bv-setup
```

`--only-binary=:all:` requires prebuilt dependencies and prevents pip from
compiling them. Internet is still needed to download any dependencies that are
not installed. The Linux `linux_x86_64` wheel is not a manylinux wheel and is
not verified on every Linux distribution.

The package has not been published to PyPI. For a prepared platform-specific
wheel, replace the path below with its actual absolute filename (Python 3.12):

```bash
python -m pip install '/absolute/path/to/opencv_bv_tools-0.10.0.dev18-PLATFORM.whl[wheel]'
opencv-bv-setup
```

For an experimental install directly from the public branch, this single pip
command builds the package from source (Git, native build tools and internet
access are required). This route has been tested on Linux x86-64:

```bash
python -m pip install 'opencv-bv-tools[wheel] @ git+https://github.com/nitin-singh12/opencv.git@mcp-source-install#subdirectory=apps/mcp'
opencv-bv-setup
```

This builds the MCP application and its native extensions from source; the
`[wheel]` extra still installs the pinned OpenCV Python wheel. Building OpenCV
itself from source is a separate option described above. This command installs
the named branch, not uncommitted changes in this checkout. Direct application
source installation on Windows and macOS remains unverified.

For an unverified Windows PowerShell source-install attempt with the virtual
environment created above, use:

```powershell
.\.venv\Scripts\python.exe -m pip install "opencv-bv-tools[wheel] @ git+https://github.com/nitin-singh12/opencv.git@mcp-source-install#subdirectory=apps/mcp"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\opencv-bv-setup.exe
```

On macOS the Linux/macOS command above can be tried in an activated Python 3.12
environment, but the direct application source-install route has not been
validated there. On both systems, Git, C++ build tools and internet access are
required.

On Windows use `py -3.12 -m pip` and the actual Windows wheel path. The setup
command asks for Codex, Claude Code, or another local MCP client, then asks
which folders the server may access. It registers Codex or Claude Code using
their CLI, or prints a local stdio configuration for another client. Reconnect
the client after setup. It does not configure Claude Desktop chat automatically.

For noninteractive setup, for example:

```bash
opencv-bv-setup --client codex --allow-dir /absolute/path/to/input --allow-dir /absolute/path/to/output
```

Folders must already exist. Use `--allow-any-path` instead for access wherever
the account and server file checks permit. A different existing registration
is preserved unless `--replace` is explicitly given. The `[wheel]` extra
installs the pinned OpenCV provider; source-built OpenCV users can omit it and
install their compatible `cv2` separately. Do not install multiple OpenCV
wheel variants in one environment.

## Install from source

Use **Python 3.12** for the tested wheel target. Package metadata permits Python
3.10 and later; that does not mean every version or platform has been tested.
The package is built separately for Linux x86-64,
Windows x64 and Apple Silicon macOS; a wheel from one platform cannot be used
on another. The tested macOS package requires macOS 13 or newer. The Linux
OpenCV source build is checked separately; source builds on Windows and macOS
still need validation.

From the repository root, enter `apps/mcp`. The recognition dictionary is included
in the application wheel; the OCR and face model weights are downloaded and
verified during setup. The server
uses OpenCV DNN for CPU inference and does not require RapidOCR.

Linux/macOS shell:

```bash
cd apps/mcp
PYTHON=python3.12 bash install.sh --install-only
```

Windows PowerShell:

```powershell
cd apps/mcp
py -3.12 scripts/install_mcp.py --install-only
```

The installer resolves and checks runtime dependencies normally. Internet access
is needed to obtain dependencies and model weights. Processing runs locally on CPU; no GPU or
hosted service is required.

The installer creates an isolated environment, calls the nine core tools through
MCP with included fixtures and confirms search tools register. It does not run
the optional image-search model. A successful run ends with:

```text
OK: 9 core MCP tools passed, search tools registered; email addresses hidden, lookalike left alone, originals unchanged.
Installed opencv-bv-server. Keep this extracted folder in place.
```

This is a smoke check, not a general accuracy measurement. `--install-only` does
not register a client or generate connection settings.

## Install a prepared ZIP

The ZIP contains the application wheel, notices, installer and smoke fixtures.
For this checkout, model weights are downloaded during setup and are not in the
wheel or ZIP. A ZIP may also include additional dependency wheels. A complete
fresh offline ZIP installation has not yet been documented and validated; it
would require both all dependencies and the verified models to be available.
The application wheel contains compiled Clipper and ZXing extensions and a PDFium native library and is specific to
the Python version, operating system and CPU architecture shown in its wheel
filename. Use a ZIP built for the target platform.
Verify the supplied SHA-256, extract it, then run the installer from
`opencv-bv-server-mcp`.

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

On Linux and macOS, folder restrictions are not a security boundary against
another local process that can rename an output directory or any of its
ancestors during an operation.
Use operating-system permissions to prevent untrusted users or processes from
moving those folders while the server is running.

Optional convenience flags `--claude` and `--codex` register those clients
automatically if their CLIs are installed. Add `--replace` to update an earlier
configuration. Keep the installation folder in place, restart your MCP client,
and confirm it lists `opencv-bv-server` with 13 MCP tool names: nine core tools
and four operations for optional image search. If installed, the client
CLIs can check registration with `claude mcp list` and `codex mcp list`.

## Try every tool

Replace `/absolute/path/to/input-files` and `/absolute/path/to/output` with
actual allowed absolute paths. Create the output folder before setup. On Windows
use actual Windows paths, for example `C:\Users\YourName\opencv-mcp-test`.
Each prompt is independent. Supply an appropriate input for the feature.

| Tool | Example prompt |
| --- | --- |
| `convert_image_format` | Use convert_image_format to convert /absolute/path/to/input-files/photo.jpg to /absolute/path/to/output/photo.png. |
| `read_qr` | Use read_qr on /absolute/path/to/input-files/qr.png. Show its decoded data, without opening any links. |
| `read_barcodes` | Use read_barcodes on /absolute/path/to/input-files/barcode.png and report the format and decoded value. |
| `compress_png` | Use compress_png on /absolute/path/to/input-files/screenshot.png and save /absolute/path/to/output/screenshot-small.png. Report the size change. |
| `blur_faces` | Use blur_faces on /absolute/path/to/input-files/group.jpg and save /absolute/path/to/output/group-blurred.png. Report how many regions were blurred. |
| `scan_document` | Use scan_document to turn /absolute/path/to/input-files/page.jpg into /absolute/path/to/output/page.pdf. |
| `find_emails` | Use find_emails on /absolute/path/to/input-files/email.png. Report the count and masked addresses without displaying the original. |
| `redact_emails` | Use redact_emails on /absolute/path/to/input-files/email.png and save /absolute/path/to/output/email-redacted.png. Do not open the original. |
| `redact_pdf` | Use redact_pdf on /absolute/path/to/input-files/emails.pdf and save /absolute/path/to/output/emails-redacted.pdf. Report the page and redaction counts. |

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

| Component | Purpose | How it is provided |
| --- | --- | --- |
| `mcp==2.3.0` | MCP protocol | Python runtime dependency installed by pip |
| `numpy==2.2.6` | Image arrays | Python runtime dependency installed by pip |
| `opencv-python-headless==5.0.0.93` | Image processing, face detection and CPU OCR inference | Python package installed by pip with the `wheel` extra |
| ZXing-C++ 3.1.1 | QR and barcode decoding | Compiled native component in the application wheel |
| Clipper 6.4.2 | OCR text-box expansion | Compiled native component in the application wheel |
| PDFium 156.0.8076.0 | PDF reading and rendering for redaction | Native library in the application wheel |

The default installer selects `all,wheel`. MCP and NumPy are Python package dependencies and may install their own dependencies. ZXing and PDFium are included as native code; their separate Python distributions are not required at runtime. The default wheel build obtains the pinned PDFium binary and notices from `pypdfium2==5.14.0`. An experimental Linux source-built PDFium route, which does not require `pypdfium2` to build, is documented in the repository's `3rdparty/pdfium/README.md`. PNG compression uses Python's built-in zlib.
Python packages can bring their own transitive dependencies. Python 3.12 is
installed separately. Model weights are data, not additional Python libraries.
Default runtime does not require pypdfium2, Pillow, pyoxipng, pyclipper,
ReportLab, requests, charset-normalizer, RapidOCR, Shapely, the Python `onnxruntime` package or a
separate Python `onnx` parser. OpenCV DNN loads the verified local ONNX models directly.
The supported OCR configuration is CPU-only PP-OCRv6 small detection and
recognition, using the unchanged RapidOCR model bytes. The recognition dictionary
is packaged at `models/ppocrv6-characters.txt`, extracted from that exact model's
metadata with [recorded provenance](src/opencv_bv_tools/models/ocr-provenance.json).
Arbitrary model downloads and CUDA OCR are
unsupported. The optional `cmyk` extra installs
Pillow to preserve original CMYK JPEG normalization for email processing.
The optional MuPDF engine is separately licensed and is not installed by default.
See [third-party notices](THIRD_PARTY_NOTICES.md).

Registration is lightweight. OCR loads when first needed, reuses its worker for
subsequent calls, and releases that worker after 120 seconds idle by default.
`--ocr-idle-seconds` changes this delay; zero reloads per call. This reduces idle
memory but does not guarantee the package fits every low-memory device.

### Model setup and offline use

`opencv-bv-setup` and the ZIP installer download the three pinned model files
from the configured public Google Drive location during setup. Files are checked
against fixed sizes and SHA-256 hashes before being saved in the user's cache.
No image or document is uploaded. Inference uses the local cache and does not
download anything. The first setup needs internet access unless the verified
models have already been provisioned in the selected cache; later use can be offline.

After installing the application, run `opencv-bv-models` to prepare the models
separately. To provision them without downloading, put the exact three `.onnx`
files in a local directory and run:

```bash
opencv-bv-models --source-dir /path/to/model-files
```

The local files are checked against the same hashes. For a custom shared cache,
set `OPENCV_BV_MODEL_DIR` to an absolute directory before setup. The generated
client configuration records that directory, so the client can find the models
even when it starts in a different environment. If configuring a client manually,
pass `--model-dir` with that directory to the server. If a model is missing or
changed, run `opencv-bv-models` again with the same cache setting.

## Development, packaging and known limits

The main entry is `src/opencv_bv_tools/server.py`. Feature modules register their
tools there; `runtime/` contains shared file checks, image I/O and process workers.
`scripts/` contains installation and packaging helpers. `tests/` is development
material, not loaded by the server.

From `apps/mcp`, use an isolated development
environment:

```bash
python3.12 -m venv .venv-dev
.venv-dev/bin/python -m pip install -c install-constraints.txt '.[all,wheel,test,png-test,scanner-test,barcode-test,qr-test,face-test]'
.venv-dev/bin/python -m pytest tests
.venv-dev/bin/python -m build --wheel
.venv-dev/bin/python scripts/build_mcp_zip.py --wheel dist/opencv_bv_tools-0.10.0.dev18-cp312-cp312-linux_x86_64.whl --source-commit "$(git rev-parse HEAD)" --unpublished
```

The wheel path above is an example for Linux x86-64 and Python 3.12; use the
filename produced by the build on your platform. On Windows use
`.venv-dev\Scripts\python.exe` and obtain the SHA with `git rev-parse HEAD`.
Test-only libraries include independent image/PDF oracles;
they are not default runtime requirements. `--unpublished` identifies a local
working-tree build. A release must use a reviewed commit and its exact artifacts.

Known limits include missed small faces and difficult email/code images.
The preceding version’s full Linux suite passed on explicitly selected CPU cores after a pure
Python QR-generator probe reproduced host instability on another core.
This does not establish unrestricted host stability or replace native testing
on other systems. Six original Kaggle barcode/QR images remain unavailable.
See [lightweight notes](docs/lightweight-branch.md) for measured coverage and
the preserved earlier failures.

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
official OpenCV MCP packaging convention. The known failures above and upstream
maintainer review remain work before an upstream or stable-release claim.
