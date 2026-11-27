# Local semantic image search (experimental)

Search a local image collection using a text description, for example “a red
car”. This is one feature exposed as four MCP operations: `index_images`,
`search_images`, `image_search_status` and `cancel_image_index`. Photos are
processed locally and are not sent to a service. Search results contain paths
and cosine similarities, not image content; a score is not a probability.

The implementation keeps the index, resume, cancellation and paging design
from `bigvisionllc/opencv-bv-server` commit
`e1b72225132b16d93dd5ef21e5793be09144f6f0`. The encoder uses the FG-CLIP2
OpenCV DNN experiment (`codex/fgclip2-opencv-standalone`) with unchanged model
weights. Its fingerprint differs from the earlier PyTorch encoder: create a new
index rather than reusing its indexes.

## Requirements and limits

- Python 3.12, OpenCV 5 and NumPy; no Pillow, PyTorch, Transformers or ONNX
  Runtime is needed for search inference. The model export itself used separate
  build dependencies and is not performed by the installed server.
- Linux, macOS and Windows have dedicated search validation in CI. Windows
  requires Python 3.12 and local NTFS folders. Check the test results for the
  exact commit you install; adding a job is not evidence that it has passed.
  Windows search rejects junctions/reparse points and network shares.
- CPU FP32 only. Model files total approximately 1.54 GB; the worker creates a
  verified private copy and may require substantial RAM. It loads lazily and
  exits after 120 seconds idle by default. This is not a low-memory guarantee.
- JPEG, PNG, WebP, BMP, TIFF, static GIF and JPEG 2000 were checked with local
  synthetic stills. Availability of codecs depends on the OpenCV build. HEIC,
  HEIF and AVIF are attempted only if that build supports them; unsupported
  files are counted as failures. They have not been validated on this build.
  Animated or multipage inputs are rejected. Check indexing status for errors.
- Search accuracy has not been established for a large or held-out photo
  library. Model-output parity checks do not measure retrieval accuracy.

## Obtain the model assets

The application wheel does not contain the search assets. After installing the
package in your activated Python environment, run the same command on Linux,
macOS or Windows:

```bash
opencv-bv-models --search
```

This optional download totals approximately 1.54 GB. It streams each file from
Google Drive into a temporary file, checks its pinned size and SHA-256, and
publishes only a complete verified file. A retry reuses good files and downloads
missing or damaged ones. No Google account or paid service is required. Leave
space for both the cache and a private worker copy (about 3.1 GB combined), plus
your image index; RAM usage is additional.

The cache contains these exact files:

```text
model-directory/
  text-raw.onnx             1129352720 bytes
  vision-body.onnx           371063171 bytes
  position-grid.npy              786560 bytes
  model/tokenizer.json         34356304 bytes
```

Their pinned SHA-256 hashes and public Drive file IDs are in `model_assets.py`. The worker
checks each file's size and hash, then loads a private verified copy. It never
downloads models during a search. The 1.54 GB search assets are separate from
the smaller OCR and face model downloads.

## Configure a client

Run setup from the same activated environment. It asks which client to use
(Codex, Claude Code, or another local MCP client) and which photo folders to
allow. `--search` also provisions the verified models and a private index:

```bash
opencv-bv-setup --search
```

For an explicit client and folder (replace the examples with your own paths):

```bash
# Linux
opencv-bv-setup --client codex --search --allow-dir "$HOME/Pictures" --replace

# macOS
opencv-bv-setup --client claude --search --allow-dir "$HOME/Pictures" --replace
```

```powershell
# Windows PowerShell
opencv-bv-setup --client other --search --allow-dir "$HOME\Pictures" --replace
```

The `other` choice prints JSON with the exact installed executable and arguments
for your MCP client's local stdio configuration. Setup does not guess where an
unknown client stores its settings. Copy that entry to the client, then restart
or reconnect. Codex and Claude Code registration is automated when their CLIs
are installed. Claude Desktop uses `other`.

The default index is `~/.opencv-search-index`. To select a different location,
add `--search-index-dir "/absolute/path/to/index"`; its parent must already
exist. Setup creates a new directory with private permissions (0700 on POSIX;
current-user-only ACL on Windows). It refuses an existing nonprivate directory
instead of changing its permissions. The index holds filenames, hashes and
embeddings and is not encrypted. Don't share it publicly.

For offline setup, copy the four assets using their cache layout and run
`opencv-bv-models --search --source-dir "/absolute/path/to/assets"` first.
`OPENCV_BV_SEARCH_MODEL_DIR` can select a different cache folder. Setup records
absolute model/index paths and includes them in the permitted server folders.
Manual client configurations can use `--search-model-dir`, `--search-index-dir`
and corresponding `--allow-dir` entries.

Without both search directories, search is disabled and the nine core tools
continue to work. The index is unencrypted and contains filenames, paths, hashes
and embeddings; keep it in private storage. Worker isolation does not prevent access by the same user, administrators or OS
backups. POSIX workers disable core dumps; Windows suppresses automatic Windows
Error Reporting but cannot override administrator-configured LocalDumps or debuggers.

## Test

1. Ask the client: `Use index_images to index /absolute/path/to/photos with index_name photos.`
2. Ask: `Use image_search_status for index photos.` Wait until the job completes.
3. Ask: `Use search_images to find "a red car" in index photos and return the top 10.`
4. To stop an active index job, use `cancel_image_index` with its returned
   `job_id`. Completed checkpoints are retained for resumption.

`search_images` returns `next_cursor` if another page is available. Pass it
back with the same query, index name and page size. Cursors expire on server
restart or index change. Existing source files are rechecked when returned;
missing or changed files are skipped. Incremental indexing rereads content and
reuses unchanged embeddings. Indexes are bounded, but a large index may need
several times its stored size in RAM.
