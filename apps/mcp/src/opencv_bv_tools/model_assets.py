# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.
"""Download pinned model assets at setup time and use them offline at runtime."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener


# These are exact model bytes; hashes do not trust the storage provider.
MODELS = {
    'face_detection_yunet_2026may.onnx': (229738, 'ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0', '1I-HqT9hMWzwDAit26ZrzkW0c3uCm4DJE'),
    'PP-OCRv6_det_small.onnx': (9929594, '090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f', '1vOMcA6YpjGNdg2kzIWt4W0kLJMQc3oil'),
    'PP-OCRv6_rec_small.onnx': (21234383, '6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884', '1JHQdWvSNge-LL_U2wB6F2GOhggpNAiRi'),
}
SEARCH_MODELS = {
    'text-raw.onnx': (1129352720, 'd9561b9660110dbab6628e3b40006fbd511c58c60b5eb0bf02509c2c555e6d83', '1yDpxSkrWp-hEcWtVqBGgdTHeFKqncj_1'),
    'vision-body.onnx': (371063171, '002ceaa60dab36e18727ed5de78c57989a981be9adb921e7d955feb34b730fc5', '1lAKtRHwuiLejCoN_ZWLbRE1P0z7TGrFh'),
    'position-grid.npy': (786560, 'd8d2a9b9fc1b04ee40059a4d159629a1a9e67d832a7afac673aea8853f22d152', '1hBbR9lOXXzg7_hmZjVjRvm7iYOnJPIYj'),
    'model/tokenizer.json': (34356304, '58a1696e79c9d97937389ed116f552a15c84811d7b8023918b86f4bc5775b1b0', '1yll0gkOIiQxQ1TSXPcp_6LMqjJ0_mqD-'),
}
BUNDLED = Path(__file__).resolve().parent / 'models'


def cache_directory():
    override = os.environ.get('OPENCV_BV_MODEL_DIR')
    if override:
        return Path(override).expanduser()
    if sys.platform == 'win32' and os.environ.get('LOCALAPPDATA'):
        root = Path(os.environ['LOCALAPPDATA'])
    elif sys.platform == 'darwin':
        root = Path.home() / 'Library' / 'Caches'
    else:
        root = Path(os.environ.get('XDG_CACHE_HOME') or (Path.home() / '.cache'))
    return root / 'opencv-bv-tools' / 'models' / 'v1'


def search_cache_directory():
    override = os.environ.get('OPENCV_BV_SEARCH_MODEL_DIR')
    return Path(override).expanduser() if override else cache_directory().parent / 'fgclip2-v1'


def _matches(path, size, expected):
    """Verify large cached models without loading them into RAM."""
    try:
        with path.open('rb') as stream:
            if os.fstat(stream.fileno()).st_size != size:
                return False
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            return digest.hexdigest() == expected
    except FileNotFoundError:
        return False


def model_path(name):
    if name not in MODELS:
        raise ValueError('Unknown model')
    cached = cache_directory() / name
    if cached.is_file() or os.environ.get('OPENCV_BV_MODEL_DIR'):
        return cached
    bundled = BUNDLED / name
    return bundled if bundled.is_file() else cached


def verified_bytes(name):
    size, digest, _ = MODELS[name]
    path = model_path(name)
    try:
        with path.open('rb') as stream:
            data = stream.read(size + 1)
    except OSError as exc:
        raise FileNotFoundError(f'Model {name} is missing. Run opencv-bv-models to download it.') from exc
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'Model {name} is invalid. Run opencv-bv-models to download it again.')
    return data


class HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urlsplit(newurl).scheme.lower() != 'https':
            raise ValueError('Model download redirected to an insecure URL')
        return super().redirect_request(request, fp, code, message, headers, newurl)


def install_models(source_dir=None, *, search=False):
    """Populate the cache with pinned bytes; never used by inference workers."""
    destination = search_cache_directory() if search else cache_directory()
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, (size, expected, file_id) in (SEARCH_MODELS if search else MODELS).items():
        target = destination / name
        if _matches(target, size, expected):
            continue
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.model-', delete=False) as tmp:
                temp_name = tmp.name
                digest = hashlib.sha256()
                total = 0
                if source_dir is not None:
                    stream = (Path(source_dir) / name).open('rb')
                else:
                    url = f'https://drive.usercontent.google.com/download?export=download&id={file_id}&confirm=t'
                    stream = build_opener(HTTPSRedirects()).open(url, timeout=60)
                with stream:
                    while chunk := stream.read(min(1024 * 1024, size + 1 - total)):
                        total += len(chunk)
                        if total > size:
                            raise ValueError(f'Downloaded model {name} has unexpected size')
                        digest.update(chunk)
                        tmp.write(chunk)
                if total != size or digest.hexdigest() != expected:
                    raise ValueError(f'Model {name} failed its size or SHA-256 check')
            os.replace(temp_name, target)
            temp_name = None
        finally:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description='Install verified models for offline use (OCR and face by default).')
    parser.add_argument('--source-dir', type=Path, help='copy verified models from a local folder instead of downloading')
    parser.add_argument('--search', action='store_true', help='install the optional 1.54 GB image-search models instead')
    args = parser.parse_args(argv)
    try:
        print(f'Models ready: {install_models(args.source_dir, search=args.search)}')
        return 0
    except (OSError, ValueError, URLError) as exc:
        print(f'ERROR: Could not install models: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
