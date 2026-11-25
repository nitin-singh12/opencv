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


def install_models(source_dir=None):
    """Populate the cache with pinned bytes; never used by inference workers."""
    destination = cache_directory()
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, (size, expected, file_id) in MODELS.items():
        target = destination / name
        if target.is_file():
            with target.open('rb') as existing:
                data = existing.read(size + 1)
            if len(data) == size and hashlib.sha256(data).hexdigest() == expected:
                continue
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination, prefix='.model-', delete=False) as tmp:
                temp_name = tmp.name
                digest = hashlib.sha256()
                total = 0
                if source_dir is not None:
                    stream = (Path(source_dir) / name).open('rb')
                else:
                    url = f'https://drive.google.com/uc?export=download&id={file_id}'
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
    parser = argparse.ArgumentParser(description='Install verified OCR and face models for offline use.')
    parser.add_argument('--source-dir', type=Path, help='copy verified models from a local folder instead of downloading')
    args = parser.parse_args(argv)
    try:
        print(f'Models ready: {install_models(args.source_dir)}')
        return 0
    except (OSError, ValueError, URLError) as exc:
        print(f'ERROR: Could not install models: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
