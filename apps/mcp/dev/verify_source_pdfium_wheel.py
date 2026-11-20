"""Verify the Linux source-built PDFium and its notices inside a wheel."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def verify(wheel):
    spec = json.loads((ROOT / 'native/pdfium-source-manifest.json').read_text())['linux-x86_64']
    with zipfile.ZipFile(wheel) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('PDFium wheel contains duplicate archive entries')
        prefix = 'opencv_bv_tools/_native/'
        def check(name, expected):
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f'PDFium wheel file has unexpected contents: {name}')
        check(prefix + spec['binary'], spec['binary_sha256'])
        notice_prefix = prefix + 'pdfium_notices/'
        actual = {name[len(notice_prefix):] for name in archive.namelist() if name.startswith(notice_prefix)}
        if actual != set(spec['licenses']):
            raise ValueError('PDFium wheel notices differ from the pinned source notice list')
        for name, digest in spec['licenses'].items():
            check(notice_prefix + name, digest)
    print(f'Source-built PDFium and {len(spec["licenses"])} notices verified: {wheel}')


if __name__ == '__main__':
    verify(Path(sys.argv[1]))
