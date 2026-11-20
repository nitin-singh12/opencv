"""Fail if the PDFium GN target gains an unmapped third-party dependency."""
from pathlib import Path
import re
import shlex
import subprocess
import sys

NOTICES = {
    'fx_agg': ('agg23.txt',), 'fx_freetype': ('freetype.txt',),
    'fx_lcms2': ('lcms.txt',), 'lcms2': ('lcms.txt',),
    'fx_libopenjpeg': ('libopenjpeg.txt',), 'libopenjpeg2': ('libopenjpeg.txt',),
    'jpeg': ('libjpeg_turbo.md', 'libjpeg_turbo.ijg'), 'zlib': ('zlib.txt',),
    'abseil-cpp': ('abseil.txt',), 'dragonbox': ('dragonbox-apache.txt', 'dragonbox-boost.txt'),
    'fast_float': ('fast_float.txt',), 'harfbuzz': ('harfbuzz.txt',),
    'icu': ('icu.txt',), 'libjpeg_turbo': ('libjpeg_turbo.md', 'libjpeg_turbo.ijg'),
    'llvm-libc': ('llvm-libc.txt',), 'nasm': ('nasm.txt',),
    'libc++': ('libcxx.txt',), 'libc++abi': ('libcxxabi.txt',),
}


def verify(gn, build):
    labels = subprocess.check_output([gn, 'desc', build, '//:pdfium', 'deps', '--all'], text=True).splitlines()
    notices = Path(__file__).resolve().parents[2] / 'apps/mcp/native/pdfium-source-notices'
    found = set()
    for label in labels:
        if label.startswith('//third_party:'):
            name = label.split(':', 1)[1]
        elif label.startswith('//third_party/'):
            name = label[len('//third_party/'):].split('/', 1)[0].split(':', 1)[0]
        elif label.startswith('//buildtools/third_party/'):
            name = label[len('//buildtools/third_party/'):].split('/', 1)[0].split(':', 1)[0]
        else:
            continue
        if name not in NOTICES:
            raise RuntimeError(f'Unmapped PDFium third-party target: {label}')
        for notice in NOTICES[name]:
            if not (notices / notice).is_file():
                raise RuntimeError(f'Missing notice for {label}: {notice}')
        found.add(name)
    if not found:
        raise RuntimeError('PDFium GN dependency graph contained no third-party targets')
    print('Mapped PDFium third-party targets: ' + ', '.join(sorted(found)))


def verify_link_inputs(build):
    build = Path(build)
    ninja = (build / 'obj/pdfium.ninja').read_text()
    rows = [line.strip()[len('libs = '):] for line in ninja.splitlines() if line.strip().startswith('libs = ')]
    if len(rows) != 1:
        raise RuntimeError('Could not determine PDFium linker libraries')
    flags = shlex.split(rows[0])
    archives = [Path(flag).name for flag in flags if flag.endswith('.a')]
    if archives != ['libclang_rt.builtins.a']:
        raise RuntimeError(f'Unmapped compiler-runtime archives: {archives}')
    allowed_flags = {'-ldl', '-lpthread', '-lrt', '-lglib-2.0', '-lgmodule-2.0', '-lgobject-2.0', '-lgthread-2.0'}
    if {flag for flag in flags if flag.startswith('-l')} - allowed_flags:
        raise RuntimeError('Unexpected PDFium system linker input')
    notice = Path(__file__).resolve().parents[2] / 'apps/mcp/native/pdfium-source-notices/llvm-libc.txt'
    if not notice.is_file():
        raise RuntimeError('Missing LLVM compiler-rt licence notice')
    dynamic = subprocess.check_output(['readelf', '-d', str(build / 'libpdfium.so')], text=True)
    needed = set(re.findall(r'\(NEEDED\).*\[(.*?)\]', dynamic))
    allowed = {'libpthread.so.0', 'libm.so.6', 'libgcc_s.so.1', 'libc.so.6', 'ld-linux-x86-64.so.2'}
    if not needed or needed - allowed:
        raise RuntimeError(f'Unexpected PDFium dynamic libraries: {sorted(needed - allowed)}')
    print('Mapped LLVM compiler-rt and checked system dynamic libraries: ' + ', '.join(sorted(needed)))


if __name__ == '__main__':
    verify(sys.argv[1], sys.argv[2])
    verify_link_inputs(sys.argv[2])
