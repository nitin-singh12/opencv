"""Check the pinned Git checkouts used by the experimental PDFium build."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys


def tree_digest(root):
    """Hash tool/sysroot contents, including symlink targets, not timestamps."""
    digest = hashlib.sha256()
    optional_clang_tools = {'bin/llvm-bcanalyzer', 'bin/llvm-cxxfilt', 'bin/llvm-dwarfdump',
                            'bin/llvm-objdump', 'bin/llvm-otool', 'cr_build_revision',
                            'llvmobjdump_build_revision'} if root.name == 'Release+Asserts' else set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted(dirs + files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if relative in optional_clang_tools:
                continue
            # Download caches and stamp files are not toolchain inputs.
            if '/' not in relative and name.startswith('.'):
                if name in dirs:
                    dirs.remove(name)
                continue
            if path.is_symlink():
                digest.update(b'L\0' + relative.encode() + b'\0' + os.readlink(path).encode() + b'\0')
            elif path.is_file():
                content = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        content.update(chunk)
                digest.update(b'F\0' + relative.encode() + b'\0' + content.digest())
    return digest.hexdigest()


def verify(source, git, revisions=None):
    check_tools = revisions is None
    if check_tools:
        revisions = json.loads(Path(__file__).with_name('dependency-revisions.json').read_text())
    for relative, expected in revisions.items():
        checkout = source / relative
        def command(*args):
            return subprocess.check_output([git, '-C', str(checkout), *args], text=True).strip()
        if command('rev-parse', 'HEAD') != expected:
            raise RuntimeError(f'PDFium dependency revision mismatch: {relative}')
        # buildtools contains downloaded, ignored CIPD files; GN is hashed
        # separately. Other pinned Git dependencies should contain no ignored
        # files that could introduce unverified build inputs.
        status = ['status', '--porcelain', '--untracked-files=all']
        if relative != 'buildtools':
            status.append('--ignored')
        if command(*status):
            raise RuntimeError(f'PDFium dependency has tracked or untracked changes: {relative}')
    if check_tools:
        trees = json.loads(Path(__file__).with_name('toolchain-hashes.json').read_text())
        for relative, expected in trees.items():
            path = source / relative
            if not path.is_dir() or tree_digest(path) != expected:
                raise RuntimeError(f'PDFium toolchain or sysroot contents differ: {relative}')


if __name__ == '__main__':
    try:
        verify(Path(sys.argv[1]), sys.argv[2])
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        sys.exit(f'PDFium dependency verification failed: {error}')
