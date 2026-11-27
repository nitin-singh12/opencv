# SPDX-License-Identifier: Apache-2.0
"""Private search-state filesystem operations, without weakening ordinary outputs."""
import contextlib
import os
from pathlib import Path
import secrets
import shutil
import sys
import tempfile

from . import safe_files

MARKER = b"opencv-image-search-owner-v1\n"


@contextlib.contextmanager
def folder(path, folders):
    """Validate and pin a directory for the duration of an operation."""
    if sys.platform == "win32":
        from . import windows_files
        path = windows_files.normalize_path(path)
        folders.check(path, "The image search folder")
        with windows_files._directories(path):
            yield path
    else:
        fd, _ = safe_files._open_folder(path, folders, create=False)
        try:
            yield safe_files._fd_path(fd)
        finally:
            os.close(fd)


def verify_private_directory(path):
    if sys.platform == "win32":
        from .windows_search import verify_private_directory as verify
        return verify(path)
    path = Path(path)
    if path.stat().st_mode & 0o077:
        raise ValueError("search_index_dir must be private (permissions 0700); choose or create a private directory.")
    return path


def create_private_directory(path):
    if sys.platform == "win32":
        from .windows_search import create_private_directory as create
        return create(path)
    path = Path(path)
    path.mkdir(mode=0o700, exist_ok=True)
    return verify_private_directory(path)


class _WindowsTemporaryDirectory:
    def __init__(self, parent):
        self.name = str(create_private_directory(Path(parent) / (".opencv-search-worker-" + secrets.token_hex(16))))

    def cleanup(self):
        shutil.rmtree(self.name)


def temporary_directory(parent):
    if sys.platform == "win32":
        return _WindowsTemporaryDirectory(parent)
    return tempfile.TemporaryDirectory(prefix=".opencv-search-worker-", dir=parent)


def claim_owner(path, folders):
    if sys.platform == "win32":
        from .windows_search import claim_owner as claim
        return claim(path, folders, MARKER)
    import fcntl
    if not path.exists():
        try:
            safe_files.write_file(MARKER, path, folders, suffix="", create_folders=False)
        except safe_files.UnsafePath:
            if not path.exists():
                raise
    source, _, _ = safe_files.open_input(path, folders)
    try:
        if source.read(64) != MARKER or os.fstat(source.fileno()).st_nlink != 1:
            raise safe_files.UnsafePath("Invalid index directory ownership marker.")
        fcntl.flock(source.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return source
    except BaseException:
        source.close()
        raise


def write_snapshot(data, path, folders, *, overwrite, before_publish):
    if sys.platform == "win32":
        from .windows_search import write_snapshot as write
        return write(data, path, folders, overwrite=overwrite, before_publish=before_publish)
    return safe_files.write_file(data, path, folders, overwrite=overwrite, suffix=".sqlite",
                                 create_folders=False, before_publish=before_publish)


def walk_windows(root, folders, recursive, check, *, extensions, max_entries):
    """Stream directory entries while ancestors are pinned against rename."""
    visited = 0
    def visit(path, depth):
        nonlocal visited
        with folder(path, folders) as real, os.scandir(real) as entries:
            for entry in entries:
                check()
                visited += 1
                if visited > max_entries:
                    raise ValueError("Folder scan exceeds one million entries; choose a smaller folder.")
                if entry.name.startswith(".opencv-search-"):
                    continue
                # is_symlink alone misses junctions and other NTFS reparse points.
                st = entry.stat(follow_symlinks=False)
                if st.st_file_attributes & 0x400:
                    continue
                if entry.is_dir(follow_symlinks=False) and recursive:
                    if depth >= 32:
                        raise ValueError("Folder depth exceeds 32; choose a smaller folder.")
                    yield from visit(real / entry.name, depth + 1)
                elif entry.is_file(follow_symlinks=False) and Path(entry.name).suffix.lower() in extensions:
                    yield real / entry.name
    yield from visit(root, 0)
