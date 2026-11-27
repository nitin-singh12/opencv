"""Portable search-state tests and explicit native Windows security acceptance."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from opencv_bv_tools.runtime import safe_files, search_files

WINDOWS = pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows/NTFS")


@pytest.fixture
def private(tmp_path):
    path = search_files.create_private_directory(tmp_path / "indexes")
    return path, safe_files.Folders.from_canonical((tmp_path,))


def test_snapshot_replacement_and_failed_publication_preserve_bytes(private):
    path, folders = private
    target = path / "photos.sqlite"
    search_files.write_snapshot(b"first", target, folders, overwrite=False, before_publish=lambda: None)
    search_files.write_snapshot(b"second", target, folders, overwrite=True, before_publish=lambda: None)
    def changed():
        raise ValueError("external change")
    with pytest.raises(ValueError, match="external change"):
        search_files.write_snapshot(b"third", target, folders, overwrite=True, before_publish=changed)
    assert target.read_bytes() == b"second"
    assert not list(path.glob(".opencv-search-*.tmp"))


def test_owner_lock_contends_between_processes_and_recovers(private):
    path, folders = private
    code = '''
from pathlib import Path
import sys
from opencv_bv_tools.runtime import search_files, safe_files
path = Path(sys.argv[1])
folders = safe_files.Folders.from_canonical((path.parent,))
owner = search_files.claim_owner(path, folders)
print("owned", flush=True)
sys.stdin.buffer.read(1)
'''
    child = subprocess.Popen([sys.executable, "-c", code, str(path / ".opencv-search-owner")],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert child.stdout.readline() == b"owned\n"
        with pytest.raises(BlockingIOError):
            search_files.claim_owner(path / ".opencv-search-owner", folders)
    finally:
        child.kill()
        child.communicate(timeout=10)
    owner = search_files.claim_owner(path / ".opencv-search-owner", folders)
    owner.close()


def test_scratch_cleanup(private):
    path, _ = private
    scratch = search_files.temporary_directory(path)
    search_files.verify_private_directory(scratch.name)
    (Path(scratch.name) / "model").mkdir()
    (Path(scratch.name) / "model" / "weights").write_bytes(b"model")
    scratch.cleanup()
    assert not Path(scratch.name).exists()


@WINDOWS
def test_windows_rejects_public_directory_without_rewriting_acl(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    with pytest.raises(safe_files.UnsafePath, match="ACL"):
        search_files.create_private_directory(public)
    assert public.is_dir()


@WINDOWS
def test_windows_private_acl_inherits_only_current_user(private):
    from opencv_bv_tools.runtime import windows_search
    path, _ = private
    windows_search.verify_private_directory(path)
    child = path / "child"
    child.mkdir()
    # Inspect the actual child DACL rather than relying on localized icacls output.
    import ctypes as C
    from opencv_bv_tools.runtime import windows_files
    handle = windows_files._open(child, windows_files.READ_ATTRIBUTES | 0x20000, directory=True)
    descriptor, owner, dacl = C.c_void_p(), C.c_void_p(), C.c_void_p()
    try:
        assert windows_search._get_security(handle, 1, 5, C.byref(owner), None,
                                             C.byref(dacl), None, C.byref(descriptor)) == 0
        assert C.cast(dacl, C.POINTER(windows_search._Acl)).contents.count == 1
        pointer = C.c_void_p()
        windows_files._checked(windows_search._get_ace(dacl, 0, C.byref(pointer)))
        ace = C.cast(pointer, C.POINTER(windows_search._Ace)).contents
        assert ace.kind == 0 and ace.flags & 0x10  # inherited allow ACE
        assert windows_search._sid_text(pointer.value + C.sizeof(windows_search._Ace)) == windows_search._user_sid()
    finally:
        if descriptor.value:
            windows_files._free(descriptor)
        windows_files._close(handle)


@WINDOWS
def test_windows_directory_pins_prevent_ancestor_rename(private):
    path, folders = private
    with search_files.folder(path, folders):
        with pytest.raises(OSError):
            path.rename(path.with_name("renamed"))
    path.rename(path.with_name("renamed"))


@WINDOWS
def test_windows_junctions_are_not_traversed(private, tmp_path):
    path, folders = private
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.png").write_bytes(b"secret")
    junction = path / "junction"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(outside)], check=True,
                   capture_output=True)
    try:
        assert list(search_files.walk_windows(path, folders, True, lambda: None,
                                              extensions={".png"}, max_entries=10)) == []
        with pytest.raises(safe_files.UnsafePath, match="reparse"):
            with search_files.folder(junction, folders):
                pass
    finally:
        junction.rmdir()


@WINDOWS
def test_windows_private_replacement_does_not_enable_ordinary_overwrite(private):
    path, folders = private
    target = path / "ordinary.png"
    safe_files.write_file(b"old", target, folders)
    with pytest.raises(safe_files.UnsafePath, match="Replacing existing"):
        safe_files.write_file(b"new", target, folders, overwrite=True)
    assert target.read_bytes() == b"old"


@WINDOWS
def test_windows_rejects_hardlinked_snapshot(private):
    path, folders = private
    target = path / "photos.sqlite"
    target.write_bytes(b"original")
    os.link(target, path / "alias")
    with pytest.raises(safe_files.UnsafePath, match="hard link"):
        search_files.write_snapshot(b"replace", target, folders, overwrite=True, before_publish=lambda: None)
    assert target.read_bytes() == b"original"


@WINDOWS
def test_windows_worker_suppresses_automatic_wer():
    code = '''
import ctypes
from opencv_bv_tools.runtime.process_support import protect_private_worker
protect_private_worker()
assert ctypes.windll.kernel32.GetErrorMode() & 2
'''
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)
