# SPDX-License-Identifier: Apache-2.0
"""Native NTFS support for private search state; ordinary outputs stay no-clobber."""
import ctypes as C
from ctypes import wintypes as W
import msvcrt
import os
from pathlib import Path
import secrets

from . import windows_files as files
from .safe_files import UnsafePath

_get_security = files._api(files.security, "GetSecurityInfo", W.DWORD, W.HANDLE, C.c_int,
                           W.DWORD, C.POINTER(C.c_void_p), C.c_void_p,
                           C.POINTER(C.c_void_p), C.c_void_p, C.POINTER(C.c_void_p))
_get_control = files._api(files.security, "GetSecurityDescriptorControl", W.BOOL,
                          C.c_void_p, C.POINTER(W.WORD), C.POINTER(W.DWORD))
_get_ace = files._api(files.security, "GetAce", W.BOOL, C.c_void_p, W.DWORD, C.POINTER(C.c_void_p))
_create_directory = files._api(files.kernel, "CreateDirectoryW", W.BOOL, W.LPCWSTR, C.c_void_p)


class _Acl(C.Structure):
    _fields_ = [("revision", W.BYTE), ("reserved", W.BYTE), ("size", W.WORD),
                ("count", W.WORD), ("reserved2", W.WORD)]


class _Ace(C.Structure):
    _fields_ = [("kind", W.BYTE), ("flags", W.BYTE), ("size", W.WORD), ("mask", W.DWORD)]


def _sid_text(pointer):
    value = W.LPWSTR()
    files._checked(files._sid_string(pointer, C.byref(value)))
    try:
        return value.value
    finally:
        files._free(C.cast(value, W.HANDLE))


def _user_sid():
    token = W.HANDLE()
    files._checked(files._token(files._current(), 8, C.byref(token)))
    try:
        size = W.DWORD()
        files._token_info(token, 1, None, 0, C.byref(size))
        buffer = C.create_string_buffer(size.value)
        files._checked(files._token_info(token, 1, buffer, size, C.byref(size)))
        return _sid_text(C.cast(buffer, C.POINTER(C.c_void_p))[0])
    finally:
        files._close(token)


def verify_private_directory(path):
    """Require current-user ownership and a protected, inheritable user-only DACL.

    This deliberately rejects even Administrators/SYSTEM allow entries. An
    administrator can still take ownership, as on POSIX with root privileges.
    """
    path = files.normalize_path(path)
    with files._directories(path):
        handle = files._open(path, files.READ_ATTRIBUTES | 0x20000, directory=True)  # READ_CONTROL
        descriptor, owner, dacl = C.c_void_p(), C.c_void_p(), C.c_void_p()
        try:
            error = _get_security(handle, 1, 5, C.byref(owner), None, C.byref(dacl), None, C.byref(descriptor))
            if error:
                raise C.WinError(error)
            control, revision = W.WORD(), W.DWORD()
            files._checked(_get_control(descriptor, C.byref(control), C.byref(revision)))
            sid = _user_sid()
            valid = bool(owner.value and dacl.value and control.value & 0x1000)  # SE_DACL_PROTECTED
            valid = valid and _sid_text(owner) == sid
            acl = C.cast(dacl, C.POINTER(_Acl)).contents if dacl.value else None
            valid = valid and acl.count == 1
            if valid:
                pointer = C.c_void_p()
                files._checked(_get_ace(dacl, 0, C.byref(pointer)))
                ace = C.cast(pointer, C.POINTER(_Ace)).contents
                # One non-inherit-only allow ACE, full control, inherited by files and folders.
                valid = (ace.kind == 0 and ace.flags & 3 == 3 and not ace.flags & 0x0c
                         and ace.mask & 0x1f01ff == 0x1f01ff
                         and _sid_text(pointer.value + C.sizeof(_Ace)) == sid)
            if not valid:
                raise UnsafePath("Search index directory requires a protected current-user-only inheritable ACL; create it with the search setup command.")
        finally:
            if descriptor.value:
                files._free(descriptor)
            files._close(handle)
    return path


def create_private_directory(path):
    path = files.normalize_path(path)
    with files._directories(path.parent):
        with files._private_descriptor(inherit=True) as descriptor:
            attributes = files._SecurityAttributes(C.sizeof(files._SecurityAttributes), descriptor, False)
            if not _create_directory(str(path), C.byref(attributes)):
                error = C.get_last_error()
                if error != 183:  # Existing folders must already be private; never change their ACL.
                    raise C.WinError(error)
        verify_private_directory(path)
    return path


def claim_owner(path, folders, marker):
    """An exclusive open handle denies other owners and deletion until close."""
    path = files.normalize_path(path)
    folders.check(path, "The index ownership marker")
    with files._directories(path.parent):
        if not path.exists():
            try:
                files.write_file(marker, path, folders, suffix="", create_folders=False)
            except (UnsafePath, OSError) as exc:
                if getattr(exc, "winerror", None) == 32:
                    raise BlockingIOError("Another server owns this index directory.") from exc
                if not path.exists():
                    raise
        try:
            handle = files._open(path, files.GENERIC_READ, share=0)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 32:
                raise BlockingIOError("Another server owns this index directory.") from exc
            raise
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            files._close(handle)
            raise
        source = os.fdopen(fd, "rb")
        try:
            if os.fstat(fd).st_nlink != 1 or source.read(64) != marker:
                raise UnsafePath("Invalid index directory ownership marker.")
            return source
        except BaseException:
            source.close()
            raise


def write_snapshot(data, path, folders, *, overwrite, before_publish):
    """Replace only a private search snapshot, using the verified new file handle."""
    path = files.normalize_path(path)
    folders.check(path.parent, "The image index folder")
    with files._directories(path.parent):
        verify_private_directory(path.parent)
        temporary = path.parent / f".opencv-search-{secrets.token_hex(8)}.tmp"
        with files._private_descriptor() as descriptor:
            handle = files._open(temporary, files.GENERIC_READ | files.GENERIC_WRITE | files.DELETE,
                                 create=True, descriptor=descriptor)
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
        except BaseException:
            try:
                files._delete_on_close(handle)
            finally:
                files._close(handle)
            raise
        published = False
        try:
            files._write_all(fd, data)
            os.fsync(fd)
            # Reject existing reparse points and hardlinks before replacement.
            try:
                current = files._open(path)
            except FileNotFoundError:
                current = None
            if current is not None:
                try:
                    info = files._FileInfo()
                    files._checked(files._info(current, C.byref(info)))
                    if info.links != 1:
                        raise UnsafePath("Image indexes must have exactly one hard link.")
                finally:
                    files._close(current)
            before_publish()
            files._publish(handle, path, replace=overwrite)
            published = True
        finally:
            try:
                if not published:
                    files._delete_on_close(handle)
            finally:
                os.close(fd)
