"""Conservative native Windows file access for the MCP server.

The first native candidate accepts plain paths on local NTFS volumes. It
rejects reparse points, network/device paths and alternate data streams.
Ancestor handles prevent directory renames during access. A private temporary
file is published by its handle, with no-clobber semantics. Existing outputs
are refused until overwrite/ACL parity has passed native acceptance tests.
"""

import contextlib
import ctypes as C
from ctypes import wintypes as W
import msvcrt
import os
from pathlib import Path, PureWindowsPath
import secrets

from .safe_files import UnsafePath, identity, _name, _write_all

kernel = C.WinDLL("kernel32", use_last_error=True)
security = C.WinDLL("advapi32", use_last_error=True)


def _api(library, name, result, *arguments):
    function = getattr(library, name)
    function.restype, function.argtypes = result, arguments
    return function


_create = _api(kernel, "CreateFileW", W.HANDLE, W.LPCWSTR, W.DWORD, W.DWORD,
               C.c_void_p, W.DWORD, W.DWORD, W.HANDLE)
_close = _api(kernel, "CloseHandle", W.BOOL, W.HANDLE)
_path = _api(kernel, "GetFinalPathNameByHandleW", W.DWORD, W.HANDLE, W.LPWSTR, W.DWORD, W.DWORD)
_long_path = _api(kernel, "GetLongPathNameW", W.DWORD, W.LPCWSTR, W.LPWSTR, W.DWORD)
_info = _api(kernel, "GetFileInformationByHandle", W.BOOL, W.HANDLE, C.c_void_p)
_type = _api(kernel, "GetFileType", W.DWORD, W.HANDLE)
_set_info = _api(kernel, "SetFileInformationByHandle", W.BOOL, W.HANDLE, C.c_int, C.c_void_p, W.DWORD)
_volume = _api(kernel, "GetVolumeInformationByHandleW", W.BOOL, W.HANDLE, W.LPWSTR, W.DWORD,
               C.c_void_p, C.c_void_p, C.c_void_p, W.LPWSTR, W.DWORD)
_current = _api(kernel, "GetCurrentProcess", W.HANDLE)
_free = _api(kernel, "LocalFree", W.HANDLE, W.HANDLE)
_token = _api(security, "OpenProcessToken", W.BOOL, W.HANDLE, W.DWORD, C.POINTER(W.HANDLE))
_token_info = _api(security, "GetTokenInformation", W.BOOL, W.HANDLE, C.c_int,
                   C.c_void_p, W.DWORD, C.POINTER(W.DWORD))
_sid_string = _api(security, "ConvertSidToStringSidW", W.BOOL, C.c_void_p, C.POINTER(W.LPWSTR))
_descriptor = _api(security, "ConvertStringSecurityDescriptorToSecurityDescriptorW",
                   W.BOOL, W.LPCWSTR, W.DWORD, C.POINTER(C.c_void_p), C.c_void_p)

READ_ATTRIBUTES = 0x80
GENERIC_READ, GENERIC_WRITE, DELETE = 0x80000000, 0x40000000, 0x10000
SHARE_READ_WRITE = 3  # Deliberately excludes FILE_SHARE_DELETE.
OPEN_REPARSE, BACKUP_SEMANTICS = 0x00200000, 0x02000000
INVALID_HANDLE = C.c_void_p(-1).value


class _FileInfo(C.Structure):
    _fields_ = [("attributes", W.DWORD), ("times", W.FILETIME * 3),
                ("volume", W.DWORD), ("size_high", W.DWORD), ("size_low", W.DWORD),
                ("links", W.DWORD), ("index_high", W.DWORD), ("index_low", W.DWORD)]


class _SecurityAttributes(C.Structure):
    _fields_ = [("size", W.DWORD), ("descriptor", C.c_void_p), ("inherit", W.BOOL)]


def _checked(value):
    if not value:
        raise C.WinError(C.get_last_error())
    return value


def normalize_path(value):
    """Accept unambiguous local paths before touching the filesystem."""
    text = os.fspath(Path(value).expanduser()).replace("/", "\\")
    parsed = PureWindowsPath(text)
    if text.startswith("\\") or (parsed.drive and not parsed.root):
        raise UnsafePath("Use a full local drive path, such as C:\\Users\\you\\Pictures\\image.png.")
    for part in parsed.parts[1:] if parsed.anchor else parsed.parts:
        if (part in {".", ".."} or part.endswith((".", " ")) or PureWindowsPath(part).is_reserved()
                or any(ch in part for ch in '<>:"|?*') or any(ord(ch) < 32 for ch in part)):
            raise UnsafePath("Use a plain Windows path without device names, streams or ambiguous components.")
    absolute = Path(os.path.abspath(text))
    if len(str(absolute)) >= 240:
        raise UnsafePath("This Windows candidate requires paths shorter than 240 characters.")
    # TEMP and user-selected folders can use 8.3 aliases (e.g. RUNNER~1).
    # Expand spelling only; every ancestor is still opened and checked below.
    existing, missing = absolute, []
    buffer = C.create_unicode_buffer(32768)
    while True:
        length = _long_path(str(existing), buffer, len(buffer))
        if length:
            if length >= len(buffer):
                raise UnsafePath("Could not expand the Windows path.")
            absolute = Path(buffer.value).joinpath(*reversed(missing))
            break
        error = C.get_last_error()
        if error not in (2, 3):  # Only missing files/folders allow a shorter prefix.
            raise C.WinError(error)
        if existing.parent == existing:
            raise UnsafePath(f"The drive for {absolute} is not available.")
        missing.append(existing.name)
        existing = existing.parent
    if len(str(absolute)) >= 240:
        raise UnsafePath("This Windows candidate requires paths shorter than 240 characters.")
    return absolute


def _open(path, access=READ_ATTRIBUTES, *, directory=False, create=False, descriptor=None):
    attributes = None
    if descriptor is not None:
        attributes = _SecurityAttributes(C.sizeof(_SecurityAttributes), descriptor, False)
    handle = _create(str(path), access, SHARE_READ_WRITE, C.byref(attributes) if attributes else None,
                     1 if create else 3, OPEN_REPARSE | (BACKUP_SEMANTICS if directory else 0), None)
    if handle == INVALID_HANDLE:
        raise C.WinError(C.get_last_error())
    try:
        info = _FileInfo()
        _checked(_info(handle, C.byref(info)))
        if info.attributes & 0x400:
            raise UnsafePath("Windows links, junctions and other reparse points are not supported yet.")
        if bool(info.attributes & 0x10) != directory or _type(handle) != 1:
            raise UnsafePath("Use a regular file, or a plain directory for a folder.")
        return handle
    except BaseException:
        _close(handle)
        raise


def _real_path(handle):
    buffer = C.create_unicode_buffer(32768)
    length = _checked(_path(handle, buffer, len(buffer), 0))
    if length >= len(buffer) or not buffer.value.startswith("\\\\?\\"):
        raise UnsafePath("Could not verify the opened Windows path.")
    return normalize_path(buffer.value[4:])


@contextlib.contextmanager
def _directories(folder, *, create=False):
    """Pin every ancestor; no directory in the chain may be renamed meanwhile."""
    handles, created = [], []
    succeeded = False
    try:
        current = Path(folder.anchor)
        for part in [None, *folder.parts[1:]]:
            if part is not None:
                current /= part
            try:
                handle = _open(current, directory=True)
            except FileNotFoundError:
                if not create:
                    raise UnsafePath(f"The folder {folder} does not exist.") from None
                try:
                    current.mkdir()
                    created.append(current)
                except FileExistsError:
                    pass
                handle = _open(current, directory=True)
            handles.append(handle)
            if _real_path(handle) != current:
                raise UnsafePath("The folder changed while it was being opened.")
        filesystem = C.create_unicode_buffer(32)
        _checked(_volume(handles[-1], None, 0, None, None, None, filesystem, len(filesystem)))
        if filesystem.value != "NTFS":
            raise UnsafePath("This Windows candidate requires a local NTFS volume.")
        yield handles[-1]
        succeeded = True
    finally:
        for handle in reversed(handles):
            _close(handle)
        if not succeeded:
            for path in reversed(created):
                with contextlib.suppress(OSError):
                    path.rmdir()


def real_folder(folder):
    path = normalize_path(folder)
    if path.exists():
        with _directories(path):
            pass
    return path


def open_input(path, folders):
    path = normalize_path(path)
    folders.check(path, f"The input file {path}")
    with _directories(path.parent):
        handle = _open(path, GENERIC_READ)
        try:
            real = _real_path(handle)
            folders.check(real, f"The input file {path}")
            fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            _close(handle)
            raise
        try:
            return os.fdopen(fd, "rb"), real, identity(os.fstat(fd))
        except BaseException:
            os.close(fd)
            raise


def _target(path, protect, overwrite):
    try:
        handle = _open(path)
    except FileNotFoundError:
        return
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        _close(handle)
        raise
    try:
        if identity(os.fstat(fd)) in protect:
            raise UnsafePath("The output is the input file or its hard link; the original is never overwritten.")
    finally:
        os.close(fd)
    if overwrite:
        raise UnsafePath("Replacing existing outputs is not enabled in this Windows candidate. Choose a new output name.")
    raise UnsafePath(f"{path.name} already exists. Choose a new output name.")


def check_output(path, folders, protect=(), overwrite=False, suffix=".png"):
    raw, _ = _name(path, suffix)
    path = normalize_path(raw)
    folders.check(path.parent, f"The output folder {path.parent}")
    existing = path.parent
    while not existing.exists():
        if existing.parent == existing:
            raise UnsafePath(f"The drive for {path} is not available.")
        existing = existing.parent
    with _directories(existing):
        if existing == path.parent:
            _target(path, protect, overwrite)


@contextlib.contextmanager
def _private_descriptor():
    """Protected DACL: the current account alone can access the new output."""
    token, sid, descriptor = W.HANDLE(), W.LPWSTR(), C.c_void_p()
    try:
        _checked(_token(_current(), 8, C.byref(token)))  # TOKEN_QUERY
        size = W.DWORD()
        _token_info(token, 1, None, 0, C.byref(size))  # TokenUser
        buffer = C.create_string_buffer(size.value)
        _checked(_token_info(token, 1, buffer, size, C.byref(size)))
        _checked(_sid_string(C.cast(buffer, C.POINTER(C.c_void_p))[0], C.byref(sid)))
        _checked(_descriptor(f"D:P(A;;FA;;;{sid.value})", 1, C.byref(descriptor), None))
        yield descriptor
    finally:
        if descriptor.value:
            _free(descriptor)
        if sid:
            _free(C.cast(sid, W.HANDLE))
        if token.value:
            _close(token)


def _publish(handle, path):
    # Rename this open file, never a re-looked-up temporary pathname. FALSE
    # means an entry appearing at the output name wins and is never replaced.
    encoded = str(path).encode("utf-16-le")
    class RenameInfo(C.Structure):
        _fields_ = [("replace", W.BOOL), ("root", W.HANDLE), ("length", W.DWORD),
                    ("name", C.c_byte * (len(encoded) + 2))]
    info = RenameInfo(False, None, len(encoded))
    C.memmove(C.addressof(info) + RenameInfo.name.offset, encoded, len(encoded))
    # Win32 path conversion expects a terminated string even though the
    # byte count excludes its terminator. Keep the two extra bytes zeroed.
    _checked(_set_info(handle, 3, C.byref(info), C.sizeof(info)))  # FileRenameInfo
    if _real_path(handle) != path:
        raise UnsafePath("The output was not published at the requested path.")


def _delete_on_close(handle):
    delete = W.BOOL(True)
    _checked(_set_info(handle, 4, C.byref(delete), C.sizeof(delete)))  # FileDispositionInfo


def write_file(data, path, folders, protect=(), overwrite=False, create_folders=True,
               suffix=".png", *, before_publish=None):
    raw, _ = _name(path, suffix)
    path = normalize_path(raw)
    folders.check(path.parent, f"The output folder {path.parent}")
    with _directories(path.parent, create=create_folders):
        _target(path, protect, overwrite)
        temporary = path.parent / f".opencv-bv-{secrets.token_hex(8)}.tmp"
        with _private_descriptor() as descriptor:
            handle = _open(temporary, GENERIC_READ | GENERIC_WRITE | DELETE,
                           create=True, descriptor=descriptor)
        try:
            fd = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
        except BaseException:
            try:
                _delete_on_close(handle)
            finally:
                _close(handle)
            raise
        published = False
        try:
            for chunk in [data] if isinstance(data, (bytes, bytearray, memoryview)) else data:
                _write_all(fd, chunk)
            os.fsync(fd)
            if before_publish is not None:
                before_publish()
            _target(path, protect, overwrite)
            written = identity(os.fstat(fd))
            _publish(handle, path)
            published = True
            return path, written
        finally:
            try:
                if not published:
                    _delete_on_close(handle)
            finally:
                os.close(fd)
