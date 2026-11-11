"""Safe reading and writing of files (screenshots and PDFs), shared by the MCP
server and the command line.

Guarantees:
  - The input is never modified, through any name: same path, symbolic
    link or hard link (checked by device and inode, not by path).
  - Only allowed folders are used. A path outside them is refused before
    anything there is opened, and every check is repeated on what was
    actually opened (a file or folder handle), so a path swapped between the
    check and the use cannot redirect it. On Linux the input is first opened
    with O_PATH, which cannot read and has no side effects, and is reopened
    for reading only after the checks pass.
  - Output is never written into an existing file. A new temporary file,
    readable only by its owner until it is complete, is written inside the
    already opened output folder and then published in one step: without
    overwrite only if the name is still free at that moment, with overwrite
    by replacing the name (other names for the old file keep their
    contents). On any failure the temporary file, and any folders created
    for it, are removed and nothing else changes.
  - Only regular files are read or replaced: no FIFOs, devices or sockets.
  - A file is only replaced if the caller could write to it, and the new
    file keeps the old file's permissions. A new file gets the normal
    permissions for the folder (umask).

Limits: on file systems without hard links (FAT, exFAT) a new file is
created directly under its final name (still never replacing anything), so
a crash can leave it incomplete. If the process is killed outright while
writing, a hidden .email-redact-*.tmp file can be left behind.

Linux is the tested platform; macOS uses F_GETPATH for the same checks.
The Windows candidate uses windows_files.py: local NTFS plain paths, private
outputs, no replacement of existing outputs pending native acceptance tests.
"""
import contextlib
import errno
import os
import secrets
import stat
import subprocess
import sys

if sys.platform != "win32":
    import fcntl
from pathlib import Path


class UnsafePath(Exception):
    """A request that would be unsafe; the message is meant for the user."""


def _fd_path(fd):
    """The real path of an open file or folder handle."""
    if os.path.isdir("/proc/self/fd"):
        return Path(os.readlink(f"/proc/self/fd/{fd}"))
    if hasattr(fcntl, "F_GETPATH"):  # macOS
        return Path(os.fsdecode(fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024)).rstrip(b"\0")))
    raise UnsafePath("This platform cannot verify which file was opened.")


def _real_folder(folder):
    """A folder's real path, in the same form _fd_path gives for handles."""
    if sys.platform == "win32":
        from .windows_files import real_folder
        return real_folder(folder)
    try:
        fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError:
        return Path(os.path.realpath(folder))  # missing for now: nothing inside can match yet
    try:
        return _fd_path(fd)
    finally:
        os.close(fd)


def _umask():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError):
        pass
    # Read the inherited mask in a small isolated child. Changing the parent
    # process's umask, even briefly, races with concurrent file creation.
    mask = int(subprocess.check_output(
        [sys.executable, "-I", "-S", "-c", "import os; print(os.umask(0o777))"],
        text=True, timeout=5).strip())
    if not 0 <= mask <= 0o777:
        raise UnsafePath("Could not determine output file permissions.")
    return mask


class Folders:
    """The folders a caller may use. None means any path (command line); an
    empty list means none at all."""

    def __init__(self, allowed):
        self.allowed = None if allowed is None else [_real_folder(Path(d).expanduser()) for d in allowed]

    def check(self, real, what):
        """what names the path as the caller gave it; the resolved path is
        never shown, so a link cannot reveal where it points."""
        if self.allowed is None:
            return
        if not any(real == d or d in real.parents for d in self.allowed):
            names = ", ".join(str(d) for d in self.allowed) or "none"
            raise UnsafePath(f"{what} is outside the folders this server may use ({names}).")


def identity(st):
    return (st.st_dev, st.st_ino)


def open_input(path, folders):
    """Open a file for reading. Returns (file object, real path,
    identity). The checks are made on the opened file itself."""
    if sys.platform == "win32":
        from .windows_files import open_input as windows_open_input
        return windows_open_input(path, folders)
    p = Path(path).expanduser()
    folders.check(Path(os.path.realpath(p)), f"The input file {p}")  # nothing outside is even opened
    # O_PATH (Linux) gives a handle that cannot read, so opening it has no
    # side effects even if the path was swapped meanwhile. Otherwise:
    # O_NONBLOCK so a FIFO cannot hang, O_NOCTTY so a terminal is not adopted.
    reading = os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY | os.O_CLOEXEC
    probe = os.O_PATH | os.O_CLOEXEC if hasattr(os, "O_PATH") and os.path.isdir("/proc/self/fd") else None
    try:
        fd = os.open(p, probe or reading)
    except FileNotFoundError:
        raise UnsafePath(f"No such file: {p}") from None
    except PermissionError:
        raise UnsafePath(f"No permission to read {p}.") from None
    except OSError as exc:
        raise UnsafePath(f"Cannot open {p}: {exc.strerror}.") from None
    try:
        st = os.fstat(fd)
        if stat.S_ISDIR(st.st_mode):
            raise UnsafePath(f"{p} is a folder, not a file.")
        if not stat.S_ISREG(st.st_mode):
            raise UnsafePath(f"{p} is not a regular file.")
        real = _fd_path(fd)
        folders.check(real, f"The input file {p}")
        if probe:
            # The checks passed; open that very file (not the path) for reading.
            try:
                rfd = os.open(f"/proc/self/fd/{fd}", reading)
            except PermissionError:
                raise UnsafePath(f"No permission to read {p}.") from None
            os.close(fd)
            fd = rfd
            if identity(os.fstat(fd)) != identity(st):
                raise UnsafePath(f"{p} changed while it was being opened.")
        os.set_blocking(fd, True)
        return os.fdopen(fd, "rb"), real, identity(st)
    except BaseException:
        os.close(fd)
        raise


def _open_folder(folder, folders, create, dry_run=False):
    """Open the output folder as a handle, creating missing folders one at a
    time (never following links) when create is true. Returns the handle and
    the folders it created. With dry_run, only check that the folder would be
    allowed; nothing is created and (None, []) is returned."""
    given = Path(folder).expanduser()
    folder = Path(os.path.realpath(given))  # as the system resolves it, links and '..' included
    folders.check(folder, f"The output folder {given}")  # nothing outside is even looked at
    existing, missing = folder, []
    while not existing.exists():
        if existing.parent == existing:
            break
        missing.insert(0, existing.name)
        existing = existing.parent
    if missing and not create:
        raise UnsafePath(f"The folder {given} does not exist.")
    try:
        fd = os.open(existing, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except (FileNotFoundError, NotADirectoryError):
        raise UnsafePath(f"{given} is not a folder.") from None
    created = []
    try:
        real = _fd_path(fd)
        folders.check(real / Path(*missing) if missing else real, f"The output folder {given}")
        if dry_run:
            os.close(fd)
            return None, []
        for part in missing:
            try:
                os.mkdir(part, 0o777, dir_fd=fd)
                created.append(_fd_path(fd) / part)
            except FileExistsError:
                pass
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            except OSError:
                raise UnsafePath(f"Cannot use {part} in {given} as a folder (it is a link or a file).") from None
            os.close(fd)
            fd = child
        folders.check(_fd_path(fd), f"The output folder {given}")
        return fd, created
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        _remove_folders(created)
        raise


def _remove_folders(created):
    for folder in reversed(created):
        with contextlib.suppress(OSError):
            os.rmdir(folder)  # only ever removes an empty folder


def _target(name, dir_fd, protect, overwrite):
    """Check what currently sits at name inside the folder. Returns its stat,
    or None when the name is free."""
    try:
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(st.st_mode):
        raise UnsafePath(f"{name} is a symbolic link; give a plain file name.")
    if stat.S_ISDIR(st.st_mode):
        raise UnsafePath(f"{name} is a folder; give a file name.")
    if not stat.S_ISREG(st.st_mode):
        raise UnsafePath(f"{name} is not a regular file.")
    if identity(st) in protect:
        raise UnsafePath(f"{name} is the input file, or another name for it such as a hard link; "
                         "the original is never overwritten.")
    if not overwrite:
        raise UnsafePath(f"{name} already exists; it is only replaced when overwrite is on "
                         "(overwrite=true, or --overwrite on the command line).")
    if not os.access(name, os.W_OK, dir_fd=dir_fd, follow_symlinks=False):
        raise UnsafePath(f"{name} is read-only, so it is not replaced.")
    return st


def _name(path, suffix):
    raw = Path(path).expanduser()
    name = raw.name
    if not name or name in (".", "..") or "\0" in str(path):
        raise UnsafePath(f"{path} is not a valid file name.")
    if suffix and not name.lower().endswith(suffix):
        why = " (a lossless format keeps the boxes solid)" if suffix == ".png" else ""
        raise UnsafePath(f"The output must end in {suffix}{why}.")
    return raw, name


def check_output(path, folders, protect=(), overwrite=False, suffix=".png"):
    """Cheap early check, so a request that would be refused fails before the
    slow work. write_file repeats every check when it writes."""
    if sys.platform == "win32":
        from .windows_files import check_output as windows_check_output
        return windows_check_output(path, folders, protect, overwrite, suffix)
    raw, name = _name(path, suffix)
    if not Path(os.path.realpath(raw.parent)).exists():
        _open_folder(raw.parent, folders, create=True, dry_run=True)
        return
    dfd, _ = _open_folder(raw.parent, folders, create=False)
    try:
        _target(name, dfd, set(protect), overwrite)
    finally:
        os.close(dfd)


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def write_file(data, path, folders, protect=(), overwrite=False, create_folders=True, suffix=".png", *, before_publish=None):
    """Write data (bytes, or a list of byte strings written in order) to
    path atomically and safely (see the module notes). protect: identities (device, inode) that must never be replaced, such as
    the input. Returns (real path written, identity of the new file)."""
    if sys.platform == "win32":
        from .windows_files import write_file as windows_write_file
        return windows_write_file(data, path, folders, protect, overwrite, create_folders, suffix, before_publish=before_publish)
    raw, name = _name(path, suffix)
    protect = set(protect)
    dfd, created = _open_folder(raw.parent, folders, create_folders)
    tmp = f".email-redact-{secrets.token_hex(8)}.tmp"
    published = False
    try:
        old = _target(name, dfd, protect, overwrite)
        tfd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=dfd)
        try:
            for chunk in [data] if isinstance(data, (bytes, bytearray, memoryview)) else data:
                _write_all(tfd, chunk)
            # A replaced file keeps its permissions; a new one gets the usual ones.
            os.fchmod(tfd, stat.S_IMODE(old.st_mode) & 0o777 if old is not None else 0o666 & ~_umask())
            os.fsync(tfd)
            if before_publish is not None:
                before_publish()
            written = identity(os.fstat(tfd))
        finally:
            os.close(tfd)
        _target(name, dfd, protect, overwrite)  # again, right before publishing
        if overwrite:
            os.replace(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd)
        else:
            try:
                os.link(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd, follow_symlinks=False)
            except FileExistsError:
                raise UnsafePath(f"{name} already exists; it is only replaced when overwrite is on "
                                 "(overwrite=true, or --overwrite on the command line).") from None
            except OSError as exc:
                if exc.errno not in (errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOSYS):
                    raise
                written = _exclusive_copy(tmp, name, dfd)  # file system without hard links (FAT, exFAT)
        published = True
        with contextlib.suppress(OSError):
            os.fsync(dfd)  # the file is already in place; some file systems cannot sync folders
        return _fd_path(dfd) / name, written
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp, dir_fd=dfd)
        os.close(dfd)
        if not published:
            _remove_folders(created)


def _exclusive_copy(tmp, name, dfd):
    try:
        out = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=dfd)
    except FileExistsError:
        raise UnsafePath(f"{name} already exists; it is only replaced when overwrite is on "
                         "(overwrite=true, or --overwrite on the command line).") from None
    try:
        src = os.open(tmp, os.O_RDONLY | os.O_CLOEXEC, dir_fd=dfd)
        try:
            while chunk := os.read(src, 1 << 20):
                _write_all(out, chunk)
            os.fchmod(out, stat.S_IMODE(os.fstat(src).st_mode))
        finally:
            os.close(src)
        os.fsync(out)
        written = identity(os.fstat(out))
    except BaseException:
        os.close(out)
        with contextlib.suppress(OSError):
            os.unlink(name, dir_fd=dfd)  # only the partial file this call created
        raise
    os.close(out)
    return written
