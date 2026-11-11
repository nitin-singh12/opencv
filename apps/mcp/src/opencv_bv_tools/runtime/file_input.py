"""Bounded regular-file reads for trusted-local command-line readers."""
import os
from pathlib import Path
import stat


def read_regular_file(path, max_bytes, error_type=ValueError):
    fd = os.open(Path(path).expanduser(), os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise error_type("Input must be a regular file.")
        if info.st_size > max_bytes:
            raise error_type(f"Input exceeds {max_bytes} bytes.")
        data = source.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise error_type(f"Input exceeds {max_bytes} bytes.")
    return data
