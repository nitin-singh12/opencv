"""Shared atomic output for trusted-local CLIs; MCP uses its access policy."""

import os
import stat
import tempfile
from pathlib import Path


class ConversionError(ValueError):
    """The input or requested conversion is unsupported or could not be decoded."""


def _check_destination(path: Path, input_stat: os.stat_result, overwrite: bool) -> None:
    try:
        existing = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(existing.st_mode):
        raise ConversionError("Output must not be a symlink or non-regular file.")
    if (existing.st_dev, existing.st_ino) == (input_stat.st_dev, input_stat.st_ino):
        raise ConversionError(
            "Input and output must be different files (including hard links)."
        )
    if not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    if not existing.st_mode & 0o222:
        raise PermissionError("Output is read-only.")


def _publish_bytes(data, destination, input_stat, overwrite, *, before_publish=None):
    """Shared trusted-directory publication; preserve the converter's semantics."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".opencv-bv-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if before_publish is not None:
            before_publish()
        _check_destination(destination, input_stat, overwrite)
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Atomic no-clobber: an output created during conversion wins.
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
