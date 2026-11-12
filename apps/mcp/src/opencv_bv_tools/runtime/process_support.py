# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Operating-system details shared by workers; no image libraries are imported."""

import contextlib
import os
import select
import selectors
import signal
import subprocess
import sys


def check_platform():
    if sys.platform not in {"linux", "darwin", "win32"}:
        raise ValueError("Use Linux, macOS or Windows for the local MCP server.")
    if sys.platform == "win32" and sys.version_info < (3, 12):
        raise ValueError("Windows requires Python 3.12 or newer for bounded pipe I/O.")


def peak_rss_kib():
    """Return peak resident memory in KiB on every platform."""
    if sys.platform == "win32":
        from .windows_process import peak_rss_kib as windows_peak
        return windows_peak()
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value / 1024) if sys.platform == "darwin" else int(value)


def limit_renderer_memory(limit):
    # Windows limits are set on the Job Object before the renderer starts.
    if sys.platform == "win32":
        return
    import resource
    for kind, value in ((resource.RLIMIT_DATA, limit), (resource.RLIMIT_AS, limit + (2 << 30))):
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(kind, (value, value))


def start_worker(command, *, pass_fds=(), memory_limit=None, **kwargs):
    """Start a worker, handing only explicitly listed PDF descriptors to it."""
    check_platform()
    if sys.platform == "win32":
        from .windows_process import start_worker as windows_start
        return windows_start(command, pass_fds=pass_fds, memory_limit=memory_limit, **kwargs)
    return subprocess.Popen(command, pass_fds=pass_fds, **kwargs)


def stop_worker(process, *, group=False, grace=0.25):
    """Reap a worker and, for OCR, its PDF helpers too."""
    if sys.platform == "win32":
        from .windows_process import stop_worker as windows_stop
        windows_stop(process)
        return
    if group:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with contextlib.suppress(ProcessLookupError):
                try:
                    os.killpg(process.pid, sig)
                except PermissionError:
                    if sys.platform != "darwin":
                        raise
                    # macOS can refuse a group signal while its exited leader
                    # is unreaped. Confirm our child's exit before one retry.
                    try:
                        process.wait(timeout=grace)
                    except subprocess.TimeoutExpired:
                        raise PermissionError("Could not stop the live worker group") from None
                    os.killpg(process.pid, sig)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=grace)
    elif process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait(timeout=grace)


class PipePoller:
    """One owner per pipe; no background readers and no unbounded buffering."""

    def __init__(self):
        if sys.platform == "win32":
            from .windows_process import PipePoller as WindowsPoller
            self._poller = WindowsPoller()
        else:
            self._poller = selectors.DefaultSelector()

    def register(self, stream, events):
        os.set_blocking(stream.fileno(), False)
        return self._poller.register(stream, events)

    def unregister(self, stream):
        return self._poller.unregister(stream)

    def get_map(self):
        return self._poller.get_map()

    def select(self, timeout):
        return self._poller.select(timeout)

    def read(self, fd, size):
        if sys.platform == "win32":
            return self._poller.read(fd, size)
        return os.read(fd, size)

    def write(self, fd, data):
        if sys.platform == "win32":
            return self._poller.write(fd, data)
        return os.write(fd, data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._poller.close()


def read_pipe(fd, size, timeout):
    """Read a bounded PDF chunk, or return None while waiting (b'' is EOF)."""
    if sys.platform == "win32":
        from .windows_process import read_pipe as windows_read
        return windows_read(fd, size, timeout)
    if select.select([fd], [], [], timeout)[0]:
        return os.read(fd, size)
    return None
