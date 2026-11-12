# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Windows worker containment and bounded binary pipes (Python 3.12+).

Workers wait on a private gate until attached to a kill-on-close Job Object.
The job also owns descendants, including PDF renderers. Pipes have one reader
or writer, use nonblocking handles, and never need background threads.
"""

import ctypes as C
from ctypes import wintypes as W
import json
import msvcrt
import os
import runpy
import selectors
import subprocess
import sys
import time


kernel = C.WinDLL("kernel32", use_last_error=True)


def _api(name, result, *arguments):
    function = getattr(kernel, name)
    function.restype, function.argtypes = result, arguments
    return function


_close = _api("CloseHandle", W.BOOL, W.HANDLE)
_current = _api("GetCurrentProcess", W.HANDLE)
_duplicate = _api("DuplicateHandle", W.BOOL, W.HANDLE, W.HANDLE, W.HANDLE,
                  C.POINTER(W.HANDLE), W.DWORD, W.BOOL, W.DWORD)
_create_job = _api("CreateJobObjectW", W.HANDLE, C.c_void_p, W.LPCWSTR)
_set_job = _api("SetInformationJobObject", W.BOOL, W.HANDLE, C.c_int, C.c_void_p, W.DWORD)
_assign = _api("AssignProcessToJobObject", W.BOOL, W.HANDLE, W.HANDLE)
_open_process = _api("OpenProcess", W.HANDLE, W.DWORD, W.BOOL, W.DWORD)
_peek = _api("PeekNamedPipe", W.BOOL, W.HANDLE, C.c_void_p, W.DWORD,
             C.POINTER(W.DWORD), C.POINTER(W.DWORD), C.POINTER(W.DWORD))
_write = _api("WriteFile", W.BOOL, W.HANDLE, C.c_void_p, W.DWORD,
              C.POINTER(W.DWORD), C.c_void_p)


class _BasicLimits(C.Structure):
    _fields_ = [("process_time", C.c_int64), ("job_time", C.c_int64), ("flags", W.DWORD),
                ("min_working_set", C.c_size_t), ("max_working_set", C.c_size_t),
                ("active_processes", W.DWORD), ("affinity", C.c_size_t),
                ("priority", W.DWORD), ("scheduling", W.DWORD)]


class _JobLimits(C.Structure):
    _fields_ = [("basic", _BasicLimits), ("io_counters", C.c_uint64 * 6),
                ("process_memory", C.c_size_t), ("job_memory", C.c_size_t),
                ("peak_process_memory", C.c_size_t), ("peak_job_memory", C.c_size_t)]


def _checked(value):
    if not value:
        raise C.WinError(C.get_last_error())
    return value


def _inheritable_copy(handle):
    duplicate = W.HANDLE()
    _checked(_duplicate(_current(), handle, _current(), C.byref(duplicate), 0, True, 2))
    return duplicate.value


def start_worker(command, *, pass_fds=(), memory_limit=None, **kwargs):
    """Gate this Python worker until containment and explicit inheritance exist."""
    if os.path.normcase(os.path.abspath(command[0])) != os.path.normcase(os.path.abspath(sys.executable)):
        raise OSError("Windows workers must use this package's Python interpreter.")
    kwargs.pop("start_new_session", None)  # Job Objects replace POSIX sessions.
    gate_read, gate_write = os.pipe()
    inherited = []
    job = process = None
    try:
        gate = _inheritable_copy(msvcrt.get_osfhandle(gate_read))
        inherited.append(gate)
        fd_map = {}
        for fd in pass_fds:
            handle = _inheritable_copy(msvcrt.get_osfhandle(fd))
            inherited.append(handle)
            fd_map[str(fd)] = handle
        job = _checked(_create_job(None, None))  # Noninheritable, owned only here.
        limits = _JobLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_limit is not None:
            limits.basic.flags |= 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
            limits.process_memory = memory_limit
        _checked(_set_job(job, 9, C.byref(limits), C.sizeof(limits)))
        startup = subprocess.STARTUPINFO()
        startup.lpAttributeList = {"handle_list": inherited}
        executable = sys.executable
        if os.path.normcase(sys.executable) != os.path.normcase(sys._base_executable):
            # Match CPython multiprocessing: bypass the venv redirector so the
            # gated interpreter itself gets the Job and the expected PID.
            executable = sys._base_executable
            kwargs["env"] = dict(kwargs.get("env") or os.environ)
            kwargs["env"]["__PYVENV_LAUNCHER__"] = sys.executable
        bootstrap = [executable, "-m", __name__, str(gate), json.dumps(fd_map), *command[1:]]
        process = subprocess.Popen(bootstrap, startupinfo=startup, close_fds=True, **kwargs)
        handle = _checked(_open_process(0x0100 | 0x0001, False, process.pid))
        try:
            _checked(_assign(job, handle))
        finally:
            _close(handle)
        process._bv_job = job
        os.write(gate_write, b"1")
        job = None  # Ownership moved to process; stop_worker closes it.
        return process
    except BaseException:
        if process is not None:
            process.kill()
            process.wait(timeout=1)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        raise
    finally:
        os.close(gate_read)
        os.close(gate_write)
        for handle in inherited:
            _close(handle)
        if job is not None:
            _close(job)


def stop_worker(process):
    job = getattr(process, "_bv_job", None)
    if job is not None:
        process._bv_job = None
        _checked(_close(job))  # Kills descendants even if the leader exited.
    elif process.poll() is None:
        process.kill()
    process.wait(timeout=1)


def _available(fd):
    available = W.DWORD()
    if _peek(msvcrt.get_osfhandle(fd), None, 0, None, C.byref(available), None):
        return available.value
    error = C.get_last_error()
    if error in {109, 233}:  # ERROR_BROKEN_PIPE / ERROR_PIPE_NOT_CONNECTED
        return -1
    if error == 232:  # Empty nonblocking pipe, not EOF.
        return 0
    raise C.WinError(error)


class PipePoller:
    def __init__(self):
        self._keys = {}
        self._progress = True
        self._write_size = 65536

    def register(self, stream, events):
        key = selectors.SelectorKey(stream, stream.fileno(), events, None)
        self._keys[key.fd] = key
        return key

    def unregister(self, stream):
        return self._keys.pop(stream.fileno())

    def get_map(self):
        return self._keys

    def select(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            reads = [(key, key.events) for key in self._keys.values()
                     if key.events == selectors.EVENT_READ and _available(key.fd) != 0]
            writes = [(key, key.events) for key in self._keys.values()
                      if key.events == selectors.EVENT_WRITE]
            # Write first so framing validation sees the updated input offset.
            if reads or (writes and self._progress):
                self._progress = False
                return writes + reads
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return writes
            time.sleep(min(0.002, remaining))
            if writes:
                return writes

    def read(self, fd, size):
        available = _available(fd)
        if available < 0:
            return b""
        if not available:
            raise BlockingIOError()
        block = os.read(fd, min(size, available))
        self._progress |= bool(block)
        return block

    def write(self, fd, data):
        # Win32 may require a whole chunk to fit even for a byte pipe. Shrink
        # after zero progress instead of assuming a particular pipe capacity.
        block = bytes(data[:self._write_size])
        written = W.DWORD()
        if not _write(msvcrt.get_osfhandle(fd), block, len(block), C.byref(written), None):
            error = C.get_last_error()
            if error in {109, 232, 233}:
                raise BrokenPipeError("Worker input pipe closed")
            raise C.WinError(error)
        if not written.value and block:
            self._write_size = max(1, self._write_size // 2)
        else:
            self._write_size = min(65536, max(self._write_size, written.value) * 2)
            self._progress |= bool(written.value)
        return written.value

    def close(self):
        self._keys.clear()


def read_pipe(fd, size, timeout):
    os.set_blocking(fd, False)
    deadline = time.monotonic() + timeout
    while True:
        available = _available(fd)
        if available < 0:
            return b""
        if available:
            return os.read(fd, min(size, available))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(0.002, remaining))


def peak_rss_kib():
    class Counters(C.Structure):
        _fields_ = [("size", W.DWORD), ("faults", W.DWORD),
                    ("peak_working_set", C.c_size_t), ("working_set", C.c_size_t),
                    ("other", C.c_size_t * 6)]
    counters = Counters()
    counters.size = C.sizeof(counters)
    get_memory = _api("K32GetProcessMemoryInfo", W.BOOL, W.HANDLE, C.c_void_p, W.DWORD)
    _checked(get_memory(_current(), C.byref(counters), counters.size))
    return counters.peak_working_set // 1024


def _bootstrap():
    gate = msvcrt.open_osfhandle(int(sys.argv[1]), os.O_RDONLY | os.O_BINARY)
    with os.fdopen(gate, "rb", buffering=0) as stream:
        if stream.read(1) != b"1":
            raise SystemExit(1)  # Parent failed/died before assigning the job.
    fd_map = json.loads(sys.argv[2])
    args = sys.argv[3:]
    for old, handle in fd_map.items():
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        os.set_inheritable(fd, False)
        # Only the descriptor immediately after the private PDF command flag.
        flag = "--draw-pages" if "--draw-pages" in args else "--redact"
        index = args.index(flag) + 1
        if args[index] != old:
            raise ValueError("Unexpected inherited PDF descriptor")
        args[index] = str(fd)
    if args[0] == "-m":
        sys.argv = args[1:]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    elif args[0] == "-c":
        sys.argv = ["-c", *args[2:]]
        exec(args[1], {"__name__": "__main__"})
    else:
        sys.argv = args
        runpy.run_path(args[0], run_name="__main__")


if __name__ == "__main__":
    _bootstrap()
