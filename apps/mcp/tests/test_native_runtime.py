"""Portable worker contracts plus tests that must run on native Windows.

Linux results do not validate skipped Win32 tests. Run these from the installed
candidate environment on each OS, followed by scripts/smoke_mcp.py.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from opencv_bv_tools.runtime import safe_files
from opencv_bv_tools.runtime.process_support import peak_rss_kib, start_worker, stop_worker
from opencv_bv_tools.runtime.worker_process import _run_worker

WINDOWS = pytest.mark.skipif(sys.platform != "win32", reason="requires actual Windows APIs")


@pytest.mark.parametrize("target,architecture,directory,suffix", [
    ("linux", "x86_64", "bin", ""), ("darwin", "arm64", "bin", ""),
    ("win32", "AMD64", "Scripts", ".exe"),
])
def test_installer_paths_and_target_selection(tmp_path, monkeypatch, target, architecture, directory, suffix):
    # Configuration unit test only; this does not emulate an OS or run its APIs.
    path = Path(__file__).resolve().parents[1] / "scripts/install_mcp.py"
    spec = importlib.util.spec_from_file_location("native_installer_test", path)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    monkeypatch.setattr(installer, "sys", SimpleNamespace(platform=target, version_info=(3, 12)))
    monkeypatch.setattr(installer.platform, "machine", lambda: architecture)
    installer.check_platform()
    assert installer.executable(tmp_path, "python") == tmp_path / directory / ("python" + suffix)
    assert installer.executable(tmp_path, "opencv-bv-mcp") == tmp_path / directory / ("opencv-bv-mcp" + suffix)
    if target == "win32":
        environment = tmp_path / ".venvs" / "checked"
        environment.mkdir(parents=True)
        installer.promote(tmp_path, environment, "fingerprint")
        assert not (tmp_path / ".venv").exists()
        assert installer.installed_environment(tmp_path, "fingerprint") is None  # No fake executable is trusted.
        assert json.loads((tmp_path / ".install-state.json").read_text())["environment"] == str(environment.relative_to(tmp_path))


def test_intel_macos_install_fails_before_dependency_download(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/install_mcp.py"
    spec = importlib.util.spec_from_file_location("native_installer_intel_test", path)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    monkeypatch.setattr(installer, "sys", SimpleNamespace(platform="darwin", version_info=(3, 12)))
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    with pytest.raises(installer.InstallError, match="Intel macOS is not supported"):
        installer.check_platform()


def test_large_binary_pipe_transfer_is_exact():
    data = bytes(range(256)) * 32768
    code = """
import hashlib,json,sys
header=json.loads(sys.stdin.buffer.readline())
data=sys.stdin.buffer.read(header['input_bytes'])
print(json.dumps({'sha256':hashlib.sha256(data).hexdigest()}),flush=True)
"""
    def validate(result, limits):
        assert result["sha256"] == hashlib.sha256(data).hexdigest()
    result = _run_worker(data, {"input_bytes": len(data), "limits": {}}, timeout=20,
                         command=[sys.executable, "-c", code], response_validator=validate)
    assert result.get("sha256") == hashlib.sha256(data).hexdigest(), result


@pytest.mark.parametrize("code,reason", [
    ("import time;time.sleep(20)", "deadline_exceeded"),
    ("import sys,time;sys.stderr.write('x'*131072);sys.stderr.flush();time.sleep(20)", "worker_diagnostic_limit"),
    ("import sys;sys.exit(2)", "worker_failed"),
])
def test_uncooperative_worker_is_bounded(code, reason):
    started = time.monotonic()
    result = _run_worker(b"x" * (4 << 20), {"limits": {}}, timeout=1,
                         command=[sys.executable, "-c", code])
    assert result["stop_reason"] == reason, result
    assert time.monotonic() - started < 5


def test_repeated_timeout_has_no_process_or_thread_leak():
    process = psutil.Process()
    before = {p.pid for p in process.children(recursive=True)}
    threads = process.num_threads()
    for _ in range(5):
        result = _run_worker(b"x", {"limits": {}}, timeout=0.15,
                             command=[sys.executable, "-c", "import time;time.sleep(20)"])
        assert result["stop_reason"] == "deadline_exceeded"
    assert {p.pid for p in process.children(recursive=True)} <= before
    assert process.num_threads() <= threads


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.parametrize("helper", [False, True])
def test_cleanup_after_group_leader_exits(helper):
    # Keep the leader unreaped; an optional helper ignores TERM and needs KILL.
    code = """
import subprocess,sys
if sys.argv[1] == 'True':
    child = subprocess.Popen([sys.executable, '-c',
        "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(30)"],
        stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == 'ready'
    print(child.pid, flush=True)
"""
    process = start_worker([sys.executable, "-c", code, str(helper)],
                           stdout=subprocess.PIPE, text=True, start_new_session=True)
    child_pid = None
    try:
        output = process.stdout.read()  # EOF confirms exit without wait/reap.
        child_pid = int(output) if helper else None
        stop_worker(process, group=True)
        assert process.returncode == 0
        if child_pid:
            deadline = time.monotonic() + 3
            while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
                if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                    break
                time.sleep(0.02)
            assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
    finally:
        process.stdout.close()
        if child_pid:
            try:
                psutil.Process(child_pid).kill()
            except psutil.NoSuchProcess:
                pass
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal error handling")
def test_live_group_permission_error_is_not_hidden(monkeypatch):
    from opencv_bv_tools.runtime import process_support
    process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    def denied(*args):
        raise PermissionError("denied")
    try:
        monkeypatch.setattr(process_support, "sys", SimpleNamespace(platform="darwin"))
        monkeypatch.setattr(process_support.os, "killpg", denied)
        with pytest.raises(PermissionError, match="live worker"):
            process_support.stop_worker(process, group=True, grace=0.05)
    finally:
        process.kill()
        process.wait(timeout=3)


def test_pipe_backpressure_with_slow_reader():
    data = bytes(range(256)) * 2048
    code = """
import hashlib,json,sys,time
header=json.loads(sys.stdin.buffer.readline())
remaining=header['input_bytes']; digest=hashlib.sha256()
while remaining:
    chunk=sys.stdin.buffer.read(min(2048,remaining))
    if not chunk: raise RuntimeError('early EOF')
    digest.update(chunk); remaining-=len(chunk); time.sleep(0.002)
print(json.dumps({'sha256':digest.hexdigest()}),flush=True)
"""
    result = _run_worker(data, {"input_bytes": len(data), "limits": {}}, timeout=10,
                         command=[sys.executable, "-c", code], response_validator=lambda *args: None)
    assert result.get("sha256") == hashlib.sha256(data).hexdigest(), result


def test_peak_memory_has_consistent_units():
    # Some POSIX launch paths retain the parent's historical high-water mark.
    # Allow that overhead while still catching a bytes-versus-KiB factor of 1024.
    parent_rss = psutil.Process().memory_info().rss
    code = """
import json,psutil
from opencv_bv_tools.runtime.process_support import peak_rss_kib
data=bytearray(24*1024*1024)
print(json.dumps([peak_rss_kib()*1024, psutil.Process().memory_info().rss]))
"""
    peak, current = json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))
    assert current * 0.8 <= peak <= max(parent_rss, current) * 16


def test_new_output_and_cancelled_publication(tmp_path):
    folder = tmp_path / "images with spaces café"
    folder.mkdir()
    policy = safe_files.Folders([folder])
    source = folder / "original.png"
    source.write_bytes(b"original")
    stream, real, ident = safe_files.open_input(source, policy)
    with stream:
        assert stream.read() == b"original"
    output = folder / "new" / "result.png"
    path, _ = safe_files.write_file(b"result", output, policy, {ident})
    assert path == output and output.read_bytes() == b"result"
    with pytest.raises(safe_files.UnsafePath):
        safe_files.write_file(b"replace", source, policy, {ident}, overwrite=True)
    def cancel():
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        safe_files.write_file(b"partial", folder / "failed" / "out.png", policy, before_publish=cancel)
    assert not (folder / "failed").exists()
    assert source.read_bytes() == b"original"
    assert not list(folder.rglob("*.tmp"))


@WINDOWS
@pytest.mark.parametrize("path", [r"C:relative.png", r"\\server\share\file.png",
                                  r"\\?\C:\file.png", r"C:\image.png:hidden",
                                  r"C:\CON\file.png", r"C:\file.png.", r"C:\a\..\file.png"])
def test_windows_ambiguous_paths_are_refused(path):
    from opencv_bv_tools.runtime.windows_files import normalize_path
    with pytest.raises(safe_files.UnsafePath):
        normalize_path(path)


@WINDOWS
def test_windows_publication_survives_handle_close(tmp_path, monkeypatch):
    from opencv_bv_tools.runtime import windows_files as win
    publish = win._publish
    observed = {}
    def inspect_publication(handle, destination):
        publish(handle, destination)
        observed.update(handle_path=str(win._real_path(handle)), destination=str(destination),
                        entries_before_close=sorted(p.name for p in tmp_path.iterdir()))
        print(json.dumps(observed), flush=True)
    monkeypatch.setattr(win, "_publish", inspect_publication)
    output = tmp_path / "published.png"
    returned, _ = safe_files.write_file(b"published", output, safe_files.Folders([tmp_path]))
    observed.update(returned=str(returned), entries_after_close=sorted(p.name for p in tmp_path.iterdir()))
    print(json.dumps(observed), flush=True)
    assert output.is_file(), observed
    assert Path(observed["handle_path"]) == returned == output
    assert output.read_bytes() == b"published"


@WINDOWS
@pytest.mark.parametrize("name", ["x.png", "small.png", "longer-output.png", "résultat-顔-🙂.png"])
def test_windows_published_filename_is_exact(tmp_path, name):
    policy = safe_files.Folders([tmp_path])
    for index in range(8):
        target = tmp_path / (str(index) + name)
        output, _ = safe_files.write_file(b"result", target, policy)
        assert output == target and target.read_bytes() == b"result"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(str(i) + name for i in range(8))


@WINDOWS
def test_windows_hardlink_and_existing_output_refused(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"original")
    alias = tmp_path / "alias.png"
    alias.hardlink_to(source)
    policy = safe_files.Folders([tmp_path])
    identity = safe_files.identity(source.stat())
    for output in (source, alias):
        with pytest.raises(safe_files.UnsafePath, match="original"):
            safe_files.write_file(b"bad", output, policy, {identity}, overwrite=True)
    existing = tmp_path / "existing.png"
    existing.write_bytes(b"keep")
    with pytest.raises(safe_files.UnsafePath, match="not enabled"):
        safe_files.write_file(b"bad", existing, policy, overwrite=True)
    assert source.read_bytes() == b"original" and existing.read_bytes() == b"keep"


@pytest.fixture
def windows_alias_folder():
    import tempfile
    from opencv_bv_tools.runtime import windows_files as win
    short_name = win._api(win.kernel, "GetShortPathNameW", win.W.DWORD,
                         win.W.LPCWSTR, win.W.LPWSTR, win.W.DWORD)
    # Use the real Windows TEMP location, where the startup failure occurred.
    with tempfile.TemporaryDirectory(prefix="mcp alias café ") as directory:
        folder = Path(directory).resolve()
        buffer = win.C.create_unicode_buffer(32768)
        win._checked(short_name(str(folder), buffer, len(buffer)))
        alias = Path(buffer.value)
        if alias == folder:
            pytest.skip("8.3 names unavailable; alias acceptance remains untested")
        yield folder, alias


@WINDOWS
@pytest.mark.parametrize("allow_alias", [False, True])
def test_windows_short_and_long_paths_share_policy(windows_alias_folder, allow_alias):
    folder, alias = windows_alias_folder
    source = folder / "original.png"
    source.write_bytes(b"original")
    policy = safe_files.Folders([alias if allow_alias else folder])
    stream, real, ident = safe_files.open_input(source if allow_alias else alias / source.name, policy)
    with stream:
        assert stream.read() == b"original"
    assert real == source
    output, _ = safe_files.write_file(b"result", alias / "new" / "nested" / "out.png", policy, {ident})
    assert output == folder / "new" / "nested" / "out.png"
    assert output.read_bytes() == b"result" and source.read_bytes() == b"original"
    with pytest.raises(safe_files.UnsafePath, match="original"):
        safe_files.write_file(b"bad", alias / source.name, policy, {ident}, overwrite=True)


@WINDOWS
def test_windows_junction_under_short_alias_is_refused(windows_alias_folder):
    folder, alias = windows_alias_folder
    outside, allowed = folder / "outside", folder / "allowed"
    outside.mkdir()
    allowed.mkdir()
    (outside / "input.png").write_bytes(b"private")
    link = allowed / "junction"
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                            capture_output=True, text=True, timeout=5)
    if result.returncode:
        pytest.skip("Junction creation unavailable; alias/junction acceptance remains untested")
    try:
        policy = safe_files.Folders([allowed])
        alias_link = alias / "allowed" / "junction"
        with pytest.raises(safe_files.UnsafePath, match="reparse"):
            safe_files.open_input(alias_link / "input.png", policy)
        with pytest.raises(safe_files.UnsafePath, match="reparse"):
            safe_files.write_file(b"bad", alias_link / "output.png", policy)
        assert not (outside / "output.png").exists()
    finally:
        link.rmdir()


@WINDOWS
def test_windows_directory_cannot_move_during_publication(tmp_path):
    folder = tmp_path / "pinned"
    folder.mkdir()
    policy = safe_files.Folders([folder])
    def attempt_swap():
        with pytest.raises(OSError):
            folder.rename(tmp_path / "moved")
    safe_files.write_file(b"safe", folder / "output.png", policy, before_publish=attempt_swap)
    assert (folder / "output.png").read_bytes() == b"safe"


@WINDOWS
def test_windows_concurrent_output_wins_without_clobber(tmp_path):
    path = tmp_path / "output.png"
    with pytest.raises((OSError, safe_files.UnsafePath)):
        safe_files.write_file(b"ours", path, safe_files.Folders([tmp_path]),
                              before_publish=lambda: path.write_bytes(b"theirs"))
    assert path.read_bytes() == b"theirs"
    assert not list(tmp_path.glob("*.tmp"))


@WINDOWS
def test_windows_job_assignment_failure_never_runs_worker(tmp_path, monkeypatch):
    from opencv_bv_tools.runtime import windows_process
    marker = tmp_path / "must-not-exist"
    monkeypatch.setattr(windows_process, "_assign", lambda *args: False)
    with pytest.raises(OSError):
        start_worker([sys.executable, "-c", f"from pathlib import Path;Path({str(marker)!r}).touch()"])
    assert not marker.exists()


@WINDOWS
def test_windows_worker_identity_environment_and_job():
    from opencv_bv_tools.runtime import windows_process as win
    # Record whether the ordinary venv launcher changes PID on this machine.
    plain = subprocess.Popen([sys.executable, "-c", "import os;print(os.getpid())"], stdout=subprocess.PIPE)
    plain_output, _ = plain.communicate(timeout=5)
    print(json.dumps({"launcher_pid": plain.pid, "interpreter_pid": int(plain_output)}), flush=True)
    code = """
import json,os,sys,time,opencv_bv_tools
print(json.dumps([os.getpid(),sys.prefix,opencv_bv_tools.__file__]),flush=True)
time.sleep(30)
"""
    process = start_worker([sys.executable, "-c", code], stdout=subprocess.PIPE)
    try:
        pid, prefix, package = json.loads(process.stdout.readline())
        assert pid == process.pid
        assert Path(prefix) == Path(sys.prefix)
        assert Path(package).is_relative_to(Path(sys.prefix))
        handle = win._checked(win._open_process(0x1000, False, pid))
        try:
            member = win.W.BOOL()
            in_job = win._api("IsProcessInJob", win.W.BOOL, win.W.HANDLE, win.W.HANDLE, win.C.POINTER(win.W.BOOL))
            win._checked(in_job(handle, process._bv_job, win.C.byref(member)))
            assert member.value, "Actual interpreter is outside its containment Job"
        finally:
            win._close(handle)
    finally:
        stop_worker(process)
        process.stdout.close()


@WINDOWS
def test_windows_parent_crash_kills_worker_and_descendant():
    child = "import time;time.sleep(30)"
    worker = f"""
import json,os,subprocess,sys,time
p=subprocess.Popen([sys.executable,'-c',{child!r}],stdout=subprocess.DEVNULL)
print(json.dumps([os.getpid(),p.pid]),flush=True)
time.sleep(30)
"""
    controller = f"""
import os,subprocess,sys
from opencv_bv_tools.runtime.process_support import start_worker
p=start_worker([sys.executable,'-c',{worker!r}],stdout=subprocess.PIPE)
print(p.stdout.readline().decode().strip(),flush=True)
os._exit(0)
"""
    result = subprocess.run([sys.executable, "-c", controller], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    pids = json.loads(result.stdout)
    deadline = time.monotonic() + 5
    while any(psutil.pid_exists(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not any(psutil.pid_exists(pid) for pid in pids)


@WINDOWS
def test_windows_missing_drive_check_is_bounded():
    import string
    missing = next((letter for letter in reversed(string.ascii_uppercase)
                    if not Path(f"{letter}:\\").exists()), None)
    if missing is None:
        pytest.skip("Every drive letter is mounted")
    missing_path = missing + ":\\output.png"
    code = f"""
from opencv_bv_tools.runtime.safe_files import Folders, UnsafePath, check_output
try:
    check_output({missing_path!r}, Folders(None))
except UnsafePath:
    pass
else:
    raise AssertionError('Missing drive was accepted')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


@WINDOWS
def test_windows_failed_descriptor_transfer_closes_handles(tmp_path, monkeypatch):
    from opencv_bv_tools.runtime import windows_files
    policy = safe_files.Folders([tmp_path])
    existing = tmp_path / "existing.png"
    existing.write_bytes(b"keep")
    def fail(*args):
        raise OSError("injected descriptor exhaustion")
    monkeypatch.setattr(windows_files.msvcrt, "open_osfhandle", fail)
    with pytest.raises(OSError, match="exhaustion"):
        safe_files.check_output(existing, policy)
    existing.rename(tmp_path / "renamed.png")  # No leaked no-delete-share handle.
    with pytest.raises(OSError, match="exhaustion"):
        safe_files.write_file(b"data", tmp_path / "new.png", policy)
    assert not list(tmp_path.glob("*.tmp"))


@WINDOWS
def test_windows_junction_is_refused(tmp_path):
    outside, allowed = tmp_path / "outside", tmp_path / "allowed"
    outside.mkdir()
    allowed.mkdir()
    (outside / "input.png").write_bytes(b"private")
    link = allowed / "junction"
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                            capture_output=True, text=True, timeout=5)
    if result.returncode:
        pytest.skip("Junction creation unavailable; this native acceptance case still needs a run")
    try:
        policy = safe_files.Folders([allowed])
        with pytest.raises(safe_files.UnsafePath, match="reparse"):
            safe_files.open_input(link / "input.png", policy)
        with pytest.raises(safe_files.UnsafePath, match="reparse"):
            safe_files.write_file(b"bad", link / "output.png", policy)
        assert not (outside / "output.png").exists()
    finally:
        link.rmdir()


@WINDOWS
def test_windows_output_has_one_private_noninherited_acl(tmp_path):
    from opencv_bv_tools.runtime import windows_files as win
    C, W = win.C, win.W
    path = tmp_path / "private.png"
    safe_files.write_file(b"private", path, safe_files.Folders([tmp_path]))
    get_security = win._api(win.security, "GetFileSecurityW", W.BOOL,
                            W.LPCWSTR, W.DWORD, C.c_void_p, W.DWORD, C.POINTER(W.DWORD))
    to_string = win._api(win.security, "ConvertSecurityDescriptorToStringSecurityDescriptorW", W.BOOL,
                         C.c_void_p, W.DWORD, W.DWORD, C.POINTER(W.LPWSTR), C.c_void_p)
    size, text = W.DWORD(), W.LPWSTR()
    get_security(str(path), 4, None, 0, C.byref(size))  # DACL_SECURITY_INFORMATION
    descriptor = C.create_string_buffer(size.value)
    win._checked(get_security(str(path), 4, descriptor, size, C.byref(size)))
    win._checked(to_string(descriptor, 1, 4, C.byref(text), None))
    try:
        assert text.value.startswith("D:P(A;;FA;;;"), text.value
        assert text.value.count("(") == 1 and text.value.count(")") == 1, text.value
    finally:
        win._free(C.cast(text, W.HANDLE))


@WINDOWS
def test_windows_renderer_memory_limit_is_applied_before_execution():
    code = """
try:
    data=bytearray(256*1024*1024)
except MemoryError:
    print('bounded',flush=True)
else:
    raise AssertionError('Job memory limit not applied')
"""
    process = start_worker([sys.executable, "-c", code], memory_limit=128 << 20,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        output, error = process.communicate(timeout=10)
        assert process.returncode == 0 and output.strip() == b"bounded", error
    finally:
        stop_worker(process)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission contract")
def test_non_proc_umask_probe_does_not_mutate_parent(tmp_path):
    # Isolate the test's own restrictive umask from the pytest process. Hide
    # /proc only for this helper to exercise macOS's fallback on Linux too.
    code = f"""
import builtins,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from opencv_bv_tools.runtime import safe_files
root=Path({str(tmp_path)!r})
policy=safe_files.Folders([root])
real_open=builtins.open
def without_proc(path,*args,**kwargs):
    if str(path)=='/proc/self/status': raise FileNotFoundError()
    return real_open(path,*args,**kwargs)
safe_files.open=without_proc
os.umask(0o077)
def forbidden(*args): raise AssertionError('parent umask was changed')
safe_files.os.umask=forbidden
def write(index):
    path=root/f'out-{{index}}.png'
    safe_files.write_file(b'private',path,policy)
    assert path.stat().st_mode & 0o777 == 0o600
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(write,range(12)))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
