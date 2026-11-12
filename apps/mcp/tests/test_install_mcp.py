"""Installer transactions and real CLI boundaries with isolated fake clients."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile
from types import SimpleNamespace

import pytest

tomllib = pytest.importorskip("tomllib")
pytestmark = pytest.mark.skipif(sys.version_info[:2] != (3, 12) or sys.platform != "linux",
                                reason="these registration fakes use POSIX executable scripts; native tests are separate")

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("install_mcp", REPO / "scripts/install_mcp.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)

FAKE_CLIENT = r'''#!/usr/bin/env python3
import json,os,sys,tomllib
from pathlib import Path
client=Path(sys.argv[0]).name
args=sys.argv[1:]
name='opencv-bv-server'
if client=='claude':
    path=Path(os.environ['CLAUDE_CONFIG_DIR'])/'.claude.json'
    data=json.loads(path.read_text()) if path.exists() else {}
    entries=data.setdefault('mcpServers',{})
else:
    path=Path(os.environ['CODEX_HOME'])/'config.toml'
    text=path.read_text() if path.exists() else ''
    data=tomllib.loads(text)
    entries=data.get('mcp_servers',{})
action=args[1]
if action=='list':
    print('\n'.join(entries));sys.exit(0)
if action=='get':
    if name not in entries:sys.exit(1)
    e=dict(entries[name])
    if client=='codex':
        e={'transport':{k:e[k] for k in ('command','args','env','cwd') if k in e},
           'enabled':e.get('enabled',True),'tool_timeout_sec':e.get('tool_timeout_sec')}
    print(json.dumps(e));sys.exit(0)
if action=='remove':
    entries.pop(name,None)
elif action=='add-json':
    entries[name]=json.loads(args[-1])
elif action=='add':
    sep=args.index('--'); command=args[sep+1]; rest=args[sep+2:]
    entries[name]={'command':command,'args':rest}
    if client=='claude':entries[name].update(type='stdio',env={})
else:sys.exit(2)
path.parent.mkdir(parents=True,exist_ok=True)
if client=='claude':
    path.write_text(json.dumps(data))
else:
    # Preserve unrelated content; fake only rewrites this server's final table.
    marker='[mcp_servers.opencv-bv-server]'
    text=text.split(marker)[0].rstrip()+'\n'
    if name in entries:
        e=entries[name]
        text+='\n'+marker+'\ncommand = '+json.dumps(e['command'])+'\nargs = '+json.dumps(e['args'])+'\n'
    path.write_text(text)
failure=os.environ.get('MCP_TEST_FAIL','')
mark=Path(os.environ['MCP_TEST_STATE'])/(client+'-failed')
if failure==client+':'+action and not mark.exists():
    mark.touch();sys.exit(19)
'''


@pytest.fixture
def clients(tmp_path, monkeypatch):
    binary = tmp_path / "fake clients"
    binary.mkdir()
    for name in ("claude", "codex"):
        executable = binary / name
        executable.write_text(FAKE_CLIENT)
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "custom codex"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "custom claude"))
    monkeypatch.setenv("MCP_TEST_STATE", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("client", ["claude", "codex"])
def test_register_repeat_replace_and_refuse(clients, client):
    command = clients / "extracted with spaces/.venvs/one/bin/opencv-bv-mcp"
    args = ["--allow-dir", str(clients / "input images"), "--allow-dir", str(clients / "redact-test")]
    installer.register(client, command, args, False)
    first = installer.read_entry(client)
    assert installer.same_entry(client, first, command, args)
    installer.register(client, command, args, False)
    assert installer.read_entry(client) == first
    changed = clients / "second extraction/.venvs/two/bin/opencv-bv-mcp"
    with pytest.raises(installer.InstallError, match="--replace"):
        installer.register(client, changed, args, False)
    assert installer.read_entry(client) == first
    installer.register(client, changed, args, True)
    assert installer.same_entry(client, installer.read_entry(client), changed, args)


@pytest.mark.parametrize("client", ["claude", "codex"])
def test_failed_registration_restores_previous_entry(clients, monkeypatch, client):
    old = clients / "old working environment/server"
    installer.register(client, old, ["--allow-dir", str(clients)], False)
    previous = installer.read_entry(client)
    monkeypatch.setenv("MCP_TEST_FAIL", client + ":add")
    with pytest.raises(installer.InstallError):
        installer.register(client, clients / "new/server", ["--allow-dir", str(clients)], True)
    assert installer.read_entry(client) == previous


def test_codex_timeout_rollback_preserves_unrelated_edit_and_symlink(clients, monkeypatch):
    path = installer.codex_config()
    path.parent.mkdir(parents=True)
    actual = clients / "linked configuration.toml"
    actual.write_text('model = "existing"\n\n[mcp_servers.other]\ncommand = "keep"\n')
    path.symlink_to(actual)
    installer.register("codex", clients / "old/server", [], False)
    previous = installer.read_entry("codex")
    def failure():
        with actual.open("a") as stream:
            stream.write('\n[mcp_servers.concurrent]\ncommand = "preserve"\n')
        raise installer.InstallError("injected timeout failure")
    monkeypatch.setattr(installer, "set_codex_timeout", failure)
    with pytest.raises(installer.InstallError):
        installer.register("codex", clients / "new/server", [], True)
    assert path.is_symlink()
    assert installer.read_entry("codex") == previous
    assert tomllib.loads(actual.read_text())["mcp_servers"]["concurrent"]["command"] == "preserve"
    assert tomllib.loads(actual.read_text())["mcp_servers"]["other"]["command"] == "keep"


@pytest.mark.parametrize("header", ['[mcp_servers.opencv-bv-server]', '[mcp_servers."opencv-bv-server"]', "[mcp_servers.'opencv-bv-server']"])
def test_timeout_replaces_existing_value_preserving_other_settings(clients, header):
    path = installer.codex_config()
    path.parent.mkdir(parents=True)
    path.write_text(header+'\ncommand = "server"\ntool_timeout_sec = 60 # old\n\n[mcp_servers.other]\ntool_timeout_sec = 42\n')
    installer.set_codex_timeout()
    data = tomllib.loads(path.read_text())["mcp_servers"]
    assert data[installer.NAME]["tool_timeout_sec"] == 1860
    assert data["other"]["tool_timeout_sec"] == 42


@pytest.fixture
def staged_project(clients, monkeypatch):
    root = clients / "bundle with spaces"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n")
    (root / "README.md").write_text("test")
    (root / "install-constraints.txt").write_text("test==1\n")
    old = root / ".venvs/working"
    (old / "bin").mkdir(parents=True)
    (old / "bin/python").write_text("existing interpreter")
    (root / ".venv").symlink_to(old)
    monkeypatch.setattr(installer, "check_environment", lambda *args: None)
    def new(*args):
        stage = root / ".venvs/new"
        (stage / "bin").mkdir(parents=True, exist_ok=True)
        (stage / "bin/python").write_text("new interpreter")
        return stage
    monkeypatch.setattr(installer, "install_environment", new)
    return root, old


@pytest.mark.parametrize("failure", ["install", "check", "interrupt"])
def test_failed_stage_does_not_change_working_environment_or_registration(staged_project, monkeypatch, failure):
    root, old = staged_project
    installer.register("codex", old / "bin/server", [], False)
    previous = installer.read_entry("codex")
    def fail(*args):
        if failure == "interrupt":
            raise KeyboardInterrupt()
        raise installer.InstallError("injected stage failure")
    monkeypatch.setattr(installer, "install_environment" if failure in {"install", "interrupt"} else "check_environment", fail)
    code = installer.main(["--codex", "--replace", "--allow-dir", str(root)], root=root)
    assert code != 0
    assert (root / ".venv").resolve() == old
    assert (old / "bin/python").read_text() == "existing interpreter"
    assert installer.read_entry("codex") == previous
    assert not (root / ".install-state.json").exists()


def test_success_promotes_then_reuses_checked_environment(staged_project, monkeypatch):
    root, old = staged_project
    flags = ["--claude", "--codex", "--replace", "--allow-dir", str(root)]
    assert installer.main(flags, root=root) == 0
    new = (root / ".venv").resolve()
    assert new != old and (old / "bin/python").exists()
    def forbidden(*args):
        raise AssertionError("repeat install should reuse the checked environment")
    monkeypatch.setattr(installer, "install_environment", forbidden)
    assert installer.main(flags, root=root) == 0
    assert (root / ".venv").resolve() == new


def test_registration_failure_reports_partial_state_without_promoting(staged_project, monkeypatch):
    root, old = staged_project
    monkeypatch.setenv("MCP_TEST_FAIL", "codex:add")
    assert installer.main(["--claude", "--codex", "--allow-dir", str(root)], root=root) == 1
    assert installer.read_entry("claude") is not None
    assert installer.read_entry("codex") is None
    assert (root / ".venv").resolve() == old


def test_selfcheck_selects_requested_pdf_engine(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(installer, "run", lambda args, **kwargs: commands.append(args) or "{}")
    installer.check_environment(tmp_path, tmp_path / "environment", True)
    assert commands[0][-2:] == ["--pdf-engine", "mupdf"]


@pytest.mark.parametrize("has_uv", [True, False])
def test_install_uses_encoded_file_uri_for_paths_with_spaces(tmp_path, monkeypatch, has_uv):
    commands = []
    root = tmp_path / "folder with spaces"
    root.mkdir()
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/fake/uv" if has_uv else None)
    monkeypatch.setattr(installer, "run", lambda args, **kwargs: commands.append(args))
    installer.install_environment(root, root / "package.whl", False)
    requirement = next(arg for arg in commands[1] if str(arg).startswith("opencv-bv-tools[all] @"))
    assert requirement.startswith("opencv-bv-tools[all] @ file://") and "%20" in requirement


def test_bundle_hash_and_path_validation(tmp_path):
    wheel = tmp_path / "package.whl"
    wheel.write_bytes(b"wheel")
    manifest = tmp_path / "bundle.json"
    import hashlib
    manifest.write_text(json.dumps({"wheel": wheel.name, "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}))
    assert installer.installation_source(tmp_path) == wheel
    wheel.write_bytes(b"changed")
    with pytest.raises(installer.InstallError, match="checksum"):
        installer.installation_source(tmp_path)
    manifest.write_text(json.dumps({"wheel": "../outside.whl"}))
    with pytest.raises(installer.InstallError, match="path"):
        installer.installation_source(tmp_path)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_zip_accepts_native_wheel_metadata_line_endings(tmp_path, monkeypatch, newline):
    spec = importlib.util.spec_from_file_location("build_mcp_zip", REPO / "scripts/build_mcp_zip.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.setattr(builder, "FILES", {})
    wheel, companion = tmp_path / "app.whl", tmp_path / "rapidocr.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("opencv_bv_tools-1.0.dist-info/METADATA", newline.join([
            "Metadata-Version: 2.4", "Name: opencv-bv-tools", "Version: 1.0",
            "Requires-Dist: rapidocr==3.10.0+bv.1", "", ""]))
    with zipfile.ZipFile(companion, "w") as archive:
        archive.writestr("rapidocr-3.10.0+bv.1.dist-info/METADATA", newline.join([
            "Metadata-Version: 2.4", "Name: rapidocr", "Version: 3.10.0+bv.1", "", ""]))
    output = tmp_path / "bundle.zip"
    builder.build(tmp_path, wheel, output, "a" * 40, rapidocr_wheel=companion)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("opencv-bv-server-mcp/bundle.json"))
        assert manifest["package_version"] == "1.0"
        assert manifest["dependency_wheels"][0]["wheel"] == "wheels/rapidocr.whl"


def test_zip_is_deterministic_and_excludes_environments(tmp_path):
    spec = importlib.util.spec_from_file_location("build_mcp_zip", REPO / "scripts/build_mcp_zip.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    version = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    wheel = REPO / f"dist/opencv_bv_tools-{version}-py3-none-any.whl"
    if not wheel.exists():
        pytest.skip("build the wheel first to test the release archive")
    first, second = tmp_path / "one.zip", tmp_path / "two.zip"
    for output in (first, second):
        builder.build(REPO, wheel, output, "22269df80dec6d2ffa68e6f97d10d5d8ce2d6d73", unpublished=True)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert all(name.startswith("opencv-bv-server-mcp/") for name in names)
        assert not any(".venv" in name or "benchmark" in name for name in names)
        manifest = json.loads(archive.read("opencv-bv-server-mcp/bundle.json"))
        assert manifest["source_state"] == "local-unpublished"
        assert manifest["package_source_commit"] is None
        assert manifest["base_commit"] == "22269df80dec6d2ffa68e6f97d10d5d8ce2d6d73"
        guide = archive.read("opencv-bv-server-mcp/README.md").decode()
        assert "Nine MCP tools" in guide
        import hashlib
        for path, digest in manifest["files_sha256"].items():
            assert hashlib.sha256(archive.read("opencv-bv-server-mcp/" + path)).hexdigest() == digest


def interactive_answers(monkeypatch, answers):
    answers = iter(answers)
    monkeypatch.setattr(installer.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    def reply(prompt):
        try:
            answer = next(answers)
        except StopIteration:
            raise EOFError()
        if isinstance(answer, BaseException):
            raise answer
        return answer
    monkeypatch.setattr("builtins.input", reply)


def test_prompt_selected_default_normalizes_and_deduplicates_paths(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "input images"
    folder.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    interactive_answers(monkeypatch, ["bad choice", "", "", "missing", "input images", "~/input images", ""])
    assert installer.choose_access([], False, True) == ([str(folder)], False)
    output = capsys.readouterr().out
    assert "Choose at least one folder" in output and "Not a folder" in output


@pytest.mark.parametrize("answers", [["2"], ["invalid", "2"]])
def test_prompt_anywhere(monkeypatch, answers):
    interactive_answers(monkeypatch, answers)
    assert installer.choose_access([], False, True) == ([], True)


@pytest.mark.parametrize("answers", [[], ["1"], ["1", ""], ["q"], [KeyboardInterrupt()], ["1", KeyboardInterrupt()]])
def test_cancelled_prompt_does_not_install_or_change_registration(staged_project, monkeypatch, answers):
    root, old = staged_project
    installer.register("codex", old / "bin/server", ["--allow-any-path"], False)
    previous = installer.read_entry("codex")
    interactive_answers(monkeypatch, answers)
    assert installer.main(["--codex", "--replace"], root=root) in {1, 130}
    assert installer.read_entry("codex") == previous
    assert (root / ".venv").resolve() == old
    assert not (root / ".venvs/new").exists()
    assert not (root / ".install-state.json").exists()


def test_partial_folder_choice_is_not_saved_on_eof(staged_project, monkeypatch):
    root, old = staged_project
    interactive_answers(monkeypatch, ["1", str(root)])
    assert installer.main(["--claude", "--codex"], root=root) == 1
    assert installer.read_entry("claude") is None and installer.read_entry("codex") is None
    assert not (root / ".venvs/new").exists()


@pytest.mark.parametrize("flags", [[], ["--allow-any-path", "--allow-dir", "."]])
def test_missing_noninteractive_or_conflicting_flags_fail_before_install(staged_project, monkeypatch, flags, capsys):
    root, old = staged_project
    monkeypatch.setattr(installer.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    if flags:
        with pytest.raises(SystemExit) as exc:
            installer.main(["--codex", *flags], root=root)
        assert exc.value.code == 2
    else:
        assert installer.main(["--codex"], root=root) == 1
        message = capsys.readouterr().err
        assert "--allow-dir" in message and "--allow-any-path" in message
    assert not (root / ".venvs/new").exists()
    assert installer.read_entry("codex") is None


@pytest.mark.parametrize("interactive", [False, True])
def test_access_switches_preserve_environment_and_register_exact_mode(staged_project, monkeypatch, interactive, capsys):
    root, old = staged_project
    if interactive:
        interactive_answers(monkeypatch, ["2", "1", str(root), str(root), "", "2"])
    else:
        def forbidden(prompt):
            raise AssertionError("explicit access flags must never prompt")
        monkeypatch.setattr("builtins.input", forbidden)
    modes = [["--allow-any-path"], ["--allow-dir", str(root)], ["--allow-any-path"]]
    environment = None
    for mode in modes:
        flags = ["--claude", "--codex", "--replace", *([] if interactive else mode)]
        assert installer.main(flags, root=root) == 0
        current = (root / ".venv").resolve()
        assert current == (environment or current)
        environment = current
        for client in ("claude", "codex"):
            assert installer.same_entry(client, installer.read_entry(client), current / "bin/opencv-bv-mcp", mode)
    output = capsys.readouterr().out
    assert "File access: anywhere" in output and "File access: selected folders" in output


@pytest.mark.parametrize("client", ["claude", "codex"])
@pytest.mark.parametrize("start_anywhere", [True, False])
def test_access_replacement_requires_replace_and_rolls_back_failure(clients, monkeypatch, client, start_anywhere):
    command = clients / "existing/server"
    before = ["--allow-any-path"] if start_anywhere else ["--allow-dir", str(clients)]
    after = ["--allow-dir", str(clients)] if start_anywhere else ["--allow-any-path"]
    installer.register(client, command, before, False)
    original = installer.read_entry(client)
    with pytest.raises(installer.InstallError, match="--replace"):
        installer.register(client, command, after, False)
    assert installer.read_entry(client) == original
    monkeypatch.setenv("MCP_TEST_FAIL", client + ":add")
    with pytest.raises(installer.InstallError):
        installer.register(client, command, after, True)
    assert installer.read_entry(client) == original


def test_install_only_needs_no_access_choice(staged_project, monkeypatch, capsys):
    root, old = staged_project
    interactive_answers(monkeypatch, [AssertionError("install-only must not prompt")])
    assert installer.main(["--install-only"], root=root) == 0
    assert installer.read_entry("claude") is None and installer.read_entry("codex") is None
    assert not (root / "mcp-config.json").exists()
    assert "connection settings were not changed" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["anywhere", "selected", "interactive"])
def test_generic_setup_needs_no_client_and_exports_exact_command(staged_project, monkeypatch, mode):
    root, old = staged_project
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)
    def forbidden(*args):
        raise AssertionError("generic setup must not read or change client registrations")
    monkeypatch.setattr(installer, "read_entry", forbidden)
    monkeypatch.setattr(installer, "register", forbidden)
    args = ["--allow-any-path"] if mode == "anywhere" else ["--allow-dir", str(root)]
    if mode == "interactive":
        interactive_answers(monkeypatch, ["1", str(root), ""])
    assert installer.main([] if mode == "interactive" else args, root=root) == 0
    command = str((root / ".venv").resolve() / "bin/opencv-bv-mcp")
    assert json.loads((root / "mcp-config.json").read_text()) == {
        "mcpServers": {installer.NAME: {"command": command, "args": args}}}


def test_generic_config_includes_selected_pdf_engine(staged_project):
    root, old = staged_project
    assert installer.main(["--allow-any-path", "--mupdf"], root=root) == 0
    entry = json.loads((root / "mcp-config.json").read_text())["mcpServers"][installer.NAME]
    assert entry["args"] == ["--allow-any-path", "--pdf-engine", "mupdf"]


@pytest.mark.parametrize("previous", ['broken JSON', '{"mcpServers":{"custom":{"command":"keep"}}}'])
def test_generic_config_requires_explicit_replacement(staged_project, previous):
    root, old = staged_project
    path = root / "mcp-config.json"
    path.write_text(previous)
    assert installer.main(["--allow-any-path"], root=root) == 1
    assert path.read_text() == previous
    assert (root / ".venv").resolve() == old
    assert installer.main(["--allow-any-path", "--replace"], root=root) == 0
    assert json.loads(path.read_text())["mcpServers"][installer.NAME]["args"] == ["--allow-any-path"]


def test_selfcheck_failure_preserves_generic_config(staged_project, monkeypatch):
    root, old = staged_project
    path = root / "mcp-config.json"
    path.write_text("previous config")
    def fail(*args):
        raise installer.InstallError("self-check failed")
    monkeypatch.setattr(installer, "check_environment", fail)
    assert installer.main(["--allow-any-path", "--replace"], root=root) == 1
    assert path.read_text() == "previous config"
    assert (root / ".venv").resolve() == old


def test_atomic_config_failure_preserves_file_and_prevents_registration(staged_project, monkeypatch):
    root, old = staged_project
    path = root / "mcp-config.json"
    path.write_text("previous config")
    def fail(*args):
        raise OSError("injected atomic replace failure")
    monkeypatch.setattr(installer.os, "replace", fail)
    assert installer.main(["--claude", "--allow-any-path", "--replace"], root=root) == 1
    assert path.read_text() == "previous config"
    assert installer.read_entry("claude") is None
    assert (root / ".venv").resolve() == old
    assert not list(root.glob(".mcp-install-*"))


def test_failed_optional_registration_leaves_checked_generic_config(staged_project, monkeypatch, capsys):
    root, old = staged_project
    monkeypatch.setenv("MCP_TEST_FAIL", "codex:add")
    assert installer.main(["--codex", "--allow-any-path"], root=root) == 1
    entry = json.loads((root / "mcp-config.json").read_text())["mcpServers"][installer.NAME]
    assert Path(entry["command"]).parent.joinpath("python").is_file()
    assert entry["args"] == ["--allow-any-path"]
    assert "Common MCP configuration example:" in capsys.readouterr().out


@pytest.mark.parametrize("flags", [["--claude"], ["--codex"], ["--allow-any-path"], ["--allow-dir", "."]])
def test_install_only_rejects_setup_flags(staged_project, flags):
    root, old = staged_project
    with pytest.raises(SystemExit) as exc:
        installer.main(["--install-only", *flags], root=root)
    assert exc.value.code == 2
    assert not (root / ".venvs/new").exists()


def test_generic_config_does_not_follow_a_link_to_client_settings(staged_project):
    root, old = staged_project
    target = root / "other-client.json"
    target.write_text("keep this client configuration")
    (root / "mcp-config.json").symlink_to(target)
    assert installer.main(["--allow-any-path", "--replace"], root=root) == 1
    assert target.read_text() == "keep this client configuration"
    assert (root / "mcp-config.json").is_symlink()


def test_companion_wheel_integrity_and_path_boundary(tmp_path):
    import hashlib

    wheel = tmp_path / "ocr.whl"
    wheel.write_bytes(b"frozen dependency")
    manifest = tmp_path / "bundle.json"
    entry = {"wheel": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
    manifest.write_text(json.dumps({"dependency_wheels": [entry]}))
    assert installer.companion_wheels(tmp_path) == [wheel]
    wheel.write_bytes(b"changed dependency")
    with pytest.raises(installer.InstallError, match="checksum mismatch"):
        installer.companion_wheels(tmp_path)
    entry["wheel"] = "../outside.whl"
    manifest.write_text(json.dumps({"dependency_wheels": [entry]}))
    with pytest.raises(installer.InstallError, match="Invalid dependency wheel path"):
        installer.companion_wheels(tmp_path)


def test_source_checkout_needs_companion_before_install(tmp_path):
    (tmp_path / "pyproject.toml").write_text('ocr = ["rapidocr==3.10.0+bv.1"]')
    with pytest.raises(installer.InstallError, match="Build the companion OCR wheel first"):
        installer.companion_wheels(tmp_path)
