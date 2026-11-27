"""Test setup against the installed wheel, away from the checkout."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_install_mcp import FAKE_CLIENT


def setup_command(tmp_path, *arguments, input_text=None):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-m", "opencv_bv_tools.client_setup", *arguments],
                          cwd=tmp_path, env=env, input=input_text, text=True, capture_output=True)


def test_other_client_uses_installed_server_and_preserves_spaced_folder(tmp_path):
    folder = tmp_path / "images with spaces"
    folder.mkdir()
    result = setup_command(tmp_path, "--client", "other", "--allow-dir", str(folder))
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout[result.stdout.index("{"):result.stdout.rindex("}") + 1])
    entry = config["mcpServers"]["opencv-bv-server"]
    assert Path(entry["command"]).is_file()
    expected = ["--allow-dir", str(folder)]
    if os.environ.get("OPENCV_BV_MODEL_DIR"):
        expected += ["--model-dir", str(Path(os.environ["OPENCV_BV_MODEL_DIR"]).expanduser().resolve())]
    assert entry["args"] == expected


def test_noninteractive_setup_requires_client(tmp_path):
    result = setup_command(tmp_path, "--allow-any-path", input_text="")
    assert result.returncode != 0
    assert "Choose a client" in result.stderr


def test_user_install_finds_user_script_directory(tmp_path, monkeypatch):
    from opencv_bv_tools import client_setup

    user_site = tmp_path / "user site"
    scripts = tmp_path / "user scripts"
    monkeypatch.setattr(client_setup, "__file__", str(tmp_path / "checkout" / "src" / "opencv_bv_tools" / "client_setup.py"))
    class Distribution:
        def locate_file(self, path):
            return user_site
    monkeypatch.setattr(client_setup.importlib.metadata, "distribution", lambda name: Distribution())
    monkeypatch.setattr(client_setup.site, "getusersitepackages", lambda: str(user_site))
    monkeypatch.setattr(client_setup.sysconfig, "get_preferred_scheme", lambda kind: "user-scheme")
    monkeypatch.setattr(client_setup.sysconfig, "get_path", lambda name, scheme=None: str(scripts) if scheme == "user-scheme" else "wrong")
    expected = "opencv-bv-mcp.exe" if sys.platform == "win32" else "opencv-bv-mcp"
    assert client_setup.server_executable() == scripts / expected


def test_virtual_environment_uses_python_scripts_directory(tmp_path, monkeypatch):
    from opencv_bv_tools import client_setup

    class Distribution:
        def locate_file(self, path):
            return tmp_path / "env" / "site"
    monkeypatch.setattr(client_setup.importlib.metadata, "distribution", lambda name: Distribution())
    monkeypatch.setattr(client_setup.site, "getusersitepackages", lambda: str(tmp_path / "different-user-site"))
    monkeypatch.setattr(client_setup.sysconfig, "get_path", lambda name: str(tmp_path / "env" / "scripts"))
    expected = "opencv-bv-mcp.exe" if sys.platform == "win32" else "opencv-bv-mcp"
    assert client_setup.server_executable() == tmp_path / "env" / "scripts" / expected


def test_editable_checkout_can_load_setup_helpers(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "src"
    if not (source / "opencv_bv_tools" / "client_setup.py").is_file():
        pytest.skip("source checkout is not included in the packaged ZIP")
    env = dict(os.environ, PYTHONPATH=str(source))
    result = subprocess.run([sys.executable, "-c", "from opencv_bv_tools.client_setup import load_installer; print(load_installer().__file__)"],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == (root / "scripts" / "install_mcp.py").resolve()


@pytest.mark.skipif(sys.platform == "win32", reason="fake CLI requires a POSIX executable")
def test_codex_setup_is_idempotent_and_preserves_existing_entry(tmp_path, monkeypatch):
    binary = tmp_path / "fake clients"
    binary.mkdir()
    fake = binary / "codex"
    fake.write_text(FAKE_CLIENT)
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(binary), str(Path(sys.executable).parent), os.environ["PATH"])))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex config"))
    monkeypatch.setenv("MCP_TEST_STATE", str(tmp_path))
    first = setup_command(tmp_path, "--client", "codex", "--allow-any-path")
    assert first.returncode == 0, first.stderr
    again = setup_command(tmp_path, "--client", "codex", "--allow-any-path")
    assert again.returncode == 0 and "Already registered" in again.stdout
    changed = setup_command(tmp_path, "--client", "codex", "--allow-dir", str(tmp_path))
    assert changed.returncode != 0 and "--replace" in changed.stderr
    config = (tmp_path / "codex config" / "config.toml").read_text()
    assert "--allow-any-path" in config and "tool_timeout_sec = 1860" in config


def test_search_setup_adds_private_index_and_verified_models(tmp_path, monkeypatch, capsys):
    from opencv_bv_tools import client_setup, model_assets
    from opencv_bv_tools.runtime.search_files import verify_private_directory
    server = tmp_path / 'server'
    server.touch()
    photos = tmp_path / 'photos'
    photos.mkdir()
    models = tmp_path / 'search models'
    models.mkdir()
    index = tmp_path / 'private index'
    monkeypatch.delenv('OPENCV_BV_MODEL_DIR', raising=False)
    monkeypatch.setattr(client_setup, 'server_executable', lambda: server)
    requested = []
    def install(*, search=False):
        requested.append(search)
        return models if search else tmp_path / 'core'
    monkeypatch.setattr(model_assets, 'install_models', install)
    assert client_setup.main(['--client', 'other', '--allow-dir', str(photos), '--search',
                              '--search-index-dir', str(index)]) == 0
    assert requested == [False, True]
    verify_private_directory(index)
    output = capsys.readouterr().out
    config = json.loads(output[output.index('{'):output.rindex('}') + 1])
    args = config['mcpServers']['opencv-bv-server']['args']
    assert args == ['--allow-dir', str(photos), '--search-model-dir', str(models),
                    '--search-index-dir', str(index), '--allow-dir', str(models), '--allow-dir', str(index)]
