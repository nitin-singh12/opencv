"""Install the unified server without modifying the previous working environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

NAME = "opencv-bv-server"
TIMEOUT = 1860


class InstallError(Exception):
    pass


def executable(environment, name):
    if sys.platform == "win32":
        return environment / "Scripts" / (name + ".exe")
    return environment / "bin" / name


def check_platform():
    if sys.platform == "darwin" and platform.machine() == "x86_64":
        raise InstallError("Intel macOS is not supported by this package: the pinned cryptography dependency "
                           "no longer provides Intel macOS wheels. Use an Apple Silicon Mac with native arm64 Python.")
    targets = {"linux": {"x86_64"}, "darwin": {"arm64"}, "win32": {"AMD64", "x86_64"}}
    if platform.machine() not in targets.get(sys.platform, set()) or sys.version_info[:2] != (3, 12):
        raise InstallError("Use Python 3.12: Linux x86-64, macOS arm64, or Windows x64. "
                           "macOS and Windows are experimental candidates pending native validation.")
    if sys.platform != "linux":
        print("Experimental native installation: all nine tools must pass the self-check. "
              "Complete the native-platform testing guide before treating this platform as validated.")


def run(args, *, capture=False, cwd=None):
    result = subprocess.run([str(a) for a in args], text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None, cwd=cwd)
    if result.returncode:
        # Registration arguments can contain credentials from a previous entry.
        raise InstallError(f"{Path(str(args[0])).name} failed (exit {result.returncode}).")
    return result.stdout if capture else None


def atomic_write(path, data, mode=0o600):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        mode = path.stat().st_mode & 0o777
    fd, temporary = tempfile.mkstemp(prefix=".mcp-install-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def codex_config():
    return (Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml").resolve()


def claude_config():
    return Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home()))) / ".claude.json"


def read_entry(client):
    if client == "codex":
        result = subprocess.run([client, "mcp", "get", NAME, "--json"], capture_output=True, text=True)
        if result.returncode:
            # Distinguish a missing entry from a broken client/configuration.
            run([client, "mcp", "list"], capture=True)
            return None
        return json.loads(result.stdout)
    path = claude_config()
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("mcpServers", {}).get(NAME)


def same_entry(client, entry, command, args):
    if not entry:
        return False
    transport = entry.get("transport", {}) if client == "codex" else entry
    matches = (transport.get("command") == str(command) and transport.get("args", []) == args
               and not any(transport.get(k) for k in ("env", "env_vars", "cwd"))
               and transport.get("type", "stdio") == "stdio")
    if client == "codex":
        matches = matches and entry.get("enabled") is not False and (entry.get("tool_timeout_sec") or 0) >= TIMEOUT
    return matches


def server_sections(text):
    """Locate this server's table and subtables, leaving other config intact."""
    headers = list(re.finditer(r"(?m)^[ \t]*\[([^\n]+)\][ \t]*(?:#.*)?$", text))
    names = (f"mcp_servers.{NAME}", f'mcp_servers."{NAME}"', f"mcp_servers.'{NAME}'")
    result = []
    for index, header in enumerate(headers):
        key = header.group(1).strip()
        if any(key == name or key.startswith(name + ".") for name in names):
            end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
            result.append((header.start(), end, text[header.start():end]))
    return result


def replace_server_sections(text, replacement):
    sections = server_sections(text)
    for start, end, _ in reversed(sections):
        text = text[:start] + text[end:]
    return text.rstrip() + ("\n\n" if text.strip() and replacement else "") + replacement


def set_codex_timeout():
    path = codex_config()
    text = path.read_text()
    sections = server_sections(text)
    bases = [(start, end, block) for start, end, block in sections
             if re.match(r"\s*\[mcp_servers\.(?:opencv-bv-server|\"opencv-bv-server\"|'opencv-bv-server')\]", block)]
    if len(bases) != 1:
        raise InstallError("Could not identify the Codex server table to set its PDF timeout.")
    start, end, block = bases[0]
    pattern = r"(?m)^[ \t]*tool_timeout_sec[ \t]*=.*$"
    if re.search(pattern, block):
        block = re.sub(pattern, f"tool_timeout_sec = {TIMEOUT}", block)
    else:
        first, rest = block.split("\n", 1) if "\n" in block else (block, "")
        block = first + f"\ntool_timeout_sec = {TIMEOUT}\n" + rest
    if path.read_text() != text:
        raise InstallError("Codex configuration changed during installation; retry registration.")
    atomic_write(path, (text[:start] + block + text[end:]).encode())


def register(client, command, args, replace):
    old = read_entry(client)
    if same_entry(client, old, command, args):
        print(f"Already registered in {client}; nothing changed.")
        return
    if old and not replace:
        raise InstallError(f"A different {NAME} is registered in {client}. Rerun with --replace.")
    if client == "claude":
        try:
            if old:
                run([client, "mcp", "remove", "--scope", "user", NAME], capture=True)
            run([client, "mcp", "add", "--scope", "user", NAME, "--", command, *args], capture=True)
            if not same_entry(client, read_entry(client), command, args):
                raise InstallError("Claude registration readback did not match.")
        except BaseException:
            current = read_entry(client)
            # Do not remove an unexpected entry another installer may have written.
            if current != old:
                if current and not same_entry(client, current, command, args):
                    raise InstallError("Claude entry changed unexpectedly; previous environment is preserved. Check registration.")
                if current:
                    run([client, "mcp", "remove", "--scope", "user", NAME], capture=True)
                if old:
                    run([client, "mcp", "add-json", "--scope", "user", NAME, json.dumps(old)], capture=True)
            raise
    else:
        path = codex_config()
        before = path.read_text() if path.exists() else ""
        previous = "".join(block for _, _, block in server_sections(before))
        try:
            run([client, "mcp", "add", NAME, "--", command, *args], capture=True)
            set_codex_timeout()
            if not same_entry(client, read_entry(client), command, args):
                raise InstallError("Codex registration readback did not match.")
        except BaseException:
            current = read_entry(client)
            transport = current.get("transport", {}) if current else {}
            if current != old:
                if current and (transport.get("command") != str(command) or transport.get("args", []) != args):
                    raise InstallError("Codex entry changed unexpectedly; previous environment is preserved. Check registration.")
                latest = path.read_text() if path.exists() else ""
                # Restore only this server; preserve unrelated concurrent config edits.
                atomic_write(path, replace_server_sections(latest, previous).encode())
            raise
    print(f"Registered in {client} as {NAME}.")


def installation_source(root):
    manifest = root / "bundle.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        wheel = (root / data["wheel"]).resolve()
        if not wheel.is_relative_to(root.resolve()) or wheel.suffix != ".whl":
            raise InstallError("Invalid wheel path in bundle.json.")
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != data["wheel_sha256"]:
            raise InstallError("Wheel checksum does not match bundle.json. Extract a fresh ZIP.")
        return wheel
    if (root / "pyproject.toml").is_file():
        return root
    raise InstallError("Run the installer from the complete project or extracted MCP ZIP.")


def companion_wheels(root):
    """Validate bundled local dependencies before handing them to the resolver."""
    manifest=root / "bundle.json"
    if manifest.exists():
        entries=json.loads(manifest.read_text()).get("dependency_wheels", [])
        paths=[]
        for entry in entries:
            path=(root / entry["wheel"]).resolve()
            if not path.is_relative_to(root.resolve()) or path.suffix != ".whl":
                raise InstallError("Invalid dependency wheel path.")
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["sha256"]:
                raise InstallError("Dependency wheel checksum mismatch. Extract a fresh ZIP.")
            paths.append(path)
        return paths
    project=root / "pyproject.toml"
    if project.exists() and "rapidocr==3.10.0+bv.1" in project.read_text():
        path=root / "wheels/rapidocr-3.10.0+bv.1-py3-none-any.whl"
        if not path.is_file():
            raise InstallError("Build the companion OCR wheel first; follow docs/lightweight-branch.md.")
        return [path]
    return []


def fingerprint(root, source, mupdf):
    digest = hashlib.sha256()
    digest.update(json.dumps([sys.executable, sys.version, platform.machine(), mupdf]).encode())
    paths = [source] if source.is_file() else [root / "pyproject.toml", root / "README.md",
            *sorted((root / "src").rglob("*.py")), *sorted((root / "src" / "opencv_bv_tools" / "models").glob("*"))]
    paths += [root / "install-constraints.txt", *companion_wheels(root)]
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def installed_environment(root, expected):
    try:
        state = json.loads((root / ".install-state.json").read_text())
        environment = (root / state["environment"]).resolve()
        if (state["fingerprint"] == expected and environment.is_relative_to((root / ".venvs").resolve())
                and executable(environment, "python").is_file()
                and (environment / ".checked").read_text() == expected):
            return environment
    except (OSError, ValueError, KeyError):
        pass
    return None


def install_environment(root, source, mupdf):
    (root / ".venvs").mkdir(exist_ok=True)
    environment = Path(tempfile.mkdtemp(prefix="install-", dir=root / ".venvs"))
    print(f"Installing into {environment}; previous environments are kept.", flush=True)
    python = executable(environment, "python")
    extras = "all,mupdf" if mupdf else "all"
    packages = [f"opencv-bv-tools[{extras}] @ {source.as_uri()}", *companion_wheels(root)]
    try:
        if shutil.which("uv"):
            run(["uv", "venv", "--python", sys.executable, environment])
            run(["uv", "pip", "install", "--python", python, "--only-binary", ":all:", "--compile-bytecode", "--constraint",
                 "install-constraints.txt", *packages], cwd=root)
            run(["uv", "pip", "check", "--python", python])
        else:
            run([sys.executable, "-m", "venv", environment])
            run([python, "-m", "pip", "install", "--only-binary=:all:", "--constraint", "install-constraints.txt", *packages], cwd=root)
            run([python, "-m", "pip", "check"])
    except BaseException:
        print(f"Installation did not complete. Preserved for diagnosis: {environment}", file=sys.stderr)
        raise
    return environment


def check_environment(root, environment, mupdf):
    print("Checking all nine tools through MCP, using temporary test files.", flush=True)
    run([executable(environment, "python"), root / "scripts/smoke_mcp.py", "--fixtures", root / "tests/fixtures",
         "--pdf-engine", "mupdf" if mupdf else "picture"])
    versions = run([executable(environment, "python"), "-c",
                    "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()},sort_keys=True,indent=2))"], capture=True)
    atomic_write(environment / "installed-versions.json", versions.encode())
    print("OK: all 9 MCP tools passed, email addresses hidden, lookalike left alone, originals unchanged.")


def promote(root, environment, expected):
    link = root / ".venv"
    if sys.platform == "win32" or (link.exists() and not link.is_symlink()):
        # Registration already points to the immutable environment. Windows
        # does not need administrator/Developer Mode privileges for a symlink.
        print(f"The installed Python is {executable(environment, 'python')}")
    else:
        temporary = root / f".venv-link-{os.getpid()}"
        try:
            temporary.symlink_to(environment.relative_to(root), target_is_directory=True)
            os.replace(temporary, link)
        finally:
            temporary.unlink(missing_ok=True)
    atomic_write(environment / ".checked", expected.encode())
    atomic_write(root / ".install-state.json", json.dumps({"fingerprint": expected,
                 "environment": str(environment.relative_to(root))}, indent=2).encode())


def existing_folder(value):
    folder = Path(value).expanduser().resolve()
    if not folder.is_dir():
        raise InstallError(f"Not a folder: {folder}. Create it before installing.")
    return str(folder)


def export_config(root, command, args, replace):
    """Write a common config example; never edit an arbitrary client's settings."""
    path = root / "mcp-config.json"
    config = {"mcpServers": {NAME: {"command": str(command), "args": args}}}
    if path.is_symlink():
        raise InstallError("mcp-config.json must be a regular generated file, not a symbolic link.")
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except ValueError:
            if not replace:
                raise InstallError("mcp-config.json is not valid JSON. Keep a copy of your edits, "
                                   "then use --replace to regenerate it.") from None
        else:
            if existing == config:
                return path
            if not replace:
                raise InstallError("A different mcp-config.json exists. Use --replace to update the generated example.")
    atomic_write(path, (json.dumps(config, indent=2) + "\n").encode())
    return path


def choose_access(allow_dirs, allow_any_path, registering):
    """Resolve explicit flags or an interactive choice before any installation."""
    folders = list(dict.fromkeys(existing_folder(folder) for folder in allow_dirs))
    if folders or allow_any_path or not registering:
        return folders, allow_any_path
    if not sys.stdin.isatty():
        raise InstallError(
            "Choose file access explicitly for noninteractive setup. Examples:\n"
            "  bash install.sh --allow-dir /path/to/input --allow-dir /path/to/output\n"
            "  bash install.sh --allow-any-path\n"
            "Add --replace to update existing settings, or use --install-only to skip connection setup.")
    print("\nWhere may the tools read inputs and save outputs?")
    print("  1. Selected folders (default; includes their subfolders)")
    print("  2. Anywhere your account and the server's file-safety checks allow")
    print("Original-file protection and explicit overwrite requirements apply in both modes.")
    try:
        while True:
            choice = input("Choose 1 or 2 [1], or q to cancel: ").strip().lower()
            if choice == "q":
                raise InstallError("Cancelled before installation; no registration changed.")
            if choice == "2":
                return [], True
            if choice in {"", "1"}:
                break
            print("Enter 1 for selected folders, 2 for anywhere, or q to cancel.")
        print("Enter one existing folder per line, including input and output folders.")
        print("Paths may contain spaces; do not add quotes. ~ means your home folder.")
        print("Press Enter on an empty line when finished, or Ctrl-C to cancel.")
        while True:
            value = input("Folder: ").strip()
            if not value:
                if folders:
                    return folders, False
                print("Choose at least one folder, or press Ctrl-C to cancel.")
                continue
            try:
                folder = existing_folder(value)
            except (InstallError, OSError, ValueError) as exc:
                print(str(exc))
                continue
            if folder not in folders:
                folders.append(folder)
    except EOFError:
        raise InstallError("Input ended before the access choice was complete; no registration changed. "
                           "Rerun with --allow-dir DIR or --allow-any-path.") from None


def main(argv=None, *, root=None):
    parser = argparse.ArgumentParser(description="Install/check all nine tools and generate connection settings for local stdio MCP clients.")
    parser.add_argument("--claude", action="store_true", help="also register automatically in Claude Code")
    parser.add_argument("--codex", action="store_true", help="also register automatically in Codex")
    parser.add_argument("--replace", action="store_true", help="replace a differing generated config or selected client registration")
    parser.add_argument("--install-only", action="store_true", help="install/check only; do not prompt, generate config or register clients")
    access = parser.add_mutually_exclusive_group()
    access.add_argument("--allow-dir", action="append", default=[], type=Path,
                        help="existing input/output folder and its subfolders; repeat for more folders")
    access.add_argument("--allow-any-path", action="store_true",
                        help="allow any path permitted by your account and the server's file-safety checks")
    parser.epilog = ("Setup without an access flag asks you to choose in an interactive terminal. "
                     "Noninteractive setup requires --allow-dir or --allow-any-path, unless --install-only is used.")
    parser.add_argument("--mupdf", action="store_true", help="install/test/select optional MuPDF (AGPL or commercial licence)")
    args = parser.parse_args(argv)
    root = (Path(root) if root else Path(__file__).resolve().parents[1]).resolve()
    clients = [client for client in ("claude", "codex") if getattr(args, client)]
    if args.install_only and (clients or args.allow_dir or args.allow_any_path):
        parser.error("--install-only cannot be combined with client registration or access flags.")
    try:
        check_platform()
        folders, allow_any_path = choose_access(args.allow_dir, args.allow_any_path, not args.install_only)
        for client in clients:
            if not shutil.which(client):
                raise InstallError(f"{client} is not on PATH. Install it first or omit --{client}.")
            read_entry(client)  # Fail on invalid client configuration before installing.
        source = installation_source(root)
        expected = fingerprint(root, source, args.mupdf)
        environment = installed_environment(root, expected)
        if environment:
            print(f"Reusing checked environment: {environment}")
        else:
            environment = install_environment(root, source, args.mupdf)
        check_environment(root, environment, args.mupdf)
        server_args = (["--allow-any-path"] if allow_any_path else
                       [value for folder in folders for value in ("--allow-dir", folder)])
        if args.mupdf:
            server_args += ["--pdf-engine", "mupdf"]
        command = executable(environment, "opencv-bv-mcp")
        if not args.install_only:
            config_path = export_config(root, command, server_args, args.replace)
            print(f"Common MCP configuration example: {config_path}", flush=True)
            print("Connection type: local stdio")
            print(f"Executable: {command}")
            print(f"Arguments (JSON array): {json.dumps(server_args)}", flush=True)
        errors = []
        for client in clients:
            try:
                register(client, command, server_args, args.replace)
            except Exception as exc:
                errors.append(f"{client}: {exc}")
        if errors:
            raise InstallError("Installed and checked, but registration did not finish:\n" + "\n".join(errors)
                               + "\nSuccessful registrations above remain active; all previous environments are kept.")
        promote(root, environment, expected)
        print(f"Installed {NAME}. Keep this extracted folder in place.")
        if not args.install_only:
            if allow_any_path:
                print("File access: anywhere your account and the server's file-safety checks allow.")
            else:
                print("File access: selected folders and their subfolders:")
                for folder in folders:
                    print(f"  {folder}")
            if not clients:
                print("Add the executable and arguments above to your MCP client's local stdio server settings.")
            print(f"Restart/reconnect your client, then ask to use the {NAME} MCP server.")
        else:
            print("Installation check complete; connection settings were not changed. "
                  "Rerun without --install-only to choose access and generate an MCP configuration example.")
        return 0
    except (InstallError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Previous environments are preserved; check any registration already reported above.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
