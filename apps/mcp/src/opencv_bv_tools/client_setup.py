# SPDX-License-Identifier: Apache-2.0
"""Connect an installed MCP server to a local client."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import site
import sys
import sysconfig


def server_executable():
    """Use the Python environment's scripts directory, including user installs."""
    package_root = Path(__file__).resolve().parent.parent
    user_site = Path(site.getusersitepackages()).resolve()
    if package_root == user_site:
        scripts = sysconfig.get_path("scripts", scheme=sysconfig.get_preferred_scheme("user"))
    else:
        scripts = sysconfig.get_path("scripts")
    suffix = ".exe" if sys.platform == "win32" else ""
    return Path(scripts) / ("opencv-bv-mcp" + suffix)


def load_installer():
    """Use the wheel helper or the checkout helper for editable installs."""
    if importlib.util.find_spec("opencv_bv_tools._installation") is not None:
        return importlib.import_module("opencv_bv_tools._installation")
    path = Path(__file__).resolve().parents[2] / "scripts" / "install_mcp.py"
    if not path.is_file():
        raise ImportError("The installation helper is missing")
    spec = importlib.util.spec_from_file_location("opencv_bv_installation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description="Connect the installed OpenCV MCP server to a client.")
    parser.add_argument("--client", choices=("codex", "claude", "other"), help="client to configure")
    parser.add_argument("--replace", action="store_true", help="replace an existing server entry")
    access = parser.add_mutually_exclusive_group()
    access.add_argument("--allow-dir", action="append", default=[], metavar="FOLDER")
    access.add_argument("--allow-any-path", action="store_true")
    args = parser.parse_args(argv)

    # Imported only after argparse, so --help stays lightweight.
    installer = load_installer()

    try:
        if importlib.util.find_spec("cv2") is None:
            raise installer.InstallError("OpenCV is missing. Install the package with its [wheel] extra, or install a compatible cv2 in this environment.")
        client = args.client
        if client is None:
            if not sys.stdin.isatty():
                raise installer.InstallError("Choose a client with --client codex, claude, or other.")
            print("Which MCP client do you want to connect?\n  1. Codex\n  2. Claude Code\n  3. Another local MCP client")
            choice = input("Choose 1, 2, 3, or q to cancel: ").strip().lower()
            if choice == "q":
                print("Setup cancelled.")
                return 0
            client = {"1": "codex", "2": "claude", "3": "other"}.get(choice)
            if client is None:
                raise installer.InstallError("Choose 1, 2, or 3 and try again.")
        folders, any_path = installer.choose_access(args.allow_dir, args.allow_any_path, True)
        server = server_executable()
        if not server.is_file():
            raise installer.InstallError(f"The installed server command was not found: {server}")
        server_args = (["--allow-any-path"] if any_path else
                       [value for folder in folders for value in ("--allow-dir", folder)])
        if client == "other":
            print("Add this local stdio server configuration to your client:")
            print(json.dumps({"mcpServers": {installer.NAME: {"command": str(server), "args": server_args}}}, indent=2))
        else:
            if not shutil.which(client):
                raise installer.InstallError(f"{client} was not found on PATH. Install it or select another client.")
            installer.register(client, server, server_args, args.replace)
        print("Restart or reconnect your MCP client, then select opencv-bv-server.")
        return 0
    except (installer.InstallError, OSError, ValueError, EOFError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Setup cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
