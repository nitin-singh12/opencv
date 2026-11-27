# SPDX-License-Identifier: Apache-2.0
"""Connect an installed MCP server to a local client."""
import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import site
import sys
import sysconfig


def server_executable():
    """Use the Python environment's scripts directory, including user installs."""
    try:
        package_root = Path(importlib.metadata.distribution("opencv-bv-tools").locate_file("")).resolve()
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("Install opencv-bv-tools before running setup.") from exc
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
    parser.add_argument('--search', action='store_true', help='enable local image search; downloads 1.54 GB of verified models')
    parser.add_argument('--search-index-dir', type=Path, help='private directory for search indexes (use with --search)')
    args = parser.parse_args(argv)
    if args.search_index_dir and not args.search:
        parser.error('--search-index-dir requires --search')

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
        if client != "other" and not shutil.which(client):
            raise installer.InstallError(f"{client} was not found on PATH. Install it or select another client.")
        from opencv_bv_tools.model_assets import install_models
        print(f"Preparing verified models: {install_models()}")
        server_args = (["--allow-any-path"] if any_path else
                       [value for folder in folders for value in ("--allow-dir", folder)])
        if os.environ.get("OPENCV_BV_MODEL_DIR"):
            from opencv_bv_tools.model_assets import cache_directory
            server_args += ["--model-dir", str(cache_directory().resolve())]
        if args.search:
            from opencv_bv_tools.runtime.search_files import create_private_directory
            search_models = install_models(search=True).resolve()
            index = args.search_index_dir or Path.home() / '.opencv-search-index'
            index = create_private_directory(index.expanduser().absolute())
            server_args += ['--search-model-dir', str(search_models), '--search-index-dir', str(index)]
            if not any_path:
                server_args += ['--allow-dir', str(search_models), '--allow-dir', str(index)]
            print(f'Search models ready: {search_models}')
            print(f'Private search indexes: {index}')
        if client == "other":
            print("Add this local stdio server configuration to your client:")
            print(json.dumps({"mcpServers": {installer.NAME: {"command": str(server), "args": server_args}}}, indent=2))
        else:
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
