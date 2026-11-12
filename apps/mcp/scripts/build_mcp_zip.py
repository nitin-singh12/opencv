# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Build the small installable MCP ZIP from an explicit file allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import re
from urllib.parse import quote, urlsplit
import zipfile

FILES = {
    'install.sh': 'install.sh',
    'README.md': 'README.md',
    'LICENSE': 'LICENSE',
    'LICENSING.md': 'LICENSING.md',
    'PROVENANCE.json': 'PROVENANCE.json',
    'THIRD_PARTY_NOTICES.md': 'THIRD_PARTY_NOTICES.md',
    'install-constraints.txt': 'install-constraints.txt',
    'scripts/build_rapidocr.py': 'scripts/build_rapidocr.py',
    'scripts/rapidocr-array.patch': 'scripts/rapidocr-array.patch',
    'scripts/install_mcp.py': 'scripts/install_mcp.py',
    'scripts/smoke_mcp.py': 'scripts/smoke_mcp.py',
    'dev/check_platform.py': 'dev/check_platform.py',
    'tests/test_native_runtime.py': 'tests/test_native_runtime.py',
    'tests/test_ocr_process.py': 'tests/test_ocr_process.py',
    'tests/fixtures/smoke/email.png': 'tests/fixtures/smoke/email.png',
    'tests/fixtures/smoke/email.pdf': 'tests/fixtures/smoke/email.pdf',
    'tests/fixtures/smoke/qr.png': 'tests/fixtures/smoke/qr.png',
    'tests/fixtures/smoke/README.md': 'tests/fixtures/smoke/README.md',
    'tests/fixtures/barcode/upce_0123456.png': 'tests/fixtures/barcode/upce_0123456.png',
    'tests/fixtures/face/grace_hopper.jpg': 'tests/fixtures/face/grace_hopper.jpg',
    'tests/fixtures/face/README.md': 'tests/fixtures/face/README.md',
    'tests/fixtures/face/provenance.json': 'tests/fixtures/face/provenance.json',
    'docs/licenses/YUNET-LICENSE.txt': 'docs/licenses/YUNET-LICENSE.txt',
    'docs/licenses/FLATBUFFERS-LICENSE.txt': 'docs/licenses/FLATBUFFERS-LICENSE.txt',
    'docs/lightweight-branch.md': 'docs/lightweight-branch.md',
}


def bundle_document(data, source, destination, unpublished=False):
    """Keep included links local; link omitted source/test evidence to the repo."""
    destination_by_source = {value: key for key, value in FILES.items()}

    def replace(match):
        label, target = match.groups()
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith(("#", "/")):
            return match.group(0)
        source_target = posixpath.normpath(posixpath.join(posixpath.dirname(source), parsed.path))
        if source_target in destination_by_source:
            link = posixpath.relpath(destination_by_source[source_target], posixpath.dirname(destination) or ".")
        elif unpublished:
            # Local preparation can contain paths that have not been published yet.
            return f"{label} (source checkout: `{source_target}`)"
        else:
            link = "https://github.com/nitin-singh12/opencv/blob/mcp/apps/mcp/" + quote(source_target)
        if parsed.fragment:
            link += "#" + parsed.fragment
        return f"[{label}]({link})"

    return re.sub(r"\[([^\]]+)\]\(([^\s)]+)\)", replace, data.decode()).encode()


def build(root, wheel, output, source_commit, unpublished=False, rapidocr_wheel=None):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("--source-commit must be a full source SHA (or base SHA with --unpublished)")
    with zipfile.ZipFile(wheel) as archive:
        metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Expected one wheel metadata file")
        text = archive.read(metadata[0]).decode()
        if not re.search(r"(?m)^Name: opencv-bv-tools$", text):
            raise ValueError("Wrong package wheel")
        version = re.search(r"(?m)^Version: (.+)$", text).group(1)
    entries = {name: (root / source).read_bytes() for name, source in FILES.items()}
    for name, source in FILES.items():
        if name.endswith(".md"):
            entries[name] = bundle_document(entries[name], source, name, unpublished)
    wheel_path = "wheels/" + wheel.name
    entries[wheel_path] = wheel.read_bytes()
    dependencies=[]
    if "rapidocr==3.10.0+bv.1" in text:
        companion = rapidocr_wheel or root / "wheels/rapidocr-3.10.0+bv.1-py3-none-any.whl"
        with zipfile.ZipFile(companion) as archive:
            companion_meta = archive.read("rapidocr-3.10.0+bv.1.dist-info/METADATA").decode()
            if not re.search(r"(?m)^Version: 3\.10\.0\+bv\.1$", companion_meta):
                raise ValueError("Wrong RapidOCR companion wheel")
        dependency_path = "wheels/" + companion.name
        entries[dependency_path] = companion.read_bytes()
        dependencies.append({"wheel": dependency_path, "sha256": hashlib.sha256(entries[dependency_path]).hexdigest()})
    manifest = {"name": "opencv-bv-server", "package_version": version,
                "package_source_commit": None if unpublished else source_commit,
                "base_commit": source_commit if unpublished else None,
                "source_state": "local-unpublished" if unpublished else "committed",
                "wheel": wheel_path, "dependency_wheels": dependencies,
                "wheel_sha256": hashlib.sha256(entries[wheel_path]).hexdigest(),
                "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(entries.items())}}
    entries["bundle.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo("opencv-bv-server-mcp/" + name, date_time=(2026, 11, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name == "install.sh" else 0o100644) << 16
            archive.writestr(info, data)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n")
    print(json.dumps({"zip": str(output.resolve()), "bytes": output.stat().st_size,
                      "sha256": digest, "files": len(entries), "source_state": manifest["source_state"],
                      "package_source_commit": manifest["package_source_commit"], "base_commit": manifest["base_commit"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--rapidocr-wheel", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--unpublished", action="store_true", help="Identify local changes against the supplied base commit")
    parser.add_argument("--output", type=Path, default=Path("dist/opencv-bv-server-mcp.zip"))
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.wheel, args.output, args.source_commit, args.unpublished, args.rapidocr_wheel)


if __name__ == "__main__":
    main()
