# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Build the small installable MCP ZIP from an explicit file allowlist."""
import argparse
from email.parser import BytesParser
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
    'docs/licenses/RAPIDOCR-MODEL-NOTICES.txt': 'docs/licenses/RAPIDOCR-MODEL-NOTICES.txt',
    'docs/licenses/FLATBUFFERS-LICENSE.txt': 'docs/licenses/FLATBUFFERS-LICENSE.txt',
    'docs/licenses/PDFIUM-BUNDLED-TERMS.txt': 'docs/licenses/PDFIUM-BUNDLED-TERMS.txt',
    'docs/licenses/zxing-apache-license.txt': 'docs/licenses/zxing-apache-license.txt',
    'docs/licenses/nanobind-bsd-license.txt': 'docs/licenses/nanobind-bsd-license.txt',
    'docs/licenses/robin-map-mit-license.txt': 'docs/licenses/robin-map-mit-license.txt',
    'docs/licenses/clipper-boost-license.txt': 'docs/licenses/clipper-boost-license.txt',
    'docs/lightweight-branch.md': 'docs/lightweight-branch.md',
}


def bundle_document(data, source, destination, source_commit, unpublished=False):
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
            link = f"https://github.com/nitin-singh12/opencv/blob/{source_commit}/apps/mcp/" + quote(source_target)
        if parsed.fragment:
            link += "#" + parsed.fragment
        return f"[{label}]({link})"

    return re.sub(r"\[([^\]]+)\]\(([^\s)]+)\)", replace, data.decode()).encode()


def build(root, wheel, output, source_commit, unpublished=False, dependency_wheel_dir=None):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("--source-commit must be a full source SHA (or base SHA with --unpublished)")
    with zipfile.ZipFile(wheel) as archive:
        metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Expected one wheel metadata file")
        metadata = BytesParser().parsebytes(archive.read(metadata[0]), headersonly=True)
        if metadata["Name"] != "opencv-bv-tools":
            raise ValueError("Wrong package wheel")
        version = metadata["Version"]
        if not version:
            raise ValueError("Missing package version")
    entries = {name: (root / source).read_bytes() for name, source in FILES.items()}
    for name, source in FILES.items():
        if name.endswith(".md"):
            entries[name] = bundle_document(entries[name], source, name, source_commit, unpublished)
    wheel_path = "wheels/" + wheel.name
    entries[wheel_path] = wheel.read_bytes()
    dependency_wheels = []
    if dependency_wheel_dir is not None:
        wheel_files = sorted(dependency_wheel_dir.glob("*.whl"))
        if not wheel_files:
            raise ValueError("Dependency wheel directory is empty")
        seen = set()
        for dependency in wheel_files:
            with zipfile.ZipFile(dependency) as archive:
                metadata_files = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
                if len(metadata_files) != 1:
                    raise ValueError(f"Invalid dependency wheel: {dependency.name}")
                info = BytesParser().parsebytes(archive.read(metadata_files[0]), headersonly=True)
                name = re.sub(r"[-_.]+", "-", info.get("Name", "")).lower()
                if not name or not info.get("Version") or name in seen or name == "opencv-bv-tools":
                    raise ValueError(f"Duplicate or invalid dependency wheel: {dependency.name}")
                seen.add(name)
            path = "wheels/dependencies/" + dependency.name
            data = dependency.read_bytes()
            entries[path] = data
            dependency_wheels.append({"wheel": path, "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"name": "opencv-bv-server", "package_version": version,
                # The standalone builder cannot prove that either supplied wheel
                # was built from this commit. The validator records that separately.
                "package_source_commit": None,
                "claimed_source_commit": None if unpublished else source_commit,
                "base_commit": source_commit if unpublished else None,
                "source_state": "local-unpublished" if unpublished else "source-claimed",
                "wheel": wheel_path, "dependency_wheels": dependency_wheels,
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
                      "claimed_source_commit": manifest["claimed_source_commit"], "base_commit": manifest["base_commit"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--unpublished", action="store_true", help="Identify local changes against the supplied base commit")
    parser.add_argument("--dependency-wheel-dir", type=Path, help="Include verified dependency wheels from this directory")
    parser.add_argument("--output", type=Path, default=Path("dist/opencv-bv-server-mcp.zip"))
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.wheel, args.output, args.source_commit, args.unpublished, args.dependency_wheel_dir)


if __name__ == "__main__":
    main()
