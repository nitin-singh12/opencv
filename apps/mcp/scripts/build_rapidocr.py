"""Build our array-only RapidOCR wheel from the unchanged, hash-pinned upstream wheel.

No network access, model changes, or dependency-resolution bypass. Download the
upstream wheel separately, then pass its path. The optional upstream image/URL
helpers remain available through the `images` and `downloads` extras.
"""

import argparse
import base64
import csv
from email.parser import BytesParser
from email.policy import compat32
import hashlib
import io
from pathlib import Path
import re
import zipfile

UPSTREAM_SHA256 = "2fc34e26cd0f48514804a94f8832283039c7334076bff60768b1fed9ad068050"
VERSION = "3.10.0+bv.1"
PATCHED = {
    "rapidocr/utils/load_image.py",
    "rapidocr/utils/vis_res.py",
    "rapidocr/utils/download_file.py",
}


def apply_patch(files, patch):
    """Apply exact unified hunks, with no fuzzy matching or external patch tool."""
    sections = re.split(r"(?m)^--- a/", patch)[1:]
    changed = set()
    for section in sections:
        lines = section.splitlines(keepends=True)
        name = lines[0].strip()
        if name not in PATCHED or lines[1].strip() != "+++ b/" + name:
            raise ValueError("Unexpected patch path")
        original = files[name].decode().splitlines(keepends=True)
        output, cursor, index = [], 0, 2
        while index < len(lines):
            match = re.fullmatch(
                r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@.*\n", lines[index]
            )
            if not match:
                raise ValueError("Invalid patch hunk")
            start = int(match[1]) - 1
            if start < cursor:
                raise ValueError("Overlapping patch hunks")
            output.extend(original[cursor:start])
            cursor = start
            index += 1
            while index < len(lines) and not lines[index].startswith("@@"):
                line = lines[index]
                index += 1
                if line[0] in " -":
                    if cursor >= len(original) or original[cursor] != line[1:]:
                        raise ValueError(f"Patch context mismatch: {name}:{cursor + 1}")
                    cursor += 1
                if line[0] in " +":
                    output.append(line[1:])
                elif line[0] != "-":
                    raise ValueError("Invalid patch line")
            # The next hunk follows, or the unchanged tail is appended below.
        output.extend(original[cursor:])
        files[name] = "".join(output).encode()
        changed.add(name)
    if changed != PATCHED:
        raise ValueError("Incomplete patch")


def build(upstream, output):
    raw = upstream.read_bytes()
    if hashlib.sha256(raw).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Upstream wheel SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    apply_patch(files, Path(__file__).with_name("rapidocr-array.patch").read_text())
    old_info = "rapidocr-3.10.0.dist-info/"
    new_info = f"rapidocr-{VERSION}.dist-info/"
    metadata = BytesParser(policy=compat32).parsebytes(files[old_info + "METADATA"])
    metadata.replace_header("Version", VERSION)
    requirements = metadata.get_all("Requires-Dist", [])
    del metadata["Requires-Dist"]
    for requirement in requirements:
        name = re.split(r"[ ;(<>=!~\[]", requirement, 1)[0].lower()
        if name in {"pillow", "requests"}:
            requirement += (
                '; extra == "' + ("images" if name == "pillow" else "downloads") + '"'
            )
        metadata["Requires-Dist"] = requirement
    metadata["Provides-Extra"] = "images"
    metadata["Provides-Extra"] = "downloads"
    files[old_info + "METADATA"] = metadata.as_bytes()
    files = {
        name.replace(old_info, new_info, 1): data
        for name, data in files.items()
        if not name.endswith("/RECORD")
    }
    files[new_info + "BV-MODIFICATIONS.txt"] = (
        f"RapidOCR {VERSION}: local OpenCV BV array-input build.\n"
        f"Upstream wheel SHA256: {UPSTREAM_SHA256}\n"
        "Modified utils/load_image.py, utils/vis_res.py and utils/download_file.py.\n"
        "Pillow and requests are optional; array OCR, algorithms and model bytes are unchanged.\n"
        "Install images for file/encoded-image/text-visualization helpers and downloads for HTTP helpers.\n"
        "URL image inputs require both extras. This is not an official upstream release.\n"
    ).encode()
    record_name = new_info + "RECORD"
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for name, data in sorted(files.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .rstrip(b"=")
            .decode()
        )
        writer.writerow([name, "sha256=" + digest, len(data)])
    writer.writerow([record_name, "", ""])
    files[record_name] = record.getvalue().encode()
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"rapidocr-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".whl.sha256").write_text(f"{digest}  {target.name}\n")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--output", type=Path, default=Path("wheels"))
    args = parser.parse_args()
    print(build(args.upstream, args.output).resolve())
