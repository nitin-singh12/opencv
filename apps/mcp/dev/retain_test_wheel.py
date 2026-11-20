# SPDX-License-Identifier: Apache-2.0
"""Retain only the exact wheel that passed native package validation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def main(work, destination):
    work, destination = Path(work), Path(destination)
    report = json.loads((work / "package-report.json").read_text(encoding="utf-8"))
    wheels = list((work / "dist").glob("opencv_bv_tools-*.whl"))
    if report.get("status") != "passed" or len(wheels) != 1:
        raise ValueError("A successful validation report and exactly one wheel are required")
    wheel = wheels[0]
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != report.get("wheel_sha256"):
        raise ValueError("Wheel checksum does not match the validation report")
    commit = os.environ.get("GITHUB_SHA")
    if commit and report.get("commit") != commit:
        raise ValueError("Validated commit does not match this workflow")
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(wheel, destination / wheel.name)
    (destination / f"{wheel.name}.sha256").write_text(f"{digest}  {wheel.name}\n", encoding="utf-8")
    keys = ("commit", "status", "os", "architecture", "python", "runner_image",
            "image_version", "package_version", "wheel_sha256", "nine_tool_smoke", "native_contracts")
    summary = {key: report.get(key) for key in keys}
    (destination / f"{wheel.name}.validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Retained validated wheel: {wheel.name} (SHA-256 {digest})")


if __name__ == "__main__":
    main(*sys.argv[1:])
