"""Original preservation, publication races/failures, and installed CLI contract."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Barrier

from PIL import Image
import pytest

from opencv_bv_tools import ConversionError
from opencv_bv_tools import image_conversion as cli


@pytest.fixture
def source(tmp_path):
    output = BytesIO()
    Image.new("RGB", (23, 13), "red").save(output, format="PNG")
    path = tmp_path / "input.png"
    path.write_bytes(output.getvalue())
    return path


def test_preserve_original_and_decode_content_not_extension(source):
    renamed = source.with_suffix(".unknown")
    source.rename(renamed)
    before = renamed.read_bytes(), renamed.stat().st_ino
    result = cli.convert_file(renamed, source.with_suffix(".WEBP"))
    assert result.input_format == "PNG"
    assert (renamed.read_bytes(), renamed.stat().st_ino) == before


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("alias", ["same", "hardlink", "symlink"])
def test_original_alias_refused(source, overwrite, alias):
    before = source.read_bytes(), source.stat().st_ino
    target = source.parent / "alias.png"
    if alias == "same":
        target = source
    elif alias == "hardlink":
        os.link(source, target)
    else:
        target.symlink_to(source)
    with pytest.raises(ConversionError):
        cli.convert_file(source, target, overwrite=overwrite)
    assert (source.read_bytes(), source.stat().st_ino) == before
    assert not list(source.parent.glob(".opencv-bv-*"))


def test_destination_symlink_refused(source):
    target = source.parent / "dangling.webp"
    target.symlink_to(source.parent / "absent")
    with pytest.raises(ConversionError):
        cli.convert_file(source, target, overwrite=True)
    assert target.is_symlink()


def test_existing_output_and_explicit_overwrite(source):
    target = source.with_suffix(".webp")
    target.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        cli.convert_file(source, target)
    assert target.read_bytes() == b"existing"
    result = cli.convert_file(source, target, overwrite=True)
    assert target.read_bytes() == result.data


@pytest.mark.parametrize("overwrite", ["false", 1, None])
def test_overwrite_requires_boolean(source, overwrite):
    with pytest.raises(ConversionError, match="boolean"):
        cli.convert_file(source, source.with_suffix(".webp"), overwrite=overwrite)


def test_write_failure_cleans_partial_temp(source, monkeypatch):
    target = source.with_suffix(".webp")
    target.write_bytes(b"existing")
    real_temporary_file = cli.tempfile.NamedTemporaryFile
    class BrokenWriter:
        def __enter__(self):
            self.handle = real_temporary_file(dir=source.parent, prefix=".opencv-bv-", delete=False)
            self.name = self.handle.name
            return self
        def write(self, data):
            self.handle.write(data[:4])
            raise OSError("injected short write")
        def __exit__(self, *args):
            self.handle.close()
    monkeypatch.setattr(cli.tempfile, "NamedTemporaryFile", lambda **kwargs: BrokenWriter())
    with pytest.raises(OSError, match="short write"):
        cli.convert_file(source, target, overwrite=True)
    assert target.read_bytes() == b"existing"
    assert not list(source.parent.glob(".opencv-bv-*"))


def test_overwrite_does_not_mutate_other_hardlink(source):
    target = source.with_suffix(".webp")
    target.write_bytes(b"existing")
    alias = source.parent / "other-link"
    os.link(target, alias)
    cli.convert_file(source, target, overwrite=True)
    assert alias.read_bytes() == b"existing"
    assert target.stat().st_ino != alias.stat().st_ino


def test_readonly_target_refused(source):
    target = source.with_suffix(".webp")
    target.write_bytes(b"readonly")
    target.chmod(0o444)
    with pytest.raises(PermissionError):
        cli.convert_file(source, target, overwrite=True)
    assert target.read_bytes() == b"readonly"


@pytest.mark.parametrize("failure", ["fsync", "replace", "link"])
def test_failure_preserves_files_and_cleans_temp(source, monkeypatch, failure):
    target = source.with_suffix(".webp")
    before = source.read_bytes()
    if failure != "link":
        target.write_bytes(b"existing")
    def broken(*args, **kwargs):
        raise OSError("injected publication failure")
    monkeypatch.setattr(cli.os, failure, broken)
    with pytest.raises(OSError, match="injected"):
        cli.convert_file(source, target, overwrite=failure != "link")
    assert source.read_bytes() == before
    assert target.read_bytes() == b"existing" if failure != "link" else not target.exists()
    assert not list(source.parent.glob(".opencv-bv-*"))


def test_invalid_image_preserves_existing_output(source):
    source.write_bytes(b"corrupt")
    target = source.with_suffix(".webp")
    target.write_bytes(b"existing")
    with pytest.raises(ConversionError):
        cli.convert_file(source, target, overwrite=True)
    assert target.read_bytes() == b"existing"
    assert not list(source.parent.glob(".opencv-bv-*"))


def test_concurrent_no_clobber(source, monkeypatch):
    barrier = Barrier(2)
    real_link = os.link
    def synchronized_link(*args, **kwargs):
        barrier.wait(timeout=10)
        return real_link(*args, **kwargs)
    monkeypatch.setattr(cli.os, "link", synchronized_link)
    target = source.with_suffix(".webp")
    def worker():
        try:
            return cli.convert_file(source, target).data
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: worker(), range(2)))
    assert results.count(None) == 1
    assert target.read_bytes() == next(value for value in results if value is not None)
    assert not list(source.parent.glob(".opencv-bv-*"))


def test_destination_created_during_conversion(source, monkeypatch):
    target = source.with_suffix(".webp")
    original_convert = cli.convert_image
    def convert_and_race(*args, **kwargs):
        result = original_convert(*args, **kwargs)
        target.write_bytes(b"concurrent writer")
        return result
    monkeypatch.setattr(cli, "convert_image", convert_and_race)
    with pytest.raises(FileExistsError):
        cli.convert_file(source, target)
    assert target.read_bytes() == b"concurrent writer"


def test_input_size_limit_before_read(source):
    with pytest.raises(ConversionError, match="bytes"):
        cli.convert_file(source, source.with_suffix(".webp"), max_input_bytes=1)


@pytest.mark.parametrize("limit", [-1, 0, True, "10"])
def test_file_invalid_size_limit(source, limit):
    with pytest.raises(ConversionError):
        cli.convert_file(source, source.with_suffix(".webp"), max_input_bytes=limit)


def test_missing_parent(source):
    with pytest.raises(FileNotFoundError):
        cli.convert_file(source, source.parent / "absent" / "out.png")


def test_nonregular_output(source):
    target = source.parent / "directory.png"
    target.mkdir()
    with pytest.raises(ConversionError):
        cli.convert_file(source, target, overwrite=True)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO")
def test_fifo_input_rejected_without_blocking(tmp_path):
    source = tmp_path / "fifo"
    os.mkfifo(source)
    with pytest.raises(ConversionError):
        cli.convert_file(source, tmp_path / "out.png")


@pytest.mark.parametrize("output,options", [
    ("out.gif", {}), ("out", {}), ("out.png", {"output_format": "JPEG"}),
])
def test_extension_contract(source, output, options):
    with pytest.raises(ConversionError):
        cli.convert_file(source, source.parent / output, **options)


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "opencv_bv_tools.image_conversion", *map(str, args)],
        capture_output=True, text=True, timeout=20,
    )


def test_cli_smoke_and_no_overwrite(source):
    target = source.with_suffix(".webp")
    result = run_cli(source, target)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["bytes_written"] == target.stat().st_size
    assert payload["width"] == 23 and payload["height"] == 13
    assert payload["output_format"] == "WEBP"
    second = run_cli(source, target)
    assert second.returncode == 2 and "already exists" in second.stderr
    assert "Traceback" not in second.stderr
    assert run_cli(source, target, "--overwrite").returncode == 0


@pytest.mark.parametrize("args", [
    ["--quality", "0"], ["--quality", "abc"], ["--jpeg-background", "256,0,0"],
    ["--jpeg-background", "red"], ["--jpeg-background", "0,0"],
])
def test_cli_invalid_options(source, args):
    target = source.with_suffix(".jpg")
    result = run_cli(source, target, *args)
    assert result.returncode == 2 and "Traceback" not in result.stderr
    assert not target.exists()


def test_cli_missing_source(tmp_path):
    result = run_cli(tmp_path / "absent", tmp_path / "out.png")
    assert result.returncode == 2 and "Traceback" not in result.stderr


def test_cli_explicit_background_and_lossy_webp(source):
    jpeg = source.with_suffix(".jpg")
    result = run_cli(source, jpeg, "--jpeg-background", "255,255,255")
    assert result.returncode == 0, result.stderr
    webp = source.with_suffix(".webp")
    result = run_cli(source, webp, "--lossy-webp", "--quality", "85")
    assert result.returncode == 0, result.stderr
    assert "Lossy WebP" in json.loads(result.stdout)["warnings"][0]
