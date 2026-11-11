"""Shared file policy, cancellation and resource limits for MCP adapters."""
import contextlib
from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import threading

import anyio.from_thread
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from opencv_bv_tools.runtime import safe_files

MAX_INPUT_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class ServerConfig:
    allow_dirs: tuple[Path, ...] = ()
    allow_any_path: bool = False
    threads: int = 4
    mode: str = "accurate"
    max_megapixels: float = 40.0
    allow_reveal: bool = False
    max_concurrent_calls: int = 1
    ocr_idle_seconds: float = 120.0
    ocr_timeout: float = 120.0
    pdf_timeout: float = 1800.0
    max_pages: int = 100
    pdf_engine: str = "picture"

    def validate(self):
        if not self.allow_dirs and not self.allow_any_path:
            raise ValueError("Pass --allow-dir DIR (repeatable) to choose the folders the server may use.")
        if type(self.threads) is not int or not 1 <= self.threads <= 64:
            raise ValueError("threads must be an integer from 1 to 64.")
        if self.mode not in {"accurate", "fast"}:
            raise ValueError("mode must be accurate or fast.")
        if type(self.max_concurrent_calls) is not int or not 1 <= self.max_concurrent_calls <= 4:
            raise ValueError("max_concurrent_calls must be an integer from 1 to 4.")
        if type(self.max_pages) is not int or not 1 <= self.max_pages <= 100:
            raise ValueError("max_pages must be an integer from 1 to 100.")
        if self.pdf_engine not in {"picture", "mupdf"}:
            raise ValueError("pdf_engine must be picture or mupdf.")
        for name, minimum, maximum in (("ocr_idle_seconds", 0, 3600), ("ocr_timeout", 1, 600),
                                        ("pdf_timeout", 1, 3600)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be finite, from {minimum} to {maximum} seconds.")
        if (type(self.max_megapixels) not in (int, float) or not math.isfinite(self.max_megapixels)
                or not 0.000001 <= self.max_megapixels <= 40):
            raise ValueError("max_megapixels must be finite, greater than zero and at most 40.")


def stop_if_cancelled():
    with contextlib.suppress(RuntimeError):  # direct library call, outside AnyIO worker
        anyio.from_thread.check_cancelled()


class CallSlots:
    """Bound decoded inputs without making cancelled queued calls wait forever."""
    def __init__(self, check, capacity=1):
        self._slots = threading.BoundedSemaphore(capacity)
        self._check = check

    def __enter__(self):
        while True:
            self._check()
            if self._slots.acquire(timeout=0.05):
                return self

    def __exit__(self, *exc):
        self._slots.release()


def annotations(title, *, writes=False):
    return ToolAnnotations(title=title, readOnlyHint=not writes, destructiveHint=writes,
                           idempotentHint=not writes, openWorldHint=False)


class ToolContext:
    def __init__(self, config):
        config.validate()
        self.config = config
        self.folders = safe_files.Folders(None if config.allow_any_path else config.allow_dirs)
        self.file_ops = safe_files
        self.stop_if_cancelled = stop_if_cancelled
        self.call_slots = CallSlots(lambda: self.stop_if_cancelled(), config.max_concurrent_calls)
        self.max_pixels = int(config.max_megapixels * 1_000_000)

    @contextlib.contextmanager
    def call(self):
        with self.call_slots:
            self.stop_if_cancelled()
            try:
                yield
            except (safe_files.UnsafePath, ValueError) as exc:
                raise ToolError(str(exc)) from None
            except OSError as exc:
                raise ToolError(f"File error: {exc.strerror or exc}") from None

    def require(self, extra, *modules):
        missing = [name for name in modules if importlib.util.find_spec(name) is None]
        if missing:
            alternative = "" if extra == "mupdf" else " or opencv-bv-tools[all]"
            raise ToolError(f"This tool needs the '{extra}' dependencies. Install opencv-bv-tools[{extra}] "
                            f"{alternative}, then restart the server.")

    def cancelled(self):
        # AnyIO cancellation is raised, allowing the worker supervisor's finally
        # block to terminate/reap the child and the call context to free its slot.
        self.stop_if_cancelled()
        return False

    def read(self, path):
        if not path:
            raise ValueError("image_path is required.")
        source, real, identity = self.file_ops.open_input(path, self.folders)
        with source:
            if os.fstat(source.fileno()).st_size > MAX_INPUT_BYTES:
                raise ValueError("Input exceeds 32 MiB.")
            data = source.read(MAX_INPUT_BYTES + 1)
        if len(data) > MAX_INPUT_BYTES:
            raise ValueError("Input exceeds 32 MiB.")
        return data, real, {identity}

    def write(self, data, output_path, protect, overwrite, suffix):
        self.stop_if_cancelled()
        written, _ = self.file_ops.write_file(data, output_path, self.folders, protect, overwrite,
                                              suffix=suffix, before_publish=self.stop_if_cancelled)
        return str(written)

    def transform(self, image_path, output_path, overwrite, suffix, function, success, **options):
        if type(overwrite) is not bool:
            raise ValueError("overwrite must be a boolean.")
        if not output_path:
            raise ValueError("output_path is required.")
        data, _, protect = self.read(image_path)
        self.file_ops.check_output(output_path, self.folders, protect, overwrite, suffix=suffix)
        result = function(data, cancelled=self.cancelled, **options)
        self.stop_if_cancelled()
        summary = {key: value for key, value in result.items() if key != "data"}
        if result["status"] in success:
            summary["output_path"] = self.write(result["data"], output_path, protect, overwrite, suffix)
        return json.dumps(summary, ensure_ascii=True)
