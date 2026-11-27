# SPDX-License-Identifier: Apache-2.0
"""Single-owner cancellable encoder process, unloaded after idle time."""
import json
import math
import os
import selectors
import subprocess
import sys
import threading
import time

from .process_support import PipePoller, start_worker, stop_worker

MESSAGES = {
    "missing_search_dependencies": "Install the OpenCV and NumPy dependencies before starting the server.",
    "missing_search_models": "The FG-CLIP2 export is incomplete. Supply the four verified model assets in --search-model-dir.",
    "device_out_of_memory": "The search worker ran out of memory.",
    "fgclip2_integrity_error": "FG-CLIP2 files do not match the approved pinned revision.",
    "encoder_failed": "The local encoder could not load or run. Check the model bundle and dependency versions.",
}


class SearchError(ValueError):
    pass


class EncoderProcess:
    def __init__(self, config):
        self.config = config
        from . import safe_files
        from pathlib import Path
        self.allowed_dirs = None if config.allow_any_path else [str(safe_files._real_folder(Path(p))) for p in config.allow_dirs]
        self.scratch_parent = None
        self.process = None
        self.scratch = None
        self.timer = None
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.request_id = 0
        self.generation = 0

    def _stop(self):
        self.generation += 1
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        if self.process is not None:
            stop_worker(self.process, group=True, grace=0.5)
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None

    def close(self):
        self.closed.set()
        with self.lock:
            self._stop()

    def _expire(self, generation):
        with self.lock:
            if self.generation == generation:
                self._stop()

    def _arm_idle(self):
        self.generation += 1
        self.timer = threading.Timer(self.config.search_idle_seconds, self._expire, (self.generation,))
        self.timer.daemon = True
        self.timer.start()

    def request(self, kind, raw=b"", check=lambda: None):
        while not self.lock.acquire(timeout=0.05):
            check()
        preserve_worker = False
        try:
            check()
            if self.closed.is_set():
                raise SearchError("Image search is closed.")
            self.generation += 1
            if self.timer:
                self.timer.cancel()
            if self.process is None or self.process.poll() is not None:
                self._stop()
                from . import safe_files
                if self.scratch_parent is None:
                    raise SearchError("Search index directory has not been configured.")
                folders = safe_files.Folders.from_canonical(self.allowed_dirs)
                from . import search_files
                with search_files.folder(self.scratch_parent, folders) as real:
                    if real != self.scratch_parent:
                        raise SearchError("Search index directory changed; restart the server.")
                    search_files.verify_private_directory(real)
                    self.scratch = search_files.temporary_directory(real)
                env = {**os.environ,
                       "OPENCV_IO_MAX_IMAGE_PIXELS": str(min(int(self.config.max_megapixels * 1_000_000), 40_000_000)),
                       "OPENCV_IO_MAX_IMAGE_WIDTH": "16384", "OPENCV_IO_MAX_IMAGE_HEIGHT": "16384",
                       **{k: str(self.config.threads) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
                self.process = start_worker([sys.executable, "-m", "opencv_bv_tools.runtime.search_worker"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    bufsize=0, env=env, start_new_session=True)
            self.request_id += 1
            options = dict(kind=kind, input_bytes=len(raw), request_id=self.request_id,
                           allowed_dirs=self.allowed_dirs,
                           snapshot_dir=self.scratch.name, model_dir=str(self.config.search_model_dir), backend=self.config.search_backend,
                           device=self.config.search_device, threads=self.config.threads,
                           max_pixels=min(int(self.config.max_megapixels * 1_000_000), 40_000_000))
            result = self._exchange(options, raw, check)
            if result.get("request_id") != self.request_id:
                raise SearchError("Invalid encoder response identity.")
            if result.get("status") == "error":
                reason = result.get("reason", "encoder_failed")
                if kind == "image" and reason in {"invalid_image", "animated_image", "image_size_limit", "unsupported_format"}:
                    # A bounded per-photo rejection does not invalidate the
                    # already loaded encoder. Keep it warm for the next photo.
                    check()
                    self._arm_idle()
                    preserve_worker = True
                raise SearchError(MESSAGES.get(reason, reason))
            if result.get("status") != "ok" or len(result.get("fingerprint", "")) != 64:
                raise SearchError("Invalid encoder response.")
            if kind != "info":
                vector = result.get("vector")
                if (not isinstance(vector, list) or not 1 <= len(vector) <= 2048 or
                        any(type(v) not in (int, float) or not math.isfinite(v) for v in vector) or
                        not 0.99 <= sum(v*v for v in vector) <= 1.01):
                    raise SearchError("Invalid encoder vector.")
            check()
            self._arm_idle()
            return result
        except BaseException:
            if not preserve_worker:
                self._stop()
            raise
        finally:
            self.lock.release()

    def _exchange(self, options, raw, check):
        chunks = [memoryview(json.dumps(options).encode() + b"\n")]
        if raw:
            chunks.append(memoryview(raw))
        part = offset = 0
        pending = bytearray()
        deadline = time.monotonic() + self.config.search_timeout
        with PipePoller() as poller:
            poller.register(self.process.stdin, selectors.EVENT_WRITE)
            poller.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                check()
                if self.closed.is_set():
                    raise SearchError("Image search is closed.")
                if time.monotonic() > deadline:
                    raise SearchError("Encoder request exceeded search_timeout.")
                for key, event in poller.select(0.02):
                    try:
                        if event == selectors.EVENT_WRITE:
                            offset += poller.write(key.fd, chunks[part][offset:offset + 65536])
                            if offset == len(chunks[part]):
                                part, offset = part + 1, 0
                            if part == len(chunks):
                                poller.unregister(self.process.stdin)
                        else:
                            block = poller.read(key.fd, 65536)
                            if not block:
                                raise SearchError("Encoder worker exited before replying.")
                            pending.extend(block)
                            if len(pending) > 131072:
                                raise SearchError("Encoder response exceeded its bound.")
                            if b"\n" in pending:
                                line, trailing = pending.split(b"\n", 1)
                                if trailing or part != len(chunks):
                                    raise SearchError("Unexpected encoder response bytes.")
                                result = json.loads(line)
                                if not isinstance(result, dict):
                                    raise SearchError("Invalid encoder response.")
                                return result
                    except BlockingIOError:
                        continue
