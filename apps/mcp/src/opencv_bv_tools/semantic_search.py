# SPDX-License-Identifier: Apache-2.0
"""Opt-in local semantic image search with bounded, incremental SQLite snapshots.

Indexes contain paths, content hashes and normalized vectors, not copied photos.
A single server owns an index directory. Jobs checkpoint completed embeddings;
cancellation retains those checkpoints and a later index call resumes safely.
"""
import array
import base64
import binascii
import hmac
from hashlib import sha256
import heapq
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import threading
import time
import uuid

from .runtime import safe_files, search_files
from .runtime.search_process import EncoderProcess, SearchError

MAX_DB_BYTES = 512 * 1024 * 1024
MAX_CURSOR_CHARS = 1024
MAX_IMAGES = 100_000
MAX_ENTRIES = 1_000_000
from .runtime.search_images import EXTENSIONS
SCHEMA = 1
META_SQL = "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
IMAGES_SQL = "CREATE TABLE images (path TEXT PRIMARY KEY, digest TEXT NOT NULL, vector BLOB NOT NULL)"


class Cancelled(Exception):
    pass


def validate_search_config(config):
    if not config.search_model_dir and not config.search_index_dir:
        return
    if not config.search_model_dir or not config.search_index_dir:
        raise ValueError("Image search requires both --search-model-dir and --search-index-dir.")
    if sys.platform not in {"linux", "darwin", "win32"} or sys.version_info < (3, 11):
        raise ValueError("Image search requires Linux/macOS with Python 3.11+, or Windows with Python 3.12+.")
    from .runtime.process_support import check_platform
    check_platform()
    if config.search_backend != "fgclip2" or config.search_device != "cpu":
        raise ValueError("OpenCV DNN search currently supports fgclip2 on CPU only.")
    for name, lower, upper in (("search_timeout", 1, 600), ("search_idle_seconds", 1, 3600), ("search_job_timeout", 60, 21600)):
        value = getattr(config, name)
        if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} must be finite, from {lower} to {upper} seconds.")


def walk_images(root, folders, recursive, check):
    """Never follow links. Verify actual open directories before listing them."""
    if sys.platform == "win32":
        yield from search_files.walk_windows(root, folders, recursive, check, extensions=EXTENSIONS, max_entries=MAX_ENTRIES)
        return
    visited = 0
    def visit(fd, depth):
        nonlocal visited
        check()
        real = safe_files._fd_path(fd)
        folders.check(real, "The image folder")
        # listdir(fd) is avoided: a hostile huge directory must not allocate an
        # unbounded Python list before the entry budget can stop enumeration.
        with os.scandir(fd) as entries:
            for entry in entries:
                check()
                visited += 1
                if visited > MAX_ENTRIES:
                    raise SearchError("Folder scan exceeds one million entries; choose a smaller folder.")
                if entry.is_symlink() or entry.name.startswith(".opencv-search-"):
                    continue
                if entry.is_dir(follow_symlinks=False) and recursive:
                    if depth >= 32:
                        raise SearchError("Folder depth exceeds 32; choose a smaller folder.")
                    child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                    try:
                        yield from visit(child, depth + 1)
                    finally:
                        os.close(child)
                elif entry.is_file(follow_symlinks=False) and Path(entry.name).suffix.lower() in EXTENSIONS:
                    # Actual-file policy is checked again when opened for bytes.
                    yield real / entry.name
    fd, _ = safe_files._open_folder(root, folders, create=False)
    try:
        yield from visit(fd, 0)
    finally:
        os.close(fd)


def new_database(meta):
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA trusted_schema=OFF")
    db.executescript(META_SQL + ";" + IMAGES_SQL + ";")
    put_meta(db, meta)
    return db


def put_meta(db, meta):
    db.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                   [(k, json.dumps(v, allow_nan=False)) for k, v in meta.items()])
    db.commit()


def load_database(data, check=lambda: None):
    if len(data) > MAX_DB_BYTES or not data.startswith(b"SQLite format 3\0"):
        raise SearchError("Invalid or oversized image-search index.")
    db = sqlite3.connect(":memory:")
    try:
        check()
        db.deserialize(data)
        db.set_progress_handler(lambda: (check(), 0)[1], 10000)
        db.execute("PRAGMA trusted_schema=OFF")
        db.execute("PRAGMA query_only=ON")
        db.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
        db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
        objects = list(db.execute("SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"))
        if set(objects) != {("table", "meta"), ("table", "images")} or len(objects) != 2:
            raise SearchError("Invalid index schema.")
        sql = dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='table'"))
        if sql != {"meta": META_SQL, "images": IMAGES_SQL}:
            raise SearchError("Invalid canonical index schema.")
        if ([x[1:3] for x in db.execute("PRAGMA table_info(meta)")] != [("key", "TEXT"), ("value", "TEXT")] or
            [x[1:3] for x in db.execute("PRAGMA table_info(images)")] != [("path", "TEXT"), ("digest", "TEXT"), ("vector", "BLOB")]):
            raise SearchError("Invalid index columns.")
        if db.execute("SELECT count(*) FROM meta").fetchone()[0] > 16:
            raise SearchError("Invalid index metadata.")
        meta = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM meta")}
        if (meta.get("schema") != SCHEMA or type(meta.get("dimension")) is not int or not 1 <= meta["dimension"] <= 2048 or
                not re.fullmatch("[0-9a-f]{64}", meta.get("fingerprint", "")) or
                type(meta.get("root")) is not str or not Path(meta["root"]).is_absolute() or
                type(meta.get("complete")) is not bool or type(meta.get("recursive")) is not bool or
                type(meta.get("encoder")) is not dict or meta["encoder"].get("fingerprint") != meta["fingerprint"]):
            raise SearchError("Invalid index metadata.")
        count = db.execute("SELECT count(*) FROM images").fetchone()[0]
        if count > MAX_IMAGES:
            raise SearchError("Index exceeds 100,000 images.")
        for row_number, (path, digest, vector) in enumerate(db.execute("SELECT path,digest,vector FROM images")):
            if row_number % 128 == 0:
                check()
            if (type(path) is not str or len(path) > 8192 or not Path(path).is_absolute() or
                    not re.fullmatch("[0-9a-f]{64}", digest) or
                    type(vector) is not bytes or len(vector) != meta["dimension"] * 4):
                raise SearchError("Invalid index entry.")
            unpack_vector(vector)  # Reject NaN/zero/non-normalized vectors before ranking.
        db.execute("PRAGMA query_only=OFF")
        return db, meta
    except BaseException:
        db.close()
        raise


def pack_vector(values):
    if not 1 <= len(values) <= 2048:
        raise SearchError("Embedding dimension exceeds limit.")
    packed = array.array("f", values)
    if sys.byteorder != "little":
        packed.byteswap()
    raw = packed.tobytes()
    unpack_vector(raw)
    return raw


def unpack_vector(raw):
    vector = array.array("f")
    vector.frombytes(raw)
    if sys.byteorder != "little":
        vector.byteswap()
    if any(not math.isfinite(v) for v in vector) or not 0.99 <= sum(v*v for v in vector) <= 1.01:
        raise SearchError("Index contains invalid normalized vectors.")
    return vector


class SearchService:
    def __init__(self, context, encoder=None):
        self.context = context
        self.config = context.config
        self.enabled = bool(self.config.search_model_dir and self.config.search_index_dir)
        self.encoder = encoder or EncoderProcess(self.config)
        self.encoder.allowed_dirs = None if context.folders.allowed is None else [str(p) for p in context.folders.allowed]
        self.lock = threading.RLock()
        self.busy = threading.Lock()
        self.cancelled = threading.Event()
        self.closed = threading.Event()
        self.thread = None
        self.job = None
        self.owner = None
        self.index_guard = None
        self.index_status = {}
        self.index_identity = {}
        self.index_revisions = {}
        self._cursor_secret = os.urandom(32)
        if self.enabled:
            # Operator opts in to model and index directories at server launch;
            # both still have to fall within allowed-folder policy.
            for path in (self.config.search_model_dir, self.config.search_index_dir):
                with search_files.folder(path, context.folders):
                    pass
            with search_files.folder(self.config.search_index_dir, context.folders) as real:
                self.index_dir = search_files.verify_private_directory(real)
            if sys.platform == "win32":
                # Keep ancestor handles open until the worker is reaped and state is closed.
                self.index_guard = search_files.folder(self.index_dir, context.folders)
                self.index_guard.__enter__()
            self.encoder.scratch_parent = self.index_dir

    def _configured(self):
        if not self.enabled:
            raise SearchError("Image search is disabled. Start with --search-model-dir and --search-index-dir inside allowed folders.")
        if self.closed.is_set():
            raise SearchError("Image search is closed.")

    def _claim(self):
        if self.owner is not None:
            return
        try:
            self.owner = search_files.claim_owner(self.index_dir / ".opencv-search-owner", self.context.folders)
        except BlockingIOError:
            raise SearchError("Another server owns this index directory.") from None

    def _path(self, name):
        if type(name) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            raise SearchError("index_name must be 1–64 letters, numbers, underscores or hyphens.")
        return self.index_dir / (name + ".sqlite")

    def _read(self, name, check=lambda: None):
        path = self._path(name)
        if not path.exists():
            self.index_identity[name] = None
            self.index_revisions[name] = None
            return None, None
        source, _, _ = safe_files.open_input(path, self.context.folders)
        with source:
            st = os.fstat(source.fileno())
            self.index_identity[name] = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
            if st.st_nlink != 1 or st.st_size > MAX_DB_BYTES:
                raise SearchError("Index must be a single-link regular file no larger than 512 MiB.")
            chunks, size = [], 0
            while chunk := source.read(1 << 20):
                check()
                size += len(chunk)
                if size > MAX_DB_BYTES:
                    raise SearchError("Index exceeds 512 MiB.")
                chunks.append(chunk)
            data = b"".join(chunks)
        try:
            db, meta = load_database(data, check)
            self.index_revisions[name] = sha256(data).hexdigest()
            with self.lock:
                self.index_status[name] = dict(meta, total=db.execute("SELECT count(*) FROM images").fetchone()[0])
            return db, meta
        except (sqlite3.DatabaseError, TypeError, KeyError, json.JSONDecodeError) as exc:
            check()  # sqlite callbacks translate cancellation into OperationalError.
            raise SearchError("Invalid image-search index.") from exc

    def _checkpoint(self, db, meta, check):
        put_meta(db, meta)
        raw = db.serialize()
        if len(raw) > MAX_DB_BYTES:
            raise SearchError("Index exceeds 512 MiB; use smaller folders or separate indexes.")
        # Cancel acknowledgement and publication share a lock: after cancel()
        # returns, this job cannot publish any further checkpoints.
        with self.lock:
            check()
            path = self._path(self.job["index_name"])
            expected = self.index_identity.get(self.job["index_name"])
            def before_publish():
                check()
                try:
                    st = path.lstat()
                    current = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
                except FileNotFoundError:
                    current = None
                if current != expected:
                    raise SearchError("Index changed externally; no checkpoint was published.")
            search_files.write_snapshot(raw, path, self.context.folders, overwrite=expected is not None,
                                        before_publish=before_publish)
            st = path.stat()
            self.index_identity[self.job["index_name"]] = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
            self.job["checkpointed"] = self.job["encoded"]
            self.index_status[self.job["index_name"]] = dict(meta, total=db.execute("SELECT count(*) FROM images").fetchone()[0])

    def start(self, root_path, index_name="photos", recursive=True, max_images=MAX_IMAGES):
        self._configured()
        if type(recursive) is not bool or type(max_images) is not int or not 1 <= max_images <= MAX_IMAGES:
            raise SearchError("recursive must be boolean; max_images must be 1 to 100,000.")
        self._path(index_name)
        with search_files.folder(root_path, self.context.folders) as root:
            pass
        if not self.busy.acquire(blocking=False):
            raise SearchError("Image search is busy; check status or cancel the current index job.")
        try:
            with self.lock:
                self._claim()
                self.cancelled.clear()
                self.job = dict(job_id=uuid.uuid4().hex, index_name=index_name, status="running", phase="loading",
                                scanned=0, encoded=0, unchanged=0, failed=0, checkpointed=0, removed=0,
                                started_at=time.time(), errors=[], error_counts={})
                self.thread = threading.Thread(target=self._run, args=(str(root), index_name, recursive, max_images), daemon=True)
                self.thread.start()
                return dict(self.job)
        except BaseException:
            self.busy.release()
            raise

    def _update(self, **kwargs):
        with self.lock:
            self.job.update(kwargs)

    def _run(self, root, name, recursive, max_images):
        db = None
        deadline = time.monotonic() + self.config.search_job_timeout
        def check():
            if self.cancelled.is_set() or self.closed.is_set():
                raise Cancelled()
            if time.monotonic() >= deadline:
                raise SearchError("Index job reached its configured time budget. Run index_images again to resume checkpoints.")
        try:
            # Background jobs explicitly acquire the same resource budget as
            # ordinary MCP calls. Status/cancel do not need this slot.
            while not self.context.call_slots._slots.acquire(timeout=0.05):
                check()
            try:
                info = self.encoder.request("info", check=check)
                db, meta = self._read(name, check)
                if db is not None and (meta["root"] != root or meta["fingerprint"] != info["fingerprint"] or meta.get("recursive") != recursive):
                    raise SearchError("Index root, recursion or encoder differs. Use a new index_name; existing data was not changed.")
                self._update(phase="indexing", encoder={k: v for k, v in info.items() if k not in {"status", "request_id"}})
                seen = set()
                # Narrow image reads to the explicit root even if server permits more.
                folders = safe_files.Folders.from_canonical((Path(root),))
                for path in walk_images(root, folders, recursive, check):
                    check()
                    if len(seen) >= max_images:
                        raise SearchError("Image count exceeds max_images. Completed checkpoints are resumable; choose a larger limit or smaller folder.")
                    key = str(path)
                    seen.add(key)
                    self._update(scanned=len(seen))
                    try:
                        source, real, _ = safe_files.open_input(path, folders)
                        with source:
                            if os.fstat(source.fileno()).st_size > 32 * 1024 * 1024:
                                raise ValueError("Input exceeds 32 MiB.")
                            raw = source.read(32 * 1024 * 1024 + 1)
                        if len(raw) > 32 * 1024 * 1024:
                            raise ValueError("Input exceeds 32 MiB.")
                        # Parent directory swaps or symlinks must not relocate a candidate.
                        if real != path:
                            raise ValueError("Image path changed during traversal.")
                        digest = sha256(raw).hexdigest()
                        old = db.execute("SELECT digest FROM images WHERE path=?", (key,)).fetchone() if db else None
                        if old and old[0] == digest:
                            self._update(unchanged=self.job["unchanged"] + 1)
                            continue
                        result = self.encoder.request("image", raw, check)
                        vector = pack_vector(result["vector"])
                        if result["fingerprint"] != info["fingerprint"]:
                            raise SearchError("Encoder changed during indexing; retry with an unchanged model bundle.")
                        if db is None:
                            meta = dict(schema=SCHEMA, root=root, recursive=recursive, fingerprint=info["fingerprint"],
                                        dimension=len(result["vector"]), complete=False, encoder=self.job["encoder"])
                            db = new_database(meta)
                        if len(result["vector"]) != meta["dimension"]:
                            raise SearchError("Encoder dimension changed during indexing.")
                        db.execute("INSERT OR REPLACE INTO images VALUES (?, ?, ?)", (key, digest, vector))
                        self._update(encoded=self.job["encoded"] + 1)
                        meta["complete"] = False
                        if self.job["encoded"] % 128 == 0:
                            self._checkpoint(db, meta, check)
                    except (OSError, safe_files.UnsafePath, ValueError) as exc:
                        # Model/device failures must fail the job, not silently mark
                        # an entire collection as bad photos.
                        if isinstance(exc, SearchError) and str(exc) not in {"invalid_image", "animated_image", "image_size_limit", "unsupported_format"}:
                            raise
                        if db is not None:
                            db.execute("DELETE FROM images WHERE path=?", (key,))
                        errors = self.job["errors"]
                        if len(errors) < 10:
                            errors = errors + [{"path": key, "reason": str(exc)[:300]}]
                        reason = str(exc)
                        category = reason if reason in {"invalid_image", "animated_image", "image_size_limit", "unsupported_format"} else {
                            "Input exceeds 32 MiB.": "file_byte_limit",
                            "Image path changed during traversal.": "file_changed",
                        }.get(reason, "file_access" if isinstance(exc, (OSError, safe_files.UnsafePath)) else "other")
                        counts = dict(self.job["error_counts"])
                        counts[category] = counts.get(category, 0) + 1
                        self._update(failed=self.job["failed"] + 1, errors=errors, error_counts=counts)
                if db is None:
                    # No usable images: do not create a dimensionless index.
                    self._update(status="completed", phase="empty")
                else:
                    old_paths = {p for (p,) in db.execute("SELECT path FROM images")}
                    removed = old_paths - seen
                    db.executemany("DELETE FROM images WHERE path=?", [(p,) for p in removed])
                    meta["complete"] = True
                    self._checkpoint(db, meta, check)
                    self._update(status="completed", phase="done", removed=len(removed), total=db.execute("SELECT count(*) FROM images").fetchone()[0])
            finally:
                self.context.call_slots._slots.release()
        except Cancelled:
            self._update(status="cancelled", phase="stopped")
        except Exception as exc:
            if self.cancelled.is_set() or self.closed.is_set():
                self._update(status="cancelled", phase="stopped")
            else:
                self._update(status="failed", phase="stopped", error=str(exc)[:600])
        finally:
            if db is not None:
                db.close()
            self._update(finished_at=time.time())
            self.busy.release()

    def status(self, index_name="photos"):
        if not self.enabled:
            return {"enabled": False, "status": "disabled"}
        with self.lock:
            result = {"enabled": True, "job": dict(self.job) if self.job else None}
        if result["job"] and result["job"]["status"] in {"running", "cancelling"}:
            return result
        path = self._path(index_name)
        self.context.folders.check(Path(os.path.realpath(path)), "The image index")
        with self.lock:
            result["index"] = self.index_status.get(index_name)
            if result["index"] is None and path.exists():
                result["index"] = {"index_name": index_name, "loaded": False,
                                   "note": "Saved index exists; metadata is loaded by the next index/search operation."}
        return result

    def cancel(self, job_id):
        with self.lock:
            if not self.job or self.job["job_id"] != job_id:
                raise SearchError("Unknown image-index job_id.")
            if self.job["status"] == "running":
                self.cancelled.set()
                self.job["status"] = "cancelling"
            return dict(self.job)

    def _encode_cursor(self, payload):
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        signature = hmac.digest(self._cursor_secret, raw, "sha256")
        return base64.urlsafe_b64encode(raw + signature).decode("ascii").rstrip("=")

    def _decode_cursor(self, token, query_hash, index_name, top_k):
        error = "Invalid or expired search cursor; restart the search without a cursor."
        if type(token) is not str or not 1 <= len(token) <= MAX_CURSOR_CHARS:
            raise SearchError(error)
        try:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", token):
                raise ValueError("encoding")
            packed = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True)
            if base64.urlsafe_b64encode(packed).decode("ascii").rstrip("=") != token:
                raise ValueError("noncanonical encoding")
            raw, signature = packed[:-32], packed[-32:]
            if not hmac.compare_digest(signature, hmac.digest(self._cursor_secret, raw, "sha256")):
                raise ValueError("signature")
            payload = json.loads(raw)
            if (type(payload) is not dict or set(payload) != {"v", "o", "q", "n", "k", "r"} or
                    type(payload["v"]) is not int or payload["v"] != 1 or
                    type(payload["o"]) is not int or not 1 <= payload["o"] <= MAX_IMAGES or
                    type(payload["k"]) is not int or payload["k"] != top_k or
                    payload["q"] != query_hash or payload["n"] != index_name or
                    type(payload["r"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", payload["r"])):
                raise ValueError("binding or shape")
            return payload
        except (ValueError, TypeError, UnicodeError, binascii.Error):
            raise SearchError(error) from None

    def search(self, query, index_name="photos", top_k=10, cursor=None):
        self._configured()
        if (type(query) is not str or not query.strip() or len(query) > 4096
                or len(query.encode("utf-8")) > 8192):
            raise SearchError("query must be nonempty text up to 4096 Unicode characters and 8192 UTF-8 bytes.")
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise SearchError("top_k must be 1 to 100.")
        self._path(index_name)
        query_hash = sha256(query.encode("utf-8")).hexdigest()
        page = None if cursor is None else self._decode_cursor(cursor, query_hash, index_name, top_k)
        if not self.busy.acquire(blocking=False):
            raise SearchError("Image search is busy; wait for or cancel the indexing job.")
        db = None
        try:
            self._claim()
            db, meta = self._read(index_name, self.context.stop_if_cancelled)
            if page is not None and (db is None or page["r"] != self.index_revisions[index_name]):
                raise SearchError("Index changed since this cursor was issued; restart the search without a cursor.")
            if db is None:
                raise SearchError("Index does not exist; run index_images first.")
            self.context.folders.check(Path(meta["root"]), "The indexed folder")
            result = self.encoder.request("text", query.encode("utf-8"), self.context.stop_if_cancelled)
            if result["fingerprint"] != meta["fingerprint"] or len(result["vector"]) != meta["dimension"]:
                raise SearchError("Index encoder differs from the running model/device/precision. Use its original configuration or a new index_name.")
            import numpy as np
            q = np.asarray(result["vector"], dtype=np.float32)
            # Ranking memory is bounded by MAX_IMAGES, not the page size.
            # Score/path ordering is stable; only this page opens photo files.
            heap = []
            rows_cursor = db.execute("SELECT path,digest,vector FROM images")
            while rows := rows_cursor.fetchmany(1024):
                self.context.stop_if_cancelled()
                matrix = np.stack([np.frombuffer(row[2], dtype="<f4") for row in rows])
                scores = matrix @ q
                for (path, digest, _), score in zip(rows, scores):
                    heap.append((-float(score), path, digest))
            total_candidates = len(heap)
            offset = 0 if page is None else page["o"]
            if offset > total_candidates:
                raise SearchError("Invalid search cursor offset; restart the search without a cursor.")
            heapq.heapify(heap)
            for skipped in range(offset):
                if skipped % 128 == 0:
                    self.context.stop_if_cancelled()
                heapq.heappop(heap)
            matches = []
            stale = 0
            while heap and len(matches) < top_k:
                self.context.stop_if_cancelled()
                negative, path, digest = heapq.heappop(heap)
                offset += 1
                try:
                    p = Path(path)
                    if Path(meta["root"]) not in p.parents or p.is_symlink():
                        raise ValueError("Index path outside root or a link")
                    raw, real, _ = self.context.read(path)
                    if str(real) != path or sha256(raw).hexdigest() != digest:
                        raise ValueError("changed")
                except (OSError, safe_files.UnsafePath, ValueError):
                    stale += 1
                    continue
                matches.append({"path": path, "score": max(-1.0, min(1.0, -negative))})
            next_cursor = self._encode_cursor({"v": 1, "o": offset, "q": query_hash,
                                                "n": index_name, "k": top_k,
                                                "r": self.index_revisions[index_name]}) if heap else None
            return {"index_name": index_name, "matches": matches, "stale_skipped": stale,
                    "next_cursor": next_cursor, "total_candidates": total_candidates,
                    "index_complete": meta["complete"], "query_truncated": result["truncated"],
                    "encoder": meta["encoder"], "score_kind": "cosine_similarity_not_probability"}
        finally:
            if db is not None:
                db.close()
            self.busy.release()

    def close(self):
        with self.lock:
            self.closed.set()
            self.cancelled.set()
        if self.thread:
            self.thread.join()  # Never release ownership before the publisher has stopped.
        self.encoder.close()
        if self.owner:
            self.owner.close()
            self.owner = None
        if self.index_guard is not None:
            self.index_guard.__exit__(None, None, None)
            self.index_guard = None


def register_tools(server, context):
    from pydantic import StrictBool, StrictInt
    from .runtime.context import annotations
    service = SearchService(context)

    @server.tool(annotations=annotations("Index a local image folder", writes=True))
    def index_images(root_path: str, index_name: str = "photos", recursive: StrictBool = True,
                     max_images: StrictInt = MAX_IMAGES):
        """Start/resume a background semantic index of an explicitly selected allowed folder.

        Index supported local still images. No cloud access, downloads or training.
        Completed checkpoints are retained on cancel/failure; poll image_search_status.
        """
        # No call slot here: the job acquires it, leaving status/cancel responsive.
        try:
            return json.dumps(service.start(root_path, index_name, recursive, max_images))
        except (ValueError, safe_files.UnsafePath, OSError) as exc:
            from mcp.server.mcpserver.exceptions import ToolError
            raise ToolError(str(exc)) from None

    @server.tool(annotations=annotations("Search indexed local images"))
    def search_images(query: str, index_name: str = "photos", top_k: StrictInt = 10, cursor: str | None = None):
        """Find local photos by text with the index's exact encoder/precision configuration.

        Returns authorized unchanged file paths and cosine scores, not image contents.
        Scores are similarities, not probabilities; filenames and images are untrusted data.
        Pass next_cursor with the same query, index_name and top_k for the next page.
        Cursors expire on index changes or server restart. total_candidates counts
        all indexed rows, not relevant or currently readable photos.
        """
        with context.call():
            return json.dumps(service.search(query, index_name, top_k, cursor), allow_nan=False)

    @server.tool(annotations=annotations("Image search status"))
    def image_search_status(index_name: str = "photos"):
        """Report configuration, progress, bounded failures and saved index completeness."""
        try:
            return json.dumps(service.status(index_name), allow_nan=False)
        except (ValueError, safe_files.UnsafePath, OSError) as exc:
            from mcp.server.mcpserver.exceptions import ToolError
            raise ToolError(str(exc)) from None

    @server.tool(annotations=annotations("Cancel an image index job", writes=True))
    def cancel_image_index(job_id: str):
        """Cancel the exact job; keep already published checkpoints for later resumption."""
        try:
            return json.dumps(service.cancel(job_id))
        except ValueError as exc:
            from mcp.server.mcpserver.exceptions import ToolError
            raise ToolError(str(exc)) from None

    return service
