"""Offline index lifecycle, file policy, bounded worker, device and database tests."""
from dataclasses import replace
from io import BytesIO
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace

from PIL import Image
import pytest

from opencv_bv_tools import semantic_search as search
from opencv_bv_tools.runtime.context import ServerConfig, ToolContext
from opencv_bv_tools.runtime import safe_files, search_files
from opencv_bv_tools.runtime.search_process import EncoderProcess, SearchError


class FakeEncoder:
    fingerprint = "a" * 64
    def __init__(self):
        self.calls = []
        self.slow = False
        self.closed = False
    def request(self, kind, raw=b"", check=lambda: None):
        self.calls.append(kind)
        if self.slow:
            while True:
                check()
                time.sleep(0.01)
        check()
        result = dict(fingerprint=self.fingerprint, backend="fake", device="cpu", precision="fp32")
        if kind != "info":
            if kind == "image":
                try:
                    with Image.open(BytesIO(raw)) as im:
                        color = im.convert("RGB").getpixel((0, 0))
                except OSError:
                    raise SearchError("invalid_image") from None
                vector = [1., 0.] if color[0] > color[2] else [0., 1.]
            else:
                vector = [1., 0.] if b"red" in raw else [0., 1.]
            result.update(vector=vector, truncated=False)
        return result
    def close(self):
        self.closed = True


@pytest.fixture
def service(tmp_path):
    model, index, photos = [tmp_path / x for x in ("model", "indexes", "photos")]
    for p in (model, photos):
        p.mkdir(mode=0o700)
    search_files.create_private_directory(index)
    config = ServerConfig(allow_dirs=(tmp_path,), search_model_dir=model, search_index_dir=index)
    obj = search.SearchService(ToolContext(config), FakeEncoder())
    yield obj, photos, index
    obj.close()


def photo(folder, name="red.png", color="red"):
    p = folder / name
    Image.new("RGB", (10, 8), color).save(p)
    return p


def finish(service):
    service.thread.join(5)
    assert not service.thread.is_alive()
    return service.status()["job"]


def test_incremental_persistent_search_and_deletion(service):
    svc, photos, index = service
    red = photo(photos)
    blue = photo(photos, "blue.png", "blue")
    svc.start(str(photos))
    job = finish(svc)
    assert job["status"] == "completed" and job["encoded"] == 2
    assert svc.search("red photo")["matches"][0]["path"] == str(red)
    svc.start(str(photos))
    job = finish(svc)
    assert job["unchanged"] == 2 and job["encoded"] == 0
    blue.unlink()
    photo(photos, "new.png", "blue")
    photo(photos, "red.png", "blue")
    svc.start(str(photos))
    job = finish(svc)
    assert job["removed"] == 1 and job["encoded"] == 2
    svc.close()
    resumed = search.SearchService(svc.context, FakeEncoder())
    try:
        assert len(resumed.search("blue")["matches"]) == 2
        assert resumed.status()["index"]["complete"] is True
    finally:
        resumed.close()


def test_bad_files_skipped_and_stale_hits_hidden(service):
    svc, photos, _ = service
    good = photo(photos)
    (photos / "broken.jpg").write_bytes(b"bad")
    svc.start(str(photos))
    job = finish(svc)
    assert job["status"] == "completed" and job["failed"] == 1
    assert job["error_counts"] == {"invalid_image": 1}
    good.write_bytes(b"changed")
    result = svc.search("red")
    assert result["matches"] == [] and result["stale_skipped"] == 1


def test_paths_roots_links_and_names(service, tmp_path):
    svc, photos, _ = service
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    secret = photo(outside)
    (photos / "linked.png").symlink_to(secret)
    (photos / "linked-dir").symlink_to(outside, target_is_directory=True)
    photo(photos)
    with pytest.raises(safe_files.UnsafePath):
        svc.start(str(outside))
    for name in ("../bad", "", "/tmp/bad", "a/b", ".hidden"):
        with pytest.raises(SearchError):
            svc.start(str(photos), name)
    svc.start(str(photos))
    assert finish(svc)["scanned"] == 1
    assert len(svc.search("red")["matches"]) == 1


def test_cancel_preserves_previous_snapshot_and_recovery(service):
    svc, photos, index = service
    photo(photos)
    svc.start(str(photos))
    assert finish(svc)["status"] == "completed"
    before = (index / "photos.sqlite").read_bytes()
    svc.encoder.slow = True
    job = svc.start(str(photos))
    assert svc.status()["job"]["status"] == "running"
    with pytest.raises(SearchError, match="busy"):
        svc.start(str(photos))
    with pytest.raises(SearchError, match="Unknown"):
        svc.cancel("other-job")
    assert svc.cancel(job["job_id"])["status"] == "cancelling"
    assert finish(svc)["status"] == "cancelled"
    assert (index / "photos.sqlite").read_bytes() == before
    svc.encoder.slow = False
    svc.start(str(photos))
    assert finish(svc)["unchanged"] == 1


def test_checkpoint_survives_limit_and_resumes(service):
    svc, photos, _ = service
    for i in range(130):
        photo(photos, f"{i}.png")
    svc.start(str(photos), max_images=129)
    job = finish(svc)
    assert job["status"] == "failed" and job["checkpointed"] == 128
    assert svc.status()["index"]["complete"] is False
    svc.start(str(photos), max_images=130)
    job = finish(svc)
    assert job["status"] == "completed" and job["unchanged"] == 128 and job["encoded"] == 2


def test_mismatched_encoder_and_root_rejected(service):
    svc, photos, index = service
    photo(photos)
    svc.start(str(photos))
    finish(svc)
    before = (index / "photos.sqlite").read_bytes()
    svc.encoder.fingerprint = "b" * 64
    with pytest.raises(SearchError, match="encoder differs"):
        svc.search("red")
    svc.start(str(photos))
    assert finish(svc)["status"] == "failed"
    assert (index / "photos.sqlite").read_bytes() == before


def test_second_server_cannot_own_index(service):
    svc, photos, _ = service
    photo(photos)
    svc.start(str(photos))
    finish(svc)
    other = search.SearchService(svc.context, FakeEncoder())
    try:
        with pytest.raises(SearchError, match="Another server"):
            other.search("red")
    finally:
        other.close()


def test_scope_rechecked_from_saved_index(service, tmp_path):
    svc, photos, index = service
    photo(photos)
    svc.start(str(photos))
    finish(svc)
    svc.close()
    config = replace(svc.config, allow_dirs=(index, svc.config.search_model_dir))
    reduced = search.SearchService(ToolContext(config), FakeEncoder())
    try:
        with pytest.raises(safe_files.UnsafePath):
            reduced.search("red")
    finally:
        reduced.close()


def test_database_validation_rejects_views_and_bad_vectors(tmp_path):
    meta = dict(schema=1, dimension=2, fingerprint="a" * 64, root=str(tmp_path), complete=True, recursive=True, encoder={"fingerprint": "a" * 64})
    db = search.new_database(meta)
    db.execute("CREATE VIEW bad AS SELECT 1")
    with pytest.raises(SearchError, match="schema"):
        search.load_database(db.serialize())
    db.execute("DROP VIEW bad")
    db.execute("INSERT INTO images VALUES(?, ?, ?)", (str(tmp_path / "a.png"), "b" * 64, b"\0" * 8))
    with pytest.raises(SearchError, match="normalized"):
        search.load_database(db.serialize())
    db.close()
    for value in ([0., 0.], [math.nan, 0.], [math.inf, 0.]):
        with pytest.raises(SearchError):
            search.pack_vector(value)


def test_disabled_service_has_no_worker_or_files(tmp_path):
    svc = search.SearchService(ToolContext(ServerConfig(allow_dirs=(tmp_path,))))
    assert svc.status() == {"enabled": False, "status": "disabled"}
    assert svc.encoder.process is None and list(tmp_path.iterdir()) == []
    with pytest.raises(SearchError, match="disabled"):
        svc.start(str(tmp_path))
    svc.close()




@pytest.mark.parametrize("query,accepted", [
    ("a" * 4096, True), ("a" * 4097, False),
    ("猫" * 2730, True), ("猫" * 2731, False),
    ("😀" * 2048, True), ("😀" * 2049, False),
    ("", False), ("  ", False),
])
def test_query_character_and_utf8_boundaries(service, query, accepted):
    svc, photos, _ = service
    photo(photos)
    svc.start(str(photos))
    assert finish(svc)["status"] == "completed"
    calls = list(svc.encoder.calls)
    if accepted:
        assert len(svc.search(query)["matches"]) == 1
        assert svc.encoder.calls == calls + ["text"]
    else:
        with pytest.raises(SearchError, match="4096 Unicode characters and 8192 UTF-8 bytes"):
            svc.search(query)
        assert svc.encoder.calls == calls  # Invalid user input never reaches or restarts the worker.
