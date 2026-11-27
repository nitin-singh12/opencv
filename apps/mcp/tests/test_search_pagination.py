"""Synthetic-only cursor paging, integrity, stale-file and resource-bound tests."""
import pytest

from opencv_bv_tools import semantic_search as search
from opencv_bv_tools.runtime.search_process import SearchError
from test_semantic_search import finish, photo, service


@pytest.fixture
def indexed(service):
    svc, photos, index = service
    paths = [photo(photos, f"{i:03}.png") for i in reversed(range(125))]
    svc.start(str(photos))
    assert finish(svc)["status"] == "completed"
    return svc, sorted(paths), index


def test_all_candidates_traversable_and_cursor_replay(indexed, monkeypatch):
    svc, paths, _ = indexed
    opened = []
    original_read = svc.context.read
    def track(path):
        opened.append(path)
        return original_read(path)
    monkeypatch.setattr(svc.context, "read", track)
    first = svc.search("red", top_k=20)
    token = first["next_cursor"]
    assert isinstance(token, str) and len(token) <= search.MAX_CURSOR_CHARS
    second = svc.search("red", top_k=20, cursor=token)
    assert len(opened) == 40  # No prior-page photos reopened.
    assert second == svc.search("red", top_k=20, cursor=token)
    matches = first["matches"] + second["matches"]
    page = second
    sizes = [20, 20]
    while page["next_cursor"] is not None:
        page = svc.search("red", top_k=20, cursor=page["next_cursor"])
        assert page["total_candidates"] == 125
        sizes.append(len(page["matches"]))
        matches.extend(page["matches"])
    assert sizes == [20, 20, 20, 20, 20, 20, 5]
    assert [m["path"] for m in matches] == [str(p) for p in paths]
    assert len({m["path"] for m in matches}) == 125


def test_exact_page_boundary_and_empty_index(service):
    svc, photos, _ = service
    for i in range(4):
        photo(photos, f"{i}.png")
    svc.start(str(photos))
    assert finish(svc)["status"] == "completed"
    first = svc.search("red", top_k=2)
    last = svc.search("red", top_k=2, cursor=first["next_cursor"])
    assert len(last["matches"]) == 2 and last["next_cursor"] is None
    for path in photos.iterdir():
        path.unlink()
    svc.start(str(photos))
    assert finish(svc)["status"] == "completed"
    empty = svc.search("red", top_k=2)
    assert empty["matches"] == [] and empty["next_cursor"] is None
    assert empty["total_candidates"] == 0


def test_stale_candidates_advance_raw_offset(indexed):
    svc, paths, _ = indexed
    first = svc.search("red", top_k=20)
    paths[20].unlink()
    paths[21].write_bytes(b"changed synthetic photo")
    second = svc.search("red", top_k=20, cursor=first["next_cursor"])
    assert second["stale_skipped"] == 2
    assert [m["path"] for m in second["matches"]] == [str(p) for p in paths[22:42]]
    third = svc.search("red", top_k=20, cursor=second["next_cursor"])
    assert third["matches"][0]["path"] == str(paths[42])
    for p in paths[62:]:
        p.unlink()
    last = svc.search("red", top_k=20, cursor=third["next_cursor"])
    assert last["matches"] == [] and last["next_cursor"] is None
    assert last["stale_skipped"] == 63


def test_snapshot_change_rejected_before_encoder(indexed):
    svc, paths, _ = indexed
    token = svc.search("red", top_k=20)["next_cursor"]
    photo(paths[0].parent, "new.png")
    svc.start(str(paths[0].parent))
    assert finish(svc)["status"] == "completed"
    calls = len(svc.encoder.calls)
    with pytest.raises(SearchError, match="Index changed.*restart"):
        svc.search("red", top_k=20, cursor=token)
    assert len(svc.encoder.calls) == calls


def test_binding_tampering_and_restart_rejected_early(indexed, monkeypatch):
    svc, _, _ = indexed
    token = svc.search("red", top_k=20)["next_cursor"]
    monkeypatch.setattr(svc, "_read", lambda *args: pytest.fail("invalid cursor read index"))
    for query, name, size, bad in [
        ("RED", "photos", 20, token), ("red", "other", 20, token),
        ("red", "photos", 21, token),
        ("red", "photos", 20, ("A" if token[0] != "A" else "B") + token[1:]),
    ]:
        with pytest.raises(SearchError, match="restart"):
            svc.search(query, name, size, bad)
    svc._cursor_secret = b"new server secret"
    with pytest.raises(SearchError, match="restart"):
        svc.search("red", top_k=20, cursor=token)


@pytest.mark.parametrize("token", ["", "!", "a", "a" * 1025, 1, True, {}, [], "é"])
def test_malformed_cursor_rejected_before_read(service, monkeypatch, token):
    svc, _, _ = service
    monkeypatch.setattr(svc, "_read", lambda *args: pytest.fail("invalid cursor read index"))
    with pytest.raises(SearchError, match="restart"):
        svc.search("red", cursor=token)


@pytest.mark.parametrize("field,value", [("o", True), ("o", -1), ("o", 0), ("o", 100001),
                                         ("o", 1.5), ("o", "20"), ("v", True), ("k", True),
                                         ("r", []), ("extra", 1)])
def test_signed_invalid_shape_rejected(service, monkeypatch, field, value):
    svc, _, _ = service
    payload = dict(v=1, o=20, q=search.sha256(b"red").hexdigest(), n="photos", k=20, r="a" * 64)
    payload[field] = value
    token = svc._encode_cursor(payload)
    monkeypatch.setattr(svc, "_read", lambda *args: pytest.fail("invalid cursor read index"))
    with pytest.raises(SearchError, match="restart"):
        svc.search("red", top_k=20, cursor=token)


def test_page_bounds_busy_and_cancellation(indexed, monkeypatch):
    svc, _, _ = indexed
    for bad in (True, 0, 101, 1.5):
        with pytest.raises(SearchError, match="top_k"):
            svc.search("red", top_k=bad)
    svc.busy.acquire()
    try:
        with pytest.raises(SearchError, match="busy"):
            svc.search("red")
    finally:
        svc.busy.release()
    def cancel():
        raise RuntimeError("test cancellation")
    monkeypatch.setattr(svc.context, "stop_if_cancelled", cancel)
    with pytest.raises(RuntimeError, match="test cancellation"):
        svc.search("red")
    assert svc.busy.acquire(blocking=False)
    svc.busy.release()

