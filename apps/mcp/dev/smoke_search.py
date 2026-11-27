# SPDX-License-Identifier: Apache-2.0
"""Real, offline CPU search acceptance against an installed wheel.

Uses the unchanged production checkpoint interval (128 actual embeddings).
Synthetic color fixtures test inference and lifecycle, not retrieval quality.
Downloads must finish separately, before starting this script.
"""
import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time


async def mcp_smoke(work, photos, model_dir):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from opencv_bv_tools.runtime.search_files import create_private_directory
    indexes = create_private_directory(work / "mcp-indexes")
    source_env = {key: os.environ[key] for key in ("OPENCV_BV_SOURCE_BUILD", "OPENCV_BV_MODEL_DIR") if key in os.environ}
    params = StdioServerParameters(command=sys.executable, args=[
        "-I", "-m", "opencv_bv_tools.server", "--allow-dir", str(work),
        "--allow-dir", str(model_dir), "--search-model-dir", str(model_dir),
        "--search-index-dir", str(indexes), "--threads", "2"], env=source_env)
    observed = {}
    with (work / "mcp-stderr.log").open("w", encoding="utf-8") as stderr:
        async with stdio_client(params, errlog=stderr) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                async def call(name, **arguments):
                    response = await session.call_tool(name, arguments)
                    assert not response.is_error, (name, response.content)
                    result = json.loads(next(block.text for block in response.content if block.type == "text"))
                    observed[name] = result
                    return result
                job = await call("index_images", root_path=str(photos))
                deadline = time.monotonic() + 180
                while True:
                    status = await call("image_search_status")
                    if status["job"]["status"] not in {"running", "cancelling"}:
                        break
                    assert time.monotonic() < deadline, status
                    await asyncio.sleep(0.1)
                assert status["job"]["status"] == "completed" and status["job"]["encoded"] == 3, status
                result = await call("search_images", query="a red image")
                assert len(result["matches"]) == 3 and result["index_complete"], result
                # Terminal cancellation is idempotent; active cancellation is
                # checked separately against the real worker below.
                terminal = await call("cancel_image_index", job_id=job["job_id"])
                assert terminal["status"] == "completed", terminal
    return observed


def wait_for(predicate, seconds, description):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for {description}")


def exercise(work, model_dir, report):
    import cv2
    import numpy as np
    import opencv_bv_tools
    from opencv_bv_tools import semantic_search as search
    from opencv_bv_tools.runtime.context import ServerConfig, ToolContext
    from opencv_bv_tools.runtime.search_encoder import ASSETS
    from opencv_bv_tools.runtime.search_files import create_private_directory
    from opencv_bv_tools.runtime.search_process import SearchError

    package = Path(opencv_bv_tools.__file__).resolve()
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    report.update(installed_package=str(package), opencv=cv2.__version__, numpy=np.__version__,
                  assets={name: dict(bytes=size, sha256=digest) for name, (size, digest) in ASSETS.items()})
    # No remote model/provider call is used: all requests use local OpenCV DNN.
    def offline(*args, **kwargs):
        raise RuntimeError("Network is disabled during real search acceptance")
    socket.create_connection = offline
    original_connect = socket.socket.connect
    def offline_connect(self, address):
        # asyncio's Windows socketpair uses numeric loopback TCP.
        if self.family in (socket.AF_INET, socket.AF_INET6) and address[0] in ("127.0.0.1", "::1"):
            return original_connect(self, address)
        return offline()
    socket.socket.connect = offline_connect
    photos, indexes = work / "photos", work / "indexes"
    photos.mkdir(mode=0o700)
    create_private_directory(indexes)
    config = ServerConfig(allow_dirs=(work, model_dir), search_model_dir=model_dir,
                          search_index_dir=indexes, threads=2, search_idle_seconds=120,
                          search_timeout=180, search_job_timeout=1800)
    svc = search.SearchService(ToolContext(config))
    def record(name, value):
        report["checks"][name] = value
        (work / "search-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"check": name, "result": value}), flush=True)
    def photo(name, color):
        target = photos / name
        pixels = np.full((48, 64, 3), color, dtype=np.uint8)
        ok, raw = cv2.imencode(".png", pixels)
        assert ok
        target.write_bytes(raw.tobytes())
        return target
    def finish():
        svc.thread.join(1800)
        assert not svc.thread.is_alive(), "Index thread did not stop within 1800 seconds"
        return svc.status()["job"]
    def index(**kwargs):
        svc.start(str(photos), **kwargs)
        return finish()
    try:
        red = photo("red.png", (0, 0, 255))
        blue = photo("blue.png", (255, 0, 0))
        photo("green.png", (0, 255, 0))
        record("mcp_search_tools", asyncio.run(mcp_smoke(work, photos, model_dir)))
        job = index()
        record("representative_index", job)
        assert job["status"] == "completed" and job["encoded"] == 3, job
        result = svc.search("a red image", top_k=2)
        record("real_text_search", result)
        assert len(result["matches"]) == 2 and result["next_cursor"], result
        second = svc.search("a red image", top_k=2, cursor=result["next_cursor"])
        assert len(second["matches"]) == 1 and not second["next_cursor"]
        assert len({hit["path"] for page in (result, second) for hit in page["matches"]}) == 3
        # Image encoding is exercised separately from text-to-index retrieval.
        embedding = svc.encoder.request("image", red.read_bytes())["vector"]
        db, _ = svc._read("photos")
        try:
            saved = search.unpack_vector(db.execute("SELECT vector FROM images WHERE path=?", (str(red),)).fetchone()[0])
        finally:
            db.close()
        similarity = sum(a*b for a, b in zip(embedding, saved))
        assert similarity > 0.9999, similarity
        record("image_embedding_self_similarity", similarity)
        job = index()
        assert job["status"] == "completed" and job["unchanged"] == 3 and job["encoded"] == 0, job
        record("incremental_unchanged", job)
        blue.unlink()
        photo("red.png", (40, 70, 110))
        job = index()
        assert job["status"] == "completed" and job["encoded"] == 1 and job["removed"] == 1, job
        record("incremental_changed_deleted", job)

        # Validate a representative sample first; only then process the batch.
        for number in range(128):
            photo(f"checkpoint-{number:03}.png", (number, 90, 255-number))
        job = index(index_name="checkpoint", max_images=129)
        record("checkpoint_limit", job)
        assert job["status"] == "failed" and job["checkpointed"] == 128, job
        assert svc.status("checkpoint")["index"]["complete"] is False
        svc.close()
        svc = search.SearchService(ToolContext(config))
        job = index(index_name="checkpoint", max_images=130)
        record("checkpoint_resume_after_restart", job)
        assert job["status"] == "completed" and job["unchanged"] == 128 and job["encoded"] == 2, job
        assert svc.search("a colorful image", index_name="checkpoint")["index_complete"] is True

        # Idle release must reap the native child and clean its private snapshot.
        svc.encoder.config = replace(config, search_idle_seconds=2)
        svc.encoder.request("info")
        child, scratch = svc.encoder.process, Path(svc.encoder.scratch.name)
        wait_for(lambda: svc.encoder.process is None and not scratch.exists(), 15, "idle release")
        assert child.poll() is not None and not scratch.exists()
        record("idle_release", {"pid": child.pid, "exit_code": child.returncode, "snapshot_removed": True})

        before = hashlib.sha256((indexes / "photos.sqlite").read_bytes()).hexdigest()
        job = svc.start(str(photos))
        child = wait_for(lambda: svc.encoder.process, 10, "worker startup before cancellation")
        svc.cancel(job["job_id"])
        cancelled = finish()
        assert cancelled["status"] == "cancelled" and child.poll() is not None, cancelled
        assert hashlib.sha256((indexes / "photos.sqlite").read_bytes()).hexdigest() == before
        record("cancel_reaps_worker_preserves_index", cancelled)

        # Observe a supported one-second budget with the real model. On very
        # fast hardware success is valid; deterministic timeout coverage lives
        # in test_search_process.py and is reported separately by pytest.
        svc.encoder.config = replace(config, search_timeout=1)
        started = time.monotonic()
        try:
            svc.encoder.request("text", b"a red image")
        except SearchError as error:
            assert "search_timeout" in str(error), str(error)
            assert svc.encoder.process is None and svc.encoder.scratch is None
            record("request_timeout", {"outcome": "timed_out_and_reaped", "error": str(error),
                                       "seconds": time.monotonic()-started})
        else:
            record("request_timeout", {"outcome": "completed_within_budget_timeout_not_exercised",
                                       "seconds": time.monotonic()-started})
            svc.close()
            svc = search.SearchService(ToolContext(config))
        svc.encoder.config = config
        result = svc.search("a colorful image", index_name="checkpoint")
        assert result["matches"] and result["index_complete"]
        record("recovery_after_timeout", {"matches": len(result["matches"]), "encoder": result["encoder"]})
    finally:
        svc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path, help="New output directory")
    parser.add_argument("--model-dir", type=Path, help="Verified cached model directory")
    args = parser.parse_args()
    from opencv_bv_tools.model_assets import search_cache_directory
    model_dir = (args.model_dir or search_cache_directory()).resolve(strict=True)
    work = args.work.resolve()
    work.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status="running", started=datetime.now(timezone.utc).isoformat(),
                  scope="real local CPU inference and lifecycle; synthetic fixtures; not retrieval quality",
                  model_dir=str(model_dir), python=sys.version, platform=platform.platform(),
                  architecture=platform.machine(), cpu_count=os.cpu_count(), threads=2,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  package=importlib.metadata.version("opencv-bv-tools"),
                  provider_capability="No provider requests: local OpenCV DNN has no Batch API",
                  provider_cost_usd=0, checks={})
    try:
        exercise(work, model_dir, report)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        (work / "search-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
