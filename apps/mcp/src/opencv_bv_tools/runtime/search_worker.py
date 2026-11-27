# SPDX-License-Identifier: Apache-2.0
"""Private bounded JSON + byte protocol for lazy offline image/text encoding."""
import json
import os
import sys

ERRORS = {
    "fgclip2_integrity_error", "image_size_limit", "animated_image", "invalid_image", "invalid_embedding",
    "unsupported_search_backend", "unsupported_format",
}


def main():
    from .process_support import protect_private_worker
    protect_private_worker()
    # Parent owns and removes the private model/code directory after reaping.
    model = None
    while header := sys.stdin.buffer.readline(16385):
        if len(header) > 16384 or not header.endswith(b"\n"):
            break
        try:
            options = json.loads(header)
            size = options["input_bytes"]
            if type(size) is not int or not 0 <= size <= 32 * 1024 * 1024:
                break
            raw = sys.stdin.buffer.read(size)
            if len(raw) != size:
                break
            if model is None:
                from .search_encoder import Encoder
                model = Encoder(options)
            result = model.info if options["kind"] == "info" else model.encode(raw, options["kind"], options["max_pixels"])
            response = {"status": "ok", "request_id": options["request_id"], **result}
        except FileNotFoundError:
            response = {"status": "error", "request_id": options["request_id"], "reason": "missing_search_models"}
        except ImportError:
            response = {"status": "error", "request_id": options["request_id"], "reason": "missing_search_dependencies"}
        except Exception as exc:
            reason = str(exc)
            if "out of memory" in reason.lower():
                reason = "device_out_of_memory"
            response = {"status": "error", "request_id": options["request_id"],
                        "reason": reason if reason in ERRORS or reason == "device_out_of_memory" else "encoder_failed"}
        sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
