"""SurgVU 2026 Category-2 algorithm entrypoint (Grand Challenge container).

Adapted from the official template (isi-challenges/surgvu2025-category2-submission):
socket dispatch, JSON I/O helpers, and path conventions are kept verbatim so the
GC evaluator contract is preserved. SURGVU_INPUT_PATH/SURGVU_OUTPUT_PATH env
overrides exist ONLY for local tests; on Grand Challenge the defaults apply.
SURGVU_FAKE_MODEL=1 short-circuits model loading (CI / fixture smoke tests).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from surgvu_vqa.predict.answer import shape_answer
from surgvu_vqa.predict.frames import sample_frames

INPUT_PATH = Path(os.environ.get("SURGVU_INPUT_PATH", "/input"))
OUTPUT_PATH = Path(os.environ.get("SURGVU_OUTPUT_PATH", "/output"))


def run():
    interface_key = get_interface_key()
    print("Inputs:", interface_key)
    handler = {
        (
            "endoscopic-robotic-surgery-video",
            "visual-context-question",
        ): interf0_handler,
    }[interface_key]
    return handler()


def interf0_handler():
    started = time.time()
    question = load_json_file(INPUT_PATH / "visual-context-question.json")
    print("Question:", question)

    frames = sample_frames(INPUT_PATH / "endoscopic-robotic-surgery-video.mp4")
    print(f"Sampled {len(frames)} frames in {time.time() - started:.1f}s")

    if os.environ.get("SURGVU_FAKE_MODEL") == "1":
        raw = "A fake answer for container testing."
    else:
        from surgvu_vqa.predict.model import QwenVqa

        t0 = time.time()
        model = QwenVqa()
        print(f"Model loaded in {time.time() - t0:.1f}s")
        t1 = time.time()
        raw = model.answer(frames, question)
        print(f"Generated in {time.time() - t1:.1f}s")
        _print_cuda_peak()

    response = shape_answer(raw)
    print("Output:", response)
    write_json_file(OUTPUT_PATH / "visual-context-response.json", response)
    print(f"Total {time.time() - started:.1f}s; output saved to {OUTPUT_PATH}")
    return 0


def get_interface_key():
    inputs = load_json_file(INPUT_PATH / "inputs.json")
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(location):
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def _print_cuda_peak():
    try:
        import torch

        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"CUDA peak memory: {peak_gb:.2f} GiB")
    except ImportError:
        pass


if __name__ == "__main__":
    raise SystemExit(run())
