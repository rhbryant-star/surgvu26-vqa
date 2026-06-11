# SurgVU26 VQA — M1 Baseline Container Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A complete, valid, T4-safe Grand Challenge algorithm container (base Qwen2.5-VL-7B-Instruct-AWQ + engineered prompt) that answers SurgVU Category-2 VQA questions, rehearsed end-to-end on CHTC under GC-equivalent conditions, and scored on the 11-clip public sample set — beating the 0.0485 dummy baseline.

**Architecture:** Inference logic lives in `surgvu_vqa/predict/` (frames → prompt → Qwen2.5-VL-AWQ → answer shaping), copied into a Docker image whose entrypoint is a thin `container/inference.py` adapted from the official template. Weights ship separately as a GC *model tarball* (extracted to `/opt/ml/model` at runtime). Because this box has **no Docker and no GPU**, images build in **GitHub Actions → GHCR**, convert to SIF via **apptainer** on the submit node, and all model execution happens in **CHTC GPU jobs** — the same artifact that will run on Grand Challenge.

**Tech Stack:** Python ≥3.9 locally (`from __future__ import annotations` everywhere); container = `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` + transformers + autoawq + opencv-headless; GitHub Actions + GHCR; HTCondor + apptainer on CHTC.

This implements the **M1 slice** of `docs/superpowers/specs/2026-06-09-surgvu26-vqa-design.md` (spec §4 online pipeline, §9 in-container inference, §10 packaging, §11 container testing, §12 milestone M1).

---

## Environment facts (verified 2026-06-10 — do not re-litigate)

- **No `docker`/`podman` on this box.** Only `apptainer`/`singularity`. → Images build in CI; `do_build.sh`/`do_test_run.sh` run in CI, not locally.
- **No GPU on this box** (`nvidia-smi` absent). → All model execution via CHTC jobs.
- **Home quota tight:** 37.6 GB used / 40 GB soft. → Weights, SIFs, tarballs live in `/staging/groups/bhaskar_opscribe/surgvu26/` (group staging, 1 TB). Never download weights into `$HOME`.
- **`gh` authenticated** as `rhbryant-star`. Repo has **no remote yet** — Task 7 creates a **public** GitHub repo (public ⇒ free Actions + GHCR; methodology rules require a public codebase at submission anyway; **challenge data never committed** — gitignore covers it).
- **CHTC rules (from project memory):** group staging is NOT osdf-compatible and needs `+WantStagingMount = true`; `container_image =` with raw `/staging/...` paths is rejected → use **vanilla universe + explicit `apptainer` invocation** in the job script. Execute dirs are noexec — never pip-install compiled packages at job runtime; everything compiled lives in the image.
- **Official template** (cloned at `~/opscribe-job/reference/surgvu2025-category2-submission`): input `/input/inputs.json` (socket list) + `/input/endoscopic-robotic-surgery-video.mp4` + `/input/visual-context-question.json` (bare JSON string); output `/output/visual-context-response.json` = bare JSON string; weights tarball → `/opt/ml/model/`; `/tmp` is a noop volume; runs `--network none`, non-root `user`, `ENTRYPOINT python inference.py`.
- **Model:** `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` (official, apache-2.0, 4-bit AWQ ≈7 GB). AWQ kernels need compute capability ≥7.5: T4 = 7.5 ✓, H200 = 9.0 ✓.
- **BLEU facts:** dummy all-"No" baseline = **0.0485** on the 11 real clips; full declarative sentences are the scoring answer form.

## Known risks & fallbacks (acknowledge, don't solve preemptively)

1. **autoawq/torch version coupling.** `autoawq-kernels` 0.0.9 wheels are built against torch 2.5.1 — hence the 2.5.1 base image; without the kernels, autoawq **silently** falls back to a several-fold-slower pure-torch path (Task 10 checks `import awq_ext` explicitly). Fallback if AWQ load fails outright: swap weights to base `Qwen/Qwen2.5-VL-7B-Instruct` fp16 + `bitsandbytes` `load_in_4bit=True` (config change in `model.py` + requirements; same interface).
2. **`apptainer --net --network none` may need privileges.** Fallback: rely on `TRANSFORMERS_OFFLINE=1`/`HF_HUB_OFFLINE=1` (baked into the image) for offline fidelity and note the difference in the rehearsal log.
3. **H200 ≠ T4.** Rehearsal can't emulate 16 GB VRAM; instead `inference.py` prints `torch.cuda.max_memory_allocated()` and Task 10 asserts **peak < 14 GB**.
4. **2026 template is registration-gated.** We build against the public 2025 interface (the 2026 submission page points to it). Re-verify I/O slugs after the user registers; the dispatch dict makes slug changes a one-line fix.
5. **Synthetic-mp4 test codec.** If `cv2.VideoWriter` with `mp4v` won't open on this box, fall back to `XVID`/`.avi` in the fixture helper only (sampling code is container-format-agnostic).

## File structure

| File | Responsibility |
|---|---|
| `surgvu_vqa/predict/__init__.py` | Subpackage marker |
| `surgvu_vqa/predict/frames.py` | Evenly-spaced frame sampling from a clip (cv2 → PIL) |
| `surgvu_vqa/predict/answer.py` | Prompt text constants + BLEU-aware answer shaping |
| `surgvu_vqa/predict/model.py` | Weight-path resolution + `QwenVqa` wrapper (lazy torch imports) |
| `container/inference.py` | GC entrypoint: socket dispatch → frames → model/fake → shaped answer |
| `container/Dockerfile`, `container/requirements.txt` | Image definition (weights NOT included) |
| `container/do_build.sh`, `do_test_run.sh` | Template scripts adapted (run in CI; template's `do_save.sh` dropped — CI saves inline) |
| `container/test/input/interf0/…` | Committed fixture: template JSONs + **synthetic** clip |
| `.dockerignore` | Keep data/, venv, git, docs out of build context |
| `.github/workflows/build-container.yml` | CI: build → push GHCR → optional save-artifact |
| `scripts/make_dummy_fixture.py` | Generate the committed synthetic fixture clip |
| `scripts/make_real_fixture.sh` | Stage real case122 into a **gitignored** `interf_real/` fixture |
| `scripts/predict_samples.py` | Batch driver: 11 clips → predictions.json (runs inside SIF) |
| `hpc/fetch_weights.sh` | Submit-node: HF download AWQ → staging + build model tarball |
| `hpc/build_sif.sh` | Submit-node: GHCR image → SIF in staging |
| `hpc/rehearse_container.sub/.sh` | CHTC job: GC-equivalent single-case run (timing, VRAM, output) |
| `hpc/eval_samples.sub/.sh` | CHTC job: predictions over the 11 sample clips |
| `docs/BASELINES.md` | Scoreboard: config → local BLEU |
| `tests/predict/…`, `tests/container/…` | CPU-only unit tests (no torch locally) |

Constant shared across tasks: staging root `STG=/staging/groups/bhaskar_opscribe/surgvu26`.

---

## Task 1: Dev deps + container scaffolding

**Files:**
- Modify: `pyproject.toml` (dev extras), `.gitignore`
- Create: `.dockerignore`, `surgvu_vqa/predict/__init__.py`, `tests/predict/__init__.py`, `tests/container/__init__.py`
- Create: `container/test/input/interf0/` fixtures (copied from the cloned template)

- [ ] **Step 1: Extend dev extras in `pyproject.toml`**

Replace the `[project.optional-dependencies]` section with:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "opencv-python-headless>=4.9",
    "numpy>=1.24",
    "pillow>=10.0",
]
```

- [ ] **Step 2: Append to `.gitignore`**

```gitignore
container/model/
container/test/output/
container/test/input/interf_real/
*.sif
*.tar
```

(`container/test/input/interf_real/` holds REAL challenge data staged locally by Task 5 — never committed. The committed `interf0/` fixture is fully synthetic.)

- [ ] **Step 3: Create `.dockerignore`**

```
.git
.venv
data
docs
reference
tests
container/test
container/model
*.zip
*.tar
*.sif
**/__pycache__
**/*.pyc
```

(`**/` prefixes matter: a bare `__pycache__` only matches at the context root, and `surgvu_vqa/__pycache__` holds stale cpython-39 `.pyc` files that must not be COPY'd into the py3.11 image.)

- [ ] **Step 4: Create empty markers** — `surgvu_vqa/predict/__init__.py`, `tests/predict/__init__.py`, `tests/container/__init__.py`

- [ ] **Step 5: Install dev deps and verify suite still green**

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/python -c "import cv2, numpy, PIL; print(cv2.__version__)"
```
Expected: `15 passed`; cv2 version prints.

- [ ] **Step 6: Build the committed fixture — JSONs from the template, video SYNTHETIC**

> ⚠️ **The template's fixture mp4 is NOT a video** — it is 30 bytes of ASCII ("This is some placeholder data"); cv2 cannot decode it (verified by the plan-review panel). We therefore copy only the two JSONs and generate a real *synthetic* clip, so every downstream fake-model smoke test exercises actual frame decoding. The synthetic clip is NOT challenge data — safe to commit.

```bash
cd /home/rhbryant/opscribe-job/surgvu26-vqa
mkdir -p container/test/input/interf0
cp /home/rhbryant/opscribe-job/reference/surgvu2025-category2-submission/test/input/interf0/inputs.json container/test/input/interf0/
cp /home/rhbryant/opscribe-job/reference/surgvu2025-category2-submission/test/input/interf0/visual-context-question.json container/test/input/interf0/
```

Create `scripts/make_dummy_fixture.py`:

```python
#!/usr/bin/env python3
"""Generate a tiny SYNTHETIC clip as the committed container test fixture.

The official template's fixture mp4 is a 30-byte ASCII placeholder, not a
video — cv2 cannot decode it. A real (but synthetic, non-challenge) clip lets
the fake-model smoke tests exercise actual frame decoding inside the image.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT = (
    Path(__file__).resolve().parents[1]
    / "container" / "test" / "input" / "interf0"
    / "endoscopic-robotic-surgery-video.mp4"
)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    assert writer.isOpened(), "no mp4v codec available"
    for i in range(30):
        writer.write(np.full((64, 64, 3), COLORS[i % len(COLORS)], dtype=np.uint8))
    writer.release()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run and verify it decodes:

```bash
.venv/bin/python scripts/make_dummy_fixture.py
.venv/bin/python -c "import cv2; cap = cv2.VideoCapture('container/test/input/interf0/endoscopic-robotic-surgery-video.mp4'); assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30; print('fixture decodes: 30 frames')"
```

(Raw cv2 check because `frames.py` doesn't exist until Task 2.)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore .dockerignore surgvu_vqa/predict tests/predict tests/container container/test scripts/make_dummy_fixture.py
git commit -m "chore: predict scaffolding, dev deps, synthetic container fixture"
```

---

## Task 2: Frame sampler (`frames.py`)

**Files:**
- Create: `surgvu_vqa/predict/frames.py`
- Test: `tests/predict/test_frames.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/predict/test_frames.py
from __future__ import annotations

import cv2
import numpy as np
import pytest

from surgvu_vqa.predict.frames import sample_frames

# Distinct, codec-survivable solid colors (BGR), widely separated values.
_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]


def _write_video(path, n_frames, size=(64, 64)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    if not writer.isOpened():  # codec fallback (see plan risk 5)
        path = path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), 10.0, size)
        assert writer.isOpened(), "no usable cv2 video codec on this machine"
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), _COLORS[i % len(_COLORS)], dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_returns_requested_count(tmp_path):
    video = _write_video(tmp_path / "clip.mp4", n_frames=30)
    frames = sample_frames(video, num_frames=8)
    assert len(frames) == 8
    assert frames[0].mode == "RGB"


def test_spans_whole_clip(tmp_path):
    video = _write_video(tmp_path / "clip.mp4", n_frames=30)
    frames = sample_frames(video, num_frames=8)
    # First sampled frame ≈ color 0 (BGR 255,0,0 → RGB 0,0,255-ish), last ≈ color of frame 29 (29%5=4 → BGR 0,255,255 → RGB ~255,255,0)
    first_px = frames[0].getpixel((32, 32))
    last_px = frames[-1].getpixel((32, 32))
    assert first_px[2] > 200 and first_px[0] < 60      # blue-dominant start
    assert last_px[0] > 200 and last_px[2] < 60        # red+green (yellow) end
    # All frames are not identical (sampling actually moved through the clip)
    assert len({f.tobytes() for f in frames}) > 1


def test_short_video_returns_all_frames(tmp_path):
    video = _write_video(tmp_path / "short.mp4", n_frames=3)
    frames = sample_frames(video, num_frames=8)
    assert len(frames) == 3


def test_unreadable_video_raises(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"this is not an mp4")
    with pytest.raises(ValueError):
        sample_frames(bogus, num_frames=8)
```

- [ ] **Step 2: Run to verify failure**

`.venv/bin/pytest tests/predict/test_frames.py -v` → FAIL with `ModuleNotFoundError: surgvu_vqa.predict.frames`.

- [ ] **Step 3: Implement**

```python
# surgvu_vqa/predict/frames.py
"""Evenly-spaced frame sampling from a video clip (cv2 → PIL RGB)."""
from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image

# 8 frames balances temporal coverage of a 30-s clip against T4 VRAM/time
# (each frame costs 256-512 visual tokens; see predict/model.py pixel budget).
DEFAULT_NUM_FRAMES = 8


def sample_frames(video_path: Path, num_frames: int = DEFAULT_NUM_FRAMES) -> list[Image.Image]:
    """Return up to `num_frames` RGB frames evenly spaced across the clip.

    Shorter clips return every frame. Raises ValueError when the file cannot
    be opened or no frame decodes.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"video reports no frames: {video_path}")
        count = min(num_frames, total)
        if count == 1:
            indices = [0]
        else:
            indices = [round(i * (total - 1) / (count - 1)) for i in range(count)]
        frames: list[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if not frames:
            raise ValueError(f"could not decode any frames: {video_path}")
        return frames
    finally:
        cap.release()
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/predict/test_frames.py -v` → 4 passed. Full suite → 19 passed.

- [ ] **Step 5: Commit**

```bash
git add surgvu_vqa/predict/frames.py tests/predict/test_frames.py
git commit -m "feat: evenly-spaced frame sampler for 30-s clips"
```

---

## Task 3: Prompt + answer shaping (`answer.py`)

**Files:**
- Create: `surgvu_vqa/predict/answer.py`
- Test: `tests/predict/test_answer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/predict/test_answer.py
from surgvu_vqa.predict.answer import (
    ANSWER_STYLE_INSTRUCTION,
    FALLBACK_ANSWER,
    SYSTEM_PROMPT,
    build_user_text,
    shape_answer,
)


def test_prompt_constants_are_nonempty():
    assert SYSTEM_PROMPT.strip()
    assert ANSWER_STYLE_INSTRUCTION.strip()
    assert FALLBACK_ANSWER.strip().endswith(".")


def test_build_user_text_contains_question_and_style():
    text = build_user_text("  Was a stapler used?  ")
    assert text.startswith("Was a stapler used?")
    assert ANSWER_STYLE_INSTRUCTION in text


def test_shape_strips_quotes_and_whitespace():
    assert shape_answer('  "A stapler was not used."  ') == "A stapler was not used."


def test_shape_collapses_newlines_keeps_first_sentence():
    raw = "A stapler was not used. The clip shows suturing.\nExtra commentary."
    assert shape_answer(raw) == "A stapler was not used."


def test_shape_appends_terminal_period():
    assert shape_answer("The forceps type is Cadiere Forceps") == "The forceps type is Cadiere Forceps."


def test_shape_empty_returns_fallback():
    assert shape_answer("   ") == FALLBACK_ANSWER
```

- [ ] **Step 2: Run to verify failure** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# surgvu_vqa/predict/answer.py
"""Prompt text + BLEU-aware answer shaping (v1 baseline).

The challenge scores answers with BLEU against 5 reference phrasings whose
canonical style is a short declarative sentence ("A stapler was not used.").
Full sentences dominate 1-token answers under BLEU-4 (a 1-token exact match
caps at ~0.18) — so both the prompt and the shaper push toward that form.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert surgical assistant analyzing a short clip from a "
    "robot-assisted surgery training session recorded on a da Vinci system. "
    "Answer questions about the clip accurately and concisely."
)

ANSWER_STYLE_INSTRUCTION = (
    "Answer with exactly one short declarative sentence that restates the "
    'answer, e.g. "A stapler was not used." or "The forceps type is Cadiere '
    'Forceps." Do not add explanations.'
)

FALLBACK_ANSWER = "The answer is not visible in the clip."

_SENTENCE_ENDS = (". ", "! ", "? ")


def build_user_text(question: str) -> str:
    """Text part of the user turn; frames are attached separately."""
    return f"{question.strip()}\n\n{ANSWER_STYLE_INSTRUCTION}"


def shape_answer(raw: str) -> str:
    """Normalize a model response into one clean declarative sentence."""
    text = raw.strip().strip('"').strip("'").strip()
    text = " ".join(text.split())
    if not text:
        return FALLBACK_ANSWER
    for sep in _SENTENCE_ENDS:
        cut = text.find(sep)
        if cut != -1:
            text = text[: cut + 1].rstrip()
            break
    if text[-1] not in ".!?":
        text += "."
    return text
```

- [ ] **Step 4: Run to verify pass** → 6 passed; full suite 25 passed.

- [ ] **Step 5: Commit**

```bash
git add surgvu_vqa/predict/answer.py tests/predict/test_answer.py
git commit -m "feat: VQA prompt constants + declarative answer shaping"
```

> **Scope note:** spec §9's "normalize tool names to the groundtruth vocabulary" is deliberately NOT in M1 — tool-vocabulary normalization lands with the full `answer_shaping` work in M3 (spec §12). M1 shaping is quotes/whitespace/first-sentence/period only.

---

## Task 4: Model wrapper (`model.py`)

**Files:**
- Create: `surgvu_vqa/predict/model.py`
- Test: `tests/predict/test_model.py` (path resolution only — generation is GPU-validated in Tasks 10–11)

- [ ] **Step 1: Write the failing test**

```python
# tests/predict/test_model.py
from surgvu_vqa.predict import model as model_mod
from surgvu_vqa.predict.model import HF_MODEL_ID, MODEL_DIR_ENV, resolve_model_path


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(MODEL_DIR_ENV, str(tmp_path))
    assert resolve_model_path() == str(tmp_path)


def test_tarball_dir_when_present(monkeypatch, tmp_path):
    monkeypatch.delenv(MODEL_DIR_ENV, raising=False)
    fake_tarball_dir = tmp_path / "qwen2.5-vl-7b-awq"
    fake_tarball_dir.mkdir()
    monkeypatch.setattr(model_mod, "TARBALL_MODEL_DIR", fake_tarball_dir)
    assert resolve_model_path() == str(fake_tarball_dir)


def test_falls_back_to_hub_id(monkeypatch, tmp_path):
    monkeypatch.delenv(MODEL_DIR_ENV, raising=False)
    monkeypatch.setattr(model_mod, "TARBALL_MODEL_DIR", tmp_path / "absent")
    assert resolve_model_path() == HF_MODEL_ID


def test_module_imports_without_torch():
    # Heavy deps must stay out of module import time: the dev box has no torch,
    # and importing surgvu_vqa.predict.model above must not have pulled it in.
    import sys
    assert "torch" not in sys.modules
```

- [ ] **Step 2: Run to verify failure** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# surgvu_vqa/predict/model.py
"""Qwen2.5-VL VQA wrapper — lazy heavy imports, offline-safe weight resolution.

Loading/prompting conventions follow OpScribe's QwenVLProvider
(opscribe_pipeline/providers/vlm/qwen_vl.py), reduced to what an offline
single-model container needs. sdpa attention (T4 has no flash-attention-2);
greedy decoding for deterministic, BLEU-stable answers.
"""
from __future__ import annotations

import os
from pathlib import Path

from surgvu_vqa.predict.answer import SYSTEM_PROMPT, build_user_text

HF_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
TARBALL_MODEL_DIR = Path("/opt/ml/model/qwen2.5-vl-7b-awq")
MODEL_DIR_ENV = "SURGVU_MODEL_DIR"

# Qwen2.5-VL pixel budget per frame (28x28 patches): 256-512 visual tokens.
# Sized for 8 frames on a 16 GB T4.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28
MAX_NEW_TOKENS = 48


def resolve_model_path() -> str:
    """Weight location: env override → GC model-tarball dir → HF hub id."""
    env = os.environ.get(MODEL_DIR_ENV, "")
    if env:
        return env
    if TARBALL_MODEL_DIR.is_dir():
        return str(TARBALL_MODEL_DIR)
    return HF_MODEL_ID


class QwenVqa:
    """Loads once in __init__; answer() is per-question inference."""

    def __init__(self, model_path: str | None = None):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        path = model_path or resolve_model_path()
        self._processor = AutoProcessor.from_pretrained(
            path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self._model.eval()

    def answer(self, frames, question: str) -> str:
        import torch

        content = [{"type": "image"} for _ in frames]
        content.append({"type": "text", "text": build_user_text(question)})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[prompt], images=list(frames), return_tensors="pt"
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
```

- [ ] **Step 4: Run to verify pass** → 4 passed; full suite 29 passed.

- [ ] **Step 5: Commit**

```bash
git add surgvu_vqa/predict/model.py tests/predict/test_model.py
git commit -m "feat: Qwen2.5-VL-AWQ wrapper with offline weight resolution"
```

---

## Task 5: Container entrypoint (`inference.py`)

**Files:**
- Create: `container/inference.py`
- Create: `scripts/make_real_fixture.sh`
- Test: `tests/container/test_inference_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/container/test_inference_io.py
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "container" / "test" / "input" / "interf0"


def _load_inference():
    spec = importlib.util.spec_from_file_location("inference", REPO / "container" / "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_input_dir(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("inputs.json", "visual-context-question.json", "endoscopic-robotic-surgery-video.mp4"):
        shutil.copy(FIXTURE / name, input_dir / name)
    return input_dir


def test_fake_model_end_to_end(tmp_path, monkeypatch):
    input_dir = _make_input_dir(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setenv("SURGVU_INPUT_PATH", str(input_dir))
    monkeypatch.setenv("SURGVU_OUTPUT_PATH", str(output_dir))
    monkeypatch.setenv("SURGVU_FAKE_MODEL", "1")

    inference = _load_inference()
    rc = inference.run()

    assert rc == 0
    out = json.loads((output_dir / "visual-context-response.json").read_text())
    assert isinstance(out, str)
    assert out == "A fake answer for container testing."


def test_unknown_interface_raises(tmp_path, monkeypatch):
    input_dir = _make_input_dir(tmp_path)
    bogus = [{"interface": {"slug": "some-unknown-socket"}}]
    (input_dir / "inputs.json").write_text(json.dumps(bogus))
    monkeypatch.setenv("SURGVU_INPUT_PATH", str(input_dir))
    monkeypatch.setenv("SURGVU_OUTPUT_PATH", str(tmp_path / "out2"))
    monkeypatch.setenv("SURGVU_FAKE_MODEL", "1")

    inference = _load_inference()
    try:
        inference.run()
        raised = False
    except KeyError:
        raised = True
    assert raised
```

Note: `inference.py` lives in `container/` (not the package) because the image copies it to `/opt/app/inference.py` — the test loads it by file path for that reason.

- [ ] **Step 2: Run to verify failure** → fails (file does not exist).

- [ ] **Step 3: Implement**

```python
# container/inference.py
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
```

- [ ] **Step 4: Run to verify pass** → 2 passed; full suite 31 passed.

- [ ] **Step 5: Create `scripts/make_real_fixture.sh`** — stages a complete REAL-input dir (canonical GC filenames + inputs.json) so apptainer/docker smoke runs can be pointed at genuine challenge video. The whole dir is **gitignored** (challenge data is never committed).

```bash
#!/usr/bin/env bash
# Stage a REAL-input fixture dir (case122) with canonical GC filenames for
# in-container smoke runs. Entirely gitignored: challenge data never committed.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/data/raw/cat2_samples/case122"
DEST="$REPO/container/test/input/interf_real"
mkdir -p "$DEST"
cp "$SRC/case122.mp4" "$DEST/endoscopic-robotic-surgery-video.mp4"
cp "$SRC/case122_question.json" "$DEST/visual-context-question.json"
cp "$REPO/container/test/input/interf0/inputs.json" "$DEST/inputs.json"
echo "Real fixture staged at $DEST (gitignored)."
```

`chmod +x scripts/make_real_fixture.sh` and run it; verify the three files exist and `git check-ignore container/test/input/interf_real/inputs.json` succeeds (exit 0 = ignored).

- [ ] **Step 6: Commit**

```bash
git add container/inference.py tests/container/test_inference_io.py scripts/make_real_fixture.sh
git commit -m "feat: GC container entrypoint with fake-model test path"
```

---

## Task 6: Dockerfile, requirements, build scripts

**Files:**
- Create: `container/Dockerfile`, `container/requirements.txt`
- Create: `container/do_build.sh`, `container/do_test_run.sh` (adapted from template; the template's `do_save.sh` is deliberately **not** carried over — CI's `save_tarball` step does the `docker save` inline, and Task 8 owns the model tarball)

- [ ] **Step 1: Create `container/requirements.txt`**

```
transformers==4.51.3
accelerate==1.6.0
autoawq==0.2.9
autoawq-kernels==0.0.9
opencv-python-headless==4.11.0.86
pillow>=10.0
numpy<2.3
```

(torch/cuda come from the base image — pinned to **2.5.1** below because `autoawq-kernels` 0.0.9 wheels are built against torch 2.5.1; without the kernels package, autoawq **silently** falls back to a pure-torch dequant path several-fold slower — fatal for the T4 time budget. If the CI build or CHTC rehearsal still hits an autoawq incompatibility, apply Risk-1 fallback: drop both autoawq lines, add `bitsandbytes==0.45.*`, switch `model.py` to base fp16 + `load_in_4bit`.)

- [ ] **Step 2: Create `container/Dockerfile`**

```dockerfile
FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Unbuffered logs; fully offline at runtime; HF cache on the writable noop volume.
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_HOME=/tmp/hf

# Deps go into SYSTEM site-packages as root: a `pip install --user` under
# /home/user breaks under apptainer, where the process runs as the invoking
# user with a different HOME and user-site is never on sys.path.
COPY container/requirements.txt /opt/app/requirements.txt
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

COPY --chown=user:user surgvu_vqa /opt/app/surgvu_vqa
COPY --chown=user:user container/inference.py /opt/app/

# Absolute path: apptainer does not honor docker WORKDIR for relative entrypoints.
ENTRYPOINT ["python", "/opt/app/inference.py"]
```

- [ ] **Step 3: Create `container/do_build.sh`**

```bash
#!/usr/bin/env bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-surgvu26-vqa-cat2}"

docker build \
  --platform=linux/amd64 \
  --file "$SCRIPT_DIR/Dockerfile" \
  --tag "$DOCKER_IMAGE_TAG" \
  "$REPO_ROOT" 2>&1
```

(Build context is the **repo root** so `surgvu_vqa/` can be copied; `.dockerignore` keeps it small.)

- [ ] **Step 4: Create `container/do_test_run.sh`** — copy the template's `do_test_run.sh` verbatim, then make exactly **four** named changes (the plan-review panel verified the first two template behaviors break under CI):

1. `DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-surgvu26-vqa-cat2}"` (env-overridable tag — CI passes a full `ghcr.io/...:sha` ref).
2. `DOCKER_NOOP_VOLUME="surgvu26-vqa-noop-volume"` (fixed name — the template derives it from the tag, and docker volume names reject the `/` and `:` in a registry ref).
3. Insert `mkdir -p "${SCRIPT_DIR}/model"` immediately before the `chmod -R -f o+rX "$INPUT_DIR" "${SCRIPT_DIR}/model"` line (`container/model/` is gitignored and absent in CI checkouts; `chmod -f` on a missing path still exits 1 under `set -e`; an empty `/opt/ml/model` mount is correct for fake-model mode).
4. Add `--env SURGVU_FAKE_MODEL="${SURGVU_FAKE_MODEL:-0}"` to the `docker run` arguments in `run_docker_forward_pass` (CI smoke-tests the image without weights via fake mode; a real GC-style run uses default `0` with the model tarball mounted).

- [ ] **Step 5: `chmod +x container/do_*.sh`; verify suite still green (`.venv/bin/pytest -q` → 31 passed).** These scripts cannot run locally (no docker) — CI validates them in Task 7.

- [ ] **Step 6: Commit**

```bash
git add container/Dockerfile container/requirements.txt container/do_build.sh container/do_test_run.sh
git commit -m "feat: T4-safe container image definition + build scripts"
```

---

## Task 7: GitHub repo + CI build → GHCR

**Files:**
- Create: `.github/workflows/build-container.yml`

- [ ] **Step 1: Create the workflow**

```yaml
# .github/workflows/build-container.yml
name: build-container

on:
  workflow_dispatch:
    inputs:
      save_tarball:
        description: "Also produce the GC upload tarball (docker save) as an artifact"
        type: boolean
        default: false
  push:
    tags: ["v*"]

env:
  IMAGE: ghcr.io/${{ github.repository_owner }}/surgvu26-vqa-cat2

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - name: Free disk space (image + save need ~25 GB)
        run: |
          sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache/CodeQL
          sudo docker image prune --all --force
          df -h /

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build image
        run: DOCKER_IMAGE_TAG="$IMAGE:${GITHUB_SHA::12}" ./container/do_build.sh

      - name: Smoke test (fake model, no network)
        run: |
          DOCKER_IMAGE_TAG="$IMAGE:${GITHUB_SHA::12}" SURGVU_FAKE_MODEL=1 ./container/do_test_run.sh
          cat container/test/output/interf0/visual-context-response.json
          python -c "import json,sys; out=json.load(open('container/test/output/interf0/visual-context-response.json')); assert isinstance(out,str) and out, out; print('smoke OK:', out)"

      - name: Push image
        run: |
          docker tag "$IMAGE:${GITHUB_SHA::12}" "$IMAGE:latest"
          docker push "$IMAGE:${GITHUB_SHA::12}"
          docker push "$IMAGE:latest"

      - name: Save GC upload tarball
        if: ${{ inputs.save_tarball }}
        run: |
          docker save "$IMAGE:${GITHUB_SHA::12}" | gzip -c > surgvu26-vqa-cat2.tar.gz
          ls -lh surgvu26-vqa-cat2.tar.gz

      - name: Upload tarball artifact
        if: ${{ inputs.save_tarball }}
        uses: actions/upload-artifact@v4
        with:
          name: surgvu26-vqa-cat2-image
          path: surgvu26-vqa-cat2.tar.gz
          retention-days: 2
```

- [ ] **Step 2: Upgrade gh token scopes (one-time, verified missing)**

The stored gh OAuth token has scopes `gist, read:org, repo` only. Pushing a history that adds `.github/workflows/*` is **refused** without the `workflow` scope, and the package checks below need `read:packages`:

```bash
gh auth refresh -h github.com -s workflow -s read:packages
gh auth status   # confirm scopes now include workflow, read:packages
```

(Interactive device-code flow — if running unattended, ask the user to complete it: `! gh auth refresh -h github.com -s workflow -s read:packages`.)

- [ ] **Step 3: Create the public GitHub repo and push**

```bash
cd /home/rhbryant/opscribe-job/surgvu26-vqa
git add .github/workflows/build-container.yml
git commit -m "ci: container build + GHCR push + GC tarball artifact"
gh repo create surgvu26-vqa --public --source . --push
```

Expected: repo `rhbryant-star/surgvu26-vqa` created, `main` pushed. **Verify no data leaked:** `git ls-files | grep -E "data/raw|case1|\.mp4$"` must return ONLY `container/test/input/interf0/endoscopic-robotic-surgery-video.mp4` (our synthetic fixture from Task 1 — not challenge data).

- [ ] **Step 4: Trigger and watch CI**

```bash
gh workflow run build-container
sleep 10 && gh run list --workflow build-container --limit 1
gh run watch "$(gh run list --workflow build-container --limit 1 --json databaseId -q '.[0].databaseId')" --exit-status
```

Expected: all steps green, including the fake-model smoke test inside docker with `--network none`. If the build fails on dependency resolution, apply Risk-1 fallback and re-trigger.

- [ ] **Step 5: Make the GHCR package PUBLIC (one-time — packages default to private even in public repos)**

```bash
gh api --method PATCH /user/packages/container/surgvu26-vqa-cat2 -f visibility=public \
  || echo "API refused — flip manually: github.com → Packages → surgvu26-vqa-cat2 → Package settings → Change visibility → Public"
gh api /user/packages/container/surgvu26-vqa-cat2 -q .visibility   # expect: public
```

Without this, Task 9's anonymous `apptainer build docker://ghcr.io/...` gets a 401. (Authenticated fallback if visibility can't be flipped: `APPTAINER_DOCKER_USERNAME=rhbryant-star APPTAINER_DOCKER_PASSWORD=$(gh auth token) ./hpc/build_sif.sh latest`.)

- [ ] **Step 6: Verify image is pullable** — `gh api /user/packages/container/surgvu26-vqa-cat2/versions -q '.[0].metadata.container.tags'` lists `latest`.

- [ ] **Step 7: Commit any fixes made during CI debugging** (workflow edits etc.). Keep messages `ci: …`.

---

## Task 8: Weights + sample data → staging; model tarball

**Files:**
- Create: `hpc/fetch_weights.sh`

All artifacts go to `STG=/staging/groups/bhaskar_opscribe/surgvu26` — never `$HOME` (quota).

- [ ] **Step 1: Create `hpc/fetch_weights.sh`**

```bash
#!/usr/bin/env bash
# Download Qwen2.5-VL-7B-Instruct-AWQ to group staging and build the GC model
# tarball (extracts to /opt/ml/model/qwen2.5-vl-7b-awq at runtime).
# Run directly on the CHTC submit node. Also mirrors the public sample clips
# to staging for the eval job.
set -euo pipefail

STG=/staging/groups/bhaskar_opscribe/surgvu26
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$STG/models" "$STG/tarballs"

python3 -m pip install --user --quiet "huggingface_hub[cli]"
python3 -m huggingface_hub.commands.huggingface_cli download \
  Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
  --local-dir "$STG/models/qwen2.5-vl-7b-awq"

echo "Building model tarball (top-level dir = qwen2.5-vl-7b-awq)..."
tar -czf "$STG/tarballs/qwen25vl7b-awq-model.tar.gz" -C "$STG/models" qwen2.5-vl-7b-awq
ls -lh "$STG/tarballs/qwen25vl7b-awq-model.tar.gz"

echo "Mirroring public sample clips to staging for the eval job..."
rsync -a "$REPO/data/raw/cat2_samples/" "$STG/cat2_samples/" --exclude "__MACOSX"
echo "Done."
```

- [ ] **Step 2: Run it** — `chmod +x hpc/fetch_weights.sh && ./hpc/fetch_weights.sh` (expect ~7 GB download; several minutes). Verify: `du -sh $STG/models/qwen2.5-vl-7b-awq` ≈ 7 GB; tarball exists; `ls $STG/cat2_samples | head` shows `case122…`.

- [ ] **Step 3: Commit the script**

```bash
git add hpc/fetch_weights.sh
git commit -m "feat: staging fetch for AWQ weights, model tarball, sample mirror"
git push
```

---

## Task 9: SIF build from GHCR

**Files:**
- Create: `hpc/build_sif.sh`

- [ ] **Step 1: Create `hpc/build_sif.sh`**

```bash
#!/usr/bin/env bash
# Convert the CI-built GHCR image into a SIF in group staging (submit node).
# Scratch AND layer cache are forced to staging: /tmp has a 256 MB quota and
# $HOME has <3.5 GB headroom — the default cache (~/.apptainer*) would blow
# the home quota on a ~10 GB image pull.
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26
TAG="${1:-latest}"
mkdir -p "$STG/tmp" "$STG/apptainer_cache"
export APPTAINER_TMPDIR="$STG/tmp"
export APPTAINER_CACHEDIR="$STG/apptainer_cache"

apptainer build --force \
  "$STG/surgvu26-vqa-cat2-${TAG}.sif" \
  "docker://ghcr.io/rhbryant-star/surgvu26-vqa-cat2:${TAG}"
ln -sf "$STG/surgvu26-vqa-cat2-${TAG}.sif" "$STG/surgvu26-vqa-cat2-current.sif"
ls -lh "$STG"/surgvu26-vqa-cat2-*.sif
```

- [ ] **Step 2: Run it** — `chmod +x hpc/build_sif.sh && ./hpc/build_sif.sh latest`. Requires Task 7 Step 5 (package made public); authenticated fallback: `APPTAINER_DOCKER_USERNAME=rhbryant-star APPTAINER_DOCKER_PASSWORD=$(gh auth token) ./hpc/build_sif.sh latest`. Expect an 8–12 GB SIF in staging.

- [ ] **Step 3: CPU sanity check on the submit node (fake model — no GPU needed):**

```bash
STG=/staging/groups/bhaskar_opscribe/surgvu26
REPO=/home/rhbryant/opscribe-job/surgvu26-vqa
mkdir -p /tmp/surgvu_io/output
apptainer run --containall \
  --bind "$REPO/container/test/input/interf0:/input:ro,/tmp/surgvu_io/output:/output" \
  --env SURGVU_FAKE_MODEL=1 \
  "$STG/surgvu26-vqa-cat2-current.sif"
cat /tmp/surgvu_io/output/visual-context-response.json
```

Expected: prints the socket tuple, samples 8 frames from the dummy fixture, writes `"A fake answer for container testing."`. This validates the apptainer execution path end-to-end before burning a GPU slot.

- [ ] **Step 4: Commit**

```bash
git add hpc/build_sif.sh
git commit -m "feat: GHCR→SIF conversion for CHTC execution"
git push
```

---

## Task 10: CHTC rehearsal (real model, single case, GC-equivalent)

**Files:**
- Create: `hpc/rehearse_container.sub`, `hpc/rehearse_container.sh`

- [ ] **Step 1: Create `hpc/rehearse_container.sh`**

```bash
#!/usr/bin/env bash
# GC-equivalent rehearsal: real weights, one real case, offline, /input + /output
# + /opt/ml/model mounted exactly as Grand Challenge does.
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26
CASE="${1:-case122}"

mkdir -p input output model
echo "Extracting model tarball (as GC does at runtime)..."
tar -xzf "$STG/tarballs/qwen25vl7b-awq-model.tar.gz" -C model

cp "$STG/cat2_samples/$CASE/$CASE.mp4" input/endoscopic-robotic-surgery-video.mp4
cp "$STG/cat2_samples/$CASE/${CASE}_question.json" input/visual-context-question.json
cat > input/inputs.json <<'EOF'
[
  {"interface": {"slug": "endoscopic-robotic-surgery-video", "kind": "MP4 file",
   "super_kind": "File", "relative_path": "endoscopic-robotic-surgery-video.mp4"}},
  {"interface": {"slug": "visual-context-question", "kind": "String",
   "super_kind": "Value", "relative_path": "visual-context-question.json"}}
]
EOF

NETFLAGS=""
if apptainer exec --net --network none "$STG/surgvu26-vqa-cat2-current.sif" true 2>/dev/null; then
  NETFLAGS="--net --network none"
  echo "Network isolation: ENABLED (--network none)"
else
  echo "Network isolation: unavailable unprivileged; relying on TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE (baked into image)"
fi

# AWQ kernel check: without awq_ext, autoawq silently uses a pure-torch
# dequant path that is several-fold slower — tolerable on H200, fatal on T4.
if apptainer exec --nv "$STG/surgvu26-vqa-cat2-current.sif" python -c "import awq_ext" 2>/dev/null; then
  echo "awq kernels OK"
else
  echo "WARNING: awq_ext missing — pure-torch fallback (slow); investigate before T4 submission"
fi

# (/usr/bin/time is absent on the EL9 nodes; inference.py prints its own
# load/generate timings and CUDA peak, which is everything Step 3 consumes.)
apptainer run --nv --containall $NETFLAGS \
  --bind "$(pwd)/input:/input:ro,$(pwd)/output:/output,$(pwd)/model:/opt/ml/model:ro" \
  "$STG/surgvu26-vqa-cat2-current.sif" 2>&1 | tee rehearsal_log.txt

echo "===== RESULT ====="
cat output/visual-context-response.json
echo
python3 - <<'PY'
import json, re, sys
out = json.load(open("output/visual-context-response.json"))
assert isinstance(out, str) and out.strip(), f"bad output: {out!r}"
log = open("rehearsal_log.txt").read()
m = re.search(r"CUDA peak memory: ([0-9.]+) GiB", log)
peak = float(m.group(1)) if m else -1
print(f"answer={out!r}  cuda_peak={peak} GiB")
assert peak > 0, "no CUDA peak reported - model did not run on GPU"
assert peak < 14.0, f"T4 BUDGET EXCEEDED: peak {peak} GiB >= 14 GiB"
print("T4 VRAM budget check: OK")
PY
rm -rf model  # don't transfer 7GB back
```

- [ ] **Step 2: Create `hpc/rehearse_container.sub`**

```
universe = vanilla
executable = rehearse_container.sh
arguments = case122
log    = rehearse_$(Cluster).log
output = rehearse_$(Cluster).out
error  = rehearse_$(Cluster).err

request_cpus = 4
request_memory = 32GB
request_disk = 40GB
request_gpus = 1
requirements = (CUDAGlobalMemoryMb >= 15000)
+WantStagingMount = true

transfer_output_files = output, rehearsal_log.txt
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
queue
```

- [ ] **Step 3: Submit and verify**

```bash
cd hpc && condor_submit rehearse_container.sub && condor_watch_q
```

When done, inspect `rehearse_*.out`: expect `awq kernels OK` (if the WARNING appears instead, the pure-torch fallback is active — investigate the kernels wheel before any T4 submission), model load time, generate time, `CUDA peak memory`, a real declarative answer for case122 ("Are there forceps being used here?" → ground truth is "No…"-family), and `T4 VRAM budget check: OK`. **Record load+generate wall time** — extrapolate to the T4 (~3–5× slower than H200): if H200 total > ~60 s, flag time-limit risk and reduce frames/pixels.

- [ ] **Step 4: Commit**

```bash
git add hpc/rehearse_container.sub hpc/rehearse_container.sh
git commit -m "feat: GC-equivalent container rehearsal job (offline, VRAM-checked)"
git push
```

---

## Task 11: 11-clip prediction run on CHTC

**Files:**
- Create: `scripts/predict_samples.py`, `hpc/eval_samples.sub`, `hpc/eval_samples.sh`

- [ ] **Step 1: Create `scripts/predict_samples.py`**

```python
#!/usr/bin/env python3
"""Batch predictions over the public sample clips (runs INSIDE the SIF).

Writes predictions.json in the eval-harness format: {clip_id: answer}.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from surgvu_vqa.predict.answer import shape_answer
from surgvu_vqa.predict.frames import sample_frames
from surgvu_vqa.predict.model import QwenVqa

CASE_DIR_PATTERN = re.compile(r"^case\d+$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("predictions.json"))
    args = parser.parse_args()

    cases = sorted(d for d in args.samples_dir.iterdir() if d.is_dir() and CASE_DIR_PATTERN.match(d.name))
    print(f"Found {len(cases)} cases")
    t0 = time.time()
    model = QwenVqa()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    predictions = {}
    for case in cases:
        question = json.loads((case / f"{case.name}_question.json").read_text())
        frames = sample_frames(case / f"{case.name}.mp4")
        t1 = time.time()
        raw = model.answer(frames, question)
        answer = shape_answer(raw)
        predictions[case.name] = answer
        print(f"{case.name} ({time.time() - t1:.1f}s) Q: {question!r} -> A: {answer!r}")

    args.out.write_text(json.dumps(predictions, indent=2))
    print(f"Wrote {len(predictions)} predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Create `hpc/eval_samples.sh`**

```bash
#!/usr/bin/env bash
# Run scripts/predict_samples.py inside the SIF over the staged sample clips.
# Binds the repo's surgvu_vqa over the baked copy so prompt iterations don't
# need an image rebuild (transfer_input_files brings the current repo copy).
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26

mkdir -p model
tar -xzf "$STG/tarballs/qwen25vl7b-awq-model.tar.gz" -C model

apptainer exec --nv --containall \
  --bind "$STG/cat2_samples:/samples:ro,$(pwd)/model:/opt/ml/model:ro,$(pwd)/surgvu_vqa:/opt/app/surgvu_vqa:ro,$(pwd):/work" \
  --pwd /work \
  "$STG/surgvu26-vqa-cat2-current.sif" \
  python /work/predict_samples.py --samples-dir /samples --out /work/predictions.json

rm -rf model
echo "===== predictions.json ====="
cat predictions.json
```

- [ ] **Step 3: Create `hpc/eval_samples.sub`**

```
universe = vanilla
executable = eval_samples.sh
log    = eval_$(Cluster).log
output = eval_$(Cluster).out
error  = eval_$(Cluster).err

request_cpus = 4
request_memory = 32GB
request_disk = 40GB
request_gpus = 1
requirements = (CUDAGlobalMemoryMb >= 15000)
+WantStagingMount = true

transfer_input_files = ../scripts/predict_samples.py, ../surgvu_vqa
transfer_output_files = predictions.json
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
queue
```

- [ ] **Step 4: Submit, wait, retrieve**

```bash
cd hpc && condor_submit eval_samples.sub && condor_watch_q
```

Expected in `eval_*.out`: 11 per-case lines with questions + shaped declarative answers; `predictions.json` lands in `hpc/`.

- [ ] **Step 5: Commit**

```bash
git add scripts/predict_samples.py hpc/eval_samples.sub hpc/eval_samples.sh
git commit -m "feat: 11-clip CHTC prediction run for local BLEU scoring"
git push
```

---

## Task 12: Score, record baseline, document

**Files:**
- Create: `docs/BASELINES.md`
- Modify: `README.md`

- [ ] **Step 1: Score the predictions locally**

```bash
cd /home/rhbryant/opscribe-job/surgvu26-vqa
.venv/bin/python -m surgvu_vqa.eval.score --truth data/raw/truth.json --predictions hpc/predictions.json
```

(If `data/raw/truth.json` is missing — fresh checkout — rebuild: `./data/download_samples.sh` then `.venv/bin/python -c "import json; from pathlib import Path; from surgvu_vqa.data.public_samples import build_truth_json; Path('data/raw/truth.json').write_text(json.dumps(build_truth_json(Path('data/raw/cat2_samples')), indent=2))"`.)

**Acceptance: mean BLEU > 0.0485** (the dummy floor). Realistic expectation for prompted base 7B: 0.15–0.45.

- [ ] **Step 2: Create `docs/BASELINES.md`**

```markdown
# Local BLEU scoreboard (11 public clips, max-over-5-refs, mean)

| Date | Config | mean BLEU | Notes |
|---|---|---|---|
| 2026-06-10 | dummy all-"No" | 0.0485 | floor — any model must beat this |
| <date> | M1: base 7B-AWQ, 8 frames, prompt v1, greedy | <score> | <commit/SIF tag> |

Caveat: local metric is for RELATIVE ranking (official tokenizer/smoothing
unpublished — see surgvu_vqa/eval/bleu.py).
```

Fill in the real score + date + image tag.

- [ ] **Step 3: Add a Container section to `README.md`**

```markdown
## Container (Grand Challenge submission)
    # build: GitHub Actions → ghcr.io/rhbryant-star/surgvu26-vqa-cat2 (no local docker needed)
    gh workflow run build-container               # build + push image
    ./hpc/build_sif.sh latest                     # GHCR → SIF in staging (submit node)
    ./hpc/fetch_weights.sh                        # one-time: AWQ weights + model tarball + sample mirror
    cd hpc && condor_submit rehearse_container.sub  # GC-equivalent rehearsal (offline, VRAM check)
    cd hpc && condor_submit eval_samples.sub        # 11-clip predictions → score locally

GC upload artifacts: trigger `build-container` with `save_tarball=true` → image
tar.gz artifact; model tarball at $STG/tarballs/qwen25vl7b-awq-model.tar.gz.
```

- [ ] **Step 4: Full suite green, commit, push**

```bash
.venv/bin/pytest -q
git add docs/BASELINES.md README.md
git commit -m "docs: record M1 baseline BLEU + container workflow"
git push
```

- [ ] **Step 5: Done-check against M1 acceptance** — container builds in CI ✅, fake-mode smoke inside docker `--network none` ✅, SIF runs offline on CHTC with model tarball ✅, VRAM peak < 14 GiB ✅, 11-clip predictions scored ✅, beats 0.0485 ✅. Anything failing → fix before declaring M1.

**Deferred to submission time (registration-gated, user action):** create the GC Algorithm, upload the image tarball + model tarball, try-out on the GC platform, prelim submission when the phase opens Jul 20.

---

## Self-Review (updated after the 3-lens adversarial critique, 2026-06-10)

**1. Spec coverage (M1 slice):** §9 inference steps (read inputs → sample frames → generate → shape → JSON-string output) → Tasks 2–5, **except tool-vocabulary normalization, explicitly deferred to M3** (note in Task 3); §10 packaging (model tarball to `/opt/ml/model`, no FA2 on T4 — `sdpa` in `model.py`, offline env vars, 4-bit **via official AWQ checkpoint — spec §10 amended** to record the AWQ-primary/bnb-fallback inversion) → Tasks 4, 6, 8; §11 container testing (`do_test_run` with `--network none` in CI; offline rehearsal; latency/VRAM measured) → Tasks 7, 10; §12 M1 milestone (valid baseline path before Jul 20) → Tasks 1–12 complete the pre-registration portion; the GC upload itself is registration-gated and explicitly deferred. Local-harness integration → Tasks 11–12.

**2. Placeholder scan:** every code step has full content; the `<score>`/`<date>` slots in BASELINES.md are fill-at-execution values produced by Step 1 of that task. The template-copy step (Task 6 Step 4) names the exact source file and exactly four changes, each with its reason.

**3. Type consistency:** `sample_frames(Path, int) -> list[Image.Image]` consumed by `inference.py` and `predict_samples.py` ✓; `shape_answer(str) -> str` ✓; `QwenVqa.answer(frames, question) -> str` ✓; predictions.json `{clip_id: str}` matches the eval-harness schema ✓; `TARBALL_MODEL_DIR` = `/opt/ml/model/qwen2.5-vl-7b-awq` matches the Task 8 tarball top-level dir and the Task 10/11 mounts ✓; image name `surgvu26-vqa-cat2` consistent across do_build/do_test_run/CI/build_sif ✓; pytest running totals recomputed: 15 → 19 → 25 → 29 → 31 ✓.

**4. Environment reality (incorporates all verified critique findings):** no docker locally — docker commands run in CI only ✓; no GPU locally — model runs in CHTC jobs ✓; home+`/tmp` quotas — weights/SIF/tarballs/apptainer cache+tmp ALL forced to group staging (Task 9 script defaults, not fallbacks) ✓; gh token scopes upgraded before the workflow push (`workflow`, `read:packages` — Task 7 Step 2) ✓; GHCR package flipped public before the SIF pull (Task 7 Step 5) ✓; system-site pip install (not `--user`) so packages resolve under both docker and apptainer ✓; committed fixture mp4 is synthetic and actually decodes (template's was an ASCII placeholder) ✓; `/usr/bin/time` absent on EL9 — not used ✓; AWQ kernels explicitly checked at rehearsal (silent slow-path guard) ✓; group staging via `+WantStagingMount`, vanilla universe + explicit apptainer, nothing pip-installed at job runtime ✓; Python 3.9 — all new modules carry `from __future__ import annotations` ✓.
