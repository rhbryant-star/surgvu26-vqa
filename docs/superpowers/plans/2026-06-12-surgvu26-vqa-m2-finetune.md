# SurgVU26 VQA — M2 Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synthesize a grounded VQA training set from the SurgVU weak labels (`tools.csv`/`tasks.csv`), LoRA-fine-tune Qwen2.5-VL-7B on it, re-quantize the merged model to AWQ, and beat the prompt-only baseline (**0.5141 mean BLEU**) on the 11 public clips.

**Architecture:** Per-case CHTC jobs stream ONLY the needed videos out of the 344 GB GCS zip via HTTP-Range (`remotezip`), cut 30-s clips into 8-frame sets, and pair them with template-synthesized Q&A whose answers use the **v3 BLEU-winning declarative style**. Frames ship as packed per-case tars (group staging is file-count-bound). Training reuses the proven LLaMA-Factory 0.9.3 phase-detector recipe (multi-image sharegpt format). Deployment: merge LoRA → **AutoAWQ self-quantization** (same artifact shape as today's checkpoint → container code unchanged), with adapter-on-AWQ as the bridge/fallback.

**Tech Stack:** Python ≥3.9 locally; `remotezip` (pure-Python, noexec-safe); LLaMA-Factory 0.9.3 (transformers==4.50.3, peft==0.15.2 — the pinned phase-detector stack); AutoAWQ 0.2.9 + transformers==4.51.3 for the quantize step; HTCondor + bhaskargpu4000 H200s.

Implements spec §7 (training-data synthesis) + §8 (model & fine-tuning) of `docs/superpowers/specs/2026-06-09-surgvu26-vqa-design.md`.

---

## Environment & research facts (verified 2026-06-12 — do not re-litigate)

- **Videos zip:** `https://storage.googleapis.com/isi-surgvu/surgvu24_videos_only.zip` = **344 GB — does NOT fit staging** (~314 GB free). `remotezip` verified against it: central directory reads fine; layout `surgvu24/case_NNN/case_NNN_video_part_PPP.mp4` (280 mp4s, deflate-compressed, ~0.8 GB/part; `__MACOSX/` junk present). Execute nodes have outbound HTTPS (proven by the GHCR pull in M1).
- **Labels:** zip at `$STG/labels/surgvu24_labels_updated_v2.zip` (and public URL `https://storage.googleapis.com/isi-surgvu/surgvu24_labels_updated_v2.zip`); 155 case dirs, each `tools.csv` (columns: `index,install_case_part,install_case_time,uninstall_case_part,uninstall_case_time,arm,commercial_toolname,groundtruth_toolname`; times `HH:MM:SS.ffffff`) + `tasks.csv` (task intervals with `start_part`/`stop_part`, `start_time`/`stop_time`, `task` column — verify exact header in Task 1 against real files).
- **🔴 LEAKAGE:** label cases `case_122`–`case_132` correspond to the public sample clips `case122`–`case132` (M1 rehearsal confirmed case_122's content matches the case122 question). **These 11 cases are EXCLUDED from train AND val.**
- **Group staging:** 686 GB/1 TB bytes, **9,266/10,000 files** → datasets ship as packed per-case tars; never loose frames. Scratch on execute nodes is quota-free.
- **Training recipe (proven by the phase detector — reuse, don't reinvent):** wrapper `/home/rhbryant/opscribe-job/finetune_phase_detector.sh` pattern: `pip install --target=./extra_packages --no-deps "llamafactory[torch,metrics]"` + pinned pure-py HF stack (`transformers==4.50.3 tokenizers==0.21.0 peft==0.15.2 trl==0.9.6 accelerate==1.7.0 datasets==3.5.0 …`), compiled wheels from `$GROUP/pypackages.tar.gz` → `/tmp/pypackages`, `PYTHONPATH` prepend, dataset paths rewritten at job start, base-model snapshot resolved from `$GROUP/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*`, launch `torchrun --nproc_per_node=$N -m llamafactory.launcher <yaml>`.
- **Multi-image format (LLaMA-Factory sharegpt):** per sample `{"messages":[{role,content}…], "images":[8 paths]}` with **8 `<image>` placeholders** in the user content; registered in a `dataset_info.json` with `"columns": {"messages":"messages","images":"images"}` + role tags.
- **Deployment research verdict:** (1°) merge LoRA → **AutoAWQ 0.2.9 self-quantize** — `Qwen2_5_VLAWQForCausalLM` is registered in v0.2.9, quantizes only the LM decoder (`modules_to_not_convert=["visual"]`), text-only calibration (official Qwen recipe: use your fine-tuning text), ~15 min on H200, output loads in the existing container UNCHANGED. (2°/bridge) PEFT adapter directly on the official AWQ base — supported by `peft 0.15.x–0.18` (`AwqLoraLinear`; **peft ≥0.19/main drops autoawq support**). **transformers ≥4.52 is FORBIDDEN everywhere** (VLM refactor renames decoder modules → breaks adapter keys and quant tooling). bnb-at-load and GPTQ rejected (startup time / version churn).
- **🔴 Our own lm_head patch applies to the self-quantized artifact too:** AutoAWQ writes `modules_to_not_convert=["visual"]` and an fp16 `lm_head` — the M1 crash recurs unless `"lm_head"` is appended to the produced `config.json` (same fix as `fetch_weights.sh`).
- **Train/infer consistency:** inference uses `MIN_PIXELS=256·28²`, `MAX_PIXELS=512·28²` per frame and 8 frames/clip — training must match (`image_max_pixels: 401408`) and `cutoff_len` must fit 8×≤512 visual tokens + text → **6144**.
- **BLEU style facts (v3, `docs/BASELINES.md`):** answers must be full declarative restatements ("No, a scalpel was not used." / "The marking pen color is blue."); single words score ~0; train targets use exactly this style.

## Shared constants (used by several tasks)

- `STG=/staging/groups/bhaskar_opscribe/surgvu26`, `GROUP=/staging/groups/bhaskar_opscribe`
- Dataset name/version: `surgvu_vqa_v1`; staging dir `$STG/datasets/surgvu_vqa_v1/`
- Adapter output: `$GROUP/adapters/surgvu_vqa_v1`
- Quantized model: `$STG/models/qwen25vl7b-surgvu-v1-awq` + tarball `$STG/tarballs/qwen25vl7b-surgvu-v1-awq-model.tar.gz` (inner top-level dir `qwen2.5-vl-7b-awq` — **same dir name as v0**, so `TARBALL_MODEL_DIR` and all job scripts work unchanged)
- **Case split (deterministic, hand-fixed here):** `EXCLUDED = case_122…case_132` (leakage). `VAL_CASES = ["case_000","case_020","case_040","case_060","case_080","case_100","case_140","case_150"]` (8 cases, spread). `TRAIN_CASES` = 40 cases: `["case_001","case_004","case_007","case_010","case_013","case_016","case_019","case_023","case_026","case_029","case_032","case_035","case_038","case_041","case_044","case_047","case_050","case_053","case_056","case_059","case_062","case_065","case_068","case_071","case_074","case_077","case_083","case_086","case_089","case_092","case_095","case_098","case_101","case_104","case_107","case_110","case_113","case_116","case_119","case_134"]` (every ~3rd case, skipping EXCLUDED and VAL; Task 1 verifies all exist in the labels and drops any missing with a logged warning).
- Per-clip frame count: 8; clip length 30 s; max clips/case 25; QA pairs/clip: ≤6.

## Known risks & fallbacks

1. **AWQ-quantized fine-tune quality** (AutoAWQ archived; VL support late). Gate: val + 11-clip BLEU vs the unquantized adapter (bridge path A measures the same adapter without quantization). If self-quant regresses badly → ship bridge: container `requirements.txt` += `peft==0.15.2`, `model.py` loads adapter from the model tarball onto the AWQ base.
2. **Clips spanning video parts:** a 30-s window crossing `part_NNN` boundaries is dropped at planning time (parts are separate mp4s; stitching is not worth it).
3. **Label noise** (installed ≠ visible): accepted — this is the challenge's intended weak supervision; mitigate with "in use during this clip" phrasing for positives that overlap the clip by ≥10 s.
4. **tasks.csv header drift:** Task 1 binds to the real header observed in the data and centralizes column names as constants (inspect-then-bind, like M0 Task 6).
5. **Style overfitting to templates:** template pools give 3–4 syntactic variants per question type, rotated deterministically; both `groundtruth_toolname` and `commercial_toolname` vocabularies are used in questions (refs in the public set use both, e.g. "large needle driver").

## File structure

| File | Responsibility |
|---|---|
| `data/download_labels.sh` | Fetch + extract label CSVs locally (dev/tests; gitignored data) |
| `surgvu_vqa/data/labels.py` | Parse one case's `tools.csv`/`tasks.csv` → typed intervals (seconds) |
| `surgvu_vqa/data/clip_plan.py` | Deterministic 30-s clip selection per case (task-stratified, part-safe) |
| `surgvu_vqa/data/qa_synthesis.py` | Clip + intervals → QA pairs in v3 declarative style |
| `scripts/build_case_dataset.py` | Per-case driver: fetch parts (remotezip) → frames → tar + qa.jsonl |
| `hpc/build_dataset.sub/.sh` | Condor fan-out: one job per case |
| `scripts/merge_qa_dataset.py` | Per-case jsonl → LLaMA-Factory train/val jsonl + dataset_info |
| `configs/finetune_surgvu_vqa.yaml` | LLaMA-Factory config (multi-image deltas applied) |
| `hpc/finetune_vqa.sub/.sh` | Training job (adapted from the phase-detector wrapper) |
| `scripts/merge_and_quantize.py` | Adapter merge → AutoAWQ quantize → lm_head patch |
| `hpc/quantize_model.sub/.sh` | GPU job running the above + tarball build |
| `tests/data/test_labels.py`, `test_clip_plan.py`, `test_qa_synthesis.py`, `tests/scripts/test_merge_qa_dataset.py` | CPU TDD |

---

## Task 1: Label parsing (`labels.py`)

**Files:** Create `surgvu_vqa/data/labels.py`, `data/download_labels.sh`; Test `tests/data/test_labels.py`

- [ ] **Step 1: Create and run `data/download_labels.sh`**

```bash
#!/usr/bin/env bash
# Fetch the public label CSVs for local dev/tests (data/raw is gitignored).
set -euo pipefail
DEST="$(cd "$(dirname "$0")" && pwd)/raw"
mkdir -p "$DEST"
curl -fsSL -o "$DEST/labels.zip" "https://storage.googleapis.com/isi-surgvu/surgvu24_labels_updated_v2.zip"
unzip -oq "$DEST/labels.zip" -d "$DEST/labels_root"
echo "cases: $(ls "$DEST/labels_root/labels" | wc -l)"
```

Run it (`chmod +x`, execute). Expected: `cases: 155`. Then **inspect the real headers** (`head -2 data/raw/labels_root/labels/case_122/tools.csv data/raw/labels_root/labels/case_122/tasks.csv`) and bind the column constants below to what is actually there (tools.csv header was verified 2026-06-10; tasks.csv assumed `start_part,start_time,stop_part,stop_time,task` — FIX the constants if reality differs, keep the interface).

- [ ] **Step 2: Write the failing test**

```python
# tests/data/test_labels.py
from __future__ import annotations

from pathlib import Path

from surgvu_vqa.data.labels import CaseLabels, parse_time_s

TOOLS_CSV = """index,install_case_part,install_case_time,uninstall_case_part,uninstall_case_time,arm,commercial_toolname,groundtruth_toolname
0,1.0,00:00:10.000000,1.0,00:01:40.000000,USM1,Large Needle Driver,needle driver
1,1.0,00:00:20.500000,2.0,00:00:30.000000,USM2,Cadiere Forceps,cadiere forceps
"""

TASKS_CSV = """index,task,start_part,start_time,stop_part,stop_time
0,Suturing,1.0,00:00:05.000000,1.0,00:02:00.000000
1,Range of Motion,2.0,00:00:00.000000,2.0,00:01:00.000000
"""


def _write_case(tmp_path: Path) -> Path:
    case = tmp_path / "case_001"
    case.mkdir()
    (case / "tools.csv").write_text(TOOLS_CSV)
    (case / "tasks.csv").write_text(TASKS_CSV)
    return case


def test_parse_time_s():
    assert parse_time_s("00:01:40.500000") == 100.5


def test_tools_parsed_per_part(tmp_path):
    labels = CaseLabels.load(_write_case(tmp_path))
    # Tool 0 lives entirely in part 1: 10s..100s
    t0 = labels.tool_intervals[0]
    assert (t0.part, t0.start_s, t0.end_s) == (1, 10.0, 100.0)
    assert t0.groundtruth == "needle driver"
    assert t0.commercial == "Cadiere Forceps" or t0.commercial == "Large Needle Driver"
    # Tool 1 spans parts (install part 1, uninstall part 2) and must be
    # represented without inventing cross-part timestamps:
    t1 = labels.tool_intervals[1]
    assert t1.part == 1 and t1.start_s == 20.5 and t1.end_s is None  # open until end of part


def test_tasks_parsed(tmp_path):
    labels = CaseLabels.load(_write_case(tmp_path))
    a = labels.task_intervals[0]
    assert (a.part, a.start_s, a.end_s, a.task) == (1, 5.0, 120.0, "Suturing")


def test_tools_in_window(tmp_path):
    labels = CaseLabels.load(_write_case(tmp_path))
    present = labels.tools_in_window(part=1, start_s=15.0, end_s=45.0, min_overlap_s=10.0)
    names = {t.groundtruth for t in present}
    assert "needle driver" in names          # overlaps 15..45 fully
    assert "cadiere forceps" in names        # 20.5..45 = 24.5s ≥ 10s
    none = labels.tools_in_window(part=1, start_s=101.0, end_s=131.0, min_overlap_s=10.0)
    assert {t.groundtruth for t in none} == {"cadiere forceps"}  # open-ended interval still active


def test_task_for_window(tmp_path):
    labels = CaseLabels.load(_write_case(tmp_path))
    assert labels.task_for_window(part=1, start_s=10.0, end_s=40.0) == "Suturing"
    assert labels.task_for_window(part=1, start_s=110.0, end_s=140.0) is None  # <50% covered
```

- [ ] **Step 3: Run to verify failure** → `ModuleNotFoundError: surgvu_vqa.data.labels`.

- [ ] **Step 4: Implement**

```python
# surgvu_vqa/data/labels.py
"""Parse one SurgVU case's tools.csv / tasks.csv into typed intervals.

Times are seconds WITHIN a video part (parts are separate mp4 files named
case_NNN_video_part_PPP.mp4). A tool installed in part P and uninstalled in a
later part is represented as an open-ended interval in part P (end_s=None):
we never invent cross-part timestamps. Column names are centralized below —
verified against the real 2024-v2 labels; adjust ONLY these if the header
drifts (Task 1 Step 1 inspection).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

TOOL_COLS = {
    "install_part": "install_case_part",
    "install_time": "install_case_time",
    "uninstall_part": "uninstall_case_part",
    "uninstall_time": "uninstall_case_time",
    "commercial": "commercial_toolname",
    "groundtruth": "groundtruth_toolname",
}
TASK_COLS = {
    "task": "task",
    "start_part": "start_part",
    "start_time": "start_time",
    "stop_part": "stop_part",
    "stop_time": "stop_time",
}


def parse_time_s(text: str) -> float:
    """'HH:MM:SS.ffffff' → seconds."""
    h, m, s = text.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


@dataclass(frozen=True)
class ToolInterval:
    part: int
    start_s: float
    end_s: float | None  # None = active until the end of this part
    groundtruth: str
    commercial: str


@dataclass(frozen=True)
class TaskInterval:
    part: int
    start_s: float
    end_s: float
    task: str


@dataclass
class CaseLabels:
    case_id: str
    tool_intervals: list[ToolInterval]
    task_intervals: list[TaskInterval]

    @classmethod
    def load(cls, case_dir: Path) -> "CaseLabels":
        case_dir = Path(case_dir)
        tools: list[ToolInterval] = []
        with open(case_dir / "tools.csv", newline="") as f:
            for row in csv.DictReader(f):
                ip = int(float(row[TOOL_COLS["install_part"]]))
                up = int(float(row[TOOL_COLS["uninstall_part"]]))
                start = parse_time_s(row[TOOL_COLS["install_time"]])
                end: float | None = parse_time_s(row[TOOL_COLS["uninstall_time"]])
                if up != ip:
                    end = None  # spans parts: open-ended within install part
                tools.append(ToolInterval(
                    part=ip, start_s=start, end_s=end,
                    groundtruth=row[TOOL_COLS["groundtruth"]].strip(),
                    commercial=row[TOOL_COLS["commercial"]].strip(),
                ))
        tasks: list[TaskInterval] = []
        with open(case_dir / "tasks.csv", newline="") as f:
            for row in csv.DictReader(f):
                sp = int(float(row[TASK_COLS["start_part"]]))
                ep = int(float(row[TASK_COLS["stop_part"]]))
                if sp != ep:
                    continue  # cross-part task segments are skipped (risk 2)
                tasks.append(TaskInterval(
                    part=sp,
                    start_s=parse_time_s(row[TASK_COLS["start_time"]]),
                    end_s=parse_time_s(row[TASK_COLS["stop_time"]]),
                    task=row[TASK_COLS["task"]].strip(),
                ))
        return cls(case_id=case_dir.name, tool_intervals=tools, task_intervals=tasks)

    def tools_in_window(self, part: int, start_s: float, end_s: float,
                        min_overlap_s: float = 10.0) -> list[ToolInterval]:
        """Tools whose installed interval overlaps [start_s, end_s) in `part`
        by at least min_overlap_s. Open-ended intervals extend to +inf."""
        out = []
        for t in self.tool_intervals:
            if t.part != part:
                continue
            t_end = t.end_s if t.end_s is not None else float("inf")
            overlap = min(t_end, end_s) - max(t.start_s, start_s)
            if overlap >= min_overlap_s:
                out.append(t)
        return out

    def task_for_window(self, part: int, start_s: float, end_s: float) -> str | None:
        """The task covering ≥50% of the window, else None."""
        need = (end_s - start_s) / 2.0
        for a in self.task_intervals:
            if a.part != part:
                continue
            overlap = min(a.end_s, end_s) - max(a.start_s, start_s)
            if overlap >= need:
                return a.task
        return None
```

- [ ] **Step 5: Run to verify pass** (5 tests), then **sanity-check against real labels**: `.venv/bin/python -c "from pathlib import Path; from surgvu_vqa.data.labels import CaseLabels; L = CaseLabels.load(Path('data/raw/labels_root/labels/case_122')); print(len(L.tool_intervals), len(L.task_intervals), L.task_intervals[0])"` — expect plausible nonzero counts; fix column constants if the real tasks.csv header differs (then re-run tests).

- [ ] **Step 6: Commit** — `git add surgvu_vqa/data/labels.py tests/data/test_labels.py data/download_labels.sh && git commit -m "feat: SurgVU label parsing (per-part tool/task intervals)"`

---

## Task 2: Clip planning (`clip_plan.py`)

**Files:** Create `surgvu_vqa/data/clip_plan.py`; Test `tests/data/test_clip_plan.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_clip_plan.py
from __future__ import annotations

from surgvu_vqa.data.clip_plan import plan_clips
from surgvu_vqa.data.labels import CaseLabels, TaskInterval, ToolInterval


def _labels(task_intervals):
    return CaseLabels(case_id="case_001", tool_intervals=[], task_intervals=task_intervals)


def test_clips_inside_task_segments_and_deterministic():
    labels = _labels([TaskInterval(part=1, start_s=0.0, end_s=300.0, task="Suturing")])
    a = plan_clips(labels, clip_len_s=30.0, max_clips=5, seed=7)
    b = plan_clips(labels, clip_len_s=30.0, max_clips=5, seed=7)
    assert a == b                                  # deterministic
    assert 1 <= len(a) <= 5
    for c in a:
        assert c.part == 1
        assert c.start_s >= 0.0 and c.start_s + 30.0 <= 300.0
        assert c.task == "Suturing"


def test_short_segments_skipped():
    labels = _labels([TaskInterval(part=1, start_s=0.0, end_s=20.0, task="Other")])
    assert plan_clips(labels, clip_len_s=30.0, max_clips=5, seed=7) == []


def test_clips_spread_across_tasks():
    labels = _labels([
        TaskInterval(part=1, start_s=0.0, end_s=120.0, task="Suturing"),
        TaskInterval(part=2, start_s=0.0, end_s=120.0, task="Range of Motion"),
    ])
    clips = plan_clips(labels, clip_len_s=30.0, max_clips=6, seed=1)
    tasks = {c.task for c in clips}
    assert tasks == {"Suturing", "Range of Motion"}     # stratified
    assert len([c for c in clips if c.part == 2]) >= 1
```

- [ ] **Step 2: Run → fails** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# surgvu_vqa/data/clip_plan.py
"""Deterministic, task-stratified 30-s clip selection for one case.

Clips never cross video-part boundaries (each part is its own mp4). Sampling
is seeded (case-stable) so re-running dataset builds is reproducible.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from surgvu_vqa.data.labels import CaseLabels


@dataclass(frozen=True)
class ClipSpec:
    part: int
    start_s: float
    end_s: float
    task: str


def plan_clips(labels: CaseLabels, clip_len_s: float = 30.0,
               max_clips: int = 25, seed: int = 0) -> list[ClipSpec]:
    """Round-robin over task segments, sampling non-overlapping windows."""
    rng = random.Random(seed)
    segments = [a for a in labels.task_intervals if a.end_s - a.start_s >= clip_len_s]
    if not segments:
        return []
    # Candidate starts per segment: aligned grid, shuffled per-segment.
    per_segment: list[list[ClipSpec]] = []
    for a in segments:
        starts = []
        s = a.start_s
        while s + clip_len_s <= a.end_s:
            starts.append(s)
            s += clip_len_s
        rng.shuffle(starts)
        per_segment.append([ClipSpec(a.part, st, st + clip_len_s, a.task) for st in starts])
    clips: list[ClipSpec] = []
    i = 0
    while len(clips) < max_clips and any(per_segment):
        bucket = per_segment[i % len(per_segment)]
        if bucket:
            clips.append(bucket.pop())
        if not any(per_segment):
            break
        i += 1
    return sorted(clips, key=lambda c: (c.part, c.start_s))
```

- [ ] **Step 4: Run to verify pass** (3 tests); full suite green.
- [ ] **Step 5: Commit** — `"feat: deterministic task-stratified clip planning"`

---

## Task 3: QA synthesis (`qa_synthesis.py`)

**Files:** Create `surgvu_vqa/data/qa_synthesis.py`; Test `tests/data/test_qa_synthesis.py`

The crux. Answers use the **v3 winning style** (full declarative restatement). Question templates rotate deterministically; tool names alternate between `groundtruth_toolname` and lowercased `commercial_toolname` (the public refs use both vocabularies).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_qa_synthesis.py
from __future__ import annotations

from surgvu_vqa.data.clip_plan import ClipSpec
from surgvu_vqa.data.labels import CaseLabels, TaskInterval, ToolInterval
from surgvu_vqa.data.qa_synthesis import ALL_GROUNDTRUTH_TOOLS, synthesize_qa


def _labels():
    return CaseLabels(
        case_id="case_001",
        tool_intervals=[
            ToolInterval(part=1, start_s=0.0, end_s=300.0, groundtruth="needle driver", commercial="Large Needle Driver"),
            ToolInterval(part=1, start_s=0.0, end_s=300.0, groundtruth="cadiere forceps", commercial="Cadiere Forceps"),
        ],
        task_intervals=[TaskInterval(part=1, start_s=0.0, end_s=300.0, task="Suturing")],
    )


CLIP = ClipSpec(part=1, start_s=30.0, end_s=60.0, task="Suturing")


def test_generates_positive_negative_step_and_which():
    qa = synthesize_qa(_labels(), CLIP, seed=3)
    kinds = {p.kind for p in qa}
    assert {"tool_yes", "tool_no", "task_what", "task_yesno"} <= kinds
    assert len(qa) <= 6


def test_positive_answer_is_declarative_yes():
    qa = [p for p in synthesize_qa(_labels(), CLIP, seed=3) if p.kind == "tool_yes"]
    assert qa, "expected a positive tool question"
    p = qa[0]
    assert p.answer.startswith("Yes, ")
    assert p.answer.endswith(".")
    assert len(p.answer.split()) >= 5            # full declarative, never terse
    # the tool named in the question is the one in the answer
    assert any(tok in p.answer.lower() for tok in p.question.lower().split() if len(tok) > 4)


def test_negative_tool_not_in_clip():
    qa = [p for p in synthesize_qa(_labels(), CLIP, seed=3) if p.kind == "tool_no"]
    assert qa
    present = {"needle driver", "cadiere forceps"}
    for p in qa:
        assert not any(t in p.question.lower() for t in present)
        assert p.answer.startswith("No, ") and p.answer.endswith(".")


def test_task_answers_in_v3_style():
    qa = {p.kind: p for p in synthesize_qa(_labels(), CLIP, seed=3)}
    assert "suturing" in qa["task_what"].answer.lower()
    assert qa["task_what"].answer.endswith(".")
    assert qa["task_yesno"].answer.startswith(("Yes, ", "No, "))


def test_deterministic():
    a = synthesize_qa(_labels(), CLIP, seed=11)
    b = synthesize_qa(_labels(), CLIP, seed=11)
    assert [(p.question, p.answer) for p in a] == [(p.question, p.answer) for p in b]


def test_negative_pool_excludes_present_tools():
    absent = [t for t in ALL_GROUNDTRUTH_TOOLS if t not in {"needle driver", "cadiere forceps"}]
    assert "stapler" in absent and "needle driver" not in absent
```

- [ ] **Step 2: Run → fails.**

- [ ] **Step 3: Implement**

```python
# surgvu_vqa/data/qa_synthesis.py
"""Synthesize grounded VQA pairs for one clip from weak labels.

Answers use the v3 BLEU-winning declarative style (docs/BASELINES.md):
Yes/No-led restatements for closed questions, subject-restating sentences for
identification. Question templates rotate deterministically per (clip, seed)
and alternate between the groundtruth and commercial tool vocabularies.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from surgvu_vqa.data.clip_plan import ClipSpec
from surgvu_vqa.data.labels import CaseLabels

# The 12 challenge tool classes (groundtruth_toolname vocabulary).
ALL_GROUNDTRUTH_TOOLS = [
    "needle driver", "cadiere forceps", "prograsp forceps",
    "monopolar curved scissors", "bipolar forceps", "stapler",
    "force bipolar", "vessel sealer", "permanent cautery hook/spatula",
    "clip applier", "tip-up fenestrated grasper", "grasping retractor",
]

_TOOL_Q = [
    "Was a {tool} used during this clip?",
    "Is a {tool} being used here?",
    "Did the surgeon use a {tool} in this clip?",
    "Was a {tool} used during the surgery?",
]
_TOOL_YES = [
    "Yes, a {tool} was used during this clip.",
    "Yes, a {tool} is being used.",
    "Yes, the surgeon used a {tool} in this clip.",
    "Yes, a {tool} was used.",
]
_TOOL_NO = [
    "No, a {tool} was not used during this clip.",
    "No, a {tool} is not being used.",
    "No, the surgeon did not use a {tool} in this clip.",
    "No, a {tool} was not used.",
]
_TASK_WHAT_Q = [
    "What surgical task is being performed in this clip?",
    "Which surgical task is shown here?",
    "What task is the surgeon performing?",
]
_TASK_WHAT_A = [
    "The surgical task being performed is {task}.",
    "The task shown here is {task}.",
    "The surgeon is performing {task}.",
]
_TASK_YESNO_Q = ["Is {task} being performed in this clip?"]
_TASK_YES = ["Yes, {task} is being performed in this clip."]
_TASK_NO = ["No, {task} is not being performed in this clip."]
_WHICH_Q = ["Which tools are being used in this clip?"]
_WHICH_A = ["The tools being used are {tools}."]


@dataclass(frozen=True)
class QAPair:
    kind: str       # tool_yes | tool_no | task_what | task_yesno | which_tools
    question: str
    answer: str


def _join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def synthesize_qa(labels: CaseLabels, clip: ClipSpec, seed: int = 0,
                  max_pairs: int = 6) -> list[QAPair]:
    rng = random.Random((seed, clip.part, int(clip.start_s)).__hash__())
    present = labels.tools_in_window(clip.part, clip.start_s, clip.end_s)
    present_names: list[str] = []
    pairs: list[QAPair] = []

    # Positives: up to 2 present tools, vocab alternating per pick.
    picks = rng.sample(present, min(2, len(present))) if present else []
    for i, t in enumerate(picks):
        name = t.groundtruth if i % 2 == 0 else t.commercial.lower()
        j = rng.randrange(len(_TOOL_Q))
        pairs.append(QAPair("tool_yes", _TOOL_Q[j].format(tool=name), _TOOL_YES[j].format(tool=name)))
    present_names = sorted({t.groundtruth for t in present})

    # Negatives: 2 tools NOT present (groundtruth vocabulary).
    absent = [t for t in ALL_GROUNDTRUTH_TOOLS if t not in set(present_names)]
    for name in rng.sample(absent, min(2, len(absent))):
        j = rng.randrange(len(_TOOL_Q))
        pairs.append(QAPair("tool_no", _TOOL_Q[j].format(tool=name), _TOOL_NO[j].format(tool=name)))

    # Task identification + task yes/no (balanced positive/negative).
    task = clip.task
    j = rng.randrange(len(_TASK_WHAT_Q))
    pairs.append(QAPair("task_what", _TASK_WHAT_Q[j], _TASK_WHAT_A[j].format(task=task)))
    if rng.random() < 0.5:
        pairs.append(QAPair("task_yesno", _TASK_YESNO_Q[0].format(task=task), _TASK_YES[0].format(task=task)))
    else:
        all_tasks = ["Suturing", "Uterine Horn", "Suspensory Ligaments", "Rectal Artery/Vein",
                     "Skills Application", "Range of Motion", "Retraction and Collision Avoidance"]
        wrong = rng.choice([t for t in all_tasks if t.lower() != task.lower()] or ["Other"])
        pairs.append(QAPair("task_yesno", _TASK_YESNO_Q[0].format(task=wrong), _TASK_NO[0].format(task=wrong)))

    # Which-tools (only when we actually know ≥1 present tool).
    if present_names and len(pairs) < max_pairs:
        pairs.append(QAPair("which_tools", _WHICH_Q[0], _WHICH_A[0].format(tools=_join(present_names))))

    return pairs[:max_pairs]
```

- [ ] **Step 4: Run to verify pass** (6 tests); full suite green.
- [ ] **Step 5: Commit** — `"feat: grounded QA synthesis in v3 declarative style"`

---

## Task 4: Per-case dataset builder (`build_case_dataset.py`)

**Files:** Create `scripts/build_case_dataset.py`; Test `tests/scripts/__init__.py` + `tests/scripts/test_build_case_dataset.py`

- [ ] **Step 1: Write the failing test** (uses a LOCAL zip — the source is abstracted so tests need no network)

```python
# tests/scripts/test_build_case_dataset.py
from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

from scripts.build_case_dataset import build_case

TOOLS_CSV = """index,install_case_part,install_case_time,uninstall_case_part,uninstall_case_time,arm,commercial_toolname,groundtruth_toolname
0,1.0,00:00:00.000000,1.0,00:02:00.000000,USM1,Large Needle Driver,needle driver
"""
TASKS_CSV = """index,task,start_part,start_time,stop_part,stop_time
0,Suturing,1.0,00:00:00.000000,1.0,00:02:00.000000
"""


def _make_video(path: Path, seconds: int = 120, fps: int = 10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    assert w.isOpened()
    for i in range(seconds * fps):
        w.write(np.full((64, 64, 3), (i % 200, 50, 100), dtype=np.uint8))
    w.release()


def test_build_case_end_to_end(tmp_path):
    # Fake the remote zip with a local one (same inner layout).
    vid = tmp_path / "v.mp4"
    _make_video(vid)
    zpath = tmp_path / "videos.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.write(vid, "surgvu24/case_001/case_001_video_part_001.mp4")
    case_labels = tmp_path / "labels" / "case_001"
    case_labels.mkdir(parents=True)
    (case_labels / "tools.csv").write_text(TOOLS_CSV)
    (case_labels / "tasks.csv").write_text(TASKS_CSV)
    out = tmp_path / "out"

    n_clips = build_case(
        case_id="case_001",
        labels_dir=case_labels,
        out_dir=out,
        zip_source=str(zpath),       # local path → zipfile; URL → remotezip
        max_clips=3,
        seed=5,
    )

    assert n_clips >= 1
    qa = [json.loads(l) for l in (out / "case_001_qa.jsonl").read_text().splitlines()]
    assert all(r["images"] and len(r["images"]) == 8 for r in qa)
    assert all(r["question"] and r["answer"].endswith(".") or r["answer"] for r in qa)
    with tarfile.open(out / "case_001_frames.tar") as t:
        names = t.getnames()
    # every referenced frame exists in the tar
    for r in qa:
        for img in r["images"]:
            assert img in names
```

- [ ] **Step 2: Run → fails.**

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python3
# scripts/build_case_dataset.py
"""Build one case's VQA training shard: frames tar + qa.jsonl.

Streams ONLY this case's needed video parts out of the 344 GB GCS zip via
HTTP Range (remotezip); never materializes the whole archive. Designed to run
inside a CHTC job (videos land on quota-free scratch and are deleted), but
works locally too. zip_source may be an http(s) URL (remotezip) or a local
path (zipfile) — tests use the local path.
"""
from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path

import cv2
from PIL import Image

from surgvu_vqa.data.clip_plan import plan_clips
from surgvu_vqa.data.labels import CaseLabels
from surgvu_vqa.data.qa_synthesis import synthesize_qa

JPEG_QUALITY = 90
FRAMES_PER_CLIP = 8


def _open_zip(zip_source: str):
    if zip_source.startswith(("http://", "https://")):
        from remotezip import RemoteZip
        return RemoteZip(zip_source)
    return zipfile.ZipFile(zip_source)


def _extract_frames(video_path: Path, start_s: float, end_s: float) -> list:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        first = min(int(start_s * fps), max(total - 1, 0))
        last = min(int(end_s * fps) - 1, total - 1)
        if last <= first:
            return []
        idxs = [round(first + i * (last - first) / (FRAMES_PER_CLIP - 1)) for i in range(FRAMES_PER_CLIP)]
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        return frames
    finally:
        cap.release()


def build_case(case_id: str, labels_dir: Path, out_dir: Path, zip_source: str,
               max_clips: int = 25, seed: int = 0) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)

    labels = CaseLabels.load(Path(labels_dir))
    clips = plan_clips(labels, max_clips=max_clips, seed=seed)
    needed_parts = sorted({c.part for c in clips})
    if not clips:
        print(f"{case_id}: no usable clips")
        return 0

    # Fetch only the needed parts.
    part_paths: dict[int, Path] = {}
    with _open_zip(zip_source) as z:
        for part in needed_parts:
            member = f"surgvu24/{case_id}/{case_id}_video_part_{part:03d}.mp4"
            print(f"{case_id}: fetching {member}")
            z.extract(member, path=work)
            part_paths[part] = work / member

    records = []
    tar_path = out_dir / f"{case_id}_frames.tar"
    with tarfile.open(tar_path, "w") as tar:
        for ci, clip in enumerate(clips):
            frames = _extract_frames(part_paths[clip.part], clip.start_s, clip.end_s)
            if len(frames) < FRAMES_PER_CLIP:
                continue
            img_names = []
            for fi, img in enumerate(frames):
                name = f"frames/{case_id}/clip{ci:04d}_f{fi}.jpg"
                tmp = work / "frame.jpg"
                img.save(tmp, "JPEG", quality=JPEG_QUALITY)
                tar.add(tmp, arcname=name)
                img_names.append(name)
            for pair in synthesize_qa(labels, clip, seed=seed + ci):
                records.append({
                    "case": case_id, "kind": pair.kind,
                    "question": pair.question, "answer": pair.answer,
                    "images": img_names,
                })

    with open(out_dir / f"{case_id}_qa.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    # Clean the (potentially GB-sized) videos.
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    print(f"{case_id}: {len(clips)} clips, {len(records)} QA pairs")
    return len(clips)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case-id", required=True)
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--zip-source", default="https://storage.googleapis.com/isi-surgvu/surgvu24_videos_only.zip")
    p.add_argument("--max-clips", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    build_case(a.case_id, a.labels_dir, a.out_dir, a.zip_source, a.max_clips, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add `remotezip` to dev extras** in `pyproject.toml` (`"remotezip>=0.12"`), `pip install -e ".[dev]"`, run tests → pass; full suite green.
- [ ] **Step 5: Commit** — `"feat: per-case dataset builder (range-streamed video, packed frames)"`

---

## Task 5: Dataset fan-out on CHTC

**Files:** Create `hpc/build_dataset.sh`, `hpc/build_dataset.sub`, `hpc/cases_train.txt`, `hpc/cases_val.txt`

- [ ] **Step 1: Write the case lists** — `hpc/cases_train.txt` = the 40 TRAIN_CASES (one per line); `hpc/cases_val.txt` = the 8 VAL_CASES. Generate from the constants in this plan; cross-check every named case exists in `data/raw/labels_root/labels/` and **assert none is in case_122..case_132**.

- [ ] **Step 2: Create `hpc/build_dataset.sh`**

```bash
#!/usr/bin/env bash
# One case's dataset shard, on an execute node (video → scratch, never staging).
set -euo pipefail
CASE="$1"
STG=/staging/groups/bhaskar_opscribe/surgvu26

mkdir -p pkgs
pip install --target=./pkgs --quiet remotezip opencv-python-headless pillow numpy
export PYTHONPATH="$(pwd)/pkgs:$(pwd):${PYTHONPATH:-}"

mkdir -p labels_root
tar -C labels_root -xzf "$STG/labels/surgvu24_labels_updated_v2.zip" 2>/dev/null || \
  (cp "$STG/labels/surgvu24_labels_updated_v2.zip" . && unzip -oq surgvu24_labels_updated_v2.zip -d labels_root)

python3 scripts/build_case_dataset.py \
  --case-id "$CASE" \
  --labels-dir "labels_root/labels/$CASE" \
  --out-dir out \
  --seed 42

mkdir -p "$STG/datasets/surgvu_vqa_v1"
cp out/${CASE}_frames.tar out/${CASE}_qa.jsonl "$STG/datasets/surgvu_vqa_v1/"
echo "done $CASE"
```

(Note: `opencv-python-headless` ships compiled `.so`s — pip --target into the job SCRATCH dir is fine because scratch is NOT noexec, unlike staging. The phase-detector wrapper proves runtime pip works on execute nodes.)

- [ ] **Step 3: Create `hpc/build_dataset.sub`**

```
universe = vanilla
executable = build_dataset.sh
arguments = $(case)
log    = build_ds_$(Cluster)_$(case).log
output = build_ds_$(Cluster)_$(case).out
error  = build_ds_$(Cluster)_$(case).err

request_cpus = 2
request_memory = 8GB
request_disk = 30GB
+WantStagingMount = true

transfer_input_files = ../scripts/build_case_dataset.py, ../surgvu_vqa
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

queue case from cases_all.txt
```

`cases_all.txt` = train + val lists concatenated (48 lines).

- [ ] **Step 4: Submit a SINGLE smoke case first** — `condor_submit build_dataset.sub -a "queue case from echo case_001 |" ` (or a one-line file). Verify `$STG/datasets/surgvu_vqa_v1/case_001_frames.tar` + `_qa.jsonl` arrive, jsonl records have 8 images each, spot-view one JPEG. Then submit the full 48.
- [ ] **Step 5: Verification sweep after the fan-out** — count tars/jsonls (expect ≈48 each, file-count adds ~96 to staging — fine), `wc -l` total QA pairs (expect ~4–7k), print kind distribution.
- [ ] **Step 6: Commit** — `"feat: dataset fan-out jobs (one per case, range-streamed)"`

---

## Task 6: Merge to LLaMA-Factory format (`merge_qa_dataset.py`)

**Files:** Create `scripts/merge_qa_dataset.py`; Test `tests/scripts/test_merge_qa_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_merge_qa_dataset.py
from __future__ import annotations

import json
from pathlib import Path

from scripts.merge_qa_dataset import merge


def _shard(d: Path, case: str, n: int):
    recs = [{"case": case, "kind": "tool_yes", "question": f"Q{i}?", "answer": f"Yes, answer {i} is here.",
             "images": [f"frames/{case}/clip{i:04d}_f{j}.jpg" for j in range(8)]} for i in range(n)]
    (d / f"{case}_qa.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_merge_emits_sharegpt_multiimage(tmp_path):
    _shard(tmp_path, "case_001", 3)
    _shard(tmp_path, "case_000", 2)
    out = tmp_path / "lf"
    n_train, n_val = merge(tmp_path, out, train_cases=["case_001"], val_cases=["case_000"])
    assert (n_train, n_val) == (3, 2)
    rec = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    assert rec["messages"][0]["role"] == "system"
    user = rec["messages"][1]
    assert user["role"] == "user" and user["content"].count("<image>") == 8
    assert rec["messages"][2]["role"] == "assistant"
    assert len(rec["images"]) == 8
    info = json.loads((out / "dataset_info.json").read_text())
    assert info["surgvu_vqa_train"]["columns"]["images"] == "images"
```

- [ ] **Step 2: Run → fails.**

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python3
# scripts/merge_qa_dataset.py
"""Merge per-case qa.jsonl shards into LLaMA-Factory sharegpt train/val files.

Output format matches the proven phase-detector recipe: messages with string
contents, 8 <image> placeholders in the user turn, and a parallel "images"
list of 8 RELATIVE frame paths (the training wrapper rewrites them to the
scratch dir after untarring the frame tars).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from surgvu_vqa.predict.answer import SYSTEM_PROMPT, build_user_text


def _to_sharegpt(rec: dict) -> dict:
    placeholders = "\n".join(["<image>"] * len(rec["images"]))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{placeholders}\n{build_user_text(rec['question'])}"},
            {"role": "assistant", "content": rec["answer"]},
        ],
        "images": rec["images"],
    }


def merge(shards_dir: Path, out_dir: Path, train_cases: list[str], val_cases: list[str]):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, cases in (("train", train_cases), ("val", val_cases)):
        n = 0
        with open(out_dir / f"{split}.jsonl", "w") as out:
            for case in cases:
                shard = Path(shards_dir) / f"{case}_qa.jsonl"
                if not shard.exists():
                    print(f"WARNING: missing shard {shard}")
                    continue
                for line in shard.read_text().splitlines():
                    out.write(json.dumps(_to_sharegpt(json.loads(line))) + "\n")
                    n += 1
        counts[split] = n
    info = {
        f"surgvu_vqa_{split}": {
            "file_name": f"{split}.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
        } for split in ("train", "val")
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"train={counts['train']} val={counts['val']}")
    return counts["train"], counts["val"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shards-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-cases", type=Path, required=True, help="file: one case id per line")
    p.add_argument("--val-cases", type=Path, required=True)
    a = p.parse_args()
    merge(a.shards_dir, a.out_dir,
          a.train_cases.read_text().split(), a.val_cases.read_text().split())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

NOTE: training the prompt WITH `build_user_text` (the v3 routed instruction) keeps train and inference prompts identical — the model learns the mapping it will actually be asked for.

- [ ] **Step 4: Run tests → pass; run for real** against `$STG/datasets/surgvu_vqa_v1/` (after Task 5 completes) with the case-list files; output to `$STG/datasets/surgvu_vqa_v1/lf/`. Report final train/val counts.
- [ ] **Step 5: Commit** — `"feat: merge shards into LLaMA-Factory sharegpt train/val"`

---

## Task 7: Training config + job

**Files:** Create `configs/finetune_surgvu_vqa.yaml`, `hpc/finetune_vqa.sh`, `hpc/finetune_vqa.sub`

- [ ] **Step 1: Create `configs/finetune_surgvu_vqa.yaml`** (deltas from the phase-detector yaml are marked)

```yaml
### model — path patched at runtime to the staged snapshot
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
image_max_pixels: 401408        # = inference MAX_PIXELS (512 visual tokens/frame)
image_min_pixels: 200704        # = inference MIN_PIXELS
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target: q_proj,k_proj,v_proj,o_proj

### dataset — registered by the wrapper's dataset_info.json
dataset: surgvu_vqa_train
eval_dataset: surgvu_vqa_val
template: qwen2_vl
cutoff_len: 6144                # 8 frames x <=512 visual tokens + text  (phase detector used 1024)
preprocessing_num_workers: 8

### output
output_dir: /staging/groups/bhaskar_opscribe/adapters/surgvu_vqa_v1
logging_steps: 10
save_steps: 200
save_total_limit: 2
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1   # 8-image samples (phase detector: 4 single-image)
gradient_accumulation_steps: 16  # eff. batch 32 on 2 GPUs
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
gradient_checkpointing: true

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 200
```

- [ ] **Step 2: Create `hpc/finetune_vqa.sh`** — copy `/home/rhbryant/opscribe-job/finetune_phase_detector.sh` and adapt (the proven skeleton stays: venv activate, llamafactory pip-into-extra_packages with the pinned stack, pypackages tarball → /tmp, PYTHONPATH). The changed sections:

```bash
# ── Data: untar all frame tars + place LF files ────────────────────────────
STG=/staging/groups/bhaskar_opscribe/surgvu26
DATA="$STG/datasets/surgvu_vqa_v1"
mkdir -p data
for t in "$DATA"/*_frames.tar; do tar -xf "$t"; done          # frames/<case>/...
cp "$DATA/lf/train.jsonl" "$DATA/lf/val.jsonl" data/

# Rewrite relative image paths to this scratch dir
python3 - <<'PY'
import json, os
root = os.getcwd()
for split in ("train", "val"):
    path = f"data/{split}.jsonl"
    out = []
    for line in open(path):
        r = json.loads(line)
        r["images"] = [os.path.join(root, p) for p in r["images"]]
        out.append(json.dumps(r))
    open(path, "w").write("\n".join(out) + "\n")
PY

# Register datasets for LLaMA-Factory
cat > data/dataset_info.json <<EOF
{
  "surgvu_vqa_train": {"file_name": "$(pwd)/data/train.jsonl", "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
             "assistant_tag": "assistant", "system_tag": "system"}},
  "surgvu_vqa_val": {"file_name": "$(pwd)/data/val.jsonl", "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
             "assistant_tag": "assistant", "system_tag": "system"}}
}
EOF

# ── Model snapshot + yaml patch (same pattern as phase detector) ──────────
HF_CACHE="/staging/groups/bhaskar_opscribe/hf_cache"
export HF_HOME="$HF_CACHE"
MODEL_SNAPSHOT=$(find "${HF_CACHE}/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)
sed -i "s|^model_name_or_path:.*|model_name_or_path: ${MODEL_SNAPSHOT}|" configs/finetune_surgvu_vqa.yaml
sed -i "s|^dataset_dir:.*||" configs/finetune_surgvu_vqa.yaml
echo "dataset_dir: $(pwd)/data" >> configs/finetune_surgvu_vqa.yaml

# ── Launch ─────────────────────────────────────────────────────────────────
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
mkdir -p /staging/groups/bhaskar_opscribe/adapters/surgvu_vqa_v1
torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29501 \
  -m llamafactory.launcher configs/finetune_surgvu_vqa.yaml
```

- [ ] **Step 3: Create `hpc/finetune_vqa.sub`**

```
universe = vanilla
executable = finetune_vqa.sh
log    = ft_$(Cluster).log
output = ft_$(Cluster).out
error  = ft_$(Cluster).err

request_cpus = 16
request_memory = 128GB
request_disk = 100GB
request_gpus = 2
requirements = (CUDAGlobalMemoryMb >= 70000)
rank = (Machine == "bhaskargpu4000.chtc.wisc.edu") * 1000
+WantStagingMount = true

transfer_input_files = ../configs/finetune_surgvu_vqa.yaml
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
queue
```

(Wrapper expects `configs/finetune_surgvu_vqa.yaml` relative — create `configs/` in scratch: add `mkdir -p configs && mv finetune_surgvu_vqa.yaml configs/` at the top of the wrapper.)

- [ ] **Step 4: Commit** — `"feat: LLaMA-Factory multi-image training config + job"`

---

## Task 8: Train + loss gate

- [ ] **Step 1: Submit** `condor_submit finetune_vqa.sub`; monitor `ft_*.out` (background `condor_wait`). Expected wall time: ~2–5 h for ~5k samples × 3 epochs on 2×H200.
- [ ] **Step 2: Gate** — training completes, `eval_loss` decreases and final `eval_loss < 1.0` (transcript→note adapters historically land ~0.5; QA-style targets should go lower). Adapter files in `$GROUP/adapters/surgvu_vqa_v1/`. If loss diverges/OOM: halve `cutoff_len` images budget (`image_max_pixels: 200704`) and resubmit (record in BASELINES notes).

---

## Task 9: Merge + AWQ self-quantization

**Files:** Create `scripts/merge_and_quantize.py`, `hpc/quantize_model.sub/.sh`

- [ ] **Step 1: Create `scripts/merge_and_quantize.py`**

```python
#!/usr/bin/env python3
"""Merge the LoRA adapter into bf16 Qwen2.5-VL-7B and AWQ-quantize the result.

MUST run with transformers==4.51.3 + autoawq==0.2.9 (the archived AutoAWQ's
last-tested combo; transformers >=4.52 renames VL decoder modules and breaks
everything). Calibration follows the official Qwen recipe: text from OUR
fine-tuning data, ChatML-formatted. The produced config.json gets the same
lm_head patch as the official checkpoint (see hpc/fetch_weights.sh).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_calibration(train_jsonl: Path, n: int = 256) -> list[str]:
    rng = random.Random(13)
    lines = train_jsonl.read_text().splitlines()
    rows = [json.loads(l) for l in rng.sample(lines, min(n, len(lines)))]
    texts = []
    for r in rows:
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        asst = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
        user_text = user.replace("<image>", "").strip()
        texts.append(
            f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{asst}<|im_end|>"
        )
    return texts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--merged-out", type=Path, required=True)
    p.add_argument("--quant-out", type=Path, required=True)
    a = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print("Loading base + adapter...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, a.adapter)
    model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(a.base)
    print("Saving merged model...")
    model.save_pretrained(a.merged_out, safe_serialization=True)
    processor.save_pretrained(a.merged_out)
    del model
    torch.cuda.empty_cache()

    print("Quantizing with AutoAWQ...")
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    awq_model = AutoAWQForCausalLM.from_pretrained(str(a.merged_out), device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(str(a.merged_out))
    calib = load_calibration(a.train_jsonl)
    awq_model.quantize(tokenizer, quant_config=quant_config, calib_data=calib)
    awq_model.save_quantized(str(a.quant_out))
    processor.save_pretrained(a.quant_out)

    # Same fix as the official checkpoint: lm_head is fp16 but the written
    # quantization_config forgets to exclude it (transformers would wrap it
    # and crash: "expected scalar type Int but found Half").
    cfg_path = Path(a.quant_out) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    mods = cfg["quantization_config"].setdefault("modules_to_not_convert", ["visual"])
    if "lm_head" not in mods:
        mods.append("lm_head")
    if "visual" not in mods:
        mods.append("visual")
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print("Patched modules_to_not_convert:", mods)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Create `hpc/quantize_model.sh`**

```bash
#!/usr/bin/env bash
# Merge adapter -> bf16, AWQ-quantize, build the v1 model tarball.
set -euo pipefail
STG=/staging/groups/bhaskar_opscribe/surgvu26
GROUP=/staging/groups/bhaskar_opscribe
export HF_HOME="$GROUP/hf_cache"

mkdir -p pkgs
# Install order matters: transformers LAST so autoawq's resolver can't downgrade it.
pip install --target=./pkgs --quiet "autoawq==0.2.9" --no-deps
pip install --target=./pkgs --quiet "peft==0.15.2" "accelerate==1.7.0" "transformers==4.51.3" \
  "tokenizers>=0.21,<0.22" "safetensors" "datasets==3.5.0" zstandard
export PYTHONPATH="$(pwd)/pkgs:${PYTHONPATH:-}"

BASE=$(find "$GROUP/hf_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)

python3 merge_and_quantize.py \
  --base "$BASE" \
  --adapter "$GROUP/adapters/surgvu_vqa_v1" \
  --train-jsonl "$STG/datasets/surgvu_vqa_v1/lf/train.jsonl" \
  --merged-out merged \
  --quant-out quant

mkdir -p "$STG/models"
rm -rf "$STG/models/qwen25vl7b-surgvu-v1-awq"
cp -r quant "$STG/models/qwen25vl7b-surgvu-v1-awq"

# Tarball with the SAME inner dir name as v0 so the container's
# TARBALL_MODEL_DIR and every job script work unchanged.
mkdir -p tarball_root
cp -r quant tarball_root/qwen2.5-vl-7b-awq
tar -czf "$STG/tarballs/qwen25vl7b-surgvu-v1-awq-model.tar.gz" -C tarball_root qwen2.5-vl-7b-awq
ls -lh "$STG/tarballs/"
rm -rf merged quant tarball_root
```

- [ ] **Step 3: Create `hpc/quantize_model.sub`** — vanilla, `request_gpus = 1`, `request_memory = 96GB`, `request_disk = 120GB`, `requirements = (CUDAGlobalMemoryMb >= 70000)`, bhaskar rank, `+WantStagingMount = true`, `transfer_input_files = ../scripts/merge_and_quantize.py`.
- [ ] **Step 4: Submit; gate** — job completes; `$STG/tarballs/qwen25vl7b-surgvu-v1-awq-model.tar.gz` ≈ 5–6 GB; the patched `config.json` lists `["visual","lm_head"]`.
- [ ] **Step 5: Commit** — `"feat: adapter merge + AWQ self-quantization (lm_head-patched)"`

---

## Task 10: Evaluate, record, decide

- [ ] **Step 1: Parameterize the eval/rehearsal tarball** — in `hpc/eval_samples.sh` and `hpc/rehearse_container.sh`, replace the hardcoded tarball path with `TARBALL="${SURGVU_TARBALL:-$STG/tarballs/qwen25vl7b-awq-model.tar.gz}"`; pass the env through the `.sub` files via `environment = "SURGVU_TARBALL=..."` comment lines documenting usage. Commit.
- [ ] **Step 2: 11-clip eval with the fine-tuned model** — `condor_submit eval_samples.sub -a 'environment = "SURGVU_TARBALL=/staging/groups/bhaskar_opscribe/surgvu26/tarballs/qwen25vl7b-surgvu-v1-awq-model.tar.gz"'`; score locally.
- [ ] **Step 3: Val-set eval** — run `scripts/predict_samples.py`-equivalent over ~100 sampled val clips… simplest: a small `scripts/eval_val_set.py` that loads `val.jsonl` (which references frames in the val tars), runs the model on each, scores with `question_bleu` against the single synthesized answer. Run inside the eval job (same SIF + binds). Report mean.
- [ ] **Step 4: Decision gate** — accept if 11-clip mean BLEU **≥ 0.5141** (no regression) AND val BLEU materially beats the base model on the same val set (run base for comparison if time permits). If the quantized model regresses vs expectations, test the bridge path (peft adapter on official AWQ base) before concluding the fine-tune failed.
- [ ] **Step 5: Record + ship** — update `docs/BASELINES.md` (new row + per-case + val numbers); if accepted: this tarball becomes the submission model artifact (container code unchanged — verify with one `rehearse_container.sub` run using the new tarball env). Update README model section. Commit, push, merge per finishing-a-development-branch.

---

## Self-Review

**1. Spec coverage:** §7 synthesis (tool-usage pos/neg with both vocabularies, step what/yes-no, which-tools, balanced negatives, held-out split, leakage exclusion, deterministic) → Tasks 1–6; §8 fine-tuning (LoRA r32/α64, ~3 epochs, frames-consistent-with-inference, messages format) → Tasks 6–8; deployment back into the T4 container (spec §10's artifact contract preserved — same tarball inner dir, same loader, lm_head patch carried) → Task 9; §11 eval gates → Task 10. Distillation (spec §7 optional) deliberately deferred to M2.5.

**2. Placeholder scan:** all code steps carry complete code; Task 1's inspect-then-bind (tasks.csv header) and Task 5's case-list cross-check are explicit verification steps with stated expected outcomes, not unknowns.

**3. Type consistency:** `CaseLabels.load(Path) -> CaseLabels`; `plan_clips(CaseLabels, …) -> list[ClipSpec]`; `synthesize_qa(CaseLabels, ClipSpec, seed) -> list[QAPair]`; shard record `{case, kind, question, answer, images[8]}` consumed by `merge_qa_dataset._to_sharegpt`; LF registration mirrors the proven phase-detector entry; tarball inner dir `qwen2.5-vl-7b-awq` = `TARBALL_MODEL_DIR` basename in `model.py` ✓.

**4. Environment reality:** 344 GB zip never fully downloaded (remotezip verified against the real archive; per-case parts ~0.8 GB to quota-free scratch); staging gains only ~98 packed files; pip-at-runtime only on execute-node scratch (not noexec staging); training stack pins = the proven phase-detector set; quantize stack pinned to AutoAWQ's last-tested combo with install-order guard; transformers 4.52+ excluded everywhere; bhaskar rank on all GPU jobs; CUDACapability ≥ 7.5 not needed for training (bf16, no AWQ kernels) but the eval jobs already carry it.
