# SurgVU 2026 — Category 2 (Surgical VQA) — Design Spec

- **Date:** 2026-06-09
- **Author:** Noah Bryant
- **Status:** Approved in brainstorming → ready for implementation planning
- **Repo:** `~/opscribe-job/surgvu26-vqa/` (standalone; reuses the OpScribe VLM core)
- **Challenge:** [SurgVU 2026](https://surgvu26.grand-challenge.org/), Category 2 — Surgical Visual Question Answering (EndoVis @ MICCAI 2026)

---

## 1. Goal & success criteria

Compete in SurgVU 2026 Category 2 (Surgical VQA). Strategy is **staged**, matching the agreed target *"ship complete, then push top-3"*:

1. **Cycle 1 — de-risk:** ship a complete, valid container early → proves the Grand Challenge path end-to-end and earns eligibility for the **$500 best-methodology-report** prize + a leaderboard spot.
2. **Cycle 2 — compete:** iterate the model toward a **top-3 2026** finish.

- **Primary metric: BLEU** — for each question, BLEU of our answer vs each of **5 reference answers**, take the **max**; final score = **mean of max-BLEU over all Q&A pairs**; uniform 1–4-gram weights (0.25 each) + smoothing. Implication: **answer phrasing/length is a first-class optimization target.**
- **Prizes (Cat 2):** 2026 1st/2nd/3rd = $1,000 / $500 / $250; "overall" (must beat 2025's best) = $3,000 / $2,000 / $1,000; best methodology report = $500 (any team). Can win a 2026 *or* an overall prize, not both.

## 2. Challenge facts (reference)

- **Organizer:** Intuitive Surgical; **sponsor:** AWS; part of EndoVis @ MICCAI 2026.
- **Timeline:** registration + training data **May 8**; preliminary phase **Jul 20 – Sep 2**; **new-registration deadline Aug 15**; final phase **Aug 21 – Sep 6** (hard submission deadline); methodology reports due **Sep 13**; challenge day **~Sep 27 / Oct 1**.
- **Task:** given a **30-second clip + an open-ended question**, output **one free-text answer**.
- **Data (shared with Cat 1):** 280 videos / 155 training sessions / ~840 hrs / >18M frames / 60 fps / 720p; da Vinci robotic surgery on **porcine** models. Test clips sampled at 1 fps.
- **Labels (weak / noisy):** `tools.csv` — tool install/uninstall intervals (≤3 tools installed at once, may be off-camera → noisy); `tasks.csv` — surgical-step start/stop intervals. Eval tool label column = `groundtruth_toolname` (not `commercial_toolname`).
- **12 tool classes:** needle driver, cadiere forceps, prograsp forceps, monopolar curved scissors, bipolar forceps, stapler, force bipolar, vessel sealer, permanent cautery hook/spatula, clip applier, tip-up fenestrated grasper, grasping retractor.
- **8 step classes:** suturing, uterine horn, suspensory ligaments, rectal artery/vein, skills application, range of motion, retraction & collision avoidance, other.
- **Public sample set** (**11 clips** — case122–case132, verified from the actual download 2026-06-10; the dataset paper said 10 — each question with **5 reference answers**): `https://storage.googleapis.com/isi-surgvu/SURGVU25_cat_2_sample_set_public.zip`. Layout: per-clip dirs `caseNNN/` holding `caseNNN.mp4`, `caseNNN_question.json` (bare JSON string), `caseNNN.json` (bare JSON array of 5 answer strings — the **first is a short canonical answer, the other four are full-sentence declarative paraphrases**). Example — Q: *"Are there forceps being used here?"* → refs *"No"*, *"No, forceps are not mentioned."*, *"No forceps are being used."*, *"No, there's no indication of forceps."*, *"No forceps are listed."*
- **Bulk data (2024/general):** videos `https://storage.googleapis.com/isi-surgvu/surgvu24_videos_only.zip`; labels `https://storage.googleapis.com/isi-surgvu/surgvu24_labels_updated_v2.zip`. Official 2026 materials to be pulled after registration.

## 3. Grand Challenge submission constraints (hard)

- **Compute:** 1 GPU — **default NVIDIA T4 (16 GB)**, A10G (24 GB) only if organizers grant it; 8 vCPU; 32 GB RAM; ~225 GB scratch.
- **Time limit:** **~5 min/case** default (organizer-set; model-load may count against it).
- **Offline:** no internet at inference. Weights baked into the image (`/opt/app/resources`) **or** shipped as a model tarball extracted to `/opt/ml/model`.
- **I/O interface** (fork of `isi-challenges/surgvu2025-category2-submission`; 2026 points to this template):
  - **Input** (read-only `/input`): `endoscopic-robotic-surgery-video.mp4` (the clip), `visual-context-question.json` (the question string), `inputs.json` (socket list).
  - **Output** (`/output`): `visual-context-response.json` = a **JSON-encoded string** (the answer text — not an object).
  - **Container:** `ENTRYPOINT ["python", "inference.py"]`, non-root `user`, workdir `/opt/app`.
- **Model-size feasibility:** Qwen2.5-VL-**7B** is feasible (4-bit ~6–8 GB on T4; fp16 ~16 GB on A10G). **72B is infeasible** on a single GC GPU (~40–48 GB even at 4-bit) → big models are **offline-only** (training-data generation on CHTC).
- **Known unknowns (do not guess):** the 2026 page defers some Category-2 format details ("Stay tuned"); a SurgVU-specific GPU/time number is unpublished; the exact free-text scoring is taken from the official evaluation-criteria page (BLEU as above) and re-verified after registration.

## 4. Architecture — two pipelines, one shared artifact

```
OFFLINE — CHTC 8xH200 (training)                    ONLINE — Grand Challenge container (inference)
─────────────────────────────                       ──────────────────────────────────────────────
SurgVU data: videos + tools.csv + tasks.csv         /input/endoscopic-robotic-surgery-video.mp4
        │                                            /input/visual-context-question.json
        ▼                                                   │
 [clip segmenter] 30-s label-aligned clips                  ▼
        │                                            [frame sampler] ~8-16 frames
        ▼                                                   │
 [QA synthesizer] (clip, question, answer)                  ▼
   from weak labels (tool-usage/step/...)            [question router]
        │              ▲                               closed-form ─► [perception + BLEU template]
        │   [72B/235B distiller] (optional)            free-form   ─► [7B VQA generate]
        ▼                                                   │
 [LoRA fine-tune Qwen2.5-VL-7B] ─► merge ─► weights         ▼
                          │                          [answer post-processor] → canonical short string
                          └──────── baked into ───►          │
                                    container                ▼
                                                     /output/visual-context-response.json (JSON string)
```

The **question router** makes the staged plan incremental: in Cycle 1 it is a no-op (everything → 7B VLM = Approach B); in Cycle 2 we add the closed-form template branch (Approach C) and swap the base 7B for the fine-tuned one (Approach A). The container shell is unchanged between cycles.

## 5. Project structure

Standalone repo. During development/training it **imports** OpScribe's VLM core (`opscribe_pipeline/providers/vlm/{base,qwen_vl,factory}.py` + `opscribe_pipeline/models/`). For the offline container it **vendors a pinned copy** of just those minimal modules — the container must be self-contained with no internet. We deliberately do **not** fork or carry the heavy OpScribe package (graph / synthesis / billing / RAG).

```
surgvu26-vqa/
├── data/
│   ├── download.sh              # fetch videos, labels, public sample set
│   ├── clips.py                 # segment videos → 30-s label-aligned clips; frame sampling
│   └── qa_synthesis.py          # (clip, Q, A) from tools.csv/tasks.csv (+ optional distillation)
├── train/                       # LoRA SFT — thin wrapper over OpScribe train_qwen_chtc.py
├── eval/                        # local BLEU harness (10 public samples + held-out split)
├── answer_shaping.py            # BLEU-tuned answer templates + post-processing
├── container/                   # fork of isi-challenges/surgvu2025-category2-submission
│   ├── inference.py             # GC entrypoint: router + VLM + templater + post-proc
│   ├── Dockerfile · build.sh · do_test_run.sh · export.sh
│   └── resources/               # vendored VLM core; weights via model-tarball (/opt/ml/model)
└── docs/superpowers/specs/      # this design doc; methodology report draft
```

## 6. Components (each testable in isolation)

| Component | Single responsibility | Depends on |
|---|---|---|
| `data/clips.py` | Cut videos into 30-s clips aligned to label intervals; sample N frames | cv2, label CSVs |
| `data/qa_synthesis.py` | Weak labels → (clip, question, answer) pairs in canonical style | clips.py, label CSVs |
| distiller (optional) | Offline 72B/235B → richer/paraphrased QA | OpScribe `QwenVLProvider` |
| `train/` | LoRA fine-tune 7B on the synthetic set | `train_qwen_chtc.py`, peft |
| `container/inference.py` | Route question → produce answer | vendored `qwen_vl`, `answer_shaping` |
| `answer_shaping.py` | (question, perception) → BLEU-optimal short string | — |
| `eval/` | Compute challenge BLEU locally to rank iterations | sacrebleu / nltk |

## 7. Training-data synthesis (the core)

For each 30-s clip, interval-overlap on the labels tells us **which tools were in use** and **which step was active**. From that we mint grounded Q&A in the challenge's style:

- **Tool-usage (yes/no):** *"Was a [tool] used during the surgery?"* → *"[Tool] was used."* / *"A [tool] was not used."* — generated for **all 12 tools per clip** so we get balanced positives **and** negatives.
- **Step:** *"What surgical step is being performed?"* → *"[step]."*; plus *"Is [step] being performed?"* (yes/no).
- **Multi-tool identification:** *"Which tools are being used?"* → list. **Avoid exact counting** (label noise).

Decisions:
- **Answer phrasing is validated empirically**, not assumed — parse the 10 public samples for canonical phrasing, then use the local BLEU harness to test short (*"No"*) vs full declarative (*"A large needle driver was not used."*). Hypothesis: the **full declarative form maximizes max-BLEU** (matches the long reference's higher-order n-grams). Confirm before committing.
- Use **`groundtruth_toolname`** vocabulary throughout.
- **Held-out video split** for our own BLEU eval — never train on the videos we test against.
- **Distillation (Cycle 2 hedge):** offline 72B / Qwen3-VL-235B on CHTC generate free-form descriptive Q&A + answer paraphrases, as insurance against open-ended hidden-test questions. Primary signal stays label-derived (grounded, no hallucination); distillation is additive.
- **Scale:** tens of thousands of balanced QA pairs, capped for training tractability.

## 8. Model & fine-tuning

- **Base:** Qwen2.5-VL-7B-Instruct; **LoRA** via `scripts/train_qwen_chtc.py` — target `q/k/v/o_proj`, rank 32 / alpha 64, lr 1e-4 cosine, ~3 epochs (mirrors the phase-detector recipe).
- **Multi-frame consistency:** train with the *same* ~8–16 sampled frames per clip used at inference; `messages` dataset format (image list + text) per `build_phase_detector_dataset.py`.
- Train on 8×H200; `merge_and_unload()` → single weights directory for the container.

## 9. In-container inference + BLEU answer-shaping

`container/inference.py` (forked from the official 2025 Cat-2 template):
1. Read `/input/inputs.json`; load the clip + question string.
2. **Frame sampler:** decode the mp4, take ~8–16 evenly spaced frames (capped for VRAM + the ~5-min limit).
3. **Question router** (no-op in Cycle 1): keyword/vocabulary match → *tool-usage* | *step* | *free-form*.
   - closed-form → fine-tuned 7B perception → compose via `answer_shaping` templates.
   - free-form → 7B VQA generate directly.
4. **Post-process:** normalize tool names to the groundtruth vocabulary; match reference case/punctuation; **greedy decode** (deterministic, stable BLEU).
5. Write the answer **as a JSON string** to `/output/visual-context-response.json`.

## 10. Weight packaging (offline) — gotchas

- **Ship weights as a GC model tarball → `/opt/ml/model`** (thin image, sidesteps any image-size ceiling). 7B 4-bit ≈ 5–8 GB.
- **🔴 Disable Flash-Attention on T4** — Turing doesn't support FA2; make the attention backend **conditional** (`sdpa`/eager on T4). OpScribe's default FA2 path would crash the container.
- **Pre-fetch at build time** (build has internet; runtime does not): code + processor/config go **in the image**, the **merged 7B weights go via the model-tarball** (`/opt/ml/model`). Default to **bnb 4-bit** (matches OpScribe's 72B 4-bit path); keep an fp16 path if A10G is granted.

## 11. Testing & evaluation

- **Local BLEU harness** replicating the exact metric (max over 5 refs, mean over Qs, uniform 0.25×4 weights + smoothing) — validate on the 10 public samples (known refs) first, then the held-out split.
- **Offline container test:** the official `do_test_run.sh` with `--network none` on provided fixtures → proves the I/O contract, offline operation, and **per-case latency** vs the time limit on T4-class hardware.
- **Ablations ranked by local BLEU:** answer-form, frame count, fine-tuned vs base, router on/off.
- **Unit tests (TDD):** `clips.py` interval alignment; `qa_synthesis.py` label→answer correctness; `answer_shaping` templates; frame sampler.

## 12. Milestones (today Jun 9 → register by Aug 15 → final Sep 6)

| Milestone | Target | What |
|---|---|---|
| **M0** | Jun 9–16 | Register on **surgvu26** + sign data agreement; download data + 10 samples; scaffold repo; build local BLEU harness |
| **M1 (Cycle 1)** | by Jul 20 (prelim opens) | Approach B container (base 7B, prompt-engineered, offline, T4-safe) → **first valid prelim submission** |
| **M2** | Jul–Aug | `clips` + `qa_synthesis` → v1 training set → LoRA fine-tune 7B (Approach A); beat B locally |
| **M3** | Aug | Add router + `answer_shaping` (Approach C); ablate; iterate prelim leaderboard (≤10 tries) |
| **M4** | by Sep 6 | Lock best config → **2-shot final submission** |
| **M5** | by Sep 13 | Methodology report + GitHub + 3-min video (report-prize eligible) |

## 13. Risks & open unknowns

- **Test-question taxonomy not fully published** ("Stay tuned") → cover the known style broadly + distillation for open-ended; re-verify after pulling official 2026 materials.
- **GPU tier (T4 vs A10G) uncertain** → design T4-safe (4-bit, no FA2, frame cap); request A10G from organizers.
- **Weak-label noise** (3 installed ≠ 3 visible) → lean on presence intervals; avoid exact-counting questions.
- **BLEU over-fitting to public samples** → held-out split; no degenerate one-size-fits-all answer.
- **Time-limit blow-out** → fallback levers: fewer frames, lower `min/max_pixels`, or pre-quantize AWQ for speed.

## 14. Manual prerequisites (user action)

- **Register on `surgvu26.grand-challenge.org`** (Join → admin approval) and **sign the Intuitive data agreement** — the existing approval was on the *2024* page and does **not** carry to 2026.
- Pull the **official 2026** data, public sample set, and Cat-2 submission template once registered; confirm any updated format/metric details.
- If T4 proves tight, **request an A10G** GPU from the organizers.

## 15. Procedure-agnostic note (OpScribe principle)

This challenge is **robotic gyn/general surgery on porcine models** — a different domain from OpScribe's orthopedic open surgery. The OpScribe ortho captioner adapter and surgical vocabulary do **not** transfer; what transfers is the **infrastructure** (Qwen2.5-VL inference, LoRA pipeline, offline container build). Keep challenge-specific tool/step vocabulary isolated in `qa_synthesis.py` / `answer_shaping.py`, not hard-coded into shared code.
