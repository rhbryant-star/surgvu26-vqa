# SurgVU 2026 Cat 2 — Submission Checklist

**Model:** M2 v2 (echo-free LoRA fine-tune, AWQ 4-bit) — local BLEU **0.5620** on the 11 public clips
(beats the v3 prompt baseline 0.5141 by +0.048). Scoreboard: `docs/BASELINES.md`.

## The two artifacts

Grand Challenge runs an **algorithm container** (our code) with a **Model** (the weights) mounted at
`/opt/ml/model` — the container is code-only, weights are separate. `model.py` loads
`/opt/ml/model/qwen2.5-vl-7b-awq`, which is exactly the tarball's top-level dir, so **no code change is
needed to swap models.**

| Artifact | What | Where |
|---|---|---|
| **Algorithm image** (code, ~3 GB) | built by CI from `main` | GHCR `ghcr.io/rhbryant-star/surgvu26-vqa-cat2:latest` + a docker-save `.tar.gz` in the `build-container` run artifacts |
| **Model** (v2 weights, 5.4 GB) | `qwen25vl7b-surgvu-v2-awq-model.tar.gz` | CHTC staging `/staging/groups/bhaskar_opscribe/surgvu26/tarballs/` — download to upload to GC |

## Steps (your Grand Challenge account)

1. Create/update the Cat 2 **Algorithm** — GC docs: <https://grand-challenge.org/documentation/create-your-own-algorithm/>
2. **Upload the image** — either point GC at the GHCR image, or upload the docker-save `.tar.gz`.
3. **Attach the Model** — upload `qwen25vl7b-surgvu-v2-awq-model.tar.gz`; GC mounts it at `/opt/ml/model`.
4. **Submit to the Preliminary Testing Phase** (open now — began Jul 20).

## Phase budget (don't waste it)

- **Preliminary: up to 10 tries.** This is the debug phase — its real job is to prove the container runs on
  GC's **T4 (16 GB, offline, ~5 min/case)**, which we have **never** tested (only H200/CHTC). Expect to burn
  a few tries shaking out container mechanics. Does **not** count toward the final ranking.
- **Final: only 2 submissions, best-of-2 counts.** Save these for a validated container.

## What to watch on the first preliminary run

1. **Does it run at all on the T4** — model load + 8-frame inference under 16 GB and the time limit. AWQ is
   ~6-8 GB so it should fit, but this is the unverified step.
2. **Output format** — `/output/visual-context-response.json` must be a bare JSON-encoded string (the CI
   smoke test checks this).
3. **Hidden-set BLEU vs our 0.5620 local** — how the proxy translates.

## Eligibility (for winning, not just scoring)

Final placement also requires a **methodology report** (due Sep 13), **GitHub codebase**, and a
**pre-recorded presentation video** — not just a leaderboard score.
