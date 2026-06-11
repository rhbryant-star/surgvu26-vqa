# Local BLEU scoreboard (11 public clips, max-over-5-refs, mean)

| Date | Config | mean BLEU | Notes |
|---|---|---|---|
| 2026-06-10 | dummy all-"No" | 0.0485 | floor — any model must beat this |
| 2026-06-11 | M1: Qwen2.5-VL-7B-AWQ, 8 frames, prompt v1, greedy | **0.1881** | image `ghcr.io/rhbryant-star/surgvu26-vqa-cat2` @ branch m1-baseline-container; eval job 7653222 (H200, ~2.7 s/clip generate; rehearsal: 8.01 GiB CUDA peak, 64 s/case total incl. 47 s load) |

## Per-case (M1 baseline)

```
case122: 0.0000  "No."                                            <- terse-answer zero
case123: 0.5411  "No, a large needle driver was not used."
case124: 0.4347  "The forceps type is Maryland Bipolar Forceps."
case125: 0.1323  "No suture is required."
case126: 0.0000  "No."                                            <- terse-answer zero
case127: 0.0408  "The forceps type is Cadiere Forceps."
case128: 0.1757  "No, a needle driver was not used."
case129: 0.0341  "The forceps type is ..."                        <- prompt-example leakage
case130: 0.0375  "The forceps type is ..."                        <- prompt-example leakage
case131: 0.0000  "No."                                            <- terse-answer zero
case132: 0.6732  "A large needle driver was used."
```

## Confirmed levers for M3 (answer shaping / prompt v2)

1. **Never emit terse answers.** "No." scores exactly 0.0 (tokenizer: `no.` ≠ `no`; no higher n-grams). Full declaratives score 0.4–0.7. Forcing sentence-form on the three zeros alone ≈ +0.10–0.15 mean.
2. **Kill prompt-example leakage.** The style example "The forceps type is Cadiere Forceps." appears verbatim in answers to unrelated questions (case129/130). Use a neutral or question-specific exemplar.
3. Question-type routing (spec §4) remains the structural fix for both.

Caveat: local metric is for RELATIVE ranking (official tokenizer/smoothing
unpublished — see surgvu_vqa/eval/bleu.py).
