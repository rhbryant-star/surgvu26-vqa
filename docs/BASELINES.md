# Local BLEU scoreboard (11 public clips, max-over-5-refs, mean)

| Date | Config | mean BLEU | Notes |
|---|---|---|---|
| 2026-06-10 | dummy all-"No" | 0.0485 | floor — any model must beat this |
| 2026-06-11 | M1: Qwen2.5-VL-7B-AWQ, 8 frames, prompt v1, greedy | 0.1881 | image `ghcr.io/rhbryant-star/surgvu26-vqa-cat2` @ branch m1-baseline-container; eval job 7653222 (H200, ~2.7 s/clip generate; rehearsal: 8.01 GiB CUDA peak, 64 s/case total incl. 47 s load) |
| 2026-06-11 | v2: type-routed prompts + single-word shaping fix | 0.3167 | eval job 7653343; yes/no cases fixed (0.45–0.67), identification still terse |
| 2026-06-12 | **v3: structural identify example + statement-form yes/no** | **0.5141** | eval job 7660269; case131 = 1.0000; predictions archived as `hpc/predictions_v3.json` |

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

## Lever history (all empirically confirmed)

1. ✅ **v2: never emit terse answers** — "No." scores exactly 0.0 (tokenizer: `no.` ≠ `no`); single-word period-strip + sentence-form prompts. +0.13 mean.
2. ✅ **v2: type-routed instructions** (yes/no vs identify vs generic) killed example leakage.
3. ✅ **v3: structural examples teach the restatement pattern** — identification questions need an example with the right SHAPE (non-challenge object: "What color is the marking pen?" → "The marking pen color is blue."); abstract instructions alone under-format. +0.20 mean.

## Remaining levers

- case129/130 (~0.10): the model over-restates into wordy multi-clause answers, diluting n-gram precision. Possible: stronger brevity cue or shaping-side trimming.
- **Correctness ≠ BLEU**: case122 answers "Yes, forceps are being used." (refs say No) yet scores 0.67 — the restatement n-grams dominate the Yes/No token. The M2 fine-tune is what buys actual correctness (and the metric may not stay BLEU-shaped forever).

Caveat: local metric is for RELATIVE ranking (official tokenizer/smoothing
unpublished — see surgvu_vqa/eval/bleu.py).
