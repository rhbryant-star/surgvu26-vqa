#!/usr/bin/env python3
# scripts/merge_and_quantize.py
"""Merge a PEFT LoRA into the bf16 Qwen2.5-VL-7B base, then AutoAWQ-quantize the
merged model to the SAME 4-bit artifact shape the inference container already
loads (decoder quantized, vision tower + lm_head left in fp16).

WHY this exact recipe (every step is load-bearing — see the M2 research findings):

  1. MERGE (peft 0.15.2):
     Qwen2.5-VL is an image-text-to-text model, so the base MUST be loaded with
     ``Qwen2_5_VLForConditionalGeneration`` (NOT ``AutoModelForCausalLM`` — that
     resolves to the wrong head and won't accept the VL config).  PEFT only wraps
     the modules named in the adapter's ``target_modules`` (q/k/v/o_proj, which
     live ONLY under ``model.layers.*.self_attn`` in the LM decoder).  The vision
     tower (``visual.*``) has no LoRA layers, so ``merge_and_unload()`` leaves it
     bit-identical to the base.  We merge in bf16 (the dtype the LoRA was trained
     against) to avoid precision drift in the B@A product, and write the merged
     model to DISK because AutoAWQ 0.2.9 cannot quantize an in-memory module — its
     ``from_pretrained`` takes a path string only.

  2. QUANTIZE (autoawq 0.2.9):
     ``AutoAWQForCausalLM.from_pretrained`` auto-dispatches on ``config.model_type
     == "qwen2_5_vl"`` to ``Qwen2_5_VLAWQForCausalLM``, whose class attribute
     ``modules_to_not_convert = ["visual"]`` auto-excludes the vision tower.  We
     pass a TOKENIZER (not a processor) plus a plain ``list[str]`` of calibration
     text built from our own fine-tuning QA — this drives the DEFAULT AwqQuantizer,
     which forwards input_ids only (no pixel_values), so the decoder is calibrated
     text-only and the vision tower is never touched.  A list[str] also BYPASSES
     ``load_dataset`` entirely, which is what keeps the job fully offline.
     AWQ kernels are fp16, so we load for quant as float16 (the bf16 merged
     weights down-cast once on load).

  3. PATCH (pure stdlib):
     After ``save_quantized`` the emitted config reads
     ``modules_to_not_convert = ["visual"]``.  transformers 4.51.3's
     ``replace_with_awq_linear`` is purely name-based and has NO automatic lm_head
     exclusion, so it would wrap our fp16 ``lm_head.weight`` as a quantized linear
     and the AWQ GEMM kernel crashes ("expected scalar type Int but found Half").
     We therefore append ``"lm_head"`` to the list IN PLACE, matching the official
     ``Qwen2.5-VL-7B-Instruct-AWQ`` deployment shape and the
     ``hpc/fetch_weights.sh`` patch already proven in prod.  The append is
     idempotent and uses only the stdlib ``json`` module.

API assumptions (verified against AutoAWQ v0.2.9 / PEFT v0.15.2 source) are tagged
inline with ``[VERIFIED:...]``; conservative fallbacks are tagged ``[UNVERIFIED]``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def log(msg: str) -> None:
    """Unbuffered progress line (the SIF sets PYTHONUNBUFFERED but be explicit)."""
    print(f"[merge_and_quantize] {msg}", flush=True)


# --- calibration text -------------------------------------------------------

# The user turn in our LF sharegpt jsonl is prefixed with one "<image>" line per
# frame (8 of them).  For TEXT-ONLY calibration we feed the decoder input_ids
# only, so we strip those placeholder lines — leaving them in would tokenize a
# vision-control token with no matching pixel_values and skew calibration.
# [VERIFIED:scripts/merge_qa_dataset.py — user content == ("<image>\n")*8 + text]
_IMAGE_PLACEHOLDER = "<image>"


def _strip_image_placeholders(text: str) -> str:
    """Drop standalone ``<image>`` placeholder lines from a user turn."""
    kept = [ln for ln in text.split("\n") if ln.strip() != _IMAGE_PLACEHOLDER]
    return "\n".join(kept).strip()


def load_calib_messages(jsonl_path: Path, max_samples: int) -> list[list[dict]]:
    """Read up to ``max_samples`` chat-message lists from a local LF jsonl.

    Each line is ``{"messages": [{role, content}, ...], "images": [...]}`` (the
    sharegpt format our merge_qa_dataset.py emits).  We keep the user+assistant
    text turns (and the system turn) and strip the ``<image>`` placeholders so
    the resulting messages are pure text, ready for ``apply_chat_template``.
    """
    messages_out: list[list[dict]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec.get("messages")
            if not msgs:
                continue
            cleaned: list[dict] = []
            for m in msgs:
                content = m.get("content", "")
                if isinstance(content, str):
                    content = _strip_image_placeholders(content)
                cleaned.append({"role": m["role"], "content": content})
            messages_out.append(cleaned)
            if len(messages_out) >= max_samples:
                break
    return messages_out


def build_calib_strings(
    tokenizer, messages_list: list[list[dict]], max_seq_len: int
) -> list[str]:
    """Render each message list to a single calibration string via the chat
    template, then trim to ``max_seq_len`` tokens.

    ``add_generation_prompt=False`` keeps the FULL question+answer turns (the
    Qwen AWQ recipe calibrates on complete fine-tuning text, not just prompts).
    [VERIFIED:Qwen AWQ docs — calibrate on fine-tuning text via chat template]
    The return value is a plain ``list[str]``; AutoAWQ's get_calib_dataset wraps
    each as ``{text_column: s}`` and tokenizes it, BYPASSING load_dataset (which
    is the offline-safety guarantee).
    [VERIFIED:AutoAWQ v0.2.9 awq/utils/calib_data.py — isinstance(data[0], str)]
    """
    out: list[str] = []
    for msgs in messages_list:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        ).strip()
        if not text:
            continue
        # Trim to max_seq_len tokens so no single calib sample blows the per-block
        # length budget. truncation=True keeps the head of the conversation.
        ids = tokenizer(
            text, truncation=True, max_length=max_seq_len, add_special_tokens=False
        )["input_ids"]
        out.append(tokenizer.decode(ids, skip_special_tokens=False))
    # Fail loud NOW if a vision-control token leaked: a surviving <|image_pad|>
    # would make the VL forward demand pixel_values=None and crash mid-quantize.
    _vt = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
    bad = sum(1 for s in out if any(t in s for t in _vt))
    if bad:
        raise SystemExit(
            f"ERROR: {bad} calibration strings contain a vision token {_vt}; "
            "text-only calibration would crash the VL forward."
        )
    return out


# --- merge ------------------------------------------------------------------

def merge_lora(base_dir: str, adapter_dir: str, merged_dir: str) -> None:
    """Load bf16 base + PEFT LoRA, merge_and_unload, save merged + processor."""
    import torch
    from peft import PeftModel
    # [VERIFIED:FINDINGS B — Qwen2.5-VL is image-text-to-text; use the VL CG class,
    #  NOT AutoModelForCausalLM] Import the explicit VL class.
    from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    log(f"loading base (bf16, cpu) from {base_dir}")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16,  # merge in the dtype the LoRA was trained in
        device_map="cpu",            # CPU merge is deterministic; ~15min budget is ample
        low_cpu_mem_usage=True,
    )

    log(f"attaching PEFT LoRA adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)

    # Sanity: the adapter must NOT have wrapped any vision-tower module, or the
    # merge would alter weights AWQ expects to leave in fp16.  Qwen2.5-VL's
    # vision blocks use fused attn.qkv/attn.proj names, so an explicit
    # q/k/v/o_proj target list cannot collide — assert it anyway.
    # [VERIFIED:FINDINGS B — verify no lora_ modules under visual.]
    leaked = [n for n, _ in model.named_modules()
              if "visual" in n and (".lora_A" in n or ".lora_B" in n)]
    if leaked:
        raise SystemExit(
            f"ERROR: adapter wrapped {len(leaked)} vision-tower modules "
            f"(e.g. {leaked[0]}); AWQ assumes 'visual' is untouched. Aborting."
        )

    log("merge_and_unload (folds B*A*scaling into base Linear weights)")
    merged = model.merge_and_unload()

    log(f"saving merged model (bf16, safetensors) to {merged_dir}")
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)

    # save_pretrained writes config.json + generation_config.json + weights, but
    # NOT the processor/tokenizer.  AutoAWQ calibration needs the tokenizer, and
    # later T4 inference needs the full processor (preprocessor_config.json, chat
    # template, vocab, merges).  Carry them through from the base snapshot.
    # [VERIFIED:FINDINGS B Q3 — save_pretrained omits processor/tokenizer]
    log("copying processor + tokenizer into merged dir")
    AutoProcessor.from_pretrained(base_dir).save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(base_dir).save_pretrained(merged_dir)

    # Free the bf16 model before the quant load re-reads from disk in fp16.
    del base, model, merged
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- quantize ---------------------------------------------------------------

def quantize(merged_dir: str, out_dir: str, calib: list[str]) -> None:
    """AutoAWQ self-quantize the merged dir to 4-bit GEMM, text-only calibration."""
    import torch
    from awq import AutoAWQForCausalLM
    from transformers import AutoProcessor, AutoTokenizer

    # [VERIFIED:FINDINGS A — exact dict AutoAWQ 0.2.9 + official Qwen recipe use;
    #  q_group_size=128 matches the official 7B-AWQ checkpoint]
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",  # serialized to config.json as "gemm" — do NOT "fix"
    }

    log(f"loading merged model into AutoAWQ (fp16) from {merged_dir}")
    # [VERIFIED:FINDINGS A — AutoAWQForCausalLM is the single entrypoint; it
    #  auto-dispatches via config.model_type to Qwen2_5_VLAWQForCausalLM]
    # torch_dtype float16: AWQ kernels are fp16; bf16 weights down-cast on load.
    # No attn_implementation: SDPA is fine and flash-attn is not in the SIF.
    model = AutoAWQForCausalLM.from_pretrained(
        merged_dir,
        torch_dtype=torch.float16,
        safetensors=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_dir)

    log(f"quantizing (4-bit GEMM, text-only calib, n={len(calib)} samples)")
    # [VERIFIED:FINDINGS A — first positional arg is the TOKENIZER; calib_data is
    #  a plain list[str]; modules_to_not_convert=['visual'] is auto-populated from
    #  the class attribute, NOT passed here]
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib)

    # [VERIFIED:FINDINGS A — set use_cache before save so the saved generation
    #  config doesn't carry use_cache=False from the from_pretrained default]
    model.model.config.use_cache = True
    model.model.generation_config.use_cache = True

    log(f"save_quantized -> {out_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # [VERIFIED:FINDINGS A — Qwen recipe overrides shard_size='4GB']
    model.save_quantized(out_dir, safetensors=True, shard_size="4GB")

    # save_quantized does NOT re-emit the tokenizer/processor for a VL model — we
    # must write both so the T4 container has preprocessor_config.json + chat
    # template + vocab/merges. Pull from the merged dir (which already has them).
    # [VERIFIED:FINDINGS A/B — VL save_quantized omits processor; replicate the
    #  official Qwen2.5-VL-7B-Instruct-AWQ root file set]
    log("writing tokenizer + processor into AWQ out dir")
    AutoTokenizer.from_pretrained(merged_dir).save_pretrained(out_dir)
    AutoProcessor.from_pretrained(merged_dir).save_pretrained(out_dir)


# --- lm_head patch (pure stdlib) -------------------------------------------

def patch_lm_head(out_dir: str) -> None:
    """Append 'lm_head' to quantization_config.modules_to_not_convert, idempotent.

    Mirrors hpc/fetch_weights.sh: AutoAWQ writes ['visual'] only, but transformers
    4.51.3 will otherwise wrap the fp16 lm_head as a quantized linear and the AWQ
    kernel crashes ('expected scalar type Int but found Half').  Reads back the
    ACTUAL written list and appends (rather than hard-writing ['visual','lm_head']
    blind) in case AutoAWQ wrote a more specific token.
    [VERIFIED:FINDINGS C — name-based skip in replace_with_awq_linear, no auto
     lm_head exclusion; matches official AWQ checkpoint deployment shape]
    """
    cfg_path = Path(out_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    qcfg = cfg.get("quantization_config")
    if qcfg is None:
        raise SystemExit(
            f"ERROR: {cfg_path} has no quantization_config — save_quantized "
            "did not embed it; cannot patch lm_head."
        )
    mods = qcfg.setdefault("modules_to_not_convert", [])
    if mods is None:  # defensive: some writers emit null
        mods = []
        qcfg["modules_to_not_convert"] = mods
    if "lm_head" not in mods:
        mods.append("lm_head")
        cfg_path.write_text(json.dumps(cfg, indent=2))
        log(f"patched modules_to_not_convert -> {mods}")
    else:
        log(f"lm_head already excluded (no-op): {mods}")


# --- main -------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Merge a PEFT LoRA into Qwen2.5-VL-7B bf16 base and AutoAWQ "
        "4-bit quantize the result (text-only calibration, offline-safe)."
    )
    p.add_argument("--base", required=True,
                   help="Local bf16 snapshot dir of Qwen/Qwen2.5-VL-7B-Instruct.")
    p.add_argument("--adapter", required=True,
                   help="PEFT LoRA dir (adapter_model.safetensors + adapter_config.json).")
    p.add_argument("--merged-dir", required=True,
                   help="Scratch dir to write the merged fp16/bf16 model to.")
    p.add_argument("--out-dir", required=True,
                   help="Output dir for the AWQ-quantized model (becomes the tarball root).")
    p.add_argument("--calib-jsonl", required=True,
                   help="Local LF sharegpt train.jsonl; user+assistant text is the calib corpus.")
    p.add_argument("--max-calib-samples", type=int, default=128,
                   help="Number of calibration samples (default 128, the AutoAWQ default).")
    p.add_argument("--max-calib-seq-len", type=int, default=512,
                   help="Per-sample token length cap (default 512, the AutoAWQ default).")
    a = p.parse_args()

    # Belt-and-suspenders: force offline even if the launcher forgot to. Set
    # BEFORE the heavy imports below so HF never reaches the hub.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    base = a.base
    adapter = a.adapter
    merged_dir = a.merged_dir
    out_dir = a.out_dir
    calib_jsonl = Path(a.calib_jsonl)

    for label, path in (("base", base), ("adapter", adapter)):
        if not Path(path).exists():
            raise SystemExit(f"ERROR: {label} dir does not exist: {path}")
    if not calib_jsonl.exists():
        raise SystemExit(f"ERROR: calib jsonl does not exist: {calib_jsonl}")

    log(f"base       = {base}")
    log(f"adapter    = {adapter}")
    log(f"merged-dir = {merged_dir}")
    log(f"out-dir    = {out_dir}")
    log(f"calib      = {calib_jsonl} (<= {a.max_calib_samples} samples, "
        f"seq_len {a.max_calib_seq_len})")

    log("=== STEP 1/3: merge LoRA into base ===")
    merge_lora(base, adapter, merged_dir)

    log("=== STEP 2/3: build calibration corpus + quantize ===")
    # Tokenizer for templating comes from the merged dir we just wrote.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(merged_dir)
    messages_list = load_calib_messages(calib_jsonl, a.max_calib_samples)
    if not messages_list:
        raise SystemExit(f"ERROR: no calibration messages parsed from {calib_jsonl}")
    calib = build_calib_strings(tok, messages_list, a.max_calib_seq_len)
    if not calib:
        raise SystemExit("ERROR: calibration corpus is empty after templating.")
    log(f"calibration corpus ready: {len(calib)} text samples")
    quantize(merged_dir, out_dir, calib)

    log("=== STEP 3/3: patch lm_head in config.json ===")
    patch_lm_head(out_dir)

    # Fail loud rather than ship a silently-unpatched checkpoint to the T4.
    final = json.loads((Path(out_dir) / "config.json").read_text())
    final_mods = final["quantization_config"]["modules_to_not_convert"]
    if "lm_head" not in final_mods:
        raise SystemExit(f"ERROR: lm_head NOT in final modules_to_not_convert: {final_mods}")
    log("DONE. AWQ model written to: " + out_dir)
    log("final modules_to_not_convert: " + str(final_mods))
    return 0


if __name__ == "__main__":
    sys.exit(main())
