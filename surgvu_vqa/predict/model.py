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
# Eval-only knobs (unset in the deployed AWQ container → behavior unchanged).
# SURGVU_ADAPTER_DIR: load a LoRA adapter on the bf16 base — the plan's
# "bridge path A", which measures the fine-tune WITHOUT quantization.
# SURGVU_DTYPE: "bfloat16" loads the non-AWQ bf16 base; default float16 (AWQ).
ADAPTER_DIR_ENV = "SURGVU_ADAPTER_DIR"
DTYPE_ENV = "SURGVU_DTYPE"

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
        dtype = (
            torch.bfloat16
            if os.environ.get(DTYPE_ENV, "").lower() in ("bf16", "bfloat16")
            else torch.float16
        )
        self._processor = AutoProcessor.from_pretrained(
            path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )
        adapter = os.environ.get(ADAPTER_DIR_ENV, "")
        if adapter:
            # Eval/bridge path: fine-tuned LoRA on the bf16 base. The AWQ
            # lm_head guard below applies only to the pre-quantized AWQ
            # checkpoint, so it is skipped here.
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        else:
            # The official AWQ checkpoint stores lm_head.weight in fp16 (NOT
            # quantized) but its quantization_config forgets to exclude lm_head,
            # so transformers wraps it as a quantized linear and the AWQ kernel
            # dies with "expected scalar type Int but found Half". Passing an
            # AwqConfig override here is IGNORED for pre-quantized checkpoints,
            # so the fix lives in the checkpoint itself: hpc/fetch_weights.sh
            # patches modules_to_not_convert to ["visual", "lm_head"] before
            # tarballing. This guard catches an unpatched checkpoint at load
            # time, not mid-answer.
            head_cls = type(model.lm_head).__name__
            if "WQLinear" in head_cls:
                raise RuntimeError(
                    f"lm_head was AWQ-wrapped ({head_cls}): the checkpoint's "
                    'quantization_config.modules_to_not_convert is missing '
                    '"lm_head" — re-run hpc/fetch_weights.sh to patch and '
                    "re-tar the weights."
                )
        self._model = model
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
