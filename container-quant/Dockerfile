# Dedicated quantization container for SurgVU M2 (Task 0b).
# Merges the LoRA adapter into the bf16 base, then AutoAWQ self-quantizes the
# result to the same 4-bit artifact shape the inference container already loads.
# Stack = the PROVEN M1 inference combo (torch 2.5.1 + transformers 4.51.3 +
# autoawq 0.2.9 + autoawq-kernels 0.0.9, which run the AWQ kernels in prod) plus
# peft 0.15.2 for the merge. transformers >=4.52 is FORBIDDEN (VLM module
# rename breaks both adapter keys and the AWQ tooling).
FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/staging/groups/bhaskar_opscribe/hf_cache \
    TOKENIZERS_PARALLELISM=false \
    PIP_NO_CACHE_DIR=1

RUN pip install \
      "transformers==4.51.3" "accelerate==1.6.0" \
      "autoawq==0.2.9" "autoawq-kernels==0.0.9" \
      "peft==0.15.2" \
      "opencv-python-headless==4.11.0.86" "pillow>=10.0" "numpy<2.3"

# Sanity: the AWQ quantizer entrypoint + Qwen2.5-VL class + peft merge stack
# must all import at build time (fail CI here, not on a GPU node).
RUN python -c "import awq, transformers, peft; \
print('awq', awq.__version__, '| tf', transformers.__version__, '| peft', peft.__version__)" && \
    python -c "from awq import AutoAWQForCausalLM; print('AutoAWQForCausalLM OK')" && \
    python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('qwen2.5-vl OK')"

ENTRYPOINT ["bash"]
