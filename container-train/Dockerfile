# Dedicated, version-pinned training container for SurgVU M2 LoRA fine-tuning.
# Pins mirror the PROVEN phase-detector recipe (LLaMA-Factory 0.9.3 +
# transformers 4.50.3 stack). Built in CI as root; CHTC runs it via apptainer.
# NOTE: deliberately NOT reusing opscribe.sif — its venv drifted to
# torch 2.10 / transformers 5.1, which breaks LLaMA-Factory 0.9.3.
FROM --platform=linux/amd64 pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/staging/groups/bhaskar_opscribe/hf_cache \
    TOKENIZERS_PARALLELISM=false \
    PIP_NO_CACHE_DIR=1

# llamafactory itself with --no-deps (torch is already in the base image), then
# the exact pinned dependency set the phase-detector job is known to work with.
RUN pip install --no-deps "llamafactory[torch,metrics]==0.9.3" && \
    pip install \
      "transformers==4.50.3" "tokenizers==0.21.0" "huggingface-hub==0.30.2" \
      "datasets==3.5.0" "trl==0.9.6" "peft==0.15.2" "accelerate==1.7.0" \
      evaluate multiprocess dill pytz python-dateutil six fsspec requests packaging \
      omegaconf "antlr4-python3-runtime==4.9.3" fire einops tyro nltk jieba rouge-chinese \
      attrs aiosignal aiohappyeyeballs async-timeout idna py-cpuinfo setuptools docstring-parser \
      sentencepiece "opencv-python-headless" pillow "numpy<2.3" pandas

# Sanity: the launcher + Qwen2.5-VL support must import at build time.
RUN python -c "import llamafactory, transformers, peft, trl, accelerate; \
print('llamafactory', llamafactory.__version__, '| tf', transformers.__version__, '| peft', peft.__version__)" && \
    python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('qwen2.5-vl OK')"

ENTRYPOINT ["bash"]
