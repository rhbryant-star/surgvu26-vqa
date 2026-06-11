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
