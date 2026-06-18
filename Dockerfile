# syntax=docker/dockerfile:1.7

# Default to a Docker Hub base image so Docker Hub automated builds can pull it
# without NGC registry configuration. For an on-prem NVIDIA Enterprise build,
# this can be overridden with nvcr.io/nvidia/pytorch:25.03-py3 or newer.
ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}

ARG HABIBI_REVISION=main

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/.cache \
    XDG_CACHE_HOME=/models/.cache \
    HABIBI_MODEL_DIR=/models/Habibi-TTS \
    HABIBI_DEFAULT_MODEL=Unified \
    HABIBI_DEFAULT_DIALECT=SAU \
    HABIBI_DEFAULT_REF_AUDIO=assets/Najdi.wav \
    HABIBI_DEFAULT_REF_TEXT="تكفى طمني انا اليوم ماني بنايم ولا هو بداخل عيني النوم الين اتطمن عليه." \
    HABIBI_OUTPUT_DIR=/tmp/habibi-tts \
    PORT=8000 \
    HABIBI_REVISION=${HABIBI_REVISION} \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN python3 -m pip install torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN python3 -m pip install \
    "habibi-tts==0.1.1" \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    soundfile \
    huggingface_hub \
    cached-path

WORKDIR /app

# Keep the model weights in the image so the runtime pod does not need to
# download them. This image should be pushed to a private registry unless the
# model license and customer policy explicitly allow public redistribution.
RUN python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

revision = os.environ.get("HABIBI_REVISION", "main")
model_dir = os.environ.get("HABIBI_MODEL_DIR", "/models/Habibi-TTS")

snapshot_download(
    repo_id="SWivid/Habibi-TTS",
    revision=revision,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    allow_patterns=[
        "README.md",
        "Unified/model_200000.safetensors",
        "Unified/vocab.txt",
        "Specialized/SAU/model_200000.safetensors",
        "Specialized/SAU/vocab.txt",
    ],
)

print("Habibi model files cached under", model_dir)
PY

# Pre-cache the default vocoder used by F5-TTS/Habibi so the service can start
# even when the runtime network is restricted.
RUN python3 - <<'PY'
from f5_tts.infer.utils_infer import load_vocoder
load_vocoder(vocoder_name="vocos", is_local=False, device="cpu")
print("Vocoder cached")
PY

COPY server.py /app/server.py
COPY smoke_test.sh /app/smoke_test.sh
COPY NOTICE.md /app/NOTICE.md

RUN chmod +x /app/smoke_test.sh && \
    habibi-tts_infer-cli --help >/tmp/habibi_cli_help.txt && \
    python3 - <<'PY'
import importlib.metadata
import torch
import torchaudio
print("habibi-tts", importlib.metadata.version("habibi-tts"))
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("cuda build", torch.version.cuda)
PY

EXPOSE 8000

CMD ["python3", "/app/server.py"]
