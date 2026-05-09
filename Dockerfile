# TDvX — transcription + diarization + translation
# Target: NVIDIA Blackwell (RTX 5060 Ti, sm_120) on Vast.ai via --ssh --direct
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYANNOTE_AUDIO_TELEMETRY=0 \
    HF_HOME=/models \
    HF_HUB_CACHE=/models/hub \
    TORCH_HOME=/models/torch \
    TORCH_CUDA_ARCH_LIST="12.0+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        ffmpeg git curl tini \
        openssh-server \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && mkdir -p /var/run/sshd /models/hub /models/torch \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch with cu128 wheels first so requirements.txt picks up the CUDA build.
# Pin setuptools<81 so pkg_resources stays importable (openai-whisper's sdist needs it).
RUN pip3 install --upgrade pip \
    && pip3 install "setuptools<81" wheel \
    && pip3 install --index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.7.*" "torchaudio==2.7.*"

COPY requirements.txt .
# --no-build-isolation: openai-whisper is sdist-only and its setup.py imports pkg_resources,
# which modern setuptools (>=81) removed. Reuse the globally-installed setuptools<81.
RUN pip3 install --no-build-isolation -r requirements.txt

COPY . .

RUN mkdir -p temp uploads

ENV HOST=0.0.0.0 \
    PORT=8000 \
    DEFAULT_MODEL=tdv1-fast \
    ENABLE_DIARIZATION=true

EXPOSE 8000 22

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# tini reaps zombies; sshd is launched by Vast's --ssh wrapper, not by CMD.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
