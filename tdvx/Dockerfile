# TDvX v3 — Transcrição + Diarização + Finetuning Dataset
# Base: NVIDIA CUDA 12.8 + cuDNN 9 + Ubuntu 22.04
FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYANNOTE_AUDIO_TELEMETRY=0

# Dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    ffmpeg \
    git \
    curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CUDA 12.8 — instala antes do requirements.txt
RUN pip3 install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Restante das dependências
COPY ../requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Código da aplicação (apenas tdvx/)
COPY . .

# Diretórios de dados
RUN mkdir -p finetuning_data/audio finetuning_data/labels

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# PYANNOTE_AUTH_TOKEN deve ser passado em runtime: --env PYANNOTE_AUTH_TOKEN=hf_xxx
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
