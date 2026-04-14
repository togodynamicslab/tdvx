#!/usr/bin/env bash
# setup_gpu.sh — STT Pipeline: ambiente GPU (Linux)
set -e

echo ""
echo " ╔══════════════════════════════════════════════════╗"
echo " ║    STT Pipeline — Ambiente GPU  (CUDA 12.8)      ║"
echo " ║    Quantizacao: int8_float16                     ║"
echo " ╚══════════════════════════════════════════════════╝"
echo ""

# ── Verificações ──────────────────────────────────────────────────────────────

if ! command -v python3 &>/dev/null; then
    echo "[ERRO] python3 nao encontrado. Instale Python 3.10+"
    exit 1
fi
echo "[OK] $(python3 --version)"

if command -v nvidia-smi &>/dev/null; then
    echo "[OK] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "[OK] VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
else
    echo "[AVISO] nvidia-smi nao encontrado — verifique drivers NVIDIA"
fi
echo ""

# ── Criar venv ────────────────────────────────────────────────────────────────

echo "[1/5] Criando ambiente virtual: venv_gpu"
if [ -d "venv_gpu" ]; then
    echo "      venv_gpu ja existe — reutilizando"
else
    python3 -m venv venv_gpu
fi

# ── Ativar venv ───────────────────────────────────────────────────────────────

echo "[2/5] Ativando venv_gpu"
source venv_gpu/bin/activate

# ── pip ───────────────────────────────────────────────────────────────────────

echo "[3/5] Atualizando pip"
pip install --upgrade pip setuptools wheel -q

# ── PyTorch CUDA ──────────────────────────────────────────────────────────────

echo "[4/5] Instalando PyTorch 2.x com CUDA 12.8 (~2.5 GB — aguarde...)"
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 -q

# ── Demais dependências ───────────────────────────────────────────────────────

echo "[5/5] Instalando dependencias do pipeline"
pip install -r requirements.txt -q

# ── Verificação final ─────────────────────────────────────────────────────────

echo ""
echo " Verificando instalacao..."
python -c "import torch; print('  PyTorch :', torch.__version__); print('  CUDA OK :', torch.cuda.is_available())" 2>/dev/null
python -c "from faster_whisper import WhisperModel; print('  faster-whisper: OK')" 2>/dev/null
python -c "from pyannote.audio import Pipeline; print('  pyannote.audio: OK')" 2>/dev/null

echo ""
echo " ╔══════════════════════════════════════════════════╗"
echo " ║   Setup concluido!                               ║"
echo " ║                                                  ║"
echo " ║   Para usar:                                     ║"
echo " ║     source venv_gpu/bin/activate                 ║"
echo " ║     python voice_model/stt_gpu.py audio.wav      ║"
echo " ╚══════════════════════════════════════════════════╝"
echo ""
