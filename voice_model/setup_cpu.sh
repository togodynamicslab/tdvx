#!/usr/bin/env bash
# setup_cpu.sh — STT Pipeline: ambiente CPU (Linux)
set -e

echo ""
echo " ╔══════════════════════════════════════════════════╗"
echo " ║    STT Pipeline — Ambiente CPU (sem GPU)         ║"
echo " ║    Quantizacao: int8  (~4x menos RAM)            ║"
echo " ╚══════════════════════════════════════════════════╝"
echo ""

# ── Verificações ──────────────────────────────────────────────────────────────

if ! command -v python3 &>/dev/null; then
    echo "[ERRO] python3 nao encontrado. Instale Python 3.10+"
    exit 1
fi
echo "[OK] $(python3 --version)"
echo "[OK] CPUs: $(nproc) nucleos"
echo ""

# ── Criar venv ────────────────────────────────────────────────────────────────

echo "[1/5] Criando ambiente virtual: venv_cpu"
if [ -d "venv_cpu" ]; then
    echo "      venv_cpu ja existe — reutilizando"
else
    python3 -m venv venv_cpu
fi

# ── Ativar venv ───────────────────────────────────────────────────────────────

echo "[2/5] Ativando venv_cpu"
source venv_cpu/bin/activate

# ── pip ───────────────────────────────────────────────────────────────────────

echo "[3/5] Atualizando pip"
pip install --upgrade pip setuptools wheel -q

# ── PyTorch CPU-only ──────────────────────────────────────────────────────────

echo "[4/5] Instalando PyTorch CPU-only (~200 MB — rapido)"
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q

# ── Demais dependências ───────────────────────────────────────────────────────

echo "[5/5] Instalando dependencias do pipeline"
pip install -r requirements_cpu.txt -q

# ── Verificação final ─────────────────────────────────────────────────────────

echo ""
echo " Verificando instalacao..."
python -c "import torch; print('  PyTorch :', torch.__version__); print('  Threads :', torch.get_num_threads())" 2>/dev/null
python -c "from faster_whisper import WhisperModel; print('  faster-whisper: OK')" 2>/dev/null
python -c "from pyannote.audio import Pipeline; print('  pyannote.audio: OK')" 2>/dev/null

echo ""
echo " ╔══════════════════════════════════════════════════╗"
echo " ║   Setup concluido!                               ║"
echo " ║                                                  ║"
echo " ║   Para usar:                                     ║"
echo " ║     source venv_cpu/bin/activate                 ║"
echo " ║     python stt_cpu.py seu_audio.wav              ║"
echo " ╚══════════════════════════════════════════════════╝"
echo ""
