#!/usr/bin/env bash
# setup_finetune.sh — Ambiente de fine-tuning TDv1 (Linux, GPU ou CPU)
set -e

echo ""
echo " ╔══════════════════════════════════════════════════════════╗"
echo " ║   TDv1 Fine-tuning — Ambiente Linux                     ║"
echo " ║   whisper-medium + Common Voice pt-BR                   ║"
echo " ╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Verificações ──────────────────────────────────────────────────────────────

if ! command -v python3 &>/dev/null; then
    echo "[ERRO] python3 nao encontrado. Instale Python 3.10+"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR" -lt 10 ]; then
    echo "[ERRO] Python 3.10+ necessario (encontrado: $PY_VER)"
    exit 1
fi
echo "[OK] Python $PY_VER"

# Detecta GPU
HAS_GPU=0
if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "[OK] GPU: $GPU_NAME (${GPU_VRAM} MB VRAM)"
    HAS_GPU=1
else
    echo "[INFO] Nenhuma GPU NVIDIA detectada — instalando PyTorch CPU"
    echo "       Fine-tuning em CPU e muito lento; considere usar GPU."
fi
echo "[OK] CPUs: $(nproc) nucleos"
echo ""

# ── Criar venv ────────────────────────────────────────────────────────────────

VENV_DIR="venv_finetune"
echo "[1/5] Criando ambiente virtual: $VENV_DIR"
if [ -d "$VENV_DIR" ]; then
    echo "      $VENV_DIR ja existe — reutilizando"
else
    python3 -m venv "$VENV_DIR"
fi

# ── Ativar venv ───────────────────────────────────────────────────────────────

echo "[2/5] Ativando $VENV_DIR"
source "$VENV_DIR/bin/activate"

# ── pip ───────────────────────────────────────────────────────────────────────

echo "[3/5] Atualizando pip"
pip install --upgrade pip setuptools wheel -q

# ── PyTorch ───────────────────────────────────────────────────────────────────

if [ "$HAS_GPU" -eq 1 ]; then
    echo "[4/5] Instalando PyTorch 2.x com CUDA 13.0 (~2.5 GB — aguarde...)"
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130 -q
else
    echo "[4/5] Instalando PyTorch 2.x CPU-only (~200 MB)"
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q
fi

# ── Dependências ──────────────────────────────────────────────────────────────

echo "[5/5] Instalando dependencias (requirements + finetune)"
pip install -r requirements.txt -q
pip install -r requirements-finetune.txt -q

# ── Verificação final ─────────────────────────────────────────────────────────

echo ""
echo " Verificando instalacao..."
python -c "
import torch
print(f'  PyTorch     : {torch.__version__}')
print(f'  CUDA        : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU         : {torch.cuda.get_device_name(0)}')
" 2>/dev/null

python -c "from transformers import WhisperForConditionalGeneration; print('  transformers: OK')" 2>/dev/null
python -c "import evaluate; print('  evaluate    : OK')" 2>/dev/null
python -c "import ctranslate2; print('  ctranslate2 : OK')" 2>/dev/null
python -c "import mlflow; print('  mlflow      : OK')" 2>/dev/null
python -c "import librosa; print('  librosa     : OK')" 2>/dev/null

echo ""
echo " ╔══════════════════════════════════════════════════════════╗"
echo " ║   Setup concluido!                                      ║"
echo " ║                                                          ║"
echo " ║   Ative o ambiente:                                      ║"
echo " ║     source venv_finetune/bin/activate                   ║"
echo " ║                                                          ║"
echo " ║   Teste rapido (valida dados sem treinar):               ║"
echo " ║     python finetune_commonvoice.py --dry-run             ║"
echo " ║                                                          ║"
echo " ║   Treino completo pt-BR:                                 ║"
echo " ║     python finetune_commonvoice.py \\                    ║"
echo " ║       --language pt \\                                   ║"
echo " ║       --run-name tdv1-cv-pt-v1 \\                       ║"
echo " ║       --num-epochs 3                                     ║"
echo " ║                                                          ║"
echo " ║   Modo LoRA (menos VRAM):                                ║"
echo " ║     python finetune_commonvoice.py --use-lora            ║"
echo " ╚══════════════════════════════════════════════════════════╝"
echo ""
