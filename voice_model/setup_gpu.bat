@echo off
setlocal EnableDelayedExpansion
title STT Pipeline - Setup GPU

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║    STT Pipeline — Ambiente GPU  (CUDA 12.8)      ║
echo  ║    Quantizacao: int8_float16                     ║
echo  ╚══════════════════════════════════════════════════╝
echo.

:: ── Verificações iniciais ────────────────────────────────────────────────────

where python >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado no PATH.
    echo        Instale Python 3.10+ em https://python.org
    pause & exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Python %PY_VER% encontrado

:: Verifica se NVIDIA driver existe
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [AVISO] nvidia-smi nao encontrado. Verifique se os drivers NVIDIA estao instalados.
    echo         Continuando mesmo assim...
) else (
    echo [OK] Driver NVIDIA detectado
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>nul
)

echo.

:: ── Criar venv ───────────────────────────────────────────────────────────────

echo [1/5] Criando ambiente virtual: venv_gpu
if exist venv_gpu (
    echo       venv_gpu ja existe — reutilizando
) else (
    python -m venv venv_gpu
    if errorlevel 1 (
        echo [ERRO] Falha ao criar venv_gpu
        pause & exit /b 1
    )
)

:: ── Ativar venv ──────────────────────────────────────────────────────────────

echo [2/5] Ativando venv_gpu
call venv_gpu\Scripts\activate.bat

:: ── pip ──────────────────────────────────────────────────────────────────────

echo [3/5] Atualizando pip
python -m pip install --upgrade pip setuptools wheel --quiet

:: ── PyTorch CUDA ─────────────────────────────────────────────────────────────

echo [4/5] Instalando PyTorch 2.x com CUDA 12.8
echo       (download ~2.5 GB — aguarde...)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet
if errorlevel 1 (
    echo [ERRO] Falha ao instalar PyTorch+CUDA
    echo        Verifique sua conexao e tente novamente
    pause & exit /b 1
)
echo       PyTorch instalado com sucesso

:: ── Demais dependências ──────────────────────────────────────────────────────

echo [5/5] Instalando dependencias do pipeline
pip install -r requirements_gpu.txt --quiet
if errorlevel 1 (
    echo [ERRO] Falha ao instalar requirements_gpu.txt
    pause & exit /b 1
)

:: ── Verificação final ────────────────────────────────────────────────────────

echo.
echo  Verificando instalacao...
python -c "import torch; print('  PyTorch :', torch.__version__); print('  CUDA OK :',torch.cuda.is_available())" 2>nul
python -c "from faster_whisper import WhisperModel; print('  faster-whisper: OK')" 2>nul
python -c "from pyannote.audio import Pipeline; print('  pyannote.audio: OK')" 2>nul

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║   Setup concluido com sucesso!                   ║
echo  ║                                                  ║
echo  ║   Para usar:                                     ║
echo  ║     venv_gpu\Scripts\activate.bat                ║
echo  ║     python stt_gpu.py seu_audio.wav              ║
echo  ║                                                  ║
echo  ║   O token HF ja esta configurado em .env         ║
echo  ╚══════════════════════════════════════════════════╝
echo.
pause
