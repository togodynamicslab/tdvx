@echo off
setlocal EnableDelayedExpansion
title STT Pipeline - Setup CPU

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║    STT Pipeline — Ambiente CPU (sem GPU)         ║
echo  ║    Quantizacao: int8  (~4x menos RAM)            ║
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

:: Info de CPU e RAM
for /f "skip=1 tokens=*" %%p in ('wmic cpu get Name 2^>nul') do (
    if not "%%p"=="" (
        echo [OK] CPU: %%p
        goto :cpu_done
    )
)
:cpu_done

echo.

:: ── Criar venv ───────────────────────────────────────────────────────────────

echo [1/5] Criando ambiente virtual: venv_cpu
if exist venv_cpu (
    echo       venv_cpu ja existe — reutilizando
) else (
    python -m venv venv_cpu
    if errorlevel 1 (
        echo [ERRO] Falha ao criar venv_cpu
        pause & exit /b 1
    )
)

:: ── Ativar venv ──────────────────────────────────────────────────────────────

echo [2/5] Ativando venv_cpu
call venv_cpu\Scripts\activate.bat

:: ── pip ──────────────────────────────────────────────────────────────────────

echo [3/5] Atualizando pip
python -m pip install --upgrade pip setuptools wheel --quiet

:: ── PyTorch CPU-only ─────────────────────────────────────────────────────────

echo [4/5] Instalando PyTorch CPU-only (download ~200 MB — muito mais rapido)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 (
    echo [ERRO] Falha ao instalar PyTorch CPU
    pause & exit /b 1
)
echo       PyTorch CPU instalado com sucesso

:: ── Demais dependências ──────────────────────────────────────────────────────

echo [5/5] Instalando dependencias do pipeline
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERRO] Falha ao instalar requirements.txt
    pause & exit /b 1
)

:: ── Verificação final ────────────────────────────────────────────────────────

echo.
echo  Verificando instalacao...
python -c "import torch; print('  PyTorch :', torch.__version__); print('  Threads :', torch.get_num_threads())" 2>nul
python -c "from faster_whisper import WhisperModel; print('  faster-whisper: OK')" 2>nul
python -c "from pyannote.audio import Pipeline; print('  pyannote.audio: OK')" 2>nul

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║   Setup concluido com sucesso!                   ║
echo  ║                                                  ║
echo  ║   Para usar:                                     ║
echo  ║     venv_cpu\Scripts\activate.bat                ║
echo  ║     python voice_model\stt_cpu.py audio.wav      ║
echo  ║                                                  ║
echo  ║   Configure HF_TOKEN no .env da raiz             ║
echo  ╚══════════════════════════════════════════════════╝
echo.
pause
