@echo off
REM setup_finetune_venv.bat — Instala dependências do fine-tuning no venv isolado
REM GPU: RTX 5060 Ti (Blackwell) | CUDA 13.2 driver | PyTorch cu128
REM ─────────────────────────────────────────────────────────────────────────────

setlocal

set VENV=.venv-finetune
set PYTHON=%VENV%\Scripts\python.exe
set PIP=%VENV%\Scripts\pip.exe

echo.
echo ════════════════════════════════════════════════════════════════
echo  TDv1 Fine-tuning — Setup do ambiente virtual
echo  GPU: RTX 5060 Ti ^| CUDA 13.2 ^| PyTorch cu128
echo ════════════════════════════════════════════════════════════════
echo.

REM ── Verifica que o venv existe ────────────────────────────────────────────────
if not exist "%PYTHON%" (
    echo [ERRO] Venv nao encontrado. Execute primeiro:
    echo   python -m venv .venv-finetune
    exit /b 1
)

echo [1/4] Atualizando pip e wheel no venv...
%PYTHON% -m pip install --upgrade pip wheel setuptools --quiet

echo.
echo [2/4] Instalando PyTorch 2.7 com CUDA 12.8 (RTX 5060 Ti / Blackwell)...
echo       (pode demorar — download de ~2.5 GB)
echo.
%PIP% install torch torchvision torchaudio ^
    --index-url https://download.pytorch.org/whl/cu128 ^
    --upgrade

if errorlevel 1 (
    echo.
    echo [AVISO] Falha com cu128. Tentando cu126...
    %PIP% install torch torchvision torchaudio ^
        --index-url https://download.pytorch.org/whl/cu126 ^
        --upgrade
)

echo.
echo [3/4] Instalando dependências de fine-tuning...
echo.
%PIP% install ^
    transformers>=4.40.0 ^
    datasets>=2.14.0 ^
    accelerate>=0.25.0 ^
    evaluate>=0.4.0 ^
    jiwer>=3.0.0 ^
    ctranslate2>=4.0.0 ^
    peft>=0.10.0 ^
    mlflow>=2.10.0 ^
    tensorboard>=2.14.0 ^
    python-dotenv ^
    numpy ^
    soundfile ^
    librosa

echo.
echo [4/4] Verificando instalação...
%PYTHON% -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponivel:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
%PYTHON% -c "import transformers; print('Transformers:', transformers.__version__)"
%PYTHON% -c "import datasets; print('Datasets:', datasets.__version__)"
%PYTHON% -c "import ctranslate2; print('CTranslate2:', ctranslate2.__version__)"
%PYTHON% -c "import peft; print('PEFT:', peft.__version__)"

echo.
echo ════════════════════════════════════════════════════════════════
echo  Setup concluido! Para usar o ambiente:
echo.
echo    .venv-finetune\Scripts\activate
echo.
echo  Depois rode o dry-run para testar:
echo    python finetune_commonvoice.py --dry-run --max-train-samples 1000
echo ════════════════════════════════════════════════════════════════
echo.
endlocal
