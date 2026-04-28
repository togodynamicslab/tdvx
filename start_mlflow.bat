@echo off
REM ════════════════════════════════════════════════════════════════════════
REM start_mlflow.bat — Inicia serviços MLflow + PostgreSQL (Windows)
REM ════════════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "COMPOSE_FILE=%SCRIPT_DIR%tdvx\docker-compose.yml"

echo ════════════════════════════════════════════════════════════════
echo TDvX — Iniciando MLflow Tracking Server
echo ════════════════════════════════════════════════════════════════

REM Verifica se .env existe
if not exist "%SCRIPT_DIR%tdvx\.env" (
    echo ⚠️  Arquivo .env não encontrado!
    echo.
    echo Copie o arquivo de exemplo:
    echo   copy tdvx\.env.example tdvx\.env
    echo.
    echo E configure as variáveis necessárias ^(PYANNOTE_AUTH_TOKEN, etc.^)
    exit /b 1
)

REM Inicia serviços
echo ▶️  Iniciando containers MLflow + PostgreSQL ...
docker-compose -f "%COMPOSE_FILE%" up -d mlflow postgres

if errorlevel 1 (
    echo ❌ Falha ao iniciar containers.
    echo Verifique se Docker Desktop está rodando.
    exit /b 1
)

REM Aguarda inicialização
echo.
echo ⏳ Aguardando inicialização do MLflow ^(10s^) ...
timeout /t 10 /nobreak >nul

REM Verifica se está rodando
echo.
echo 🔍 Verificando status dos containers ...
docker-compose -f "%COMPOSE_FILE%" ps mlflow postgres

REM Testa conexão
echo.
echo 🔍 Testando conexão com MLflow API ...
curl -f -s http://localhost:5000/health >nul 2>&1
if errorlevel 1 (
    echo ❌ MLflow não está respondendo.
    echo.
    echo Verifique os logs:
    echo   docker-compose -f "%COMPOSE_FILE%" logs mlflow
    exit /b 1
)

echo ✅ MLflow está acessível!

echo.
echo ════════════════════════════════════════════════════════════════
echo ✅ MLflow Tracking Server iniciado com sucesso!
echo ════════════════════════════════════════════════════════════════
echo.
echo 📊 Acesse a UI: http://localhost:5000
echo 🗄️  PostgreSQL: localhost:5432 ^(user: mlflow^)
echo.
echo Para ver logs em tempo real:
echo   docker-compose -f "%COMPOSE_FILE%" logs -f mlflow
echo.
echo Para parar os serviços:
echo   docker-compose -f "%COMPOSE_FILE%" down
echo.
echo Próximo passo: executar fine-tuning
echo   python run_finetune_with_mlflow.py --sprint 1
echo.

endlocal
