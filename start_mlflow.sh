#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# start_mlflow.sh — Inicia serviços MLflow + PostgreSQL
# ═══════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/tdvx/docker-compose.yml"

echo "════════════════════════════════════════════════════════════════"
echo "TDvX — Iniciando MLflow Tracking Server"
echo "════════════════════════════════════════════════════════════════"

# Verifica se .env existe
if [ ! -f "${SCRIPT_DIR}/tdvx/.env" ]; then
    echo "⚠️  Arquivo .env não encontrado!"
    echo ""
    echo "Copie o arquivo de exemplo:"
    echo "  cp tdvx/.env.example tdvx/.env"
    echo ""
    echo "E configure as variáveis necessárias (PYANNOTE_AUTH_TOKEN, etc.)"
    exit 1
fi

# Inicia serviços
echo "▶️  Iniciando containers MLflow + PostgreSQL ..."
docker-compose -f "$COMPOSE_FILE" up -d mlflow postgres

# Aguarda inicialização
echo ""
echo "⏳ Aguardando inicialização do MLflow (10s) ..."
sleep 10

# Verifica se está rodando
echo ""
echo "🔍 Verificando status dos containers ..."
docker-compose -f "$COMPOSE_FILE" ps mlflow postgres

# Testa conexão
echo ""
echo "🔍 Testando conexão com MLflow API ..."
if curl -f -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ MLflow está acessível!"
else
    echo "❌ MLflow não está respondendo."
    echo ""
    echo "Verifique os logs:"
    echo "  docker-compose -f $COMPOSE_FILE logs mlflow"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ MLflow Tracking Server iniciado com sucesso!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Acesse a UI: http://localhost:5000"
echo "🗄️  PostgreSQL: localhost:5432 (user: mlflow)"
echo ""
echo "Para ver logs em tempo real:"
echo "  docker-compose -f $COMPOSE_FILE logs -f mlflow"
echo ""
echo "Para parar os serviços:"
echo "  docker-compose -f $COMPOSE_FILE down"
echo ""
echo "Próximo passo: executar fine-tuning"
echo "  python run_finetune_with_mlflow.py --sprint 1"
echo ""
