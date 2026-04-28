# MLflow — Guia Rápido de Comandos

Referência rápida de comandos para gerenciar o stack MLflow do TDv1.

---

## Gerenciamento de Containers

### Iniciar Serviços

```bash
# Todos os serviços MLflow
cd tdvx
docker-compose up -d mlflow postgres

# Com pgAdmin (opcional)
docker-compose --profile tools up -d

# Usando script auxiliar
./start_mlflow.sh        # Linux/Mac
start_mlflow.bat         # Windows
```

### Parar Serviços

```bash
# Parar containers (dados preservados)
docker-compose stop

# Parar e remover containers (dados preservados)
docker-compose down

# CUIDADO: Remove containers E volumes (apaga tudo!)
docker-compose down -v
```

### Verificar Status

```bash
# Lista containers rodando
docker-compose ps

# Verifica saúde do MLflow
curl http://localhost:5000/health
# Esperado: {"status":"ok"}

# Logs em tempo real
docker-compose logs -f mlflow

# Logs do PostgreSQL
docker-compose logs -f postgres

# Logs de todos os serviços
docker-compose logs -f
```

### Reiniciar Serviços

```bash
# Reinicia todos
docker-compose restart

# Reinicia apenas MLflow
docker-compose restart mlflow

# Reinicia PostgreSQL (causará downtime do MLflow)
docker-compose restart postgres
```

---

## Fine-tuning

### Execução Simples

```bash
# Wrapper recomendado
python run_finetune_with_mlflow.py --sprint 1

# Com parâmetros customizados
python run_finetune_with_mlflow.py \
    --sprint 2 \
    --base-model openai/whisper-large-v3 \
    --max-steps 1000 \
    --learning-rate 1e-5
```

### Validação sem Treino

```bash
# Valida dataset (dry-run)
python run_finetune_with_mlflow.py --sprint 1 --dry-run

# Script original
cd tdvx
python finetune.py --dry-run
```

### Controle Total (Script Original)

```bash
cd tdvx
python finetune.py \
    --mlflow-uri http://localhost:5000 \
    --run-name tdv1-sprint-3 \
    --model-name tdv1-pt-en \
    --max-steps 2000 \
    --batch-size 16 \
    --convert-ct2
```

---

## Acesso às UIs

| Serviço | URL | Credenciais |
|---------|-----|-------------|
| MLflow UI | http://localhost:5000 | — |
| pgAdmin | http://localhost:5050 | Ver `.env` |
| PostgreSQL | `localhost:5432` | user: `mlflow`, pass: ver `.env` |

---

## Gerenciamento de Dados

### Backup

```bash
# Backup do PostgreSQL
docker-compose exec postgres pg_dump -U mlflow mlflow > \
    mlflow_backup_$(date +%Y%m%d).sql

# Backup de artefatos (Linux/Mac)
docker run --rm \
    -v tdvx_mlflow_artifacts:/data \
    -v $(pwd):/backup \
    alpine tar czf /backup/mlflow_artifacts_$(date +%Y%m%d).tar.gz -C /data .

# Backup de artefatos (Windows PowerShell)
docker run --rm `
    -v tdvx_mlflow_artifacts:/data `
    -v ${PWD}:/backup `
    alpine tar czf /backup/mlflow_artifacts_$(Get-Date -Format 'yyyyMMdd').tar.gz -C /data .
```

### Restauração

```bash
# Restaurar banco
cat mlflow_backup_20260428.sql | \
    docker-compose exec -T postgres psql -U mlflow mlflow

# Restaurar artefatos (Linux/Mac)
docker run --rm \
    -v tdvx_mlflow_artifacts:/data \
    -v $(pwd):/backup \
    alpine tar xzf /backup/mlflow_artifacts_20260428.tar.gz -C /data

# Restaurar artefatos (Windows PowerShell)
docker run --rm `
    -v tdvx_mlflow_artifacts:/data `
    -v ${PWD}:/backup `
    alpine tar xzf /backup/mlflow_artifacts_20260428.tar.gz -C /data
```

### Limpeza

```bash
# Remove runs antigos (via Python)
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()

# Deleta runs com WER > 20%
for exp in client.search_experiments():
    for run in client.search_runs(exp.experiment_id):
        wer = run.data.metrics.get('eval_wer', 0)
        if wer > 20.0:
            client.delete_run(run.info.run_id)
            print(f'Deleted: {run.info.run_name} (WER={wer})')
"

# Limpa volumes órfãos
docker volume prune

# Remove TUDO (CUIDADO!)
docker-compose down -v
docker volume rm tdvx_mlflow_db tdvx_mlflow_artifacts tdvx_pgadmin_data
```

---

## Model Registry

### Listar Modelos

```bash
# Via CLI
mlflow models list --tracking-uri http://localhost:5000

# Via Python
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()
for model in client.search_registered_models():
    print(f'{model.name} — latest: v{model.latest_versions[0].version}')
"
```

### Promover Versão

```bash
# Via Python
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()

# Promove versão 3 para Production
client.transition_model_version_stage(
    name='tdv1-pt-en',
    version=3,
    stage='Production',
    archive_existing_versions=True
)
print('Model promoted to Production')
"
```

### Carregar Modelo

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# Versão específica
model = mlflow.transformers.load_model("models:/tdv1-pt-en/3")

# Versão em Production
model = mlflow.transformers.load_model("models:/tdv1-pt-en/Production")

# Versão em Staging
model = mlflow.transformers.load_model("models:/tdv1-pt-en/Staging")
```

---

## Debugging

### Verificar Conexão MLflow

```bash
# Health check
curl -f http://localhost:5000/health && echo "✓ OK" || echo "✗ FALHOU"

# Lista experimentos (via API)
curl http://localhost:5000/api/2.0/mlflow/experiments/search

# Via Python
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
print(mlflow.search_experiments())
"
```

### Logs Detalhados

```bash
# Logs do MLflow com timestamp
docker-compose logs -f --tail=100 mlflow

# Logs do PostgreSQL
docker-compose logs -f --tail=100 postgres

# Inspeciona container
docker-compose exec mlflow sh
# Dentro do container:
#   ls /mlflow/artifacts
#   ps aux
```

### Testar PostgreSQL

```bash
# Conecta via psql
docker-compose exec postgres psql -U mlflow

# Dentro do psql:
# \dt                           -- lista tabelas
# SELECT * FROM experiments;    -- vê experimentos
# SELECT * FROM runs LIMIT 5;   -- vê runs
# \q                            -- sai
```

---

## Integração com TDvX

### Executar Fine-tuning no Container

```bash
# Entra no container TDvX
docker-compose exec tdvx bash

# Dentro do container
python /app/finetune.py \
    --run-name tdv1-sprint-1-docker \
    --model-name tdv1-pt-en

# MLflow URI já configurado via .env: http://mlflow:5000
```

### Mapear Modelo Fine-tunado

```python
# app/services/engine.py
import os
import mlflow

class STTEngine:
    def __init__(self):
        # Carrega do Model Registry
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        model_path = mlflow.transformers.download_artifacts(
            artifact_uri="models:/tdv1-pt-en/Production"
        )
        
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_path, device="cuda")
```

---

## Troubleshooting Rápido

| Problema | Comando |
|----------|---------|
| MLflow não inicia | `docker-compose logs mlflow` |
| Porta 5000 em uso | Edite `docker-compose.yml`: `"5001:5000"` |
| Connection refused | `curl http://localhost:5000/health` |
| PostgreSQL não conecta | `docker-compose logs postgres` |
| Volumes corrompidos | `docker-compose down -v` + `docker-compose up -d` |
| Experimento sumiu | Verifique `.env` → `MLFLOW_TRACKING_URI` |
| Modelo não registra | Adicione `--model-name` ao comando |

---

## Comandos Docker Úteis

```bash
# Lista todos os volumes
docker volume ls

# Inspeciona volume
docker volume inspect tdvx_mlflow_db

# Remove volume específico (CUIDADO!)
docker volume rm tdvx_mlflow_artifacts

# Remove containers parados
docker container prune

# Espaço usado por Docker
docker system df

# Limpeza geral (CUIDADO!)
docker system prune -a --volumes
```

---

## Variáveis de Ambiente Importantes

```env
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_FINETUNE_EXPERIMENT=tdv1-finetune
MLFLOW_DB_PASSWORD=mlflow123

# PostgreSQL
PGADMIN_EMAIL=admin@tdvx.local
PGADMIN_PASSWORD=admin123

# HuggingFace (para push de modelos)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
PYANNOTE_AUTH_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Referências Rápidas

- **UI do MLflow**: http://localhost:5000
- **Documentação MLflow**: https://mlflow.org/docs/latest/
- **Docker Compose Docs**: https://docs.docker.com/compose/
- **PostgreSQL Docs**: https://www.postgresql.org/docs/

---

**Última atualização**: 2026-04-28
