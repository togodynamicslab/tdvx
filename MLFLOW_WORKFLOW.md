# Fluxo de Trabalho — Fine-tuning com MLflow

Este documento descreve o fluxo completo de trabalho para fine-tuning do TDv1 com rastreamento MLflow.

---

## Diagrama de Sequência

```
┌─────────┐         ┌──────────────┐         ┌─────────┐         ┌────────────┐
│  User   │         │  Wrapper     │         │ MLflow  │         │ PostgreSQL │
│         │         │  Script      │         │ Server  │         │            │
└────┬────┘         └──────┬───────┘         └────┬────┘         └─────┬──────┘
     │                     │                      │                    │
     │ run_finetune.py     │                      │                    │
     │ --sprint 4          │                      │                    │
     │────────────────────>│                      │                    │
     │                     │                      │                    │
     │                     │ Health check         │                    │
     │                     │─────────────────────>│                    │
     │                     │                      │                    │
     │                     │<─────────────────────│                    │
     │                     │ 200 OK               │                    │
     │                     │                      │                    │
     │                     │ Create experiment    │                    │
     │                     │─────────────────────>│                    │
     │                     │                      │                    │
     │                     │                      │ INSERT experiment  │
     │                     │                      │───────────────────>│
     │                     │                      │                    │
     │                     │                      │<───────────────────│
     │                     │                      │ experiment_id      │
     │                     │<─────────────────────│                    │
     │                     │ experiment_id        │                    │
     │                     │                      │                    │
     │                     │ Start run            │                    │
     │                     │─────────────────────>│                    │
     │                     │                      │                    │
     │                     │                      │ INSERT run         │
     │                     │                      │───────────────────>│
     │                     │                      │                    │
     │                     │                      │<───────────────────│
     │                     │<─────────────────────│ run_id             │
     │                     │ run_id               │                    │
     │                     │                      │                    │
     ├─────────────────────┼──────────────────────┼────────────────────┼─────────┐
     │                     │  FINE-TUNING LOOP    │                    │         │
     │                     │                      │                    │         │
     │                     │ Log params           │                    │         │
     │                     │─────────────────────>│                    │         │
     │                     │                      │ INSERT params      │         │
     │                     │                      │───────────────────>│         │
     │                     │                      │                    │         │
     │                     │ [Training step 1]    │                    │         │
     │                     │                      │                    │         │
     │                     │ Log metric           │                    │         │
     │                     │ (train/loss=0.234)   │                    │         │
     │                     │─────────────────────>│                    │         │
     │                     │                      │ INSERT metric      │         │
     │                     │                      │───────────────────>│         │
     │                     │                      │                    │         │
     │                     │ [Training step 2]    │                    │         │
     │                     │ ...                  │                    │         │
     │                     │                      │                    │         │
     │                     │ [Eval step 200]      │                    │         │
     │                     │                      │                    │         │
     │                     │ Log metrics          │                    │         │
     │                     │ (eval/wer=12.3%)     │                    │         │
     │                     │─────────────────────>│                    │         │
     │                     │                      │ INSERT metrics     │         │
     │                     │                      │───────────────────>│         │
     │                     │                      │                    │         │
     │                     │ [More training...]   │                    │         │
     ├─────────────────────┼──────────────────────┼────────────────────┼─────────┘
     │                     │                      │                    │
     │                     │ Log artifacts        │                    │
     │                     │ (eval_results.json)  │                    │
     │                     │─────────────────────>│                    │
     │                     │                      │                    │
     │                     │                      │ Save to volume     │
     │                     │                      │ /mlflow/artifacts  │
     │                     │                      │                    │
     │                     │<─────────────────────│                    │
     │                     │                      │                    │
     │                     │ Register model       │                    │
     │                     │ (tdv1-pt-en)         │                    │
     │                     │─────────────────────>│                    │
     │                     │                      │                    │
     │                     │                      │ INSERT model       │
     │                     │                      │ INSERT version     │
     │                     │                      │───────────────────>│
     │                     │                      │                    │
     │                     │                      │<───────────────────│
     │                     │<─────────────────────│                    │
     │                     │ model_version=3      │                    │
     │                     │                      │                    │
     │                     │ End run              │                    │
     │                     │─────────────────────>│                    │
     │                     │                      │                    │
     │                     │                      │ UPDATE run         │
     │                     │                      │ (status=FINISHED)  │
     │                     │                      │───────────────────>│
     │                     │                      │                    │
     │<────────────────────│                      │                    │
     │ ✓ Fine-tuning done  │                      │                    │
     │ Run ID: abc123      │                      │                    │
     │ Model: tdv1-pt-en v3│                      │                    │
     │                     │                      │                    │
┌────┴────┐         ┌──────┴───────┐         ┌────┴────┐         ┌─────┴──────┐
│  User   │         │  Wrapper     │         │ MLflow  │         │ PostgreSQL │
│         │         │  Script      │         │ Server  │         │            │
└─────────┘         └──────────────┘         └─────────┘         └────────────┘
```

---

## Fluxo Detalhado por Etapa

### 1. Preparação

```bash
# Usuário inicia MLflow stack
./start_mlflow.sh

# Docker Compose sobe 3 containers:
# - postgres (database)
# - mlflow (tracking server)
# - pgadmin (opcional, profile tools)
```

**Estados dos serviços**:
- ✅ PostgreSQL: rodando, healthy, porta 5432
- ✅ MLflow: rodando, healthy, porta 5000
- ✅ Volumes: mlflow_db, mlflow_artifacts criados

### 2. Coleta de Dataset

```bash
# Usuário executa transcrições normalmente com TDvX
# Cada transcrição salva automaticamente em:
# tdvx/finetuning_data/manifest.jsonl
```

**Critérios de salvamento**:
- ✅ Confidence > 0.70 (configurável)
- ✅ Duração entre 0.5s e 28s
- ✅ Diarização bem-sucedida

### 3. Validação de Dataset

```bash
# Usuário valida dataset antes de treinar
python run_finetune_with_mlflow.py --sprint 4 --dry-run
```

**Wrapper verifica**:
1. `manifest.jsonl` existe?
2. Mínimo de 10 amostras?
3. Arquivos de áudio existem?
4. Distribuição de idiomas OK?

**Saída esperada**:
```
✓ Dataset encontrado: 1.240 amostras
✓ Distribuição: PT=980, EN=260
✓ Duração total: 8.5h
```

### 4. Execução de Fine-tuning

```bash
# Wrapper constrói comando e executa finetune.py
python run_finetune_with_mlflow.py --sprint 4
```

**Sequência interna**:

#### 4.1. Setup MLflow
```python
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tdv1-finetune")
run = mlflow.start_run(run_name="tdv1-sprint-4")
```

**Requests HTTP**:
- `POST /api/2.0/mlflow/experiments/create` (se não existe)
- `POST /api/2.0/mlflow/runs/create`

**PostgreSQL**:
```sql
-- experiments table
INSERT INTO experiments (name, artifact_location) 
VALUES ('tdv1-finetune', '/mlflow/artifacts');

-- runs table
INSERT INTO runs (run_uuid, experiment_id, status, start_time)
VALUES ('abc123...', 1, 'RUNNING', 1714291200);
```

#### 4.2. Log de Parâmetros
```python
mlflow.log_params({
    "base_model": "openai/whisper-medium",
    "learning_rate": 1e-5,
    "batch_size": 8,
    "languages": "portuguese+english",
    "dataset_total": 1240,
    "train_size": 1116,
    "eval_size": 124
})
```

**PostgreSQL**:
```sql
INSERT INTO params (run_uuid, key, value)
VALUES 
    ('abc123...', 'base_model', 'openai/whisper-medium'),
    ('abc123...', 'learning_rate', '1e-05'),
    ('abc123...', 'batch_size', '8'),
    ...
```

#### 4.3. Loop de Treinamento

**Step 1-25**:
```python
# Trainer (HuggingFace) chama automaticamente:
mlflow.log_metric("train/loss", 0.234, step=1)
mlflow.log_metric("train/loss", 0.198, step=2)
...
```

**PostgreSQL**:
```sql
INSERT INTO metrics (run_uuid, key, value, timestamp, step)
VALUES 
    ('abc123...', 'train/loss', 0.234, 1714291201, 1),
    ('abc123...', 'train/loss', 0.198, 1714291202, 2),
    ...
```

**Step 200 (Avaliação)**:
```python
# Trainer executa eval, computa WER
mlflow.log_metrics({
    "eval/loss": 0.156,
    "eval/wer": 15.2,
    "eval/runtime": 42.3
}, step=200)
```

#### 4.4. Salvamento de Artefatos
```python
mlflow.log_artifact("eval_results.json", artifact_path="eval")
```

**Filesystem (volume Docker)**:
```
/mlflow/artifacts/
└── 1/                          # experiment_id
    └── abc123.../              # run_id
        └── eval/
            └── eval_results.json
```

#### 4.5. Registro no Model Registry
```python
mlflow.transformers.log_model(
    transformers_model=str(output_dir),
    artifact_path="model"
)
mlflow.register_model(
    model_uri=f"runs:/{run.info.run_id}/model",
    name="tdv1-pt-en"
)
```

**PostgreSQL**:
```sql
-- registered_models table
INSERT INTO registered_models (name, creation_time)
VALUES ('tdv1-pt-en', 1714291500);

-- model_versions table
INSERT INTO model_versions (name, version, run_id, source, current_stage)
VALUES ('tdv1-pt-en', 3, 'abc123...', 'runs:/abc123.../model', 'None');
```

#### 4.6. Finalização
```python
mlflow.end_run()
```

**PostgreSQL**:
```sql
UPDATE runs 
SET status='FINISHED', end_time=1714295800
WHERE run_uuid='abc123...';
```

### 5. Visualização de Resultados

**Usuário acessa UI**: http://localhost:5000

**Queries executadas pela UI**:
```sql
-- Lista runs do experimento
SELECT * FROM runs 
WHERE experiment_id=1 
ORDER BY start_time DESC;

-- Carrega métricas de um run
SELECT key, value, step, timestamp
FROM metrics
WHERE run_uuid='abc123...'
ORDER BY key, step;

-- Carrega parâmetros
SELECT key, value FROM params
WHERE run_uuid='abc123...';
```

**UI renderiza**:
- 📊 Gráfico de `train/loss` vs step
- 📊 Gráfico de `eval/wer` vs step
- 📋 Tabela de parâmetros
- 📁 Lista de artefatos

### 6. Promoção de Modelo

**Usuário promove via UI ou CLI**:

Via UI:
1. Models → `tdv1-pt-en`
2. Version 3 → Stage → **Production**

Via Python:
```python
client = MlflowClient("http://localhost:5000")
client.transition_model_version_stage(
    name="tdv1-pt-en",
    version=3,
    stage="Production",
    archive_existing_versions=True
)
```

**PostgreSQL**:
```sql
-- Arquiva versões antigas em produção
UPDATE model_versions
SET current_stage='Archived'
WHERE name='tdv1-pt-en' AND current_stage='Production';

-- Promove nova versão
UPDATE model_versions
SET current_stage='Production'
WHERE name='tdv1-pt-en' AND version=3;
```

### 7. Integração com Aplicação

**engine.py carrega modelo do Registry**:
```python
import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Baixa artefatos da versão em produção
model_path = mlflow.transformers.download_artifacts(
    artifact_uri="models:/tdv1-pt-en/Production"
)

# Carrega com faster-whisper
from faster_whisper import WhisperModel
model = WhisperModel(model_path, device="cuda")
```

**HTTP Request**:
- `GET /api/2.0/mlflow/model-versions/get-download-uri?name=tdv1-pt-en&stage=Production`

**Resposta**:
```json
{
  "artifact_uri": "file:///mlflow/artifacts/1/abc123.../model"
}
```

---

## Persistência de Dados

### PostgreSQL (Metadados)

**Localização**: Volume Docker `mlflow_db`  
**Path interno**: `/var/lib/postgresql/data/pgdata`

**Tabelas principais**:
- `experiments` — lista de experimentos
- `runs` — execuções de treino
- `params` — parâmetros de cada run
- `metrics` — métricas (loss, WER, etc.)
- `tags` — metadados adicionais
- `registered_models` — modelos registrados
- `model_versions` — versões de cada modelo

**Backup**:
```bash
docker-compose exec postgres pg_dump -U mlflow mlflow > backup.sql
```

### Artefatos (Modelos e Logs)

**Localização**: Volume Docker `mlflow_artifacts`  
**Path interno**: `/mlflow/artifacts`

**Estrutura**:
```
/mlflow/artifacts/
├── 1/                          # experiment_id=1
│   ├── abc123.../              # run_id 1
│   │   ├── eval/
│   │   │   └── eval_results.json
│   │   └── ct2/
│   │       └── config.json
│   └── def456.../              # run_id 2
│       └── ...
└── 2/                          # experiment_id=2
    └── ...
```

**Backup**:
```bash
docker run --rm \
    -v mlflow_artifacts:/data \
    -v $(pwd):/backup \
    alpine tar czf /backup/artifacts.tar.gz -C /data .
```

---

## Troubleshooting por Componente

### PostgreSQL

**Sintoma**: MLflow falha ao iniciar com erro de conexão DB

**Debug**:
```bash
# 1. Verifica se PostgreSQL está rodando
docker-compose ps postgres

# 2. Testa conexão
docker-compose exec postgres pg_isready -U mlflow

# 3. Verifica logs
docker-compose logs postgres

# 4. Conecta manualmente
docker-compose exec postgres psql -U mlflow
```

**Solução comum**: Aguardar health check
```bash
docker-compose up -d postgres
sleep 10
docker-compose up -d mlflow
```

### MLflow Server

**Sintoma**: API retorna 500 Internal Server Error

**Debug**:
```bash
# 1. Logs detalhados
docker-compose logs -f mlflow

# 2. Health check
curl -v http://localhost:5000/health

# 3. Verifica backend store
docker-compose exec mlflow env | grep BACKEND
```

**Solução comum**: Reiniciar com logs
```bash
docker-compose restart mlflow
docker-compose logs -f mlflow
```

### Wrapper Script

**Sintoma**: Script falha com "MLflow not accessible"

**Debug**:
```python
# Teste manual de conexão
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
print(mlflow.search_experiments())
```

**Solução comum**: Verificar `MLFLOW_TRACKING_URI` no `.env`

---

**Última atualização**: 2026-04-28  
**Versão**: 1.0
