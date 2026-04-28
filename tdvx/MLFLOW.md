# MLflow Integration — TDv1 Fine-tuning Tracking

Este documento descreve a integração do MLflow com o pipeline de fine-tuning do TDv1, permitindo rastreamento completo de experimentos, métricas e versionamento de modelos.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Quick Start](#quick-start)
- [Configuração](#configuração)
- [Uso](#uso)
- [MLflow UI](#mlflow-ui)
- [Model Registry](#model-registry)
- [Troubleshooting](#troubleshooting)

---

## Visão Geral

O MLflow é uma plataforma open-source para gerenciar o ciclo de vida de Machine Learning, incluindo:

- **Tracking**: Registro de parâmetros, métricas e artefatos de experimentos
- **Model Registry**: Versionamento e gestão de modelos em produção
- **Projects**: Empacotamento de código ML reproduzível
- **Models**: Formato padrão para deploy de modelos

### Por que usar MLflow no TDv1?

1. **Rastreabilidade**: Todos os experimentos de fine-tuning ficam registrados
2. **Comparação**: Compare facilmente diferentes configurações de treino
3. **Reprodutibilidade**: Parâmetros e versões são automaticamente rastreados
4. **Governança**: Model Registry permite promoção controlada (Staging → Production)
5. **Colaboração**: Equipe inteira vê os mesmos experimentos em tempo real

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                         TDv1 Fine-tuning                        │
│                                                                 │
│  ┌──────────────┐    logs params/metrics    ┌───────────────┐  │
│  │ finetune.py  │ ─────────────────────────> │  MLflow API   │  │
│  └──────────────┘                            └───────┬───────┘  │
│                                                      │          │
│                                                      ▼          │
└──────────────────────────────────────────────────────┼──────────┘
                                                       │
                ┌──────────────────────────────────────┼──────────┐
                │         MLflow Tracking Server       │          │
                │                                      │          │
                │  ┌─────────────┐          ┌─────────▼────────┐ │
                │  │  MLflow UI  │          │   PostgreSQL     │ │
                │  │ (port 5000) │          │  (metadata DB)   │ │
                │  └─────────────┘          └──────────────────┘ │
                │                                                │
                │          ┌──────────────────────┐              │
                │          │  Artifact Storage    │              │
                │          │  (models, logs, etc) │              │
                │          └──────────────────────┘              │
                └─────────────────────────────────────────────────┘
```

### Componentes

1. **PostgreSQL**: Armazena metadados (runs, params, metrics)
2. **MLflow Server**: API REST para tracking + servidor de artefatos
3. **MLflow UI**: Interface web para visualizar experimentos (porta 5000)
4. **Artifact Storage**: Volume Docker para modelos e artefatos

---

## Quick Start

### 1. Configurar ambiente

```bash
# Copie o arquivo de exemplo
cd tdvx
cp .env.example .env

# Edite .env e configure as variáveis MLflow:
# MLFLOW_TRACKING_URI=http://localhost:5000
# MLFLOW_DB_PASSWORD=mlflow123
```

### 2. Iniciar serviços MLflow

```bash
# Inicia MLflow + PostgreSQL
docker-compose up -d mlflow postgres

# Verifica se está rodando
docker-compose ps

# Logs em tempo real
docker-compose logs -f mlflow
```

### 3. Verificar MLflow UI

Abra no navegador: **http://localhost:5000**

Você deve ver a interface do MLflow sem experimentos ainda.

### 4. Executar fine-tuning com tracking

```bash
# Usando o wrapper (recomendado)
python run_finetune_with_mlflow.py --sprint 1

# Ou diretamente
cd tdvx
python finetune.py \
    --mlflow-uri http://localhost:5000 \
    --run-name tdv1-sprint-1 \
    --model-name tdv1-pt-en
```

### 5. Visualizar resultados

1. Acesse http://localhost:5000
2. Clique no experimento **tdv1-finetune**
3. Veja métricas (WER, loss) e parâmetros registrados

---

## Configuração

### Variáveis de Ambiente

Configure no arquivo `.env`:

```bash
# ── MLflow Tracking ──
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_FINETUNE_EXPERIMENT=tdv1-finetune
MLFLOW_DB_PASSWORD=mlflow123

# ── HuggingFace (opcional) ──
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Docker Compose

O `docker-compose.yml` já está configurado com 3 serviços MLflow:

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.2
    ports:
      - "5000:5000"
    # ... backend PostgreSQL + artifact storage

  postgres:
    image: postgres:15-alpine
    # ... database para metadados do MLflow

  pgadmin:  # opcional
    image: dpage/pgadmin4:latest
    profiles: [tools]
    ports:
      - "5050:80"
```

### Iniciar serviços específicos

```bash
# Apenas MLflow + PostgreSQL
docker-compose up -d mlflow postgres

# Incluir pgAdmin (gerenciamento de DB)
docker-compose --profile tools up -d pgadmin

# Parar serviços
docker-compose down

# Parar E remover dados (CUIDADO!)
docker-compose down -v
```

---

## Uso

### Opção 1: Wrapper Simplificado (Recomendado)

O script `run_finetune_with_mlflow.py` automatiza todo o processo:

```bash
# Fine-tuning da sprint 4 com tracking automático
python run_finetune_with_mlflow.py --sprint 4

# Com parâmetros customizados
python run_finetune_with_mlflow.py \
    --sprint 5 \
    --base-model openai/whisper-large-v3 \
    --max-steps 1000 \
    --learning-rate 1e-5 \
    --register-model

# Sem iniciar MLflow automaticamente
python run_finetune_with_mlflow.py \
    --sprint 6 \
    --no-auto-start-mlflow

# Dry run (valida dataset sem treinar)
python run_finetune_with_mlflow.py --sprint 7 --dry-run
```

**Features do wrapper:**
- ✅ Verifica se MLflow está rodando
- ✅ Inicia containers automaticamente se necessário
- ✅ Nomenclatura padronizada de runs (`tdv1-sprint-N`)
- ✅ Registro automático no Model Registry
- ✅ Validação de dataset antes de treinar

### Opção 2: Script Original

Use `finetune.py` diretamente para controle total:

```bash
cd tdvx

# Com tracking MLflow
python finetune.py \
    --mlflow-uri http://localhost:5000 \
    --mlflow-experiment tdv1-finetune \
    --run-name tdv1-sprint-4 \
    --model-name tdv1-pt-en \
    --max-steps 2000

# Sem MLflow (tracking desabilitado)
python finetune.py --max-steps 2000
```

### Dentro do Container Docker

```bash
# Entra no container TDv1
docker-compose exec tdvx bash

# Executa fine-tuning (usa MLFLOW_TRACKING_URI do .env)
python /app/finetune.py \
    --run-name tdv1-sprint-4-docker \
    --model-name tdv1-pt-en

# Ou usando o wrapper
python /app/run_finetune_with_mlflow.py --sprint 4
```

---

## MLflow UI

### Acessar Interface

**URL**: http://localhost:5000

### Recursos da UI

#### 1. **Experiments** — Listagem de experimentos

- Navegue até o experimento **tdv1-finetune**
- Veja todos os runs executados
- Compare métricas lado a lado

#### 2. **Runs** — Detalhes de cada execução

Para cada run você encontra:

**Parameters (Parâmetros)**:
```
base_model         = openai/whisper-medium
languages          = portuguese+english
learning_rate      = 1e-05
batch_size         = 8
max_steps          = 2000
min_confidence     = 0.4
dataset_total      = 1523
train_size         = 1370
eval_size          = 153
```

**Metrics (Métricas)**:
```
train/loss         = 0.0234   (a cada 25 steps)
eval/loss          = 0.0456   (a cada 200 steps)
eval/wer           = 12.34%   (Word Error Rate — quanto menor, melhor)
eval/runtime       = 45.2s
```

**Artifacts (Artefatos)**:
- `eval/eval_results.json` — métricas finais de avaliação
- `ct2/config.json` — config do modelo CTranslate2

**System Metrics** (se habilitado):
- GPU utilization
- CPU usage
- Memory consumption

#### 3. **Compare Runs** — Comparação

1. Selecione 2+ runs (checkbox)
2. Clique em **Compare**
3. Veja side-by-side:
   - Parâmetros diferentes
   - Evolução de métricas (gráficos)
   - Diferenças de configuração

#### 4. **Charts** — Visualizações

Gráficos automáticos:
- **Loss curves**: `train/loss` e `eval/loss` ao longo do tempo
- **WER progression**: `eval/wer` por step
- **Parameter importance**: quais params mais impactam WER

---

## Model Registry

O **Model Registry** é um repositório versionado de modelos prontos para deploy.

### Fluxo de Versionamento

```
Fine-tuning → Registro → Staging → Production
```

### Registrar Modelo Automaticamente

```bash
# Via wrapper (registro automático)
python run_finetune_with_mlflow.py \
    --sprint 5 \
    --register-model \
    --model-name tdv1-pt-en

# Via finetune.py
python finetune.py \
    --model-name tdv1-pt-en \
    --run-name tdv1-sprint-5
```

Isso cria:
- **Model**: `tdv1-pt-en`
- **Version**: v1, v2, v3... (automático a cada run)

### Gerenciar no UI

1. Acesse http://localhost:5000
2. Menu lateral → **Models**
3. Clique no modelo (ex: `tdv1-pt-en`)
4. Veja todas as versões

### Promover Modelo

#### Via UI:
1. Acesse a versão desejada
2. Botão **Stage** → selecione:
   - **None**: versão experimental
   - **Staging**: em homologação
   - **Production**: em produção
   - **Archived**: descontinuada

#### Via API (Python):

```python
from mlflow.tracking import MlflowClient

client = MlflowClient("http://localhost:5000")

# Promove versão 3 para produção
client.transition_model_version_stage(
    name="tdv1-pt-en",
    version=3,
    stage="Production",
    archive_existing_versions=True  # arquiva versões antigas em prod
)
```

### Carregar Modelo do Registry

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# Carrega versão específica
model = mlflow.transformers.load_model("models:/tdv1-pt-en/3")

# Carrega versão em Production
model = mlflow.transformers.load_model("models:/tdv1-pt-en/Production")

# Carrega versão em Staging
model = mlflow.transformers.load_model("models:/tdv1-pt-en/Staging")
```

### Integrar com engine.py

Atualize `tdvx/app/services/engine.py` para carregar do Registry:

```python
import mlflow
import os

class TranscriptionEngine:
    def __init__(self):
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Carrega modelo em produção do Registry
        self.model_path = mlflow.transformers.download_artifacts(
            artifact_uri="models:/tdv1-pt-en/Production"
        )
        
        # Usa com faster-whisper (CTranslate2)
        from faster_whisper import WhisperModel
        self.model = WhisperModel(self.model_path, device="cuda")
```

---

## Métricas Rastreadas

### Parâmetros (logged automaticamente)

| Parâmetro | Descrição |
|-----------|-----------|
| `base_model` | Modelo Whisper base usado |
| `languages` | Idiomas treinados (pt+en) |
| `learning_rate` | Taxa de aprendizado |
| `batch_size` | Tamanho do batch |
| `grad_accum` | Gradient accumulation steps |
| `warmup_steps` | Steps de warmup |
| `max_steps` | Total de steps de treino |
| `eval_steps` | Frequência de avaliação |
| `min_confidence` | Filtro de qualidade do dataset |
| `dataset_total` | Total de amostras |
| `dataset_duration_h` | Duração total em horas |
| `train_size` | Amostras de treino |
| `eval_size` | Amostras de avaliação |
| `lang_portuguese` | Amostras em português |
| `lang_english` | Amostras em inglês |

### Métricas (logged a cada step/epoch)

| Métrica | Descrição | Frequência |
|---------|-----------|------------|
| `train/loss` | Loss de treino | A cada 25 steps |
| `train/learning_rate` | LR atual (com warmup) | A cada 25 steps |
| `train/epoch` | Época atual | A cada step |
| `eval/loss` | Loss de validação | A cada eval_steps |
| `eval/wer` | Word Error Rate (%) | A cada eval_steps |
| `eval/runtime` | Tempo de avaliação | A cada eval_steps |
| `eval/samples_per_second` | Throughput de eval | A cada eval_steps |

### Artefatos Salvos

| Arquivo | Descrição |
|---------|-----------|
| `eval/eval_results.json` | Métricas finais de avaliação |
| `ct2/config.json` | Config do modelo CTranslate2 |

**Nota**: O modelo completo HuggingFace **NÃO** é salvo no MLflow por padrão (tamanho grande). Use `--hub-model-id` para publicar no HuggingFace Hub.

---

## pgAdmin (Opcional)

Interface web para gerenciar o banco PostgreSQL do MLflow.

### Iniciar

```bash
docker-compose --profile tools up -d pgadmin
```

### Acessar

**URL**: http://localhost:5050  
**Email**: `admin@tdvx.local` (configurável no `.env`)  
**Senha**: `admin123` (configurável no `.env`)

### Conectar ao PostgreSQL

1. Login no pgAdmin
2. Add New Server
3. **General** tab:
   - Name: `MLflow DB`
4. **Connection** tab:
   - Host: `postgres`
   - Port: `5432`
   - Database: `mlflow`
   - Username: `mlflow`
   - Password: `mlflow123` (do `.env`)
5. Save

### Explorar Dados

Você verá as tabelas MLflow:
- `experiments` — experimentos criados
- `runs` — todas as execuções
- `params` — parâmetros de cada run
- `metrics` — métricas logadas
- `tags` — tags de metadados
- `registered_models` — modelos no Registry
- `model_versions` — versões de cada modelo

---

## Troubleshooting

### MLflow não inicia

```bash
# Verifica logs
docker-compose logs mlflow postgres

# Problemas comuns:
# 1. PostgreSQL não está ready
docker-compose up -d postgres
# aguarde 10s
docker-compose up -d mlflow

# 2. Porta 5000 em uso
# Edite docker-compose.yml: "5001:5000"
# Edite .env: MLFLOW_TRACKING_URI=http://localhost:5001
```

### "Connection refused" ao executar finetune.py

```bash
# 1. Verifica se MLflow está rodando
curl http://localhost:5000/health
# Deve retornar: {"status":"ok"}

# 2. Se não estiver, inicia
docker-compose up -d mlflow postgres

# 3. Aguarda inicialização
sleep 10
curl http://localhost:5000/health
```

### Erro de autenticação do PostgreSQL

```bash
# Verifica senha configurada no .env
cat .env | grep MLFLOW_DB_PASSWORD

# Recria containers com senha correta
docker-compose down -v  # CUIDADO: apaga dados!
docker-compose up -d mlflow postgres
```

### "No such experiment" error

O experimento é criado automaticamente no primeiro run. Se removido:

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.create_experiment("tdv1-finetune")
```

Ou simplesmente execute `finetune.py` novamente.

### Modelo não aparece no Registry

Certifique-se de usar `--model-name`:

```bash
python finetune.py --model-name tdv1-pt-en
```

Sem isso, o modelo é salvo mas não registrado.

### Limpar todos os dados MLflow

```bash
# ATENÇÃO: Remove TODOS os experimentos, runs e modelos!
docker-compose down -v
docker-compose up -d mlflow postgres
```

---

## Referências

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Docker](https://mlflow.org/docs/latest/docker.html)
- [HuggingFace + MLflow](https://mlflow.org/docs/latest/python_api/mlflow.transformers.html)

---

## Próximos Passos

1. **Produção**: Deploy do MLflow em servidor dedicado
2. **S3 Storage**: Migrar artifacts para S3/MinIO
3. **Auth**: Adicionar autenticação ao MLflow UI
4. **CI/CD**: Automatizar fine-tuning via GitHub Actions
5. **Alertas**: Notificações quando WER atinge threshold

---

**Versão**: 1.0  
**Última atualização**: 2026-04-28  
**Mantido por**: TDvX Team
