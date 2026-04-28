# Integração MLflow — Notas de Implementação

## Resumo da Integração

Esta integração adiciona rastreamento completo de experimentos de fine-tuning usando MLflow, com backend PostgreSQL para persistência de metadados e armazenamento de artefatos em volumes Docker.

**Data da implementação**: 2026-04-28

---

## Componentes Adicionados

### 1. Docker Compose — Stack MLflow

**Arquivo**: `tdvx/docker-compose.yml`

**Serviços adicionados**:

- **mlflow**: Tracking Server MLflow v2.10.2
  - Porta: 5000
  - Backend: PostgreSQL
  - Artifact storage: volume Docker local
  - Health check configurado

- **postgres**: Banco de dados PostgreSQL 15
  - Porta: 5432 (interna)
  - Database: `mlflow`
  - User: `mlflow`
  - Volume persistente

- **pgadmin**: Interface web para PostgreSQL (opcional, profile `tools`)
  - Porta: 5050
  - Permite visualizar/gerenciar tabelas do MLflow

**Volumes criados**:
- `mlflow_artifacts` — artefatos de modelos
- `mlflow_db` — dados do PostgreSQL
- `pgadmin_data` — configuração do pgAdmin

**Network**:
- `tdvx-network` — permite comunicação entre serviços

### 2. Script Wrapper de Fine-tuning

**Arquivo**: `run_finetune_with_mlflow.py`

**Funcionalidades**:
- Verifica automaticamente se MLflow está rodando
- Inicia containers Docker se necessário (flag `--auto-start-mlflow`)
- Valida dataset antes de treinar
- Nomenclatura padronizada de runs (`tdv1-sprint-N`)
- Registro automático no Model Registry
- Parâmetros simplificados vs. `finetune.py` original

**Uso básico**:
```bash
python run_finetune_with_mlflow.py --sprint 4
```

### 3. Scripts de Inicialização

**Arquivos**:
- `start_mlflow.sh` — Linux/Mac
- `start_mlflow.bat` — Windows

**O que fazem**:
- Verificam se `.env` existe
- Iniciam containers MLflow + PostgreSQL
- Aguardam inicialização (10s)
- Testam health check
- Exibem links úteis e próximos passos

### 4. Configuração de Ambiente

**Arquivo**: `tdvx/.env.example`

**Novas variáveis MLflow**:
```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_FINETUNE_EXPERIMENT=tdv1-finetune
MLFLOW_DB_PASSWORD=mlflow123
PGADMIN_EMAIL=admin@tdvx.local
PGADMIN_PASSWORD=admin123
```

### 5. Documentação

**Arquivo**: `tdvx/MLFLOW.md`

Documentação completa incluindo:
- Arquitetura do sistema
- Quick start
- Configuração detalhada
- Uso da UI
- Model Registry workflows
- Troubleshooting
- Exemplos práticos

### 6. Dependências

**Arquivo**: `requirements-finetune.txt`

**Adicionado**:
```
mlflow>=2.10.0
psycopg2-binary>=2.9.0
requests>=2.31.0
```

---

## Atualizações no Código Existente

### `tdvx/finetune.py`

**Sem alterações necessárias** — O script já tinha suporte a MLflow implementado via:
- Classe `MlflowTracker` (linhas 309-435)
- Integração com `Seq2SeqTrainingArguments.report_to`
- Registro automático de params, metrics e artefatos

**Configuração de conexão**:
O script detecta automaticamente o MLflow se:
1. `MLFLOW_TRACKING_URI` está definido no `.env`, ou
2. `--mlflow-uri` é passado via CLI

### `tdvx/docker-compose.yml`

**Serviço TDvX atualizado**:
- Adicionada variável `MLFLOW_TRACKING_URI=http://mlflow:5000`
- Volumes mapeados: `./finetuning_data` e `./models`
- Conectado à network `tdvx-network`
- Dependency em `mlflow`

**Benefício**: O container TDvX pode executar fine-tuning e conectar automaticamente ao MLflow sem configuração manual.

### `.gitignore`

**Adicionado**:
```
# MLflow
mlruns/
mlartifacts/
mlflow.db
.mlflow/
```

### `README.md` (raiz)

**Atualizado**:
- Estrutura de diretórios reflete novos scripts
- Seção sobre MLflow tracking
- Link para documentação completa

### `tdvx/README.md`

**Atualizado**:
- Seção "MLflow — tracking de experimentos" expandida
- Instruções Docker Compose adicionadas
- Documentação do wrapper `run_finetune_with_mlflow.py`
- Comparação entre opções de deployment do MLflow

---

## Fluxo de Uso Recomendado

### 1. Setup Inicial (uma vez)

```bash
# Copia configuração
cd tdvx
cp .env.example .env

# Edita .env e define PYANNOTE_AUTH_TOKEN, MLFLOW_TRACKING_URI, etc.
```

### 2. Iniciar MLflow

```bash
# Opção 1: Script auxiliar
./start_mlflow.sh         # Linux/Mac
start_mlflow.bat          # Windows

# Opção 2: Docker Compose direto
cd tdvx
docker-compose up -d mlflow postgres
```

### 3. Coletar Dataset

Execute transcrições normalmente com o TDvX. Dados são salvos automaticamente em `tdvx/finetuning_data/`.

### 4. Fine-tuning

```bash
# Usando wrapper (recomendado)
python run_finetune_with_mlflow.py --sprint 1

# Ou script original
cd tdvx
python finetune.py --run-name tdv1-sprint-1 --model-name tdv1-pt-en
```

### 5. Visualizar Resultados

- **UI do MLflow**: http://localhost:5000
- **Experimentos**: Aba "Experiments" → `tdv1-finetune`
- **Modelos**: Aba "Models" → `tdv1-pt-en`

### 6. Promover Modelo

Na UI do MLflow:
1. Models → `tdv1-pt-en`
2. Seleciona versão
3. Stage → **Production**

---

## Métricas Rastreadas Automaticamente

| Categoria | Exemplo |
|-----------|---------|
| **Parâmetros** | `base_model`, `learning_rate`, `batch_size`, `languages`, `dataset_total`, `min_confidence` |
| **Métricas de Treino** | `train/loss`, `train/learning_rate`, `train/epoch` (a cada 25 steps) |
| **Métricas de Eval** | `eval/loss`, `eval/wer`, `eval/runtime` (a cada 200 steps) |
| **Artefatos** | `eval_results.json`, `ct2/config.json` |
| **Tags** | `base_model`, `languages`, `sprint` |

---

## Volumes Docker e Persistência

### Dados Persistidos

1. **PostgreSQL** (`mlflow_db`):
   - Metadados de todos os experimentos
   - Parâmetros e métricas
   - Model Registry

2. **Artifacts** (`mlflow_artifacts`):
   - Arquivos de avaliação
   - Configs de modelos
   - Plots e visualizações

### Backup

```bash
# Backup do banco PostgreSQL
docker-compose exec postgres pg_dump -U mlflow mlflow > mlflow_backup_$(date +%Y%m%d).sql

# Backup de artefatos
docker run --rm -v tdvx_mlflow_artifacts:/data -v $(pwd):/backup \
    alpine tar czf /backup/mlflow_artifacts_$(date +%Y%m%d).tar.gz -C /data .
```

### Restauração

```bash
# Restaurar banco
cat mlflow_backup_20260428.sql | docker-compose exec -T postgres psql -U mlflow mlflow

# Restaurar artefatos
docker run --rm -v tdvx_mlflow_artifacts:/data -v $(pwd):/backup \
    alpine tar xzf /backup/mlflow_artifacts_20260428.tar.gz -C /data
```

### Limpar Tudo (CUIDADO!)

```bash
# Remove containers E volumes (apaga todos os dados)
cd tdvx
docker-compose down -v
```

---

## Troubleshooting

### MLflow não inicia

**Sintoma**: `docker-compose up -d mlflow` falha

**Soluções**:
1. Verifica logs: `docker-compose logs mlflow postgres`
2. Garante que PostgreSQL está healthy primeiro:
   ```bash
   docker-compose up -d postgres
   sleep 10
   docker-compose up -d mlflow
   ```
3. Porta 5000 em uso? Edite `docker-compose.yml` para `5001:5000`

### "Connection refused" no finetune.py

**Sintoma**: Script falha ao conectar no MLflow

**Soluções**:
1. Verifica se MLflow está rodando:
   ```bash
   curl http://localhost:5000/health
   # Esperado: {"status":"ok"}
   ```
2. Verifica `MLFLOW_TRACKING_URI` no `.env`
3. Testa manualmente:
   ```python
   import mlflow
   mlflow.set_tracking_uri("http://localhost:5000")
   print(mlflow.search_experiments())
   ```

### Porta 5000 conflita com outro serviço

**Sintoma**: `Bind for 0.0.0.0:5000 failed: port is already allocated`

**Solução**:
```yaml
# docker-compose.yml
services:
  mlflow:
    ports:
      - "5001:5000"  # expõe na porta 5001 do host
```

E atualize `.env`:
```env
MLFLOW_TRACKING_URI=http://localhost:5001
```

### Experimentos/runs desapareceram

**Sintoma**: UI do MLflow mostra "No experiments"

**Causas possíveis**:
1. Volumes Docker foram removidos (`docker-compose down -v`)
2. PostgreSQL resetado
3. Conectando em URI errado

**Solução**: Restaurar do backup (veja seção Backup acima)

### Modelo não aparece no Registry

**Sintoma**: Run foi executado mas modelo não está em "Models"

**Causa**: Faltou `--model-name` no comando

**Solução**:
```bash
python finetune.py --model-name tdv1-pt-en
# ou
python run_finetune_with_mlflow.py --sprint 1 --register-model
```

---

## Próximos Passos Sugeridos

### Curto Prazo
- [ ] Testar fine-tuning end-to-end com dataset real
- [ ] Validar conversão CTranslate2 com modelo do Registry
- [ ] Documentar processo de rollback de modelo

### Médio Prazo
- [ ] Configurar autenticação no MLflow (basic auth ou OAuth)
- [ ] Migrar artifact storage para S3/MinIO (produção)
- [ ] Automatizar fine-tuning via CI/CD (GitHub Actions)
- [ ] Criar dashboards customizados no MLflow

### Longo Prazo
- [ ] Deploy do MLflow em servidor dedicado (produção)
- [ ] Integração com Kubernetes (MLflow + PostgreSQL HA)
- [ ] Sistema de alertas (Slack/email) quando WER atinge threshold
- [ ] A/B testing de modelos em produção

---

## Referências

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [MLflow Docker Deployment](https://mlflow.org/docs/latest/docker.html)
- [PostgreSQL Backend Store](https://mlflow.org/docs/latest/tracking.html#backend-stores)
- [Model Registry Workflows](https://mlflow.org/docs/latest/model-registry.html#transitioning-an-mlflow-models-stage)
- [Transformers + MLflow](https://mlflow.org/docs/latest/python_api/mlflow.transformers.html)

---

**Mantido por**: TDvX Team  
**Última atualização**: 2026-04-28  
**Versão da integração**: 1.0
