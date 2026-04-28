# 📊 Integração MLflow — TDv1 Fine-tuning

Documentação completa da integração do MLflow com o pipeline de fine-tuning do TDv1.

---

## 🎯 O que foi Implementado

### ✅ Infraestrutura Docker

- **MLflow Tracking Server** (porta 5000)
  - Backend: PostgreSQL 15
  - Artifact storage: Volume Docker local
  - Health checks configurados
  
- **PostgreSQL** (porta 5432)
  - Database: `mlflow`
  - Persistência via volume Docker
  - Health checks configurados

- **pgAdmin** (porta 5050, opcional)
  - Interface web para gerenciar PostgreSQL
  - Profile: `tools`

### ✅ Scripts e Ferramentas

- **run_finetune_with_mlflow.py**
  - Wrapper inteligente para fine-tuning
  - Verifica e inicia MLflow automaticamente
  - Nomenclatura padronizada de runs
  - Validação de dataset
  - Registro automático no Model Registry

- **start_mlflow.sh / start_mlflow.bat**
  - Scripts de inicialização rápida
  - Verificação de pré-requisitos
  - Health checks automáticos
  - Linux/Mac e Windows

### ✅ Configuração

- **.env.example**
  - Template completo de variáveis
  - Documentação inline
  - Defaults seguros

- **requirements-finetune.txt**
  - MLflow + dependências
  - PostgreSQL driver
  - Bibliotecas de tracking

### ✅ Documentação

| Arquivo | Propósito |
|---------|-----------|
| [MLFLOW.md](tdvx/MLFLOW.md) | Documentação completa (arquitetura, setup, uso) |
| [MLFLOW_INTEGRATION.md](MLFLOW_INTEGRATION.md) | Notas técnicas da implementação |
| [MLFLOW_QUICK_REFERENCE.md](MLFLOW_QUICK_REFERENCE.md) | Comandos rápidos e troubleshooting |
| [MLFLOW_WORKFLOW.md](MLFLOW_WORKFLOW.md) | Fluxo detalhado e diagramas |

---

## 🚀 Quick Start

### 1. Configuração Inicial

```bash
# Copie o template de configuração
cd tdvx
cp .env.example .env

# Edite .env e configure:
# - PYANNOTE_AUTH_TOKEN
# - MLFLOW_TRACKING_URI=http://localhost:5000
# - MLFLOW_DB_PASSWORD=mlflow123
```

### 2. Inicie o MLflow

```bash
# Opção 1: Script auxiliar
./start_mlflow.sh        # Linux/Mac
start_mlflow.bat         # Windows

# Opção 2: Docker Compose direto
cd tdvx
docker-compose up -d mlflow postgres
```

### 3. Acesse a UI

Abra no navegador: **http://localhost:5000**

### 4. Execute Fine-tuning

```bash
# Wrapper simplificado (recomendado)
python run_finetune_with_mlflow.py --sprint 1

# Com parâmetros customizados
python run_finetune_with_mlflow.py \
    --sprint 2 \
    --base-model openai/whisper-large-v3 \
    --max-steps 1000 \
    --register-model
```

### 5. Visualize Resultados

1. Acesse http://localhost:5000
2. Navegue até **Experiments** → `tdv1-finetune`
3. Compare runs, veja métricas e parâmetros
4. Promova modelos via **Models** → `tdv1-pt-en`

---

## 📖 Guias de Uso

### Para Desenvolvedores

1. **Setup do Ambiente**: [MLFLOW.md#quick-start](tdvx/MLFLOW.md#quick-start)
2. **Executar Fine-tuning**: [MLFLOW.md#uso](tdvx/MLFLOW.md#uso)
3. **Entender o Fluxo**: [MLFLOW_WORKFLOW.md](MLFLOW_WORKFLOW.md)

### Para DevOps

1. **Gerenciar Containers**: [MLFLOW_QUICK_REFERENCE.md#gerenciamento-de-containers](MLFLOW_QUICK_REFERENCE.md#gerenciamento-de-containers)
2. **Backup e Restore**: [MLFLOW_QUICK_REFERENCE.md#backup](MLFLOW_QUICK_REFERENCE.md#backup)
3. **Troubleshooting**: [MLFLOW.md#troubleshooting](tdvx/MLFLOW.md#troubleshooting)

### Para Data Scientists

1. **Model Registry**: [MLFLOW.md#model-registry](tdvx/MLFLOW.md#model-registry)
2. **Comparar Experimentos**: [MLFLOW.md#mlflow-ui](tdvx/MLFLOW.md#mlflow-ui)
3. **Carregar Modelos**: [MLFLOW_QUICK_REFERENCE.md#model-registry](MLFLOW_QUICK_REFERENCE.md#model-registry)

---

## 🔧 Arquitetura

```
┌──────────────────────────────────────────────────────────────────┐
│                      TDv1 Fine-tuning                            │
│                                                                  │
│  ┌─────────────────┐        logs metrics/params   ┌──────────┐  │
│  │  finetune.py    │ ─────────────────────────────>│  MLflow  │  │
│  │  (HF Trainer)   │                               │   API    │  │
│  └─────────────────┘                               └────┬─────┘  │
│                                                          │        │
└──────────────────────────────────────────────────────────┼────────┘
                                                           │
              ┌────────────────────────────────────────────┼────────┐
              │          MLflow Stack (Docker)             │        │
              │                                            │        │
              │  ┌────────────┐              ┌────────────▼──────┐ │
              │  │  MLflow UI │              │   PostgreSQL      │ │
              │  │ (port 5000)│              │  - experiments    │ │
              │  └────────────┘              │  - runs           │ │
              │                              │  - params/metrics │ │
              │                              │  - model_registry │ │
              │                              └───────────────────┘ │
              │                                                    │
              │         ┌─────────────────────────┐               │
              │         │  Artifact Storage       │               │
              │         │  (Docker volume)        │               │
              │         │  - eval results         │               │
              │         │  - model configs        │               │
              │         └─────────────────────────┘               │
              └─────────────────────────────────────────────────────┘
```

**Detalhes**: Veja [MLFLOW_WORKFLOW.md](MLFLOW_WORKFLOW.md)

---

## 📊 Métricas Rastreadas

### Parâmetros Automáticos

- Modelo base (whisper-medium, whisper-large-v3)
- Hiperparâmetros (learning_rate, batch_size, etc.)
- Configuração de dataset (total, duração, idiomas)
- Filtros de qualidade (min_confidence, min_duration)

### Métricas de Treino

- `train/loss` — loss de treino (a cada 25 steps)
- `train/learning_rate` — learning rate atual
- `train/epoch` — época atual

### Métricas de Avaliação

- `eval/loss` — loss de validação (a cada 200 steps)
- `eval/wer` — Word Error Rate % (quanto menor, melhor)
- `eval/runtime` — tempo de avaliação
- `eval/samples_per_second` — throughput

### Artefatos

- `eval/eval_results.json` — métricas finais
- `ct2/config.json` — config do modelo CTranslate2

---

## 🎓 Conceitos-chave

### Experiments

Agrupam runs relacionados. Padrão: `tdv1-finetune`

Cada sprint ou configuração diferente pode ter seu próprio experimento.

### Runs

Uma execução de fine-tuning. Contém:
- Parâmetros usados
- Métricas ao longo do tempo
- Artefatos gerados
- Código hash (opcional)

### Model Registry

Repositório versionado de modelos prontos para deploy.

**Stages**:
- **None** — experimental
- **Staging** — em homologação
- **Production** — em produção
- **Archived** — descontinuado

**Workflow**:
```
Fine-tuning → v1 (None) → v2 (Staging) → v3 (Production)
```

---

## 🔐 Segurança e Boas Práticas

### ✅ Fazer

- ✅ Usar `.env` para credenciais (nunca commitar)
- ✅ Fazer backup regular do PostgreSQL
- ✅ Usar volumes Docker para persistência
- ✅ Nomear runs de forma descritiva (`tdv1-sprint-N`)
- ✅ Promover modelos gradualmente (None → Staging → Production)
- ✅ Documentar mudanças em cada versão de modelo

### ❌ Evitar

- ❌ Commitar `.env` no git
- ❌ Usar `docker-compose down -v` sem backup
- ❌ Rodar fine-tuning sem validar dataset primeiro
- ❌ Promover modelo direto para Production sem testar
- ❌ Deletar runs antigos sem backup

---

## 🐛 Troubleshooting

### Problemas Comuns

| Sintoma | Solução Rápida |
|---------|----------------|
| MLflow não inicia | `docker-compose logs mlflow` |
| Porta 5000 em uso | Edite `docker-compose.yml`: `"5001:5000"` |
| Connection refused | `curl http://localhost:5000/health` |
| PostgreSQL não conecta | `docker-compose up -d postgres; sleep 10` |
| Modelo não registra | Adicione `--model-name` ao comando |

**Guia completo**: [MLFLOW.md#troubleshooting](tdvx/MLFLOW.md#troubleshooting)

---

## 📚 Documentação Completa

| Arquivo | Descrição |
|---------|-----------|
| [MLFLOW.md](tdvx/MLFLOW.md) | **Guia principal** — arquitetura, configuração, uso completo |
| [MLFLOW_INTEGRATION.md](MLFLOW_INTEGRATION.md) | Notas técnicas de implementação e decisões de design |
| [MLFLOW_QUICK_REFERENCE.md](MLFLOW_QUICK_REFERENCE.md) | Comandos rápidos, Docker, troubleshooting |
| [MLFLOW_WORKFLOW.md](MLFLOW_WORKFLOW.md) | Diagramas de sequência e fluxo detalhado |

---

## 🚀 Próximos Passos

### Curto Prazo

- [ ] Testar fine-tuning end-to-end
- [ ] Validar backup/restore
- [ ] Integrar modelo do Registry com engine.py

### Médio Prazo

- [ ] Adicionar autenticação ao MLflow UI
- [ ] Migrar artifacts para S3/MinIO
- [ ] CI/CD para fine-tuning automático

### Longo Prazo

- [ ] Deploy MLflow em produção (servidor dedicado)
- [ ] A/B testing de modelos
- [ ] Alertas automáticos (Slack/email)

---

## 🙏 Contribuindo

Para reportar bugs ou sugerir melhorias na integração MLflow, abra uma issue ou PR.

---

## 📄 Licença

Mesma licença do projeto TDvX principal.

---

**Versão**: 1.0  
**Data**: 2026-04-28  
**Mantido por**: TDvX Team
