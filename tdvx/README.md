# TDvX v3 — Transcrição + Diarização + Finetuning Dataset

Serviço de transcrição de áudio com identificação de falantes, salvamento automático de dataset para finetuning e interface web embutida.

**Stack:** FastAPI · Faster-Whisper (CTranslate2) · Pyannote 3.1 · WebRTC VAD · WebSocket

---

## Como Rodar

### Pré-requisitos

- Python 3.10+
- FFmpeg: `winget install ffmpeg`
- Token Hugging Face com termos aceitos em:
  - https://huggingface.co/pyannote/speaker-diarization-3.1
  - https://huggingface.co/pyannote/embedding

### Setup (uma vez)

```bat
cd "C:\...\voice_model"

:: GPU (CUDA 12.8) — recomendado
setup_gpu.bat

:: ou CPU
setup_cpu.bat
```

Configure o token no `.env` da raiz do repo:
```env
PYANNOTE_AUTH_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
WHISPER_COMPUTE_TYPE=int8_float16
```

### Subir o servidor

```bat
cd "C:\...\voice_model"
venv_gpu\Scripts\activate.bat
cd tdvx
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Um único comando sobe tudo: API REST, WebSocket e GUI.

---

## Usar só a API (sem browser)

Acesse os endpoints diretamente via curl, Python ou qualquer cliente HTTP/WebSocket.

### Health check
```bash
curl http://localhost:8000/health
```

```json
{
  "status": "running",
  "model": "medium",
  "compute_type": "int8_float16",
  "device": "cuda",
  "diarization": true,
  "finetuning_enabled": true
}
```

### Transcrever arquivo (upload)
```bash
curl -X POST http://localhost:8000/transcribe \
     -F "file=@audio.wav"
```

### Transcrever arquivo local (path no servidor)
```bash
curl "http://localhost:8000/transcribe?path=C:/audios/reuniao.wav"
```

### Estatísticas do dataset de finetuning
```bash
curl http://localhost:8000/finetuning/stats
```

### Remover entrada do dataset
```bash
curl -X DELETE http://localhost:8000/finetuning/{entry_id}
```

### WebSocket — stream PCM float32 (clientes nativos)
```
ws://localhost:8000/ws/transcribe
```
- Enviar chunks `float32 little-endian, 16 kHz, mono`
- Enviar `b""` (bytes vazio) para sinalizar fim de stream
- Receber JSON por segmento em tempo real

```python
# Exemplo com websockets
import asyncio, websockets, numpy as np, librosa, json

async def transcribe(path):
    audio, _ = librosa.load(path, sr=16000, mono=True)
    async with websockets.connect("ws://localhost:8000/ws/transcribe") as ws:
        chunk = 4096
        for i in range(0, len(audio), chunk):
            await ws.send(audio[i:i+chunk].astype("float32").tobytes())
        await ws.send(b"")  # fim de stream
        async for msg in ws:
            print(json.loads(msg))

asyncio.run(transcribe("audio.wav"))
```

### WebSocket — stream WebM/Opus (browser)
```
ws://localhost:8000/transcribe/live
```

---

## Usar a GUI (browser)

Com o servidor rodando, abra no browser:

| URL | O que faz |
|---|---|
| `http://localhost:8000/upload.html` | Upload de arquivo — transcreve e mostra resultado + info do dataset |
| `http://localhost:8000/live` | Transcrição ao vivo via microfone |
| `http://localhost:8000/finetuning/stats` | JSON com estatísticas do dataset |
| `http://localhost:8000/docs` | Documentação interativa (Swagger UI) |

---

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status do serviço |
| `GET` | `/models` | Modelo ativo e quantização |
| `POST` | `/transcribe` | Upload de arquivo → transcrição + salva dataset |
| `GET` | `/transcribe?path=` | Arquivo local → transcrição + salva dataset |
| `GET` | `/finetuning/stats` | Estatísticas do dataset acumulado |
| `DELETE` | `/finetuning/{id}` | Remove entrada do dataset (WAV + manifest) |
| `GET` | `/upload.html` | GUI de upload |
| `GET` | `/live` | GUI de transcrição ao vivo |
| `WS` | `/ws/transcribe` | Stream PCM float32 16kHz |
| `WS` | `/transcribe/live` | Stream WebM/Opus do browser |

---

## Resposta de transcrição

```json
{
  "timestamp": "2025-03-17T14:30:00.123456",
  "language": "pt",
  "duration": 45.2,
  "segments": [
    {
      "speaker": "Speaker 0",
      "start": 0.500,
      "end": 3.200,
      "text": "Bom dia, como você está?",
      "confidence": 0.94
    },
    {
      "speaker": "Speaker 1",
      "start": 3.500,
      "end": 6.800,
      "text": "Estou bem, obrigado.",
      "confidence": 0.97
    }
  ],
  "finetuning": {
    "session_id": "abc123def456",
    "entries_saved": 2,
    "entry_ids": ["abc123def456_Speaker0_0.500_3.200", "abc123def456_Speaker1_3.500_6.800"]
  }
}
```

O campo `finetuning` aparece quando `FINETUNING_ENABLED=true` (padrão).

---

## Dataset de Finetuning

Cada transcrição salva automaticamente em `finetuning_data/`:

```
finetuning_data/
├── audio/{session_id}/
│   ├── Speaker0_0.500_3.200.wav    ← segmento cortado por speaker
│   └── Speaker1_3.500_6.800.wav
├── labels/{session_id}.json        ← transcrição completa da sessão
└── manifest.jsonl                  ← uma linha JSON por segmento
                                       (compatível com HuggingFace / NeMo / Whisper)
```

Formato de cada linha do `manifest.jsonl`:
```json
{
  "id": "abc123_Speaker0_0.500_3.200",
  "audio_filepath": "audio/abc123/Speaker0_0.500_3.200.wav",
  "text": "Bom dia, como você está?",
  "speaker": "Speaker 0",
  "duration": 2.7,
  "confidence": 0.94,
  "language": "pt",
  "source": "reuniao.wav",
  "session_id": "abc123def456",
  "timestamp": "2025-03-17T14:30:00"
}
```

Para desativar: `FINETUNING_ENABLED=false` no `.env`.

---

## Configuração (`.env` na raiz do repo)

```env
# OBRIGATÓRIO
PYANNOTE_AUTH_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Whisper
WHISPER_MODEL=medium               # small | medium | large-v3
WHISPER_COMPUTE_TYPE=int8_float16  # int8_float16 (GPU) | int8 (CPU) | auto

# Diarização
PYANNOTE_MIN_SPEAKERS=1
PYANNOTE_MAX_SPEAKERS=10
PYANNOTE_CLUSTERING_THRESHOLD=0.5

# VAD
ENABLE_VAD=true
VAD_AGGRESSIVENESS=1               # 0-3

# Finetuning
FINETUNING_ENABLED=true
# FINETUNING_DATA_DIR=             # vazio = tdvx/finetuning_data
```

---

## Fine-tuning do modelo proprietário

### Como os dados são coletados

Toda transcrição feita pelo app salva automaticamente os segmentos em `finetuning_data/` (relativo à raiz do serviço):

```
finetuning_data/
├── audio/
│   └── {session_id}/
│       ├── Speaker0_0.500_3.200.wav   ← segmento cortado por speaker, 16kHz mono PCM
│       └── Speaker1_3.500_6.800.wav
├── labels/
│   └── {session_id}.json              ← transcrição completa da sessão
└── manifest.jsonl                     ← uma linha JSON por segmento (índice global)
```

Cada linha do `manifest.jsonl`:
```json
{
  "id": "abc123_Speaker0_0.500_3.200",
  "audio_filepath": "audio/abc123/Speaker0_0.500_3.200.wav",
  "text": "Bom dia, como você está?",
  "speaker": "Speaker 0",
  "duration": 2.7,
  "confidence": 0.94,
  "language": "pt",
  "source": "reuniao.wav",
  "session_id": "abc123def456",
  "timestamp": "2025-03-17T14:30:00"
}
```

Para desativar a coleta: `FINETUNING_ENABLED=false` no `.env`.

---

### Rodando o fine-tuning (após fechar a sprint)

#### 1. Instale as dependências de treino

```bash
pip install -r ../requirements-finetune.txt
```

#### 2. Verifique o dataset antes de treinar

```bash
python finetune.py --dry-run
```

Saída esperada:
```
Manifest: 1.240 entradas carregadas, 0 ignoradas
Filtro: 1.187/1.240 entradas mantidas | descartadas → {'duration': 31, 'confidence': 22, ...}
Distribuição de idiomas: {'portuguese': 980, 'english': 207}
[dry-run] 1.187 entradas prontas para treino. Encerrando.
```

#### 3. Treine

```bash
# Padrão: whisper-medium, PT + EN, 3 épocas
python finetune.py

# Modelo de maior qualidade (TDv1)
python finetune.py --base-model openai/whisper-large-v3

# Limitar steps (útil para testar o pipeline)
python finetune.py --max-steps 500

# Especificar onde estão os dados e onde salvar
python finetune.py \
  --data-dir /dados/finetuning_data \
  --output-dir /modelos/tdv1-v2
```

#### 4. Onde o modelo é salvo

```
models/
├── tdv1-finetuned/          ← modelo HuggingFace (usado para re-treinos futuros)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── tdv1-finetuned-ct2/      ← modelo convertido para CTranslate2 (usado pelo app)
    ├── model.bin
    ├── config.json
    └── vocabulary.json
```

A conversão para CTranslate2 roda automaticamente ao final do treino.

#### 5. Ativar o modelo fine-tunado no app

Em [app/services/engine.py](app/services/engine.py), passe o caminho do diretório `-ct2` para o `STTEngine`:

```python
# Antes (modelo base do HuggingFace Hub)
self.model = WhisperModel("medium", device=self.device, ...)

# Depois (modelo proprietário fine-tunado)
self.model = WhisperModel("models/tdv1-finetuned-ct2", device=self.device, ...)
```

Ou via variável de ambiente no `.env`:
```env
WHISPER_MODEL=models/tdv1-finetuned-ct2
```

---

### MLflow — tracking de experimentos

O fine-tuning integra com MLflow para registrar parâmetros, métricas por step e versionar o modelo no Model Registry.

#### Subir o servidor MLflow

```bash
# Instala (junto com as outras deps de treino)
pip install -r ../requirements-finetune.txt

# Sobe o servidor (deixe rodando em segundo plano)
mlflow server --host 0.0.0.0 --port 5000

# UI disponível em: http://localhost:5000
```

Configure no `.env`:
```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_FINETUNE_EXPERIMENT=tdv1-finetune
```

#### O que é registrado automaticamente

| O quê | Onde aparece na UI |
|---|---|
| Parâmetros de treino (lr, batch, epochs...) | aba **Parameters** |
| Stats do dataset (total, duração, idiomas) | aba **Parameters** |
| `train/loss` a cada 25 steps | aba **Metrics** (gráfico) |
| `eval/loss` e `eval/wer` a cada `--eval-steps` | aba **Metrics** (gráfico) |
| WER final | aba **Metrics** |
| `eval_results.json` | aba **Artifacts** |
| Modelo registrado (se `--model-name`) | seção **Models** |

#### Rodar com tracking completo

```bash
python finetune.py \
    --run-name tdv1-sprint-4 \
    --model-name tdv1-pt-en \
    --hub-model-id minha-empresa/tdv1-pt-en-v2
```

Cada sprint gera uma nova versão no Model Registry (`v1 → v2 → v3`), comparável via UI.

#### Promover uma versão para Production

Na UI do MLflow (`http://localhost:5000`), aba **Models → tdv1-pt-en**:
1. Selecione a versão desejada
2. Clique em **Stage → Production**

Ou via código:
```python
from mlflow import MlflowClient
client = MlflowClient("http://localhost:5000")
client.transition_model_version_stage(
    name="tdv1-pt-en", version="3", stage="Production"
)
```

---

### Referência de parâmetros CLI

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--data-dir` | `finetuning_data/` | Diretório com `manifest.jsonl` |
| `--output-dir` | `models/tdv1-finetuned/` | Saída do modelo HuggingFace |
| `--base-model` | `openai/whisper-medium` | Modelo base (`openai/whisper-large-v3` para TDv1) |
| `--languages` | `portuguese english` | Idiomas a incluir no treino |
| `--max-steps` | `0` (usa épocas) | Máximo de steps de treino |
| `--num-epochs` | `3` | Épocas de treino |
| `--batch-size` | `8` | Amostras por batch por GPU |
| `--min-confidence` | `0.40` | Confiança mínima para usar como ground-truth |
| `--eval-steps` | `200` | Frequência de avaliação e checkpoint |
| `--mlflow-uri` | `MLFLOW_TRACKING_URI` do `.env` | URI do servidor MLflow |
| `--mlflow-experiment` | `tdv1-finetune` | Nome do experimento |
| `--run-name` | `tdv1-YYYYMMDD-HHMM` | Nome do run (ex.: `tdv1-sprint-4`) |
| `--model-name` | — | Nome no Model Registry (ex.: `tdv1-pt-en`) |
| `--hub-model-id` | — | `usuario/repo` no HuggingFace Hub |
| `--dry-run` | — | Valida o dataset sem treinar |

---

## Versionamento do modelo

Os arquivos de modelo são grandes demais para o git (centenas de MB a vários GB). A estratégia adotada usa o **HuggingFace Hub** como repositório de modelos, já que o token HF já está configurado no projeto.

### Convenção de nomes

```
{org}/tdv1-{idiomas}-v{N}
```

Exemplos:
- `minha-empresa/tdv1-pt-en-v1` — primeiro modelo treinado
- `minha-empresa/tdv1-pt-en-v2` — após a sprint 2

### Publicar após o treino

```bash
python finetune.py --hub-model-id minha-empresa/tdv1-pt-en-v2
```

O script faz push do modelo HuggingFace (não do CTranslate2) para um repositório **privado** no Hub, criando um commit rastreável com as métricas de WER.

### Carregar uma versão específica no app

```python
# engine.py — carrega direto do Hub (requer HF_TOKEN no .env)
self.model = WhisperModel(
    "minha-empresa/tdv1-pt-en-v2",
    device=self.device,
    compute_type=self.compute_type,
)
```

Ou baixe localmente e use o caminho:
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('minha-empresa/tdv1-pt-en-v2', local_dir='models/tdv1-v2-ct2')
"
```

### Criar repositório privado (uma vez)

```bash
python -c "
from huggingface_hub import HfApi
HfApi().create_repo('tdv1-pt-en-v1', private=True, repo_type='model')
"
```

### Histórico de versões

Mantenha um `MODEL_VERSIONS.md` na raiz do repo git para registrar o que mudou em cada versão:

```markdown
## v2 — Sprint 4 (2025-04-30)
- hub: minha-empresa/tdv1-pt-en-v2
- base: openai/whisper-medium
- dados: 1.187 segmentos PT + 207 EN (14h de áudio)
- WER: PT 6.2% | EN 8.1%
- mudanças: primeira versão bilíngue

## v1 — Sprint 2 (2025-03-15)
- hub: minha-empresa/tdv1-pt-en-v1
- base: openai/whisper-medium
- dados: 543 segmentos PT (6h de áudio)
- WER: PT 9.4%
```

---

## Ferramentas incluídas

| Arquivo | O que faz |
|---|---|
| `finetune.py` | Fine-tuning do modelo TDv1 a partir do dataset acumulado |
| `benchmark.py` | Mede RTF e qualidade do pipeline em um arquivo de áudio |
| `test_live_transcription.py` | Testa o WebSocket enviando um arquivo em chunks |
| `example_client.py` | Cliente Python de exemplo para WebSocket |
| `load_test/locustfile.py` | Teste de carga HTTP + WebSocket (requer `locust`) |

```bash
# Fine-tuning (após fechar sprint)
python finetune.py --dry-run          # valida dataset
python finetune.py                    # treina e converte

# Benchmark
python benchmark.py audio.wav

# Teste WebSocket
python test_live_transcription.py audio.wav

# Teste de carga (interface web em :8089)
locust -f load_test/locustfile.py --host=http://localhost:8000
```

---

## Estrutura

```
tdvx/
├── app/
│   ├── main.py              ← FastAPI: todas as rotas
│   ├── config.py            ← Settings (lê .env da raiz do repo)
│   ├── models/
│   │   ├── response.py      ← TranscriptionResponse, FinetuningInfo
│   │   └── model_config.py  ← ModelConfig
│   └── services/
│       ├── engine.py        ← STTEngine, DiarizationEngine, SpeakerIndexer
│       ├── processor.py     ← TranscriptionProcessor (orquestra pipeline)
│       ├── saver.py         ← FinetuningDatasetSaver
│       ├── audio_buffer.py  ← SpeechBuffer com WebRTC VAD
│       └── vad_service.py   ← VADService
├── static/
│   ├── upload.html          ← GUI de upload
│   └── index.html           ← GUI de transcrição ao vivo
├── finetuning_data/         ← dataset acumulado (audio + labels + manifest)
├── load_test/
│   └── locustfile.py        ← teste de carga Locust
├── benchmark.py
├── example_client.py
├── test_live_transcription.py
├── Dockerfile
└── docker-compose.yml
```

---

## Pipeline de Processamento

```
Audio (float32, 16kHz, mono)
  │
  ▼  1. STT
  STTEngine (Faster-Whisper medium, int8_float16)
  → [{start, end, text, confidence}]  +  language
  │
  ▼  2. Diarização
  DiarizationEngine (Pyannote 3.1)
  → [{start, end, SPEAKER_00 | SPEAKER_01 | ...}]
  │
  ▼  3. Re-ID de Speaker
  SpeakerIndexer (embeddings 192-d, cosine similarity >= 0.80)
  → "Speaker 0" / "Speaker 1" / ...
  │
  ▼  4. Merge temporal + filtro de alucinações
  TranscriptionProcessor
  → TranscriptionResponse JSON
  │
  ▼  5. Finetuning (opcional, FINETUNING_ENABLED=true)
  FinetuningDatasetSaver
  → WAV por segmento + manifest.jsonl
```
