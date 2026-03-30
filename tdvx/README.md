# TDvX — Transcrição com Diarização de Speaker

API de transcrição de áudio com identificação de falantes, construída com FastAPI, Faster-Whisper e Pyannote.

---

## Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Pipeline de Processamento](#pipeline-de-processamento)
- [Rotas da API](#rotas-da-api)
- [Modelos Utilizados](#modelos-utilizados)
- [Configuração](#configuração)
- [Como Rodar](#como-rodar)
- [Estrutura de Arquivos](#estrutura-de-arquivos)

---

## Visão Geral

O TDvX é um serviço que recebe um arquivo de áudio e retorna a transcrição completa com:

- **Texto** de cada fala
- **Identificação do falante** (`Speaker 0`, `Speaker 1`, etc.)
- **Timestamps** de início e fim de cada segmento
- **Confiança** da transcrição (0.0 a 1.0)
- **Idioma** detectado automaticamente

---

## Arquitetura

```
                    ┌──────────────────────────────────────────┐
                    │              TDvX FastAPI                │
                    │                                          │
   Audio File  ───► │  POST /transcribe                        │
                    │        │                                 │
                    │        ▼                                 │
                    │  ┌─────────────┐    ┌──────────────────┐ │
                    │  │  STTEngine  │    │ DiarizationEngine│ │
                    │  │  (Whisper)  │    │   (Pyannote)     │ │
                    │  └──────┬──────┘    └────────┬─────────┘ │
                    │         │                    │           │
                    │         ▼                    ▼           │
                    │  segments[]           speaker turns[]    │
                    │  [{start,end,text}]   [{start,end,who}]  │
                    │         │                    │           │
                    │         └──────────┬─────────┘           │
                    │                   ▼                      │
                    │         TranscriptionProcessor           │
                    │         (merge por overlap temporal)     │
                    │                   │                      │
                    └───────────────────┼──────────────────────┘
                                        ▼
                             TranscriptionResponse JSON
```

### Componentes Principais

| Componente | Arquivo | Responsabilidade |
|---|---|---|
| `STTEngine` | `app/services/engine.py` | Wrapper do Faster-Whisper. Transcreve áudio → segmentos de texto com timestamps |
| `DiarizationEngine` | `app/services/engine.py` | Wrapper do Pyannote. Detecta quem fala quando |
| `SpeakerIndexer` | `app/services/engine.py` | Mantém registro de embeddings de voz. Mapeia IDs brutos para "Speaker 0", "Speaker 1" etc. |
| `TranscriptionProcessor` | `app/services/processor.py` | Orquestra as 4 etapas do pipeline. Faz o merge temporal entre STT e diarização |
| `Settings` | `app/config.py` | Lê configurações do `.env` via Pydantic Settings |

---

## Pipeline de Processamento

O processamento ocorre em 4 etapas sequenciais:

### Etapa 1 — Speech-to-Text (Whisper)

```
audio (numpy float32, 16kHz) ──► STTEngine.transcribe()
                                        │
                                        ▼
                           segments: [{
                             start: float,
                             end: float,
                             text: str,
                             confidence: float
                           }]
                           language: str  ("pt", "en", etc.)
```

- **Modelo**: `faster-whisper/medium` (CTranslate2 backend)
- **Quantização**: `int8_float16` (GPU) ou `int8` (CPU)
- **Velocidade**: ~4–6s por 10s de áudio

### Etapa 2 — Diarização (Pyannote)

```
waveform dict ──► DiarizationEngine.diarize()
                          │
                          ▼
             turns: [{
               start: float,
               end: float,
               raw_speaker: "SPEAKER_00" | "SPEAKER_01" | ...
             }]
```

- **Modelo**: `pyannote/speaker-diarization-3.1`
- **Input**: `{"waveform": Tensor(1, T), "sample_rate": 16000}`
- **Clustering threshold**: `0.5` (configurável em `.env`)

### Etapa 3 — Mapeamento de Speaker por Embedding

```
Para cada raw_speaker único:
  1. Pega o turn mais longo desse speaker
  2. Extrai embedding de 192 dimensões (pyannote/embedding)
  3. SpeakerIndexer.identify(embedding):
     - Se similaridade coseno ≥ 0.80 com centroide existente → reutiliza ID
     - Caso contrário → cria "Speaker N" novo
```

- **Objetivo**: Mapeamento estável de speakers entre chamadas
- **Threshold**: `0.80` (configurável via `SPEAKER_SIMILARITY_THRESHOLD`)

### Etapa 4 — Merge Temporal

```
Para cada segmento Whisper [start, end]:
  - Calcula overlap com cada turn de diarização
  - Atribui ao speaker com maior overlap
  - Filtra alucinações do Whisper (chars/palavras repetidas)
```

---

## Rotas da API

### `GET /health`

Verifica se o serviço está ativo.

**Resposta:**
```json
{
  "status": "running",
  "model": "medium",
  "device": "cuda",
  "diarization": true
}
```

---

### `GET /models`

Retorna a configuração do modelo ativo.

**Resposta:**
```json
{
  "active_model": "tdv1-fast",
  "name": "TDv1-Fast",
  "whisper_model": "medium",
  "description": "Pipeline rápido com Faster-Whisper Medium",
  "estimated_speed": "~4-6s por 10s de áudio",
  "quantization": "int8_float16 (GPU) / int8 (CPU)"
}
```

---

### `POST /transcribe`

Recebe um arquivo de áudio e retorna a transcrição completa com diarização.

**Request:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3"
```

**Formatos aceitos:** WAV, MP3, M4A, FLAC, OGG (via FFmpeg)

**Limite de tamanho:** 100MB (configurável via `MAX_AUDIO_FILE_SIZE_MB`)

**Resposta:**
```json
{
  "timestamp": "2025-03-17T14:30:00.123456",
  "language": "pt",
  "duration": 45.2,
  "segments": [
    {
      "speaker": "Speaker 0",
      "start": 0.5,
      "end": 3.2,
      "text": "Bom dia, como você está?",
      "confidence": 0.94
    },
    {
      "speaker": "Speaker 1",
      "start": 3.5,
      "end": 6.8,
      "text": "Estou bem, obrigado.",
      "confidence": 0.97
    }
  ]
}
```

**Campos da resposta:**

| Campo | Tipo | Descrição |
|---|---|---|
| `timestamp` | datetime | Momento do processamento |
| `language` | string | Código ISO 639-1 do idioma detectado |
| `duration` | float | Duração total do áudio em segundos |
| `segments[].speaker` | string | Identificador do falante ("Speaker 0", "Speaker 1", ...) |
| `segments[].start` | float | Início do segmento em segundos |
| `segments[].end` | float | Fim do segmento em segundos |
| `segments[].text` | string | Texto transcrito |
| `segments[].confidence` | float | Confiança de 0.0 a 1.0 |

---

### `GET /upload.html`

Serve a interface web para upload de arquivos.

---

### `GET /`

Redireciona para `/upload.html`.

---

## Modelos Utilizados

| Modelo | Finalidade | Parâmetros | VRAM (GPU) |
|---|---|---|---|
| `faster-whisper/medium` | Speech-to-Text | ~769M | ~3GB (int8_float16) |
| `pyannote/speaker-diarization-3.1` | Detecção de falantes | ~80M | incluído acima |
| `pyannote/embedding` | Fingerprint de voz (192-d) | ~29M | incluído acima |

**Seleção de dispositivo:** automática. Usa CUDA se disponível, CPU caso contrário.

**Quantização:**
- GPU: `int8_float16` — pesos em int8, ativações em float16 (2× menos VRAM)
- CPU: `int8` — pesos e ativações em int8 (4× menos RAM)

---

## Configuração

Crie um arquivo `.env` na raiz do projeto (pasta `tdvx/`):

```env
# OBRIGATÓRIO — token do Hugging Face para baixar modelos Pyannote
# Aceite os termos em: https://huggingface.co/pyannote/speaker-diarization-3.1
PYANNOTE_AUTH_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

# Servidor
HOST=0.0.0.0
PORT=8000

# Whisper
WHISPER_MODEL=medium         # small | medium | large-v3
CPU_THREADS=0                # 0 = auto (os.cpu_count())

# Limites de arquivo
MAX_AUDIO_FILE_SIZE_MB=100

# VAD (Voice Activity Detection)
ENABLE_VAD=true
VAD_AGGRESSIVENESS=1         # 0=permissivo, 3=agressivo

# Thresholds
SPEAKER_SIMILARITY_THRESHOLD=0.80    # similaridade coseno para re-ID de speaker
MIN_SEGMENT_CONFIDENCE=0.30          # confiança mínima do Whisper

# Pyannote — Diarização
PYANNOTE_MIN_SPEAKERS=1      # 0 ou 1 = auto-detect
PYANNOTE_MAX_SPEAKERS=10
PYANNOTE_CLUSTERING_THRESHOLD=0.5    # menor = mais splits de speaker
PYANNOTE_LIVE_CLUSTERING_THRESHOLD=0.65
```

---

## Como Rodar

### Pré-requisitos

- Python 3.10+
- FFmpeg instalado (`winget install ffmpeg` no Windows)
- Token do Hugging Face com acesso aos modelos Pyannote

### Instalação

```bash
cd tdvx
pip install -r requirements.txt
# Crie o .env com seu PYANNOTE_AUTH_TOKEN
```

### Execução

```bash
# Desenvolvimento (com reload automático)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Produção
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Nota:** Use `--workers 1` em produção com GPU. Múltiplos workers tentariam carregar o modelo duplicado na VRAM.

### Teste rápido

```bash
# Health check
curl http://localhost:8000/health

# Transcrever arquivo
curl -X POST http://localhost:8000/transcribe -F "file=@seu_audio.wav"
```

---

## Estrutura de Arquivos

```
tdvx/
├── app/
│   ├── main.py                  # FastAPI — rotas e startup
│   ├── config.py                # Settings via Pydantic (lê .env)
│   ├── models/
│   │   ├── response.py          # TranscriptionResponse, TranscriptionSegment
│   │   └── model_config.py      # Configuração dos modelos disponíveis
│   └── services/
│       ├── engine.py            # STTEngine, DiarizationEngine, SpeakerIndexer
│       ├── processor.py         # TranscriptionProcessor (orquestra o pipeline)
│       ├── audio_buffer.py      # SpeechBuffer, AudioBuffer (chunking)
│       └── vad_service.py       # VADService (WebRTC VAD)
├── static/
│   └── upload.html              # Interface web de upload
├── requirements.txt
├── .env                         # Variáveis de ambiente (não versionar)
└── README.md
```

---

## Filtro de Alucinações

O Whisper às vezes gera texto repetitivo. O pipeline filtra automaticamente segmentos que:

- Têm 7+ caracteres iguais consecutivos (ex: `"aaaaaaa"`)
- Têm a mesma palavra repetida 5+ vezes (ex: `"obrigado obrigado obrigado..."`)
- Têm 6+ palavras e todas são iguais

---

## Performance

| Cenário | Tempo por 10s de áudio |
|---|---|
| GPU (RTX 3060+) | 4–6s |
| CPU (8 cores) | 20–40s |
| Overhead diarização | +2–5s fixo por arquivo |

**RTF** (Real-Time Factor): quanto tempo de processamento por segundo de áudio. RTF < 1 = mais rápido que real-time.
