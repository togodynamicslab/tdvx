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

## Ferramentas incluídas

| Arquivo | O que faz |
|---|---|
| `benchmark.py` | Mede RTF e qualidade do pipeline em um arquivo de áudio |
| `test_live_transcription.py` | Testa o WebSocket enviando um arquivo em chunks |
| `example_client.py` | Cliente Python de exemplo para WebSocket |
| `load_test/locustfile.py` | Teste de carga HTTP + WebSocket (requer `locust`) |

```bash
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
