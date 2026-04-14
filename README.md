# TDvX — Voice Pipe API

Plataforma de transcrição de áudio com identificação de falantes e coleta de dataset para finetuning.
Roda 100% local — sem enviar áudio para serviços externos.

**Stack:** FastAPI · Faster-Whisper (CTranslate2) · Pyannote 3.1 · WebRTC VAD · WebSocket

---

## Estrutura

```
voice_model/
├── tdvx/               ← serviço principal (API + GUI + finetuning)
├── venv_gpu/           ← ambiente virtual GPU (criado pelo setup_gpu.bat)
├── requirements.txt    ← dependências Python (sem torch)
├── setup_gpu.bat       ← setup Windows GPU (CUDA 12.8)
├── setup_gpu.sh        ← setup Linux GPU
├── setup_cpu.bat       ← setup Windows CPU
├── setup_cpu.sh        ← setup Linux CPU
└── .env                ← variáveis de ambiente (não versionar)
```

---

## Setup (uma vez)

```bat
:: Windows + GPU NVIDIA
setup_gpu.bat

:: Windows + CPU
setup_cpu.bat
```

Configure o `.env`:
```env
PYANNOTE_AUTH_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
WHISPER_COMPUTE_TYPE=int8_float16
```

---

## Rodar

```bat
venv_gpu\Scripts\activate.bat
cd tdvx
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Usar só a API (curl / SDK)

```bash
# Health
curl http://localhost:8000/health

# Transcrever arquivo
curl -X POST http://localhost:8000/transcribe -F "file=@audio.wav"

# Transcrever arquivo local (path no servidor)
curl "http://localhost:8000/transcribe?path=C:/audios/reuniao.wav"

# Estatísticas do dataset de finetuning
curl http://localhost:8000/finetuning/stats
```

WebSocket PCM float32:
```
ws://localhost:8000/ws/transcribe
```

WebSocket WebM/Opus (browser):
```
ws://localhost:8000/transcribe/live
```

---

## Usar a GUI (browser)

| URL | O que faz |
|---|---|
| `http://localhost:8000/upload.html` | Upload de arquivo → transcrição + dataset |
| `http://localhost:8000/live` | Transcrição ao vivo via microfone |
| `http://localhost:8000/docs` | Swagger UI |

---

## Documentação completa

Ver [tdvx/README.md](tdvx/README.md)
