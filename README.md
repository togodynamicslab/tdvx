# TDvX — Voice Pipe API

Plataforma de transcrição de áudio com identificação de falantes e coleta de dataset para finetuning.
Roda 100% local — sem enviar áudio para serviços externos.

**Stack:** FastAPI · Faster-Whisper (CTranslate2) · Pyannote 3.1 · WebRTC VAD · WebSocket

---

## Estrutura

```
voice_model/
├── tdvx/                         ← serviço principal (API + GUI + finetuning)
│   ├── app/                      ← FastAPI application
│   ├── finetuning_data/          ← dataset coletado automaticamente
│   ├── models/                   ← modelos fine-tunados
│   ├── finetune.py               ← script de fine-tuning
│   ├── docker-compose.yml        ← MLflow + PostgreSQL + TDvX
│   ├── .env.example              ← template de configuração
│   └── MLFLOW.md                 ← documentação completa do MLflow
├── run_finetune_with_mlflow.py   ← wrapper para fine-tuning com MLflow
├── start_mlflow.sh               ← inicia stack MLflow (Linux/Mac)
├── start_mlflow.bat              ← inicia stack MLflow (Windows)
├── venv_gpu/                     ← ambiente virtual GPU (criado pelo setup)
├── requirements.txt              ← dependências Python (sem torch)
├── requirements-finetune.txt     ← dependências de fine-tuning
├── setup_gpu.bat                 ← setup Windows GPU (CUDA 12.8)
├── setup_gpu.sh                  ← setup Linux GPU
├── setup_cpu.bat                 ← setup Windows CPU
├── setup_cpu.sh                  ← setup Linux CPU
└── .env                          ← variáveis de ambiente (não versionar)
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
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## MLflow Tracking (Opcional)

Para rastreamento de experimentos de fine-tuning:

```bash
# Inicia MLflow + PostgreSQL via Docker
./start_mlflow.sh        # Linux/Mac
start_mlflow.bat         # Windows

# Ou manualmente
cd tdvx
docker-compose up -d mlflow postgres
```

**UI**: http://localhost:5000

**Documentação completa**: [tdvx/MLFLOW.md](tdvx/MLFLOW.md)

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

## Fine-tuning (após fechar a sprint)

```bash
cd tdvx
pip install -r ../requirements-finetune.txt

python finetune.py --dry-run   # valida o dataset
python finetune.py             # treina + converte para CTranslate2

# Com publicação no HuggingFace Hub (versionamento)
python finetune.py --hub-model-id minha-empresa/tdv1-pt-en-v2
```

Os dados são coletados automaticamente em `tdvx/finetuning_data/` a cada transcrição.
O modelo treinado é salvo em `tdvx/models/tdv1-finetuned-ct2/`.

---

## Documentação completa

Ver [tdvx/README.md](tdvx/README.md)
