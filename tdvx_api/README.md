# tdvx_api — API de Finetuning

Versão standalone do TDvX focada em **coletar dados para finetuning**.

Cada áudio processado tem seus segmentos salvos automaticamente como pares `(WAV, label)`.

---

## Como Funciona

```
Audio enviado
     │
     ▼
Pipeline TDvX (Whisper + Pyannote)
     │
     ▼
Para cada segmento [{speaker, start, end, text}]:
  ├─ Corta o WAV original nos timestamps
  ├─ Salva em finetuning_data/audio/{session_id}/{speaker}_{start}_{end}.wav
  ├─ Salva labels completos em finetuning_data/labels/{session_id}.json
  └─ Adiciona linha no finetuning_data/manifest.jsonl
     │
     ▼
Retorna JSON com transcrição + IDs das entradas salvas
```

---

## Rotas

### `POST /transcribe`
Envia arquivo de áudio via upload.

```bash
curl -X POST http://localhost:8001/transcribe \
  -F "file=@audio.mp3"
```

### `GET /transcribe?path=...`
Processa arquivo já presente no servidor pelo caminho.

```bash
curl "http://localhost:8001/transcribe?path=/home/user/audios/reuniao.wav"
# Windows:
curl "http://localhost:8001/transcribe?path=C:/audios/entrevista.mp3"
```

### `GET /health`
Status do serviço e contagem do dataset.

### `GET /dataset/stats`
Estatísticas do dataset acumulado (speakers, duração, idiomas).

### `GET /dataset/manifest`
Download do `manifest.jsonl` completo.

### `DELETE /dataset/{entry_id}`
Remove uma entrada (apaga WAV + remove do manifest).

---

## Estrutura do Dataset Gerado

```
finetuning_data/
├── audio/
│   └── abc123def456/
│       ├── Speaker0_0.500_3.200.wav
│       ├── Speaker1_3.500_6.800.wav
│       └── Speaker0_7.100_9.500.wav
├── labels/
│   └── abc123def456.json          ← transcrição completa da sessão
└── manifest.jsonl                 ← uma linha por segmento (formato HuggingFace)
```

### Exemplo de linha do manifest.jsonl

```json
{
  "id": "abc123_Speaker0_0.500_3.200",
  "audio_filepath": "audio/abc123def456/Speaker0_0.500_3.200.wav",
  "text": "Bom dia, como você está?",
  "speaker": "Speaker 0",
  "duration": 2.7,
  "confidence": 0.94,
  "language": "pt",
  "source": "reuniao.mp3",
  "session_id": "abc123def456",
  "timestamp": "2025-03-17T14:30:00"
}
```

O `manifest.jsonl` é compatível com:
- **HuggingFace** `datasets.load_dataset("json", data_files="manifest.jsonl")`
- **NeMo** ASR training
- **Whisper** finetuning scripts

---

## Rodar

```bash
cd tdvx_api

# Instala apenas as dependências extras (o tdvx já deve estar instalado)
pip install -r requirements.txt

# Roda na porta 8001 (tdvx usa 8000)
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

> Requer que o `tdvx/` esteja na pasta pai com `.env` configurado (PYANNOTE_AUTH_TOKEN).
