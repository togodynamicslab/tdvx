# STT Pipeline — Speech-to-Text + Diarização Local

100% local. O áudio nunca sai da máquina.

---

## Pré-requisitos

- Python 3.10+
- Conta HuggingFace com os termos aceitos em:
  - https://hf.co/pyannote/speaker-diarization-3.1
  - https://hf.co/pyannote/embedding

---

## Setup (fazer apenas uma vez)

### Windows — GPU (NVIDIA)
```bat
setup_gpu.bat
```

### Windows — CPU (sem GPU)
```bat
setup_cpu.bat
```

### Linux
```bash
chmod +x setup_cpu.sh && ./setup_cpu.sh
# ou
chmod +x setup_gpu.sh && ./setup_gpu.sh
```

O setup cria o ambiente virtual (`venv_gpu` ou `venv_cpu`), instala todas as dependências e configura o `.env` com o token HuggingFace.

---

## Ativar o ambiente

```bat
# GPU (Windows)
venv_gpu\Scripts\activate.bat

# CPU (Windows)
venv_cpu\Scripts\activate.bat
```

```bash
# Linux
source venv_cpu/bin/activate
source venv_gpu/bin/activate
```

---

## Como rodar

### 1. Arquivo de áudio (WAV, MP3, FLAC…)

```bash
# CPU
python stt_cpu.py audio.wav --model small

# GPU
python stt_gpu.py audio.wav --model medium
```

### 2. Microfone — grava e transcreve

```bash
# Grava até pressionar ENTER, depois transcreve
python record_mic.py --model small

# Grava 30 segundos fixos
python record_mic.py --duration 30

# Ver microfones disponíveis
python record_mic.py --list-devices
```

### 3. Microfone — tempo real (texto aparece enquanto fala)

```bash
# Texto aparece em tempo real, speakers identificados ao final
python record_mic_live.py --model small

# Salvar o áudio gravado também
python record_mic_live.py --keep-wav
```

---

## Modelos disponíveis

| Modelo | RAM/VRAM | Velocidade CPU | Precisão |
|--------|----------|----------------|----------|
| `tiny` | ~390 MB | Muito rápido | Baixa |
| `base` | ~550 MB | Rápido | Média |
| `small` | ~1 GB | Médio | Boa |
| `medium` | ~3 GB | Lento | Muito boa |
| `large-v3` | ~6 GB | Muito lento | Máxima |

> Para uso em tempo real no CPU, recomendado: `small` ou `tiny`.

---

## Saída JSON

```json
{
  "segments": [
    {
      "timestamp_start": 1.2,
      "timestamp_end": 4.8,
      "user_id": "Speaker 0",
      "text": "Bom dia, tudo bem?",
      "language_detected": "pt",
      "confidence": 0.87
    }
  ],
  "speaker_registry": { "Speaker 0": [...] },
  "metadata": { "unique_speakers": 2, "total_segments": 12 }
}
```

---

## Modo offline (após 1º download dos modelos)

Descomente no `.env`:

```
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Com isso, zero chamadas de rede em qualquer execução.
