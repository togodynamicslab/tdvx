# TDvX — Motor de Transcrição pt-BR

Pipeline completo de transcrição automática de fala em português brasileiro, baseado em Whisper fine-tunado e convertido para CTranslate2 int8.

## Estrutura

```
tdvx/
├── eval/                    # Avaliação de WER + tracking MLflow
│   ├── synthetic_eval.py    # Pipeline principal de avaliação
│   ├── texts_pt_br.txt      # 30 frases de teste pt-BR
│   └── requirements-eval.txt
├── labeler/                 # Interface web de rotulação
│   ├── app.py               # API FastAPI + frontend WaveSurfer
│   ├── static/              # HTML/CSS/JS da interface
│   └── requirements-labeler.txt
├── models/
│   └── tdvx-v4-cv-pt-ct2/  # Modelo Whisper medium int8 (740 MB)
└── .venv/                   # Ambiente virtual (criar com setup abaixo)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r eval/requirements-eval.txt
```

> O `requirements-eval.txt` inclui todas as dependências do labeler também (faster-whisper, librosa, soundfile).

---

## Labeler

Interface web para transcrever e rotular áudios/vídeos. Gera pares `WAV + TXT` por segmento, prontos para retreino.

```bash
source .venv/bin/activate
python labeler/app.py
# Abrir http://localhost:7860
```

**Fluxo:**
1. Upload de vídeo ou áudio (qualquer formato)
2. Extração automática de áudio WAV mono 16 kHz via ffmpeg
3. Transcrição com o modelo TDvX v4
4. Edição de segmentos no waveform interativo
5. Export por segmento em `labeler/exports/<job_id>/`

**Requisitos de sistema:** `ffmpeg` instalado e no PATH.

---

## Avaliação com MLflow

Pipeline que gera áudio sintético via TTS (edge-tts), aplica 12 tipos de ruído/degradação e mede o WER do modelo. Os resultados são registrados no MLflow.

### Rodar avaliação

```bash
source .venv/bin/activate

# Avaliação completa (30 frases × 12 ruídos = 360 inferências)
python eval/synthetic_eval.py \
  --model-path models/tdvx-v4-cv-pt-ct2 \
  --texts-file eval/texts_pt_br.txt \
  --mlflow-uri sqlite:///mlflow.db \
  --mlflow-experiment tdvx-eval \
  --run-name tdvx-v4-noise-bench

# Teste rápido (5 frases, só clean)
python eval/synthetic_eval.py \
  --max-texts 5 \
  --noise-types clean \
  --mlflow-uri sqlite:///mlflow.db
```

### Ver resultados na UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Abrir http://127.0.0.1:5000
```

### Tipos de ruído testados

| Tipo | Descrição |
|---|---|
| `clean` | Áudio limpo (baseline) |
| `white_noise_high/mid/low` | Ruído branco com SNR 20/10/5 dB |
| `pink_noise` | Ruído rosa — fundo ambiente |
| `reverb_small` / `reverb_large` | Reverb de sala pequena / grande |
| `low_bitrate` | Compressão telefônica (8 kHz → 16 kHz) |
| `speed_slow` / `speed_fast` | Fala 20% mais lenta / mais rápida |
| `pitch_down` / `pitch_up` | Pitch −3 / +3 semitons |

### Métricas registradas no MLflow

| Métrica | Descrição |
|---|---|
| `wer_<noise_type>` | WER médio por tipo de ruído |
| `wer_overall` | WER médio global |
| `inference_time_<noise_type>` | Tempo médio de inferência por ruído |
| `inference_time_mean_s` | Tempo médio global |
| `pct_perfeito` / `pct_bom` / `pct_ruim` | Distribuição de qualidade (%) |

### Resultado de referência — TDvX v4 (CPU, 5 frases)

| Ruído | WER |
|---|---|
| clean, white_noise_*, pink_noise, reverb_small, low_bitrate, speed_*, pitch_* | 0% |
| reverb_large | ~17% |

### Opções do CLI

```
--model-path        Caminho do modelo CT2 (padrão: Systran/faster-whisper-medium)
--texts-file        Arquivo .txt com frases (uma por linha, # ignora)
--noise-types       Tipos de ruído a testar (padrão: todos os 12)
--max-texts         Limitar número de frases (útil para testes rápidos)
--save-audio        Salvar áudios gerados em eval/audio/
--output-dir        Diretório dos relatórios (padrão: eval/results/)
--device            auto | cuda | cpu
--tts-voice         Voz edge-tts (padrão: pt-BR-FranciscaNeural)
--mlflow-uri        URI do MLflow (ex.: sqlite:///mlflow.db ou http://localhost:5000)
--mlflow-experiment Nome do experimento (padrão: tdvx-synthetic-eval)
--run-name          Nome do run (padrão: eval-<timestamp>)
```

---

## Modelo

**TDvX v4** — Whisper medium fine-tunado em pt-BR, convertido para CTranslate2 int8.

| Atributo | Valor |
|---|---|
| Base | openai/whisper-medium |
| Formato | CTranslate2 int8 |
| Tamanho | 740 MB |
| Idioma fixo | pt (pt-BR) |
| Backend | faster-whisper |

---

## Variáveis de ambiente (`.env`)

```env
# MLflow (opcional)
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EVAL_EXPERIMENT=tdvx-eval
```
