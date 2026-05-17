# TDvX Labeler

Interface web para transcrição automática + validação/rotulação de áudio e vídeo.

## Fluxo

1. Faz upload de vídeo ou áudio (MP4, MOV, MP3, WAV, etc.)
2. Extrai o áudio e transcreve automaticamente com o modelo TDvX CT2
3. Exibe waveform com segmentos clicáveis — edite o texto, ajuste os tempos
4. Exporta pares `seg_XXXX.wav` + `seg_XXXX.txt` prontos para o próximo treino

## Requisitos

- ffmpeg instalado no sistema
- Modelo CT2 em `tdvx/models/tdvx-v4-cv-pt-ct2` (ou `tdv3-cv-pt-v1-ct2`)

## Instalação e uso

```bash
cd tdvx/labeler
pip install -r requirements-labeler.txt

python app.py
# Abrir http://localhost:7860
```

Porta diferente:
```bash
PORT=8080 python app.py
```

## Saída dos exports

Os segmentos validados são salvos em `labeler/exports/<job_id>/`:

```
exports/
  a1b2c3d4/
    seg_0000.wav   ← áudio do segmento
    seg_0000.txt   ← transcrição validada
    seg_0001.wav
    seg_0001.txt
    ...
```

Esses pares podem ser adicionados diretamente ao corpus para novo treino.
