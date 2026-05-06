# TDvX — Fine-tuning Whisper pt-BR

Fine-tuning do `whisper-medium` no Mozilla Common Voice pt-BR, com augmentação de dados, amostras anti-alucinação e conversão CTranslate2 para produção.

---

## Requisitos da VM

| Item | Mínimo |
|---|---|
| GPU | NVIDIA RTX 5090 / A10 / A100 (16GB+ VRAM) |
| RAM | 32 GB |
| Disco | 150 GB (corpus ~70 GB extraído) |
| SO | Ubuntu 22.04+ |
| Python | 3.10+ (3.12 recomendado) |

---

## Passo a passo

### 1. Dependências do sistema

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip \
    ffmpeg git wget curl build-essential libsndfile1 libsndfile1-dev jq
```

### 2. Clonar o repositório

```bash
git clone -b finetune https://github.com/togodynamicslab/tdvx.git ~/tdvx
cd ~/tdvx
```

### 3. Ambiente Python

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. PyTorch com suporte à GPU

```bash
# RTX 5090 / Blackwell (sm_120) — requer CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Confirmar que a GPU foi reconhecida
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

> Para outras GPUs consulte: https://pytorch.org/get-started/locally

### 5. Dependências do projeto

```bash
pip install -r requirements-finetune.txt
```

### 6. Baixar o corpus Common Voice pt-BR

```bash
mkdir -p ~/tdvx/cvss

RESPONSE=$(curl -sf -X POST \
  "https://mozilladatacollective.com/api/datasets/cmn29f4cb017bmm07pd9yd8mw/download" \
  -H "Authorization: Bearer 200b0ff9a75977f821683b7ce76bdc8ccfd48477daa93228ff045f35e4d9214f" \
  -H "Content-Type: application/json")

DOWNLOAD_URL=$(echo $RESPONSE | jq -r '.downloadUrl')
curl -L -o ~/tdvx/cvss/cv-corpus-pt.tar.gz "$DOWNLOAD_URL"
```

> O arquivo pesa ~50–70 GB. Use `nohup curl ... &` para rodar em background.

### 7. Extrair o corpus

```bash
tar -xzf ~/tdvx/cvss/cv-corpus-pt.tar.gz -C ~/tdvx/cvss/

# Confirmar estrutura
find ~/tdvx/cvss -name "validated.tsv"
# Esperado: ~/tdvx/cvss/cv-corpus-25.0-.../pt/validated.tsv
```

### 8. Dry-run (valida ambiente antes do treino longo)

```bash
python finetune_commonvoice.py \
  --language pt \
  --cvss-dir ./cvss \
  --max-train-samples 500 \
  --dry-run
```

### 9. Treino completo

```bash
nohup python finetune_commonvoice.py \
  --language pt \
  --cvss-dir ./cvss \
  --run-name tdv1-cv-pt-v2 \
  --model-name tdv1-pt-proprietario \
  --num-epochs 3 \
  > finetune.log 2>&1 &

# Acompanhar o log
tail -f finetune.log

# Monitorar GPU (outro terminal)
watch -n 2 nvidia-smi
```

### 10. Retomar de checkpoint (se a VM cair)

```bash
python finetune_commonvoice.py \
  --language pt \
  --cvss-dir ./cvss \
  --resume-from-checkpoint auto
```

---

## Opções avançadas

```bash
# LoRA — menos VRAM, treino mais rápido
python finetune_commonvoice.py --language pt --cvss-dir ./cvss --use-lora --batch-size 16

# Desativar augmentação de áudio
python finetune_commonvoice.py --language pt --cvss-dir ./cvss --no-augment

# BFloat16 (A100 / H100)
python finetune_commonvoice.py --language pt --cvss-dir ./cvss --bf16
```

---

## Saída do treino

```
models/
  tdv1-cv-pt/          ← modelo HuggingFace (Transformers)
  tdv1-cv-pt-ct2/      ← modelo CTranslate2 int8 (produção)
```

### Copiar o modelo para o Mac

```bash
# No Mac
scp -r root@<ip-da-vm>:~/tdvx/models/ ./models/
```

### Usar o modelo no engine.py

```python
from faster_whisper import WhisperModel
model = WhisperModel("./models/tdv1-cv-pt-ct2", device="cuda", compute_type="int8_float16")
```

---

## Avaliação com dados sintéticos

Após o treino, avalie o modelo com áudios gerados e diferentes tipos de ruído:

```bash
pip install -r eval/requirements-eval.txt

python eval/synthetic_eval.py \
  --model-path ./models/tdv1-cv-pt-ct2 \
  --save-audio
```

Resultados salvos em `eval/results/` (JSON + CSV + tabela no terminal).

---

## Tempos estimados (RTX 5090)

| Etapa | Tempo |
|---|---|
| Download corpus | 30–90 min |
| Extração tar.gz | 5–15 min |
| Dry-run 500 amostras | ~2 min |
| Treino completo 3 epochs | 2–5h |
