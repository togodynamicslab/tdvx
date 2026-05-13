# TDvX v4 — Motor de Transcrição pt-BR Proprietário

Fine-tuning do `whisper-medium` com:
- **Mozilla Common Voice 25.0** (scripted speech pt-BR)
- **CORAA v1.1** (290h espontâneo pt-BR, HuggingFace)
- **LaPS BM — FalaBrasil** (eval adicional)
- Saída: CTranslate2 int8 pronto para produção em `engine.py`

---

## Requisitos da máquina de treino

| Item | Especificação |
|---|---|
| GPU | 8× NVIDIA RTX 5090 (32 GB VRAM cada = 256 GB total) |
| RAM | 128 GB+ |
| Disco | 300 GB (corpus CV ~70 GB + CORAA ~50 GB + checkpoints) |
| SO | Ubuntu 22.04+ |
| Python | **3.11** (NeMo não suporta 3.12+) |
| CUDA | 12.8+ |

---

## Passo a passo completo

### 1. Dependências do sistema

```bash
sudo apt update && sudo apt install -y \
    python3.11 python3.11-venv python3-pip \
    ffmpeg git wget curl build-essential \
    libsndfile1 libsndfile1-dev jq
```

### 2. Clonar o repositório

```bash
git clone -b finetune https://github.com/togodynamicslab/tdvx.git ~/tdvx
cd ~/tdvx
```

### 3. Ambiente Python 3.11

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 4. PyTorch com CUDA 12.8 (RTX 5090 / Blackwell)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Confirmar GPUs
python -c "import torch; print(torch.cuda.device_count(), 'GPUs —', torch.cuda.get_device_name(0))"
```

### 5. Dependências do projeto

```bash
pip install -r requirements-finetune.txt
pip install librosa jiwer python-dotenv datasets accelerate
```

### 6. Baixar o corpus Common Voice 25.0 pt-BR

```bash
mkdir -p ~/tdvx/cvss
cd ~/tdvx/cvss

# Obter URL de download autenticada
RESPONSE=$(curl -s -X POST \
  "https://mozilladatacollective.com/api/datasets/cmn29f4cb017bmm07pd9yd8mw/download" \
  -H "Authorization: Bearer $CV_API_TOKEN" \
  -H "Content-Type: application/json")

DOWNLOAD_URL=$(echo "$RESPONSE" | jq -r '.downloadUrl')

# Baixar em background (50–70 GB, ~30–90 min)
nohup curl -L -o "Common Voice Scripted Speech 25.0 - Portuguese.tar.gz" \
  "$DOWNLOAD_URL" > ~/tdvx/cv_download.log 2>&1 &

echo "Download rodando em background. Acompanhe: tail -f ~/tdvx/cv_download.log"
```

> `CV_API_TOKEN` — token da Mozilla Data Collective.
> Exporte antes: `export CV_API_TOKEN=seu_token_aqui`

### 7. Aguardar o download e verificar

```bash
# Ver progresso
tail -f ~/tdvx/cv_download.log

# Confirmar arquivo
ls -lh ~/tdvx/cvss/
# Esperado: "Common Voice Scripted Speech 25.0 - Portuguese.tar.gz" ~50–70 GB
```

> A **extração acontece automaticamente** quando o treino inicia. Não precisa extrair manualmente.

### 8. Configurar token HuggingFace (para CORAA)

```bash
# Crie um token em https://huggingface.co/settings/tokens (papel: read)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Ou salve no .env para persistir entre sessões
echo "HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx" >> ~/tdvx/.env
```

### 9. Credenciais do Google Drive (OAuth2 — recomendado)

> **Service Account não funciona** para upload ao Drive pessoal (sem quota).
> Use OAuth2 com suas credenciais pessoais.

**Passo 1 — Criar credenciais OAuth2 no Google Cloud Console:**

1. Acesse [console.cloud.google.com](https://console.cloud.google.com)
2. APIs & Services → **Enable APIs** → ativar **Google Drive API**
3. APIs & Services → **Credentials** → **Create Credentials** → **OAuth 2.0 Client ID**
   - Tipo: **Desktop app** → OK
4. Baixar JSON → salvar como `client_secrets.json` (no seu Mac)
5. OAuth consent screen → **Add test user** → seu e-mail do Google

**Passo 2 — Copiar `client_secrets.json` para a VM:**

```bash
# No Mac (ajuste a porta e IP da sua VM)
scp -P 40280 client_secrets.json root@<IP_DA_VM>:~/tdvx/
```

**Passo 3 — Gerar `token.json` na VM (modo headless, sem browser):**

```bash
cd ~/tdvx
source /venv/main/bin/activate   # ou .venv se existir
pip install google-auth-oauthlib google-auth-httplib2

python auth_gdrive.py --secrets client_secrets.json --console
# Imprime uma URL → abra no seu browser → autorize → copie o código → cole no terminal
```

O arquivo `token.json` será salvo em `~/tdvx/token.json`.

**Passo 4 — Verificar o upload antes de rodar o treino:**

```bash
python test_gdrive_upload.py \
  --folder-id SEU_FOLDER_ID \
  --credentials token.json
```

### 10. Dry-run (valida ambiente antes do treino longo)

```bash
cd ~/tdvx
source .venv/bin/activate

python finetune_commonvoice.py \
  --language pt \
  --cvss-dir ./cvss \
  --max-train-samples 500 \
  --dry-run
```

> Deve terminar sem erro em ~2 min. Confirma extração do corpus + pré-tokenização.

### 11. Treino completo TDvX v4 (8× RTX 5090)

```bash
cd ~/tdvx
source .venv/bin/activate

nohup torchrun --nproc_per_node=8 finetune_commonvoice.py \
  --language pt \
  --cvss-dir ./cvss \
  --coraa \
  --lapsbm-eval \
  --no-gradient-checkpointing \
  --torch-compile \
  --run-name tdvx-v4 \
  --model-name tdvx-v4-pt-proprietario \
  --num-epochs 3 \
  --convert-ct2 \
  --gdrive-folder-id SEU_FOLDER_ID_DO_DRIVE \
  --gdrive-credentials token.json \
  > finetune.log 2>&1 &

echo "Treino iniciado (PID $!). Acompanhe: tail -f finetune.log"
```

> **FOLDER_ID** = parte final da URL da pasta no Drive:
> `https://drive.google.com/drive/folders/` **`1AbCdEfGhIjKl...`**

```bash
# Acompanhar o log
tail -f ~/tdvx/finetune.log

# Monitorar GPUs (outro terminal)
watch -n 2 nvidia-smi
```

### 12. Retomar de checkpoint (se a VM cair)

```bash
nohup torchrun --nproc_per_node=8 finetune_commonvoice.py \
  --language pt \
  --cvss-dir ./cvss \
  --coraa \
  --lapsbm-eval \
  --no-gradient-checkpointing \
  --run-name tdvx-v4 \
  --gdrive-folder-id SEU_FOLDER_ID_DO_DRIVE \
  --gdrive-credentials token.json \
  --resume-from-checkpoint auto \
  > finetune_resume.log 2>&1 &
```

---

## O que acontece automaticamente durante o treino

| Etapa | Detalhe |
|---|---|
| Extração CV | Descomprime o `.tar.gz` em `cvss/` na primeira execução |
| CORAA download | HuggingFace baixa e cacheia em `~/.cache/huggingface/` |
| LaPS BM download | Igual, usado só para eval |
| Treino | DDP em 8 GPUs, BF16 automático, batch efetivo 512 |
| Conversão CT2 | `ct2-transformers-converter` → `models/tdvx-v4-cv-pt-ct2/` |
| Upload Drive | Modelo HF + CT2 enviados para a pasta configurada |

---

## Saída do treino

```
models/
  tdvx-v4-cv-pt/          ← modelo HuggingFace (Transformers)
  tdvx-v4-cv-pt-ct2/      ← modelo CTranslate2 int8 (produção → engine.py)
```

No Google Drive:
- `tdvx-v4-hf/` — modelo HuggingFace completo
- `tdvx-v4-ct2/` — modelo CTranslate2 int8

---

## Copiar o modelo para uso local

```bash
# No Mac / máquina local
scp -r usuario@<ip-da-vm>:~/tdvx/models/tdvx-v4-cv-pt-ct2 ./tdvx/models/
```

---

## Tempos estimados (8× RTX 5090)

| Etapa | Tempo estimado |
|---|---|
| Download Common Voice | 30–90 min |
| Download CORAA (HuggingFace) | 60–180 min |
| Extração tar.gz | 5–15 min |
| Dry-run 500 amostras | ~2 min |
| Treino completo 3 epochs | 1–3 h |
| Conversão CTranslate2 | ~5 min |
| Upload para Google Drive | 20–60 min |

---

## Opções avançadas

```bash
# Limitar amostras do CORAA (teste rápido)
torchrun --nproc_per_node=8 finetune_commonvoice.py \
  --language pt --coraa --max-coraa-samples 10000

# LoRA — menos VRAM (não necessário com 5090, mas possível)
python finetune_commonvoice.py --language pt --use-lora --batch-size 16

# Sem augmentação de áudio (treino mais rápido, menos robusto)
python finetune_commonvoice.py --language pt --no-augment
```
