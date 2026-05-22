#!/usr/bin/env python3
"""
Conversão CTranslate2 int8 para modelos Whisper fine-tunados.

Contorna o bug de incompatibilidade entre ctranslate2 e transformers>=4.46
onde ctranslate2 passa `dtype=torch.float32` para from_pretrained(), mas
transformers repassa esse kwarg para WhisperForConditionalGeneration.__init__()
que não aceita o parâmetro.

Uso:
    python convert_ct2.py [hf_dir] [ct2_dir]

    Padrão:
        hf_dir  = models/tdvx-v1.5/tdvx-v1.5-hf
        ct2_dir = models/tdvx-v1.5/tdvx-v1.5-ct2
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent

HF_DIR  = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "models" / "tdvx-v1.5" / "tdvx-v1.5-hf"
CT2_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else _ROOT / "models" / "tdvx-v1.5" / "tdvx-v1.5-ct2"

print(f"Modelo HF  : {HF_DIR}")
print(f"Saída CT2  : {CT2_DIR}")

if not HF_DIR.exists():
    print(f"ERRO: pasta não encontrada — {HF_DIR}", file=sys.stderr)
    sys.exit(1)

# ── Patch: remove 'dtype' de kwargs antes de chegar ao __init__ do Whisper ──
import transformers.modeling_utils as _mu

_orig_fn = _mu.PreTrainedModel.from_pretrained.__func__

@classmethod
def _safe_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    kwargs.pop("dtype", None)  # ctranslate2 passa dtype=torch.float32 mas __init__ não aceita
    return _orig_fn(cls, pretrained_model_name_or_path, *args, **kwargs)

_mu.PreTrainedModel.from_pretrained = _safe_from_pretrained
# ─────────────────────────────────────────────────────────────────────────────

from ctranslate2.converters.transformers import TransformersConverter  # noqa: E402

print("\nIniciando conversão int8 ...")
converter = TransformersConverter(
    model_name_or_path=str(HF_DIR),
    low_cpu_mem_usage=True,
)
converter.convert(str(CT2_DIR), quantization="int8", force=True)

print("\nArquivos gerados em", CT2_DIR)
for f in sorted(CT2_DIR.iterdir()):
    size_mb = os.path.getsize(f) / 1024 / 1024
    print(f"  {f.name:40s} {size_mb:7.1f} MB")

print("\nConversão CT2 concluída.")
