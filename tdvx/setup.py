"""
setup.py — Verifica prerequisitos do TDvX v3.

Uso:
    python setup.py          (rode de dentro de tdvx/)
"""
import os
import sys
import subprocess
from pathlib import Path

# .env esta na raiz do repo (um nivel acima de tdvx/)
_ROOT_ENV = Path(__file__).parent.parent / ".env"


def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    print(f"  {'OK' if ok else 'ERRO'} Python {v.major}.{v.minor}.{v.micro} {'(requer 3.10+)' if not ok else ''}")
    return ok


def check_env():
    if _ROOT_ENV.exists():
        token = None
        for line in _ROOT_ENV.read_text(encoding="utf-8").splitlines():
            if line.startswith("PYANNOTE_AUTH_TOKEN="):
                token = line.split("=", 1)[1].strip()
        if token and token not in ("", "your_huggingface_token_here"):
            print(f"  OK  .env encontrado com token configurado ({_ROOT_ENV})")
            return True
        print(f"  AVISO  .env encontrado mas PYANNOTE_AUTH_TOKEN nao configurado")
        print(f"         Edite: {_ROOT_ENV}")
        return False
    print(f"  ERRO  .env nao encontrado em {_ROOT_ENV}")
    print(f"        Execute: cp ../.env.example ../.env  e configure o token")
    return False


def check_ffmpeg():
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        ok = r.returncode == 0
        print(f"  {'OK' if ok else 'ERRO'} FFmpeg {'encontrado' if ok else 'nao encontrado'}")
        if not ok:
            print("       Instale: winget install ffmpeg")
        return ok
    except FileNotFoundError:
        print("  ERRO  FFmpeg nao encontrado — instale: winget install ffmpeg")
        return False


def check_torch():
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "nenhuma"
        print(f"  OK  PyTorch {torch.__version__} | CUDA: {'sim' if cuda else 'nao'} | GPU: {gpu}")
        return True
    except ImportError:
        print("  ERRO  PyTorch nao instalado")
        print("        Execute: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128")
        return False


def check_deps():
    deps = ["faster_whisper", "pyannote.audio", "fastapi", "uvicorn", "webrtcvad", "soundfile", "librosa"]
    all_ok = True
    for dep in deps:
        try:
            __import__(dep)
            print(f"  OK  {dep}")
        except ImportError:
            print(f"  MISS {dep}")
            all_ok = False
    if not all_ok:
        print("\n  Execute: pip install -r ../requirements.txt")
    return all_ok


def main():
    print("=" * 60)
    print("  TDvX v3 — Verificacao de prerequisitos")
    print("=" * 60)

    checks = [
        ("Python 3.10+",  check_python),
        ("Arquivo .env",  check_env),
        ("FFmpeg",        check_ffmpeg),
        ("PyTorch/CUDA",  check_torch),
        ("Dependencias",  check_deps),
    ]

    results = []
    for name, fn in checks:
        print(f"\n{name}:")
        results.append(fn())

    print("\n" + "=" * 60)
    if all(results):
        print("  Tudo OK! Para subir o servidor:")
        print()
        print("    venv_gpu\\Scripts\\activate.bat   (Windows)")
        print("    cd tdvx")
        print("    uvicorn app.main:app --host 0.0.0.0 --port 8000")
    else:
        print("  Alguns checks falharam. Revise as mensagens acima.")
    print("=" * 60)


if __name__ == "__main__":
    main()
