#!/usr/bin/env python3
"""
test_gdrive_upload.py — Testa o upload para Google Drive sem precisar treinar.

Cria uma pasta temporária com arquivos dummy (simula a estrutura de um modelo),
faz upload para o Drive e mostra o link final.

Uso:
    python test_gdrive_upload.py \
        --folder-id  SEU_FOLDER_ID_AQUI \
        --credentials gdrive-key.json
"""
import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    from google.oauth2 import service_account as _gdrive_sa
    from google.oauth2.credentials import Credentials as _OAuthCredentials
    from googleapiclient.discovery import build as _gdrive_build
    from googleapiclient.http import MediaFileUpload as _MediaFileUpload
except ImportError:
    print("ERRO: instale as dependencias:\n  pip install google-api-python-client google-auth")
    sys.exit(1)


def _build_drive_service(credentials_path: Path):
    """Detecta automaticamente se é token OAuth2 ou Service Account."""
    import json
    data = json.loads(credentials_path.read_text(encoding="utf-8"))
    _SCOPES = ["https://www.googleapis.com/auth/drive"]

    if data.get("type") == "service_account":
        creds = _gdrive_sa.Credentials.from_service_account_file(
            str(credentials_path), scopes=_SCOPES
        )
    else:
        # token.json gerado pelo auth_gdrive.py
        creds = _OAuthCredentials.from_authorized_user_file(str(credentials_path), _SCOPES)
        if creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())

    return _gdrive_build("drive", "v3", credentials=creds, cache_discovery=False)


def upload_dir_to_gdrive(
    local_dir: Path,
    parent_folder_id: str,
    credentials_path: Path,
    folder_name: Optional[str] = None,
) -> Optional[str]:
    _CHUNK_SIZE = 100 * 1024 * 1024

    try:
        service = _build_drive_service(credentials_path)
    except Exception as exc:
        log.error("Falha ao autenticar: %s", exc)
        return None

    folder_name = folder_name or local_dir.name

    def _make_folder(name: str, parent_id: str) -> str:
        meta = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        return service.files().create(body=meta, fields="id").execute()["id"]

    def _upload_file(file_path: Path, parent_id: str) -> None:
        size_mb = file_path.stat().st_size / 1e6
        log.info("  Enviando %-40s (%.1f MB) ...", file_path.name, size_mb)
        media   = _MediaFileUpload(str(file_path), resumable=True, chunksize=_CHUNK_SIZE)
        request = service.files().create(
            body={"name": file_path.name, "parents": [parent_id]},
            media_body=media,
            fields="id",
        )
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                log.info("    %d%%", int(status.progress() * 100))
        log.info("  ✓ %s", file_path.name)

    def _upload_dir(dir_path: Path, parent_id: str) -> None:
        for item in sorted(dir_path.iterdir()):
            if item.is_dir():
                sub_id = _make_folder(item.name, parent_id)
                _upload_dir(item, sub_id)
            elif item.is_file():
                _upload_file(item, parent_id)

    try:
        log.info("Criando pasta '%s' no Drive ...", folder_name)
        drive_folder_id = _make_folder(folder_name, parent_folder_id)
        drive_url       = f"https://drive.google.com/drive/folders/{drive_folder_id}"
        log.info("Upload iniciado → %s", drive_url)
        _upload_dir(local_dir, drive_folder_id)
        log.info("Upload concluído → %s", drive_url)
        return drive_url
    except Exception as exc:
        log.error("Falha no upload: %s", exc)
        return None


def create_dummy_model(base_dir: Path) -> Path:
    """Cria estrutura de diretório que imita um modelo HuggingFace salvo."""
    model_dir = base_dir / "test-model-hf"
    model_dir.mkdir(parents=True)

    # Arquivos que um modelo Whisper real teria (conteúdo fictício para o teste)
    files = {
        "config.json":            b'{"model_type": "whisper", "test": true}',
        "tokenizer_config.json":  b'{"tokenizer_class": "WhisperTokenizer"}',
        "vocab.json":             b'{"<pad>": 0, "<s>": 1}',
        "merges.txt":             b"#version: 0.2\n",
        "preprocessor_config.json": b'{"feature_size": 80}',
        # Simula um arquivo grande (5 MB) representando os pesos
        "model.safetensors":      b"\x00" * 5 * 1024 * 1024,
    }

    for name, content in files.items():
        (model_dir / name).write_bytes(content)

    # Subdiretório (como acontece com alguns tokenizers)
    special = model_dir / "special_tokens_map.json"
    special.write_bytes(b'{"bos_token": "<s>"}')

    total_mb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 1e6
    print(f"\nPasta de teste criada: {model_dir}")
    print(f"Arquivos: {list(f.name for f in model_dir.iterdir())}")
    print(f"Tamanho total: {total_mb:.1f} MB\n")
    return model_dir


def main():
    p = argparse.ArgumentParser(description="Teste de upload para Google Drive")
    p.add_argument("--folder-id",    required=True,
                   help="ID da pasta destino no Drive (da URL /folders/<ID>)")
    p.add_argument("--credentials",  required=True,
                   help="Caminho para o JSON da Service Account")
    p.add_argument("--folder-name",  default="test-gdrive-upload",
                   help="Nome da subpasta a criar no Drive")
    args = p.parse_args()

    creds_path = Path(args.credentials).resolve()
    if not creds_path.exists():
        print(f"ERRO: arquivo de credenciais não encontrado: {creds_path}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = create_dummy_model(Path(tmp))

        print("Iniciando upload para Google Drive ...")
        url = upload_dir_to_gdrive(
            local_dir=model_dir,
            parent_folder_id=args.folder_id,
            credentials_path=creds_path,
            folder_name=args.folder_name,
        )

    if url:
        print(f"\nOK Upload concluido com sucesso!")
        print(f"  Acesse: {url}")
    else:
        print("\nFALHA Upload falhou - veja os logs acima para detalhes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
