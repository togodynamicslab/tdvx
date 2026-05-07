#!/usr/bin/env python3
"""
auth_gdrive.py — Autoriza acesso ao Google Drive via OAuth2.

Fluxo em dois passos (sem precisar de input interativo):

  Passo 1 — gera a URL de autorizacao:
      python auth_gdrive.py --secrets client_secrets.json

  Passo 2 — abre a URL no browser, autoriza, copia o 'code=' da URL de redirect
  e passa como argumento:
      python auth_gdrive.py --secrets client_secrets.json --code SEU_CODIGO_AQUI

  O token.json sera salvo e usado automaticamente pelo script de upload.
"""
import argparse
import json
import sys
from pathlib import Path

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
except ImportError:
    print("Instale: pip install google-auth-oauthlib")
    sys.exit(1)

SCOPES     = ["https://www.googleapis.com/auth/drive"]
STATE_FILE = Path(__file__).parent / ".auth_state.json"


PORT = 8080


def run_auth_flow(secrets_path: Path, token_path: Path) -> None:
    """Abre o browser, aguarda o callback em localhost:8080 e salva o token."""
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), SCOPES)
    print()
    print("Abrindo o browser para autorizacao...")
    print("(se nao abrir automaticamente, copie a URL que aparece no terminal)")
    print()
    creds = flow.run_local_server(port=PORT, open_browser=True, prompt="consent")
    token_path.write_text(creds.to_json(), encoding="utf-8")
    print()
    print(f"Token salvo em: {token_path}")
    print()
    print("Agora rode o teste de upload:")
    print(f"  python test_gdrive_upload.py --folder-id SEU_FOLDER_ID --credentials {token_path.name}")


def main():
    p = argparse.ArgumentParser(description="Autoriza Google Drive via OAuth2")
    p.add_argument("--secrets", default="client_secrets.json",
                   help="JSON do OAuth2 Client ID (baixado do Google Cloud Console)")
    p.add_argument("--token",   default="token.json",
                   help="Onde salvar o token gerado")
    args = p.parse_args()

    secrets_path = Path(args.secrets)
    token_path   = Path(args.token)

    if not secrets_path.exists():
        print(f"Arquivo nao encontrado: {secrets_path}")
        sys.exit(1)

    # Token ja existe e ainda e valido?
    if token_path.exists() and not args.code:
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds.valid:
            print(f"Token valido encontrado em: {token_path}")
            print("Nada a fazer — o upload ja esta configurado.")
            return
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")
            print(f"Token renovado: {token_path}")
            return

    run_auth_flow(secrets_path, token_path)


if __name__ == "__main__":
    main()
