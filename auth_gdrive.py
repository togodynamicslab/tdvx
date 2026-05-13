#!/usr/bin/env python3
"""
auth_gdrive.py — Autoriza acesso ao Google Drive via OAuth2.

Uso local (abre o browser automaticamente):
    python auth_gdrive.py --secrets client_secrets.json

Uso remoto / VM sem browser (imprime URL, você cola o código):
    python auth_gdrive.py --secrets client_secrets.json --console

O token.json gerado substitui o gdrive_credentials.json no comando de treino:
    --gdrive-credentials token.json

Pré-requisitos no Google Cloud Console:
  1. APIs & Services → Enable → "Google Drive API"
  2. APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID
     Tipo: "Desktop app"  →  baixar JSON  →  salvar como client_secrets.json
  3. OAuth consent screen → Add test user: seu e-mail
"""
import argparse
import sys
from pathlib import Path

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
except ImportError:
    print("Instale: pip install google-auth-oauthlib google-auth-httplib2")
    sys.exit(1)

SCOPES = ["https://www.googleapis.com/auth/drive"]
PORT   = 8080


def _refresh_if_needed(token_path: Path) -> bool:
    """Tenta renovar token existente. Retorna True se ainda válido após renovação."""
    if not token_path.exists():
        return False
    try:
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds.valid:
            print(f"Token válido encontrado: {token_path} — nada a fazer.")
            return True
        if creds.expired and creds.refresh_token:
            print("Token expirado — renovando ...")
            creds.refresh(Request())
            token_path.write_text(creds.to_json(), encoding="utf-8")
            print(f"Token renovado: {token_path}")
            return True
    except Exception as exc:
        print(f"Token existente inválido ({exc}) — reautorizando ...")
    return False


def run_local(secrets_path: Path, token_path: Path) -> None:
    """Abre o browser local e aguarda o callback em localhost:8080."""
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), SCOPES)
    print()
    print("Abrindo browser para autorização ...")
    print("(se não abrir, copie a URL impressa no terminal)")
    print()
    creds = flow.run_local_server(port=PORT, open_browser=True, prompt="consent")
    token_path.write_text(creds.to_json(), encoding="utf-8")
    _print_success(token_path)


def run_console(secrets_path: Path, token_path: Path) -> None:
    """
    Fluxo headless: imprime a URL de autorização e pede o código.
    Funciona em VMs remotas sem browser.
    """
    flow = InstalledAppFlow.from_client_secrets_file(
        str(secrets_path),
        SCOPES,
        redirect_uri="urn:ietf:wg:oauth:2.0:oob",
    )
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )

    print()
    print("=" * 70)
    print("Abra esta URL no seu navegador (pode ser em qualquer máquina):")
    print()
    print(auth_url)
    print()
    print("Autorize o acesso e copie o código exibido na página.")
    print("=" * 70)
    print()

    code = input("Cole o código aqui e pressione Enter: ").strip()
    flow.fetch_token(code=code)
    creds = flow.credentials
    token_path.write_text(creds.to_json(), encoding="utf-8")
    _print_success(token_path)


def _print_success(token_path: Path) -> None:
    print()
    print(f"Token salvo em: {token_path}")
    print()
    print("Use no treino com:")
    print(f"  --gdrive-credentials {token_path.name}")
    print()
    print("Teste o upload antes de rodar o treino completo:")
    print(f"  python test_gdrive_upload.py --folder-id SEU_FOLDER_ID --credentials {token_path.name}")


def main():
    p = argparse.ArgumentParser(
        description="Autoriza Google Drive via OAuth2 (local ou headless/VM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--secrets", default="client_secrets.json",
                   help="JSON do OAuth2 Client ID (baixado do Google Cloud Console)")
    p.add_argument("--token",   default="token.json",
                   help="Onde salvar o token OAuth2 gerado")
    p.add_argument("--console", action="store_true",
                   help="Modo headless: imprime URL e pede o código (para VMs sem browser)")
    p.add_argument("--force",   action="store_true",
                   help="Força nova autorização mesmo que o token exista e seja válido")
    args = p.parse_args()

    secrets_path = Path(args.secrets)
    token_path   = Path(args.token)

    if not secrets_path.exists():
        print(f"Arquivo não encontrado: {secrets_path}")
        print()
        print("Para criar:")
        print("  1. console.cloud.google.com → APIs & Services → Credentials")
        print("  2. Create Credentials → OAuth 2.0 Client ID → Desktop app")
        print("  3. Baixar JSON → salvar como client_secrets.json")
        sys.exit(1)

    if not args.force and _refresh_if_needed(token_path):
        return

    if args.console:
        run_console(secrets_path, token_path)
    else:
        run_local(secrets_path, token_path)


if __name__ == "__main__":
    main()
