#!/usr/bin/env python3
"""
run_finetune_with_mlflow.py — Wrapper para executar fine-tuning com MLflow tracking
=====================================================================================

Este script facilita a execução do fine-tuning do TDv1 com rastreamento
automático no MLflow, seja em ambiente local ou dentro do container Docker.

Uso:

1. Executar localmente (conectando ao MLflow no container):
   python run_finetune_with_mlflow.py --sprint 4 --base-model openai/whisper-medium

2. Executar dentro do container Docker:
   docker-compose exec tdvx python /app/run_finetune_with_mlflow.py --sprint 4

3. Ver experimentos no MLflow UI:
   http://localhost:5000

Recursos:
- Configuração automática do MLflow tracking URI
- Nomenclatura padronizada de runs (tdv1-sprint-X)
- Registro automático no Model Registry
- Logs estruturados para debugging
- Validação de pré-requisitos (dataset, containers)

Dependências:
    pip install -r requirements-finetune.txt
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Configuração de logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("run_finetune")


# ─────────────────────────────────────────────────────────────────────────────
# Configuração de ambiente
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent
_TDVX_DIR = _ROOT / "tdvx"
_FINETUNE_SCRIPT = _TDVX_DIR / "finetune.py"
_DATA_DIR = _TDVX_DIR / "finetuning_data"
_MODELS_DIR = _TDVX_DIR / "models"

# MLflow defaults
_DEFAULT_MLFLOW_URI = "http://localhost:5000"
_DEFAULT_EXPERIMENT = "tdv1-finetune"
_DEFAULT_MODEL_REGISTRY_NAME = "tdv1-pt-en"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def check_mlflow_server(uri: str) -> bool:
    """Verifica se o servidor MLflow está acessível."""
    try:
        import requests
        response = requests.get(f"{uri}/health", timeout=5)
        return response.status_code == 200
    except Exception as exc:
        log.warning("MLflow não acessível em %s: %s", uri, exc)
        return False


def check_docker_container(container_name: str) -> bool:
    """Verifica se um container Docker está rodando."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return container_name in result.stdout
    except FileNotFoundError:
        return False


def start_mlflow_containers() -> bool:
    """Inicia os containers do MLflow via docker-compose."""
    compose_file = _TDVX_DIR / "docker-compose.yml"
    if not compose_file.exists():
        log.error("docker-compose.yml não encontrado em %s", _TDVX_DIR)
        return False

    log.info("Iniciando containers MLflow + PostgreSQL ...")
    try:
        subprocess.run(
            ["docker-compose", "-f", str(compose_file), "up", "-d", "mlflow", "postgres"],
            check=True,
            cwd=str(_TDVX_DIR),
        )
        log.info("Containers iniciados. Aguardando 10s para inicialização ...")
        import time
        time.sleep(10)
        return True
    except subprocess.CalledProcessError as exc:
        log.error("Falha ao iniciar containers: %s", exc)
        return False


def validate_dataset() -> bool:
    """Valida se o dataset de fine-tuning existe."""
    manifest = _DATA_DIR / "manifest.jsonl"
    if not manifest.exists():
        log.error(
            "Dataset não encontrado: %s\n"
            "Execute o TDv1 com FINETUNING_ENABLED=true para coletar dados primeiro.",
            manifest,
        )
        return False

    # conta linhas
    with open(manifest, encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())

    log.info("Dataset encontrado: %d amostras em %s", count, manifest)
    if count < 10:
        log.warning(
            "Dataset muito pequeno (%d amostras). Recomendado: >100 para fine-tuning efetivo.",
            count,
        )
    return True


def build_finetune_command(args: argparse.Namespace, mlflow_uri: str) -> List[str]:
    """Constrói o comando para executar finetune.py com os parâmetros corretos."""
    cmd = [sys.executable, str(_FINETUNE_SCRIPT)]

    # Localização
    cmd.extend(["--data-dir", str(_DATA_DIR)])
    output_dir = _MODELS_DIR / f"tdv1-sprint-{args.sprint}"
    cmd.extend(["--output-dir", str(output_dir)])

    # Modelo base
    cmd.extend(["--base-model", args.base_model])

    # Idiomas
    if args.languages:
        cmd.extend(["--languages"] + args.languages)

    # Treino
    if args.max_steps:
        cmd.extend(["--max-steps", str(args.max_steps)])
    else:
        cmd.extend(["--num-epochs", str(args.num_epochs)])

    cmd.extend(["--batch-size", str(args.batch_size)])
    cmd.extend(["--learning-rate", str(args.learning_rate)])
    cmd.extend(["--eval-steps", str(args.eval_steps)])

    # Qualidade do dataset
    cmd.extend(["--min-confidence", str(args.min_confidence)])

    # MLflow
    cmd.extend(["--mlflow-uri", mlflow_uri])
    cmd.extend(["--mlflow-experiment", args.mlflow_experiment])

    run_name = f"tdv1-sprint-{args.sprint}"
    if args.run_suffix:
        run_name += f"-{args.run_suffix}"
    cmd.extend(["--run-name", run_name])

    # Model Registry
    if args.register_model:
        model_name = args.model_name or _DEFAULT_MODEL_REGISTRY_NAME
        cmd.extend(["--model-name", model_name])

    # CTranslate2
    if args.convert_ct2:
        cmd.append("--convert-ct2")
        cmd.extend(["--ct2-quantization", args.ct2_quantization])

    # HuggingFace Hub
    if args.hub_model_id:
        cmd.extend(["--hub-model-id", args.hub_model_id])

    # Dry run
    if args.dry_run:
        cmd.append("--dry-run")

    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> int:
    """Executa o pipeline de fine-tuning com MLflow tracking."""

    log.info("═" * 70)
    log.info("TDv1 Fine-tuning com MLflow Tracking")
    log.info("Sprint: %d | Modelo base: %s", args.sprint, args.base_model)
    log.info("═" * 70)

    # ── 1. Validação de pré-requisitos ────────────────────────────────────────
    if not _FINETUNE_SCRIPT.exists():
        log.error("Script de fine-tuning não encontrado: %s", _FINETUNE_SCRIPT)
        return 1

    if not validate_dataset():
        return 1

    # ── 2. MLflow setup ───────────────────────────────────────────────────────
    mlflow_uri = args.mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI") or _DEFAULT_MLFLOW_URI

    # Se não estiver rodando dentro do container, verifica se o MLflow está up
    is_in_container = os.path.exists("/.dockerenv")
    if not is_in_container:
        log.info("Verificando servidor MLflow em %s ...", mlflow_uri)
        if not check_mlflow_server(mlflow_uri):
            log.warning("MLflow não acessível.")

            # Tenta iniciar via docker-compose
            if args.auto_start_mlflow:
                log.info("Tentando iniciar containers MLflow automaticamente ...")
                if not start_mlflow_containers():
                    log.error("Falha ao iniciar MLflow. Inicie manualmente:")
                    log.error("  cd %s && docker-compose up -d mlflow postgres", _TDVX_DIR)
                    return 1

                # Revalida
                if not check_mlflow_server(mlflow_uri):
                    log.error("MLflow ainda não está acessível após iniciar containers.")
                    return 1
            else:
                log.warning(
                    "Use --auto-start-mlflow para iniciar automaticamente ou execute:\n"
                    "  cd %s && docker-compose up -d mlflow postgres",
                    _TDVX_DIR,
                )
                log.info("Continuando sem MLflow tracking ...")
                mlflow_uri = ""

        if mlflow_uri:
            log.info("✓ MLflow servidor acessível")
            log.info("  UI: %s", mlflow_uri)

    # ── 3. Constrói comando de fine-tuning ────────────────────────────────────
    cmd = build_finetune_command(args, mlflow_uri)

    log.info("─" * 70)
    log.info("Comando de fine-tuning:")
    log.info("  %s", " ".join(cmd))
    log.info("─" * 70)

    if args.print_command:
        return 0

    # ── 4. Executa fine-tuning ────────────────────────────────────────────────
    log.info("Iniciando fine-tuning ...")
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            log.error("Fine-tuning falhou com código %d", result.returncode)
            return result.returncode

        log.info("✓ Fine-tuning concluído com sucesso!")

        # ── 5. Pós-processamento ──────────────────────────────────────────────
        if mlflow_uri and not args.dry_run:
            log.info("─" * 70)
            log.info("📊 Veja os resultados no MLflow UI:")
            log.info("   %s/#/experiments", mlflow_uri)
            if args.register_model:
                model_name = args.model_name or _DEFAULT_MODEL_REGISTRY_NAME
                log.info("   Model Registry: %s/#/models/%s", mlflow_uri, model_name)
            log.info("─" * 70)

        return 0

    except KeyboardInterrupt:
        log.warning("\nFine-tuning interrompido pelo usuário.")
        return 130
    except Exception as exc:
        log.exception("Erro inesperado: %s", exc)
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrapper para fine-tuning do TDv1 com MLflow tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Identificação da sprint
    parser.add_argument(
        "--sprint", type=int, required=True,
        help="Número da sprint (usado para nomear run e modelo: tdv1-sprint-N)",
    )
    parser.add_argument(
        "--run-suffix", default="",
        help="Sufixo adicional para o nome do run MLflow (ex.: --run-suffix retrain)",
    )

    # Modelo base
    parser.add_argument(
        "--base-model", default="openai/whisper-medium",
        help="Modelo Whisper base para fine-tuning",
    )

    # Idiomas
    parser.add_argument(
        "--languages", nargs="+", default=["portuguese", "english"],
        help="Idiomas para treinar (ex.: --languages pt en)",
    )

    # Treino
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="Número de steps de treino (0 = usa --num-epochs)",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=3,
        help="Número de épocas (ignorado se --max-steps > 0)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size por dispositivo",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5,
        help="Taxa de aprendizado",
    )
    parser.add_argument(
        "--eval-steps", type=int, default=200,
        help="Frequência de avaliação e checkpoint",
    )

    # Dataset
    parser.add_argument(
        "--min-confidence", type=float, default=0.40,
        help="Confiança mínima do Whisper para aceitar amostra",
    )

    # MLflow
    parser.add_argument(
        "--mlflow-uri", default="",
        help=f"URI do servidor MLflow (padrão: {_DEFAULT_MLFLOW_URI})",
    )
    parser.add_argument(
        "--mlflow-experiment", default=_DEFAULT_EXPERIMENT,
        help="Nome do experimento MLflow",
    )
    parser.add_argument(
        "--auto-start-mlflow", action="store_true", default=True,
        help="Inicia containers MLflow automaticamente se não estiverem rodando",
    )
    parser.add_argument(
        "--no-auto-start-mlflow", dest="auto_start_mlflow", action="store_false",
        help="Desabilita início automático do MLflow",
    )

    # Model Registry
    parser.add_argument(
        "--register-model", action="store_true", default=True,
        help="Registra modelo no MLflow Model Registry",
    )
    parser.add_argument(
        "--no-register-model", dest="register_model", action="store_false",
        help="Não registra no Model Registry",
    )
    parser.add_argument(
        "--model-name", default="",
        help=f"Nome do modelo no Registry (padrão: {_DEFAULT_MODEL_REGISTRY_NAME})",
    )

    # CTranslate2
    parser.add_argument(
        "--convert-ct2", action="store_true", default=True,
        help="Converte para CTranslate2 após treino",
    )
    parser.add_argument(
        "--no-convert-ct2", dest="convert_ct2", action="store_false",
        help="Não converte para CTranslate2",
    )
    parser.add_argument(
        "--ct2-quantization", default="int8_float16",
        choices=["int8", "int8_float16", "float16", "int4"],
        help="Quantização do CTranslate2",
    )

    # HuggingFace Hub
    parser.add_argument(
        "--hub-model-id", default="",
        help="Publica no HF Hub (ex.: minha-empresa/tdv1-pt-en-v2)",
    )

    # Utilitários
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Valida dataset sem executar treino",
    )
    parser.add_argument(
        "--print-command", action="store_true",
        help="Imprime comando e sai (para debug)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
