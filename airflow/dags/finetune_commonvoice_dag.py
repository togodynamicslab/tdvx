"""
finetune_commonvoice_dag.py — Pipeline Airflow para fine-tuning TDv1
=====================================================================

Orquestra o fine-tuning do whisper-medium no Common Voice pt-BR,
rodando em uma VM remota com GPU via SSH.

Fluxo de tasks:
    check_vm_ready
        → download_corpus (sensor: pula se .tar.gz já existe)
            → extract_corpus (sensor: pula se já extraído)
                → validate_corpus
                    → run_dry_run
                        → run_finetune
                            → convert_ct2
                                → register_model_mlflow
                                    → notify_complete

Variáveis Airflow necessárias (Admin > Variables):
    TDVX_VM_HOST       — IP ou hostname da VM com GPU
    TDVX_VM_USER       — usuário SSH (padrão: root)
    TDVX_VM_KEY_PATH   — caminho da chave SSH no container Airflow
    TDVX_VM_PROJECT    — caminho do projeto na VM (padrão: ~/tdvx)

Conexão Airflow necessária (Admin > Connections):
    tdvx_gpu_vm        — tipo SSH, apontando para a VM com GPU

Acionamento manual (UI ou CLI):
    airflow dags trigger finetune_commonvoice \\
      --conf '{"num_epochs": 3, "use_lora": false, "max_train_samples": null}'
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.providers.ssh.hooks.ssh import SSHHook
from airflow.utils.trigger_rule import TriggerRule

# ─────────────────────────────────────────────────────────────────────────────
# Configuração via Airflow Variables (com fallbacks)
# ─────────────────────────────────────────────────────────────────────────────

VM_HOST       = Variable.get("TDVX_VM_HOST",     default_var="localhost")
VM_USER       = Variable.get("TDVX_VM_USER",     default_var="root")
VM_PROJECT    = Variable.get("TDVX_VM_PROJECT",  default_var="~/tdvx")
SSH_CONN_ID   = "tdvx_gpu_vm"

CV_TARBALL    = f"{VM_PROJECT}/cvss/cv-corpus-pt.tar.gz"
CV_CORPUS_DIR = f"{VM_PROJECT}/cvss"
VENV_PYTHON   = f"{VM_PROJECT}/.venv/bin/python"
FINETUNE_SCRIPT = f"{VM_PROJECT}/finetune_commonvoice.py"

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

default_args = {
    "owner": "tdvx",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers Python
# ─────────────────────────────────────────────────────────────────────────────

def _build_finetune_cmd(**context) -> str:
    """Monta o comando de fine-tuning com base nos parâmetros do DAG run."""
    conf = context["dag_run"].conf or {}
    num_epochs        = conf.get("num_epochs", 3)
    use_lora          = conf.get("use_lora", False)
    max_train_samples = conf.get("max_train_samples")
    run_name          = conf.get("run_name", f"tdv1-cv-pt-{context['ds_nodash']}")
    model_name        = conf.get("model_name", "tdv1-pt-proprietario")

    cmd = (
        f"cd {VM_PROJECT} && source .venv/bin/activate && "
        f"nohup python {FINETUNE_SCRIPT} "
        f"  --language pt "
        f"  --cvss-dir {CV_CORPUS_DIR} "
        f"  --run-name {run_name} "
        f"  --model-name {model_name} "
        f"  --num-epochs {num_epochs} "
    )

    if use_lora:
        cmd += "  --use-lora "
    if max_train_samples:
        cmd += f"  --max-train-samples {max_train_samples} "

    cmd += f"> {VM_PROJECT}/finetune.log 2>&1"
    return cmd


def _push_finetune_cmd(**context):
    cmd = _build_finetune_cmd(**context)
    context["ti"].xcom_push(key="finetune_cmd", value=cmd)


def _is_dry_run(**context) -> bool:
    conf = context["dag_run"].conf or {}
    return bool(conf.get("dry_run", False))


# ─────────────────────────────────────────────────────────────────────────────
# DAG
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="finetune_commonvoice",
    description="Fine-tuning Whisper pt-BR via Common Voice na VM GPU",
    schedule=None,                    # somente acionamento manual
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["tdvx", "finetuning", "whisper", "pt-br"],
    params={
        "num_epochs": 3,
        "use_lora": False,
        "max_train_samples": None,
        "dry_run": False,
        "run_name": "",
        "model_name": "tdv1-pt-proprietario",
    },
) as dag:

    # ── 1. Verifica se a VM está acessível e GPU disponível ───────────────────
    check_vm_ready = SSHOperator(
        task_id="check_vm_ready",
        ssh_conn_id=SSH_CONN_ID,
        command="nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
        cmd_timeout=30,
    )

    # ── 2. Verifica se corpus já foi baixado (ShortCircuit pula download) ─────
    check_corpus_exists = SSHOperator(
        task_id="check_corpus_exists",
        ssh_conn_id=SSH_CONN_ID,
        command=f"test -f {CV_TARBALL} && echo 'EXISTS' || echo 'MISSING'",
        cmd_timeout=15,
    )

    # ── 3. Download do corpus Common Voice pt-BR ──────────────────────────────
    download_corpus = SSHOperator(
        task_id="download_corpus",
        ssh_conn_id=SSH_CONN_ID,
        command=f"""
            mkdir -p {CV_CORPUS_DIR}
            RESPONSE=$(curl -sf -X POST \\
                "https://mozilladatacollective.com/api/datasets/cmn29f4cb017bmm07pd9yd8mw/download" \\
                -H "Authorization: Bearer ${{TDVX_CV_TOKEN}}" \\
                -H "Content-Type: application/json")
            DOWNLOAD_URL=$(echo $RESPONSE | python3 -c "import sys,json; print(json.load(sys.stdin)['downloadUrl'])")
            curl -L -o {CV_TARBALL} "$DOWNLOAD_URL"
        """,
        cmd_timeout=7200,   # 2h — arquivo grande
    )

    # ── 4. Verifica se corpus já foi extraído ─────────────────────────────────
    check_extracted = SSHOperator(
        task_id="check_extracted",
        ssh_conn_id=SSH_CONN_ID,
        command=f"find {CV_CORPUS_DIR} -name 'validated.tsv' | head -1",
        cmd_timeout=30,
    )

    # ── 5. Extrai o tar.gz ────────────────────────────────────────────────────
    extract_corpus = SSHOperator(
        task_id="extract_corpus",
        ssh_conn_id=SSH_CONN_ID,
        command=f"tar -xzf {CV_TARBALL} -C {CV_CORPUS_DIR}/",
        cmd_timeout=1800,   # 30 min
    )

    # ── 6. Valida estrutura do corpus ─────────────────────────────────────────
    validate_corpus = SSHOperator(
        task_id="validate_corpus",
        ssh_conn_id=SSH_CONN_ID,
        command=f"""
            TSV=$(find {CV_CORPUS_DIR} -name 'validated.tsv' | head -1)
            if [ -z "$TSV" ]; then
                echo "ERRO: validated.tsv nao encontrado em {CV_CORPUS_DIR}"
                exit 1
            fi
            LINES=$(wc -l < "$TSV")
            echo "Corpus OK — $TSV — $LINES linhas"
        """,
        cmd_timeout=30,
    )

    # ── 7. Dry-run com 500 amostras (valida ambiente antes do treino longo) ───
    run_dry_run = SSHOperator(
        task_id="run_dry_run",
        ssh_conn_id=SSH_CONN_ID,
        command=(
            f"cd {VM_PROJECT} && source .venv/bin/activate && "
            f"python {FINETUNE_SCRIPT} "
            f"  --language pt "
            f"  --cvss-dir {CV_CORPUS_DIR} "
            f"  --max-train-samples 500 "
            f"  --dry-run "
            f"  2>&1 | tail -20"
        ),
        cmd_timeout=600,    # 10 min
    )

    # ── 8. Monta e persiste o comando de treino no XCom ──────────────────────
    build_finetune_cmd = PythonOperator(
        task_id="build_finetune_cmd",
        python_callable=_push_finetune_cmd,
    )

    # ── 9. Treino completo (pode demorar horas) ───────────────────────────────
    run_finetune = SSHOperator(
        task_id="run_finetune",
        ssh_conn_id=SSH_CONN_ID,
        command="{{ ti.xcom_pull(task_ids='build_finetune_cmd', key='finetune_cmd') }}",
        cmd_timeout=43200,  # 12h
    )

    # ── 10. Aguarda conclusão e exibe métricas finais ─────────────────────────
    wait_and_show_metrics = SSHOperator(
        task_id="wait_and_show_metrics",
        ssh_conn_id=SSH_CONN_ID,
        command=f"tail -50 {VM_PROJECT}/finetune.log | grep -E 'WER|wer|eval|Saved|Conclu'",
        cmd_timeout=60,
    )

    # ── 11. Verifica saída do modelo HuggingFace ──────────────────────────────
    check_model_output = SSHOperator(
        task_id="check_model_output",
        ssh_conn_id=SSH_CONN_ID,
        command=f"find {VM_PROJECT}/models -name 'config.json' | head -5",
        cmd_timeout=30,
    )

    # ── 12. Notifica conclusão (log + futuro: Slack/email) ────────────────────
    notify_complete = SSHOperator(
        task_id="notify_complete",
        ssh_conn_id=SSH_CONN_ID,
        command=(
            f"echo '✅ Fine-tuning TDv1 concluído em '$(date '+%Y-%m-%d %H:%M') && "
            f"du -sh {VM_PROJECT}/models/"
        ),
        trigger_rule=TriggerRule.ALL_SUCCESS,
        cmd_timeout=30,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Dependências
    # ─────────────────────────────────────────────────────────────────────────

    (
        check_vm_ready
        >> check_corpus_exists
        >> download_corpus
        >> check_extracted
        >> extract_corpus
        >> validate_corpus
        >> run_dry_run
        >> build_finetune_cmd
        >> run_finetune
        >> wait_and_show_metrics
        >> check_model_output
        >> notify_complete
    )
