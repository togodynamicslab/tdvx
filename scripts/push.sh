#!/bin/bash
# rsync local code to the rented instance and restart uvicorn.
# Usage: ./scripts/push.sh [--no-restart] [--no-deps]

source "$(dirname "$0")/_common.sh"

NO_RESTART=0
for arg in "$@"; do
    case "$arg" in
        --no-restart) NO_RESTART=1 ;;
        --no-deps) ;;  # no-op: deps are baked into the image now
        *) die "Unknown arg: $arg" ;;
    esac
done

refresh_ssh

# Safety net: pull validations + Deepgram refs FIRST, so even if rsync's
# excludes ever drift, we have a fresh local backup before pushing. The
# excludes below should already protect these files, but belt + suspenders.
info "Pulling latest validations + references from server (backup)…"
rsync -az \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
    --include='*/' \
    --include='.validations.json' \
    --include='.reference/' \
    --include='.reference/*.json' \
    --exclude='*' \
    "root@$SSH_HOST:$REMOTE_DIR/tests/corpus/" \
    "$REPO_ROOT/tests/corpus/" 2>/dev/null || info "  (nothing to pull or first run)"

info "Syncing code to $SSH_HOST:$SSH_PORT:$REMOTE_DIR"
rsync -az --delete \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude '.vastai_instance' \
    --exclude '/models/' \
    --exclude '/distillation/student-init/' \
    --exclude '/distillation/distil-whisper-pt-en/' \
    --exclude '/web/node_modules/' \
    --exclude '/tests/corpus/**/.validations.json' \
    --exclude '/tests/corpus/**/.reference/' \
    --exclude '/results/' \
    --exclude '/web/dist/' \
    --exclude '*.log' \
    --exclude '.env' \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
    "$REPO_ROOT/" "root@$SSH_HOST:$REMOTE_DIR/"

if [[ -f "$REPO_ROOT/.env" ]]; then
    info "Syncing .env (separately so rsync's --exclude doesn't drop it)..."
    scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
        "$REPO_ROOT/.env" "root@$SSH_HOST:$REMOTE_DIR/.env"
fi

if [[ $NO_RESTART -eq 1 ]]; then
    info "Skipping server restart (--no-restart)."
    exit 0
fi

info "Restarting uvicorn (multi-GPU if launcher present, else single-worker fallback)..."

if ssh_cmd "test -x $REMOTE_DIR/scripts/remote-start-multigpu.sh"; then
    info "Using 8-GPU launcher: $REMOTE_DIR/scripts/remote-start-multigpu.sh"
    ssh_cmd "bash $REMOTE_DIR/scripts/remote-start-multigpu.sh >/workspace/tdvx.log 2>&1" || true
else
    info "Falling back to single-worker uvicorn on :8000 (no multi-GPU launcher found)"
    ssh_cmd "cat > /workspace/restart.sh <<'EOF'
#!/bin/bash
pkill -f 'uvicorn app.main' 2>/dev/null
sleep 1
cd $REMOTE_DIR
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
EOF
chmod +x /workspace/restart.sh"
    ssh_cmd "nohup setsid /workspace/restart.sh >/workspace/tdvx.log 2>&1 </dev/null & sleep 2; pgrep -af 'uvicorn app.main' | head -3" || true
fi

# Extract public endpoint
RAW=$(vastai show instance "$INSTANCE_ID" --raw)
PUBLIC_IP=$(echo "$RAW" | jq -r '.public_ipaddr // empty')
MAPPED_PORT=$(echo "$RAW" | jq -r '.ports["8000/tcp"][0].HostPort // empty')

info ""
info "Pushed. Server endpoints:"
if [[ -n "$PUBLIC_IP" && -n "$MAPPED_PORT" ]]; then
    info "  http://$PUBLIC_IP:$MAPPED_PORT"
    info "  ws://$PUBLIC_IP:$MAPPED_PORT/ws/transcribe"
else
    info "  (direct port mapping not yet visible; try again in 30s or check 'vastai show instance $INSTANCE_ID --raw')"
fi
info "  Tail logs: ./scripts/logs.sh"
