#!/bin/bash
# start_api.sh — Sobe 8 instâncias da API (uma por GPU) + nginx load balancer
set -e

MODEL_PATH="${MODEL_PATH:-/workspace/tdvx/tdvx/models/tdvx-v1.6/tdvx-v1.6-ct2}"
BASE_PORT=8001
N_GPUS=8
NGINX_PORT=8000
LOG_DIR="logs"

mkdir -p "$LOG_DIR" results

# ── Mata instâncias anteriores ────────────────────────────────────────────────
pkill -f "api_test:app" 2>/dev/null || true
sleep 1

# ── Sobe uma instância por GPU ────────────────────────────────────────────────
echo "Subindo $N_GPUS instâncias da API..."
for i in $(seq 0 $((N_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    CUDA_VISIBLE_DEVICES=$i \
    MODEL_PATH="$MODEL_PATH" \
    COMPUTE_TYPE="int8_float16" \
    nohup uvicorn api_test:app \
        --host 127.0.0.1 \
        --port "$PORT" \
        --workers 1 \
        > "$LOG_DIR/api_gpu${i}.log" 2>&1 &
    echo "  GPU $i → porta $PORT (PID $!)"
done

# ── Configura nginx ───────────────────────────────────────────────────────────
NGINX_CONF="/tmp/tdvx_nginx.conf"

cat > "$NGINX_CONF" << EOF
worker_processes auto;
error_log /tmp/tdvx_nginx_error.log;
pid /tmp/tdvx_nginx.pid;

events { worker_connections 4096; }

http {
    upstream tdvx_api {
        least_conn;
$(for i in $(seq 0 $((N_GPUS - 1))); do
    echo "        server 127.0.0.1:$((BASE_PORT + i));"
done)
    }

    server {
        listen $NGINX_PORT;
        client_max_body_size 100M;
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;

        location / {
            proxy_pass http://tdvx_api;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
        }
    }
}
EOF

# Para nginx anterior se estiver rodando
nginx -s stop -c "$NGINX_CONF" 2>/dev/null || true
sleep 1

nginx -c "$NGINX_CONF"
echo ""
echo "nginx rodando na porta $NGINX_PORT (load balancer → GPUs 0-$((N_GPUS-1)))"

# ── Aguarda todas as instâncias subirem ───────────────────────────────────────
echo ""
echo "Aguardando instâncias ficarem prontas..."
for i in $(seq 0 $((N_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    until curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
        sleep 2
    done
    echo "  GPU $i OK (porta $PORT)"
done

echo ""
echo "Todas prontas! Testando load balancer..."
curl -s http://localhost:$NGINX_PORT/health | python3 -m json.tool
