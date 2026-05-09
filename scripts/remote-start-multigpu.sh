#!/usr/bin/env bash
# Launch N uvicorn workers per GPU + nginx load balancer on a Vast.ai 8x GPU box.
#
# Ports: 18000+100*gpu + slot  → e.g. GPU 0 slot 0..2 = 18000/18001/18002,
#                                    GPU 1 slot 0..2 = 18100/18101/18102, etc.
# That keeps them contiguous-per-GPU (easier to debug) and leaves room if we
# ever go above 100 slots per GPU.
set -u

APP_DIR=/workspace/tdvx
APP_MODULE=app.main:app
HOST=127.0.0.1
NUM_GPUS=8
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"  # override via env; batching makes >1 redundant

export HF_HOME=/root/.cache/huggingface
mkdir -p "$HF_HOME"

gpu_port() {  # args: gpu slot
    echo $((18000 + 100 * $1 + $2))
}

echo "== killing old uvicorn workers (incl. orphaned zombies) =="
pkill -9 -f 'uvicorn app.main' 2>/dev/null || true
sleep 1
# Force-kill anything bound to the port ranges we use. Covers the legacy
# 18000..18007 single-worker-per-GPU layout too.
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    for slot in $(seq 0 9); do  # up to 10 slots/GPU just in case
        port=$(gpu_port "$gpu" "$slot")
        fuser -k -9 "$port/tcp" 2>/dev/null || true
    done
done
# Legacy cleanup for 18000..18007 layout (safe no-op if ports already freed).
for p in 18000 18001 18002 18003 18004 18005 18006 18007; do
    fuser -k -9 "$p/tcp" 2>/dev/null || true
done
sleep 2

cd "$APP_DIR" || { echo "ERROR: $APP_DIR not found"; exit 1; }

TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
echo "== launching $TOTAL_WORKERS workers ($NUM_GPUS GPUs × $WORKERS_PER_GPU per GPU) =="
UPSTREAM_LINES=""
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    for slot in $(seq 0 $((WORKERS_PER_GPU - 1))); do
        port=$(gpu_port "$gpu" "$slot")
        log=/workspace/tdvx-worker-g${gpu}-s${slot}.log

        nohup setsid env \
            CUDA_VISIBLE_DEVICES=$gpu \
            HF_HOME=/root/.cache/huggingface \
            OMP_NUM_THREADS=4 \
            WORKER_TAG="g${gpu}-s${slot}" \
            uvicorn "$APP_MODULE" --host "$HOST" --port "$port" --workers 1 \
            >"$log" 2>&1 < /dev/null &

        pid=$!
        disown "$pid" 2>/dev/null || true
        echo "  launched PID $pid for GPU $gpu slot $slot on port $port"
        UPSTREAM_LINES+="    server 127.0.0.1:${port} max_fails=0 fail_timeout=0;"$'\n'
    done
done

sleep 5
echo "== live uvicorn processes =="
pgrep -af 'uvicorn app.main' | wc -l | awk '{print "  count: "$1}'
pgrep -af 'uvicorn app.main' | head -3

echo ""
echo "== configuring nginx on :8000 =="
if ! command -v nginx >/dev/null; then
    apt-get update -qq && apt-get install -y -qq nginx
fi
rm -f /etc/nginx/sites-enabled/default

if grep -q 'worker_connections' /etc/nginx/nginx.conf; then
    sed -i -E 's/^[[:space:]]*worker_connections[[:space:]]+[0-9]+;/\tworker_connections 8192;/' /etc/nginx/nginx.conf
fi
if ! grep -q 'worker_rlimit_nofile' /etc/nginx/nginx.conf; then
    sed -i '1a worker_rlimit_nofile 16384;' /etc/nginx/nginx.conf
fi
ulimit -n 16384 2>/dev/null || true

# Build the upstream block with all $TOTAL_WORKERS backends.
cat > /etc/nginx/sites-enabled/tdvx <<NGINX_EOF
map \$http_upgrade \$connection_upgrade {
    default upgrade;
    ''      close;
}

upstream tdvx_backends {
    least_conn;
${UPSTREAM_LINES}
    keepalive 128;
    keepalive_requests 10000;
    keepalive_timeout 60s;
}

server {
    listen 8000 default_server;
    listen [::]:8000 default_server;
    server_name _;

    access_log /workspace/nginx-access.log;
    error_log  /workspace/nginx-error.log warn;

    client_max_body_size 200m;
    gzip off;
    proxy_buffering off;
    proxy_request_buffering off;

    location / {
        proxy_pass http://tdvx_backends;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_connect_timeout 300s;
        proxy_next_upstream off;
        proxy_ignore_client_abort on;
    }

    location /ws/ {
        proxy_pass http://tdvx_backends;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \$connection_upgrade;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_connect_timeout 60s;
    }
}
NGINX_EOF

nginx -t && { pkill -9 nginx 2>/dev/null; sleep 1; nginx; }
echo "== nginx status =="
pgrep -af nginx | head -3 || echo "WARNING: nginx not running"

echo ""
echo "== done. server live at :8000 ($TOTAL_WORKERS backends) =="
