#!/bin/bash
# Re-arm the self-stop timer on a running instance.
# Needed after resume.sh, since 'at' jobs don't survive stop/start cycles.
# Usage: ./scripts/arm_stop.sh [hours]

source "$(dirname "$0")/_common.sh"

require_instance
HOURS="${1:-12}"

VAST_API_KEY="${VAST_API_KEY:-}"
for keyfile in "$HOME/.config/vastai/vast_api_key" "$HOME/.vastai/vast_api_key"; do
    if [[ -z "$VAST_API_KEY" && -f "$keyfile" ]]; then
        VAST_API_KEY="$(cat "$keyfile")"
    fi
done
[[ -n "$VAST_API_KEY" ]] || die "VAST_API_KEY missing. Expected env var or ~/.config/vastai/vast_api_key."

info "Arming self-stop in ${HOURS}h on instance $INSTANCE_ID..."

ssh_cmd "bash -s" <<EOF
set -e
command -v at >/dev/null || apt-get install -y at
service atd start 2>/dev/null || true
mkdir -p /workspace

cat > /workspace/self_stop.sh <<'STOP'
#!/bin/bash
INSTANCE_ID=\$(cat /etc/vastai_instance_id 2>/dev/null || echo "$INSTANCE_ID")
echo "[\$(date -u +%FT%TZ)] self-stop firing for \$INSTANCE_ID" >> /workspace/self_stop.log
curl -sS -X PUT "https://console.vast.ai/api/v0/instances/\$INSTANCE_ID/" \\
    -H "Authorization: Bearer \$VAST_API_KEY_SELFSTOP" \\
    -H "Content-Type: application/json" \\
    -d '{"state":"stopped"}' >> /workspace/self_stop.log 2>&1
STOP
chmod +x /workspace/self_stop.sh

echo "export VAST_API_KEY_SELFSTOP='$VAST_API_KEY'" > /workspace/.self_stop_env
echo "$INSTANCE_ID" > /etc/vastai_instance_id

# Clear prior 'at' jobs for root (avoid double-stops)
atq | awk '{print \$1}' | xargs -r atrm

echo "[\$(date -u +%FT%TZ)] scheduling self-stop in ${HOURS}h" >> /workspace/self_stop.log
echo "source /workspace/.self_stop_env && /workspace/self_stop.sh" | at now + $HOURS hours
EOF

info "Self-stop armed. Instance will stop in ${HOURS}h."
